import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .decoder import Decoder
from .encoder import Encoder

from selfModules.embedding import Embedding
from selfModules.perplexity import Perplexity

from utils.functional import kld_coef, parameters_allocation_check, fold


class RVAE_dilated(nn.Module):
    def __init__(self, params, prefix=''):
        super(RVAE_dilated, self).__init__()

        self.params = params

        self.embedding = Embedding(self.params, '', prefix)

        self.encoder = Encoder(self.params)

        self.context_to_mu = nn.Linear(self.params.encoder_rnn_size * 2, self.params.latent_variable_size)
        self.context_to_logvar = nn.Linear(self.params.encoder_rnn_size * 2, self.params.latent_variable_size)

        self.decoder = Decoder(self.params)

        params_size = 0
        params_num = 0
        for p in self.parameters():
            param_size = 1
            for s in p.size():
                param_size = param_size * s
            if p.requires_grad: params_size = params_size + param_size
            if p.requires_grad: params_num = params_num + 1
            #if p.requires_grad: print('Grad Param', type(p.data), p.size())
        print('RVAE parameters num[%s] size[%s]'%(params_num, params_size))

    def forward(self, drop_prob,
                encoder_word_input=None, encoder_character_input=None,
                decoder_word_input=None,
                z=None):
        """
        :param encoder_word_input: An tensor with shape of [batch_size, seq_len] of Long type
        :param encoder_character_input: An tensor with shape of [batch_size, seq_len, max_word_len] of Long type
        :param decoder_word_input: An tensor with shape of [batch_size, max_seq_len + 1] of Long type

        :param drop_prob: probability of an element of decoder input to be zeroed in sense of dropout

        :param z: context if sampling is performing

        :return: unnormalized logits of sentence words distribution probabilities
                    with shape of [batch_size, seq_len, word_vocab_size]
                 kld loss estimation
        """

        assert parameters_allocation_check(self), \
            'Invalid CUDA options. Parameters should be allocated in the same memory'
        use_cuda = self.embedding.word_embed.weight.is_cuda

        if not self.params.word_is_char:
            assert z is None and fold(lambda acc, parameter: acc and parameter is not None,
                                      [encoder_word_input, encoder_character_input, decoder_word_input],
                                      True) \
                   or (z is not None and decoder_word_input is not None), \
                "Invalid input. If z is None then encoder and decoder inputs should be passed as arguments"

        if z is None:
            ''' Get context from encoder and sample z ~ N(mu, std)
            '''
            [batch_size, _] = encoder_word_input.size()

            encoder_input = self.embedding(encoder_word_input, encoder_character_input)

            context = self.encoder(encoder_input)

            mu = self.context_to_mu(context)
            logvar = self.context_to_logvar(context)
            std = t.exp(0.5 * logvar)

            z = Variable(t.randn([batch_size, self.params.latent_variable_size]))
            if use_cuda:
                z = z.cuda()

            z = z * std + mu

            kld = (-0.5 * t.sum(logvar - t.pow(mu, 2) - t.exp(logvar) + 1, 1)).mean().squeeze()
        else:
            kld = None

        decoder_input = self.embedding.word_embed(decoder_word_input)
        out = self.decoder(decoder_input, z, drop_prob)

        return out, kld, z

    def learnable_parameters(self):

        # word_embedding is constant parameter thus it must be dropped from list of parameters for optimizer
        return [p for p in self.parameters() if p.requires_grad]

    def trainer(self, optimizer, batch_loader):
        perplexity = Perplexity()

        def train(i, batch_size, use_cuda, dropout):
            input = batch_loader.next_batch(batch_size, 'train')
            input = [(Variable(t.from_numpy(var)) if var is not None else None) for var in input]
            input = [(var.long() if var is not None else None) for var in input]
            input = [(var.cuda() if var is not None and use_cuda else var) for var in input]

            [encoder_word_input, encoder_character_input, decoder_word_input, _, target] = input

            logits, kld, _ = self(dropout,
                               encoder_word_input, encoder_character_input,
                               decoder_word_input,
                               z=None)

            logits = logits.view(-1, self.params.word_vocab_size)
            target = target.view(-1)
            cross_entropy = F.cross_entropy(logits, target)

            # since cross enctropy is averaged over seq_len, it is necessary to approximate new kld
            loss = 79 * cross_entropy + kld

            logits = logits.view(batch_size, -1, self.params.word_vocab_size)
            target = target.view(batch_size, -1)
            ppl = perplexity(logits, target).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #del loss, encoder_word_input, target
            return ppl, kld

        return train

    def validater(self, batch_loader):
        perplexity = Perplexity()

        def validate(batch_size, use_cuda):
            input = batch_loader.next_batch(batch_size, 'valid')
            input = [Variable(t.from_numpy(var)) if var is not None else None for var in input]
            input = [var.long() if var is not None else None for var in input]
            input = [var.cuda() if use_cuda and var is not None else var for var in input]

            [encoder_word_input, encoder_character_input, decoder_word_input, _, target] = input

            logits, kld, _ = self(0.,
                               encoder_word_input, encoder_character_input,
                               decoder_word_input,
                               z=None)

            ppl = perplexity(logits, target).mean()
            #del encoder_word_input, target
            return ppl, kld

        return validate

    def style(self, batch_loader, seq, use_cuda, sample_size=30):
        decoder_word_input_np, _ = batch_loader.go_input(1)
        encoder_word_input_np = [[]]
        for i in range(len(seq)):
            word = seq[i]
            word = np.array([[batch_loader.word_to_idx[word]]])
            decoder_word_input_np = np.append(decoder_word_input_np, word, 1)
            encoder_word_input_np = np.append(encoder_word_input_np, word, 1) 
        encoder_word_input_np = encoder_word_input_np[:,::-1]       
        decoder_word_input = Variable(t.from_numpy(decoder_word_input_np).long())
        encoder_word_input = Variable(t.from_numpy(encoder_word_input_np).long())
        decoder_word_input = t.cat([decoder_word_input]*sample_size, 0)
        encoder_word_input = t.cat([encoder_word_input]*sample_size, 0) 
        if use_cuda:
            decoder_word_input = decoder_word_input.cuda()
            encoder_word_input = encoder_word_input.cuda()               
        if self.params.word_is_char:   #TODO only for chinese word right now
            logits, kld, z = self(0.,
                               encoder_word_input, None, 
                               decoder_word_input,
                               z=None)
            return z.data.cpu().numpy()
        return None

    def sample(self, batch_loader, seq_len, seeds, use_cuda, template=None, beam_size=50):
        (z_num, _) = seeds.shape
        print("z sample size", z_num, "beam size", beam_size)
        beam_sent_wids, _ = batch_loader.go_input(1)
        results = []
        end_token_id = batch_loader.word_to_idx[batch_loader.end_token]

        for i in range(seq_len):
            beam_sent_num = len(beam_sent_wids)
            if beam_sent_num == 0:
                break
            if len(results) >= beam_size:
                break
            beam_z_sent_wids = np.repeat(beam_sent_wids, [z_num], axis=0) if z_num > 1 else beam_sent_wids
            decoder_word_input = Variable(t.from_numpy(beam_z_sent_wids).long())
            decoder_word_input = decoder_word_input.cuda() if use_cuda else decoder_word_input
            beam_seeds = Variable(t.from_numpy(seeds).float())
            beam_seeds = t.cat([beam_seeds]*beam_sent_num, 0) if beam_sent_num > 1 else beam_seeds
            beam_seeds = beam_seeds.cuda() if use_cuda else beam_seeds

            beam_sent_logps = None
            if template and len(template) > i and template[i] != '#':
                beam_sent_wids = np.column_stack((beam_sent_wids, [batch_loader.word_to_idx[template[i]]]*beam_sent_num))
            else:
                logits, _, _ = self(0., None, None,
                                 decoder_word_input,
                                 beam_seeds)
                [b_z_n, sl, _] = logits.size()
                logits = logits.view(-1, self.params.word_vocab_size)
                prediction = F.softmax(logits)
                prediction = prediction.view(beam_sent_num, z_num, sl, -1)
                # take mean of sentence vocab probs for each beam group
                beam_sent_vps = np.mean(prediction.data.cpu().numpy(), 1)
                # get vocab probs of the sentence last word for each beam group
                beam_last_vps = beam_sent_vps[:,-1]
                beam_last_word_size = min(batch_loader.words_vocab_size, beam_size)
                # choose last word candidate ids for each beam group
                beam_choosed_wids = np.array([np.random.choice(range(batch_loader.words_vocab_size), beam_last_word_size, replace=False, p=last_vps.ravel()).tolist() for last_vps in beam_last_vps])
                # dumplicate beam sentence word ids for choosed last word size
                beam_sent_wids = np.repeat(beam_sent_wids, [beam_last_word_size], axis=0)
                beam_sent_wids = np.column_stack((beam_sent_wids, beam_choosed_wids.reshape(-1)))
                # get sentence word probs
                beam_sent_wps = []
                for i, sent in enumerate(beam_sent_wids):
                    beam_sent_wps.append([])
                    for j, wid in enumerate(sent[1:]):
                        beam_sent_wps[i].append(beam_sent_vps[i//beam_last_word_size][j][wid])
                # desc sort sum of the beam sentence log probs
                beam_sent_logps = np.sum(np.log(beam_sent_wps), axis=1)
                beam_sent_ids = np.argsort(beam_sent_logps)[-(beam_size-len(results)):][::-1]                
                # get the top beam size sentences
                beam_sent_wids = beam_sent_wids[beam_sent_ids]
                beam_sent_logps = np.exp(beam_sent_logps[beam_sent_ids])
            # check whether some sentence is ended
            keep = []
            for i, sent in enumerate(beam_sent_wids):
                if sent[-1] == end_token_id:
                    results.append(sent)
                    self.show(batch_loader, sent, beam_sent_logps[i] if beam_sent_logps is not None and len(beam_sent_logps)>i else None)
                else:
                    keep.append(i)
            beam_sent_wids = beam_sent_wids[keep]
        lack_num = beam_size - len(results)
        if lack_num > 0:
            results = results + beam_sent_wids[:lack_num].tolist()
            for i, sent in enumerate(results[-lack_num:]):
                self.show(batch_loader, sent, beam_sent_logps[i] if beam_sent_logps is not None and len(beam_sent_logps)>i else None)
        return results

    def show(self, batch_loader, sent_wids, sent_logp):
        print(u'%s==%s'%(("" if self.params.word_is_char else " ").join([batch_loader.idx_to_word[wid] for wid in sent_wids]), sent_logp))