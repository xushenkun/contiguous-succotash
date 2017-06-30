import numpy as np
import torch as t
import torch.nn as nn
from torch.nn import Parameter

from .tdnn import TDNN


class Embedding(nn.Module):
    def __init__(self, params, path='../../../', prefix=''):
        super(Embedding, self).__init__()

        self.params = params

        word_embed = np.load(path + 'data/' + prefix + 'word_embeddings.npy')

        self.word_embed = nn.Embedding(self.params.word_vocab_size, self.params.word_embed_size)
        if not self.params.word_is_char:
            self.char_embed = nn.Embedding(self.params.char_vocab_size, self.params.char_embed_size)
        self.word_embed.weight = Parameter(t.from_numpy(word_embed).float(), requires_grad=False)
        if not self.params.word_is_char:
            self.char_embed.weight = Parameter(
                t.Tensor(self.params.char_vocab_size, self.params.char_embed_size).uniform_(-1, 1))
        if not self.params.word_is_char:
            self.TDNN = TDNN(self.params)

    def forward(self, word_input, character_input):
        """
        :param word_input: [batch_size, seq_len] tensor of Long type
        :param character_input: [batch_size, seq_len, max_word_len] tensor of Long type
        :return: input embedding with shape of [batch_size, seq_len, word_embed_size + sum_depth]
        """
        if not self.params.word_is_char:
            assert word_input.size()[:2] == character_input.size()[:2], \
                'Word input and character input must have the same sizes, but {} and {} found'.format(
                    word_input.size(), character_input.size())

        [batch_size, seq_len] = word_input.size()

        word_input = self.word_embed(word_input)

        if not self.params.word_is_char:
            character_input = character_input.view(-1, self.params.max_word_len)
            character_input = self.char_embed(character_input)
            character_input = character_input.view(batch_size,
                                                   seq_len,
                                                   self.params.max_word_len,
                                                   self.params.char_embed_size)

            character_input = self.TDNN(character_input)

            result = t.cat([word_input, character_input], 2)
        else:
            result = word_input
        return result

    def similarity(self, input):
        """
        :param input: An tensor with shape of [batch_size, word_embed_size] 
        :return: An tensor with shape [batch_size, word_vocab_size] with estimated similarity values
        """
        batch_size, _ = input.size()

        input = input.unsqueeze(1).repeat(1, self.params.word_vocab_size, 1)

        embed = self.word_embed.weight.unsqueeze(0).repeat(batch_size, 1, 1)

        result = t.pow(embed - input, 2).mean(2).squeeze(2)

        return t.cat([t.max(result,1)[0]]*self.params.word_vocab_size, 1) - result