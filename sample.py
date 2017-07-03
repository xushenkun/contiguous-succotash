# -*- coding:utf-8 -*-

import argparse
import os

import numpy as np
import torch as t

from utils.batch_loader import BatchLoader
from utils.parameters import Parameters
from model.rvae_dilated import RVAE_dilated

if __name__ == '__main__':    

    parser = argparse.ArgumentParser(description='Sampler')
    parser.add_argument('--use-cuda', type=bool, default=True, metavar='CUDA',
                        help='use cuda (default: True)')
    parser.add_argument('--beam-size', type=int, default=50, metavar='BS',
                        help='beam search size (default: 50)')
    parser.add_argument('--z-size', type=int, default=30, metavar='ZS',
                        help='z sample size (default: 30)')
    parser.add_argument('--seq-len', type=int, default=50, metavar='ZS',
                        help='seq length (default: 50)')
    parser.add_argument('--use-trained', default='', metavar='UT',
                        help='load pretrained model (default: None)')
    args = parser.parse_args()

    prefix = 'poem'
    word_is_char = True

    batch_loader = BatchLoader('', prefix, word_is_char)
    if args.use_trained:
        checkpoint_filename = args.use_trained
    else:
        checkpoint_filename = './data/'+batch_loader.prefix+'trained_last_RVAE'
    assert os.path.exists(checkpoint_filename), \
        'trained model not found'

    parameters = Parameters(batch_loader.max_word_len,
                            batch_loader.max_seq_len,
                            batch_loader.words_vocab_size,
                            batch_loader.chars_vocab_size, word_is_char)

    rvae = RVAE_dilated(parameters, batch_loader.prefix)
    checkpoint = t.load(checkpoint_filename)
    rvae.load_state_dict(checkpoint['state_dict'])
    if args.use_cuda and t.cuda.is_available():
        rvae = rvae.cuda()

    seed = np.random.normal(size=[1, parameters.latent_variable_size])
    #seed = rvae.style(batch_loader, u'青林口，白铁匠，生红炉，烧黑炭，坐南朝北打东西。', args.use_cuda and t.cuda.is_available(), sample_size=args.z_size)
    seed = rvae.style(batch_loader, u'床前看月光，疑是地上霜。举头望山月，低头思故乡。', args.use_cuda and t.cuda.is_available(), sample_size=args.z_size)
    if seed is not None:
        #result = rvae.sample(batch_loader, args.seq_len, seed, args.use_cuda and t.cuda.is_available(), u'床####，疑####。举####，低####。', args.beam_size)
        result = rvae.sample(batch_loader, args.seq_len, seed, args.use_cuda and t.cuda.is_available(), u'', args.beam_size)
        #print(result)
        print()