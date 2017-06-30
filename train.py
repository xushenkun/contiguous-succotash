import argparse
import os
import shutil

import numpy as np
import torch as t
from torch.optim import Adam

from utils.batch_loader import BatchLoader
from utils.parameters import Parameters
from model.rvae_dilated import RVAE_dilated

if __name__ == "__main__":    

    parser = argparse.ArgumentParser(description='RVAE_dilated')
    parser.add_argument('--num-epochs', type=int, default=2500, metavar='ES',
                        help='num epochs (default: 2500)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='E',
                    help='manual epoch index (useful on restarts)')
    parser.add_argument('--batch-size', type=int, default=450, metavar='BS',
                        help='batch size (default: 450)')
    parser.add_argument('--use-cuda', type=bool, default=True, metavar='CUDA',
                        help='use cuda (default: True)')
    parser.add_argument('--learning-rate', type=float, default=0.0005, metavar='LR',
                        help='learning rate (default: 0.0005)')
    parser.add_argument('--dropout', type=float, default=0.3, metavar='DR',
                        help='dropout (default: 0.3)')
    parser.add_argument('--use-trained', default='', metavar='UT',
                        help='load pretrained model (default: None)')
    parser.add_argument('--ret-result', default='', metavar='CE',
                        help='ce result path (default: '')')
    parser.add_argument('--kld-result', default='', metavar='KLD',
                        help='ce result path (default: '')')

    args = parser.parse_args()

    prefix = 'poem'
    word_is_char = True

    batch_loader = BatchLoader('', prefix, word_is_char)

    best_ret = 9999999
    is_best = False

    if not os.path.exists('data/' + batch_loader.prefix + 'word_embeddings.npy'):
        raise FileNotFoundError("word embeddings file was't found")

    parameters = Parameters(batch_loader.max_word_len,
                            batch_loader.max_seq_len,
                            batch_loader.words_vocab_size,
                            batch_loader.chars_vocab_size, word_is_char)

    rvae = RVAE_dilated(parameters, batch_loader.prefix)
    optimizer = Adam(rvae.learnable_parameters(), args.learning_rate)

    if args.use_trained:
        checkpoint = t.load(args.use_trained)
        args.start_epoch = checkpoint['epoch']
        best_ret = checkpoint['best_ret']        
        rvae.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    if args.use_cuda and t.cuda.is_available():
        rvae = rvae.cuda()

    train_step = rvae.trainer(optimizer, batch_loader)
    validate = rvae.validater(batch_loader)

    ret_result = []
    kld_result = []

    for epoch in range(args.start_epoch, args.num_epochs):

        train_ret, train_kld, train_kld_coef = train_step(epoch, args.batch_size, args.use_cuda and t.cuda.is_available(), args.dropout)
        train_ret = train_ret.data.cpu().numpy()[0]
        train_kld = train_kld.data.cpu().numpy()[0]

        valid_ret, valid_kld = validate(args.batch_size, args.use_cuda and t.cuda.is_available())
        valid_ret = valid_ret.data.cpu().numpy()[0]
        valid_kld = valid_kld.data.cpu().numpy()[0]

        ret_result += [valid_ret]
        kld_result += [valid_kld]

        is_best = valid_ret < best_ret
        best_ret = min(valid_ret, best_ret)

        print('[%s]---TRAIN-ret[%s]kld[%s]------VALID-ret[%s]kld[%s]'%(epoch, train_ret, train_kld, valid_ret, valid_kld))

        if epoch != 1 and epoch % 10 == 9:
            seed = np.random.normal(size=[1, parameters.latent_variable_size])
            sample = rvae.sample(batch_loader, 50, seed, args.use_cuda and t.cuda.is_available(), None, 1)
            print('[%s]---SAMPLE: %s'%(epoch, sample))

        if epoch != 0 and epoch % 100 == 99:
            checkpoint_filename = './data/%strained_%s_RVAE'%(batch_loader.prefix, epoch+1)
            t.save({'epoch': epoch+1, 
                'state_dict': rvae.state_dict(), 
                'best_ret': best_ret, 
                'optimizer': optimizer.state_dict()}, checkpoint_filename)
            oldest = epoch+1-3*100
            oldest_checkpoint_filename = './data/%strained_%s_RVAE'%(batch_loader.prefix, oldest) if oldest>0 else None
            if oldest_checkpoint_filename and os.path.isfile(oldest_checkpoint_filename):
                os.remove(oldest_checkpoint_filename)
            if is_best:
                shutil.copyfile(checkpoint_filename, './data/'+batch_loader.prefix+'trained_best_RVAE')

    t.save({'epoch': args.num_epochs, 
            'state_dict': rvae.state_dict(), 
            'best_ret': best_ret, 
            'optimizer': optimizer.state_dict()}, './data/'+batch_loader.prefix+'trained_last_RVAE')

    np.save(batch_loader.prefix+'ret_result_{}.npy'.format(args.ret_result), np.array(ret_result))
    np.save(batch_loader.prefix+'kld_result_npy_{}'.format(args.kld_result), np.array(kld_result))
