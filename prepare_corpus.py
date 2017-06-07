# -*- coding:utf-8 -*-

import collections
import os
import re

import numpy as np
from six.moves import cPickle

from utils.batch_loader import BatchLoader

def split_corpus_line(line, max_seq_len=200, seg_char=u'。', seg_len=4):
    lines = []
    if len(line) <= max_seq_len:
        return line
    else:
        segs = line.split(seg_char)
        cur = []
        for i,seg in enumerate(segs):            
            if i!=0 and i%seg_len==0:
                if len(cur) > 0:
                    lines.append(cur)
                cur =[]
            cur.append(seg)
        length = len(lines)
        if length >= 2:
            if len(lines[-1]) < seg_len:
                lines[-2] = lines[-2] + lines[-1]
                lines = lines[0:-1]
        return '\n'.join([u'。'.join(line)+u'。' for line in lines])

def reduce_seq_len(filename):
    import codecs
    with codecs.open('./data/'+filename, "r", encoding="utf-8") as fi:
        data = fi.read()
    lines = data.split('\n')
    new_lines = []
    for line in lines:
        new_lines.append(split_corpus_line(line))
    with codecs.open('./data/'+filename+'.ok', "w", encoding="utf-8") as fo:
        fo.write('\n'.join(new_lines))

def generate_tensor_file(prefix, word_is_char, gen_tensors):    
    batch_loader = BatchLoader('', prefix, word_is_char, gen_tensors)

if __name__ == '__main__':
    #filename='poem_test.txt'
    #reduce_seq_len(filename)
    prefix = 'poem'
    word_is_char = True
    gen_tensors = True
    generate_tensor_file(prefix, word_is_char, gen_tensors)