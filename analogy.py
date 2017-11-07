#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np
import argparse
import os
from PIL import Image
import chainer
from chainer import serializers
from chainer import Variable
from chainer import cuda
import dataset
import network
import utils
    
def generate():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--gen', type=str, default=None)
    parser.add_argument('--depth', '-d', type=int, default=0)
    parser.add_argument('--out', '-o', type=str, default='img/')
    parser.add_argument('--num', '-n', type=int, default=10)
    args = parser.parse_args()
    
    gen = network.Generator(depth=args.depth)
    print('loading generator model from ' + args.gen)
    serializers.load_npz(args.gen, gen)

    if args.gpu >= 0:
        cuda.get_device_from_id(0).use()
        gen.to_gpu()

    xp = gen.xp
        
    z1 = gen.z(1)
    z2 = gen.z(1)

    for i in range(args.num):
        print(i)
        p = i / (args.num-1)
        z = z1 * p + z2 * (1 - p)
        x = gen(z, alpha=1.0)
        x = chainer.cuda.to_cpu(x.data)
        
        img = x[0].copy()
        filename = os.path.join(args.out, 'gen_%04d.png'%i)
        utils.save_image(img, filename)

if __name__ == '__main__':
    generate()
