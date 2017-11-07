#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np
import argparse
import chainer
from chainer import training
from chainer import iterators, optimizers, serializers
from chainer import cuda
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
import dataset
import network
import utils
from updater import WganGpUpdater
import os
import shutil

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--dir', type=str, default='./train_images/')
    parser.add_argument('--gen', type=str, default=None)
    parser.add_argument('--dis', type=str, default=None)
    parser.add_argument('--optg', type=str, default=None)
    parser.add_argument('--optd', type=str, default=None)
    parser.add_argument('--epoch', '-e', type=int, default=3)
    parser.add_argument('--lr', '-l', type=float, default=0.001)
    parser.add_argument('--beta1', type=float, default=0)
    parser.add_argument('--beta2', type=float, default=0.99)
    parser.add_argument('--batch', '-b', type=int, default=16)
    parser.add_argument('--depth', '-d', type=int, default=0)
    parser.add_argument('--alpha', type=float, default=0)
    parser.add_argument('--delta', type=float, default=0.00005)
    parser.add_argument('--out', '-o', type=str, default='img/')
    parser.add_argument('--num', '-n', type=int, default=10)
    args = parser.parse_args()

    train = dataset.YuiDataset(directory=args.dir, depth=args.depth)
    train_iter = iterators.SerialIterator(train, batch_size=args.batch)

    gen = network.Generator(depth=args.depth)
    if args.gen is not None:
        print('loading generator model from ' + args.gen)
        serializers.load_npz(args.gen, gen)

    dis = network.Discriminator(depth=args.depth)
    if args.dis is not None:
        print('loading discriminator model from ' + args.dis)
        serializers.load_npz(args.dis, dis)
        
    if args.gpu >= 0:
        cuda.get_device_from_id(0).use()
        gen.to_gpu()
        dis.to_gpu()

    opt_g = optimizers.Adam(alpha=args.lr, beta1=args.beta1, beta2=args.beta2)
    opt_g.setup(gen)
    if args.optg is not None:
        print('loading generator optimizer from ' + args.optg)
        serializers.load_npz(args.optg, opt_g)
    
    opt_d = optimizers.Adam(alpha=args.lr, beta1=args.beta1, beta2=args.beta2)
    opt_d.setup(dis)
    if args.optd is not None:
        print('loading discriminator optimizer from ' + args.optd)
        serializers.load_npz(args.optd, opt_d)


    updater = WganGpUpdater(alpha=args.alpha,
                            delta=args.delta,
                            models=(gen, dis),
                            iterator={'main': train_iter},
                            optimizer={'gen': opt_g, 'dis': opt_d},
                            device=args.gpu)

    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out='results')

    if os.path.isdir(args.out):
        shutil.rmtree(args.out)
    os.makedirs(args.out)
    for i in range(args.num):
        img = train.get_example(i)
        filename = os.path.join(args.out, 'real_%d.png'%i)
        utils.save_image(img, filename)
    
    def output_image(gen, depth, out, num):
        @chainer.training.make_extension()
        def make_image(trainer):
            z = gen.z(num)
            x = gen(z, alpha=trainer.updater.alpha)
            x = chainer.cuda.to_cpu(x.data)

            for i in range(args.num):
                img = x[i].copy()
                filename = os.path.join(out, '%d_%d.png' % (trainer.updater.epoch, i))
                utils.save_image(img, filename)

        return make_image
            
    
    trainer.extend(extensions.LogReport(trigger=(1, 'epoch')))
    trainer.extend(extensions.PrintReport(['epoch', 'gen_loss', 'loss_d', 'loss_l', 'loss_dr', 'dis_loss', 'alpha']))
    trainer.extend(extensions.snapshot_object(gen, 'gen'), trigger=(10, 'epoch'))
    trainer.extend(extensions.snapshot_object(dis, 'dis'), trigger=(10, 'epoch'))
    trainer.extend(extensions.snapshot_object(opt_g, 'opt_g'), trigger=(10, 'epoch'))
    trainer.extend(extensions.snapshot_object(opt_d, 'opt_d'), trigger=(10, 'epoch'))
    trainer.extend(output_image(gen, args.depth, args.out, args.num), trigger=(1, 'epoch'))
    trainer.extend(extensions.ProgressBar(update_interval=1))    
    
    trainer.run()

    modelname = './results/gen'
    print( 'saving generator model to ' + modelname )
    serializers.save_npz(modelname, gen)

    modelname = './results/dis'
    print( 'saving discriminator model to ' + modelname )
    serializers.save_npz(modelname, dis)

    optname = './results/opt_g'
    print( 'saving generator optimizer to ' + optname )
    serializers.save_npz(optname, opt_g)

    optname = './results/opt_d'
    print( 'saving generator optimizer to ' + optname )
    serializers.save_npz(optname, opt_d)

if __name__ == '__main__':
    train()
