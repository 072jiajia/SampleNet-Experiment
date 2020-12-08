from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from torch.utils.data import DataLoader
# custom module
from data import S3DIS_cls
from model import DGCNN
from util import IOStream, AverageMeter, MultiClassBCE, calculate_ACC


os.environ["CUDA_VISIBLE_DEVICES"] = "7"


def _init_():
    # Training settings
    parser = argparse.ArgumentParser(description='SampleNet-Experiment')
    parser.add_argument('--exp_name', type=str, default='pretrained', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--test_area', type=str, default='6', metavar='N',
                        choices=['1', '2', '3', '4', '5', '6', 'all'])
    parser.add_argument('--batch_size', type=int, default=12, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=40, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--num_points', type=int, default=4096,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    args = parser.parse_args()

    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/'+args.exp_name):
        os.makedirs('checkpoints/'+args.exp_name)
    return args


def main(args, io):
    train_loader = DataLoader(S3DIS_cls(partition='train', num_points=args.num_points, test_area=args.test_area),
                              num_workers=8, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(S3DIS_cls(partition='test', num_points=args.num_points, test_area=args.test_area),
                             num_workers=8, batch_size=args.batch_size, shuffle=False, drop_last=False)

    model = DGCNN(args)
    model.cuda()
    model = nn.DataParallel(model)

    print("Let's use", torch.cuda.device_count(), "GPUs!")

    opt = optim.SGD(model.parameters(), lr=args.lr,
                    momentum=0.9, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(opt, args.epochs)

    best_test_loss = 1e10
    for epoch in range(args.epochs):
        io.cprint('Epoch [%d]' % (epoch + 1))
        train(model, train_loader, opt, io)
        scheduler.step()
        test_loss = test(model, test_loader, io)

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            # save in torch==1.4 readible style
            torch.save(model.state_dict(), 'checkpoints/%s/model_%s.t7' %
                       (args.exp_name, args.test_area),
                       _use_new_zipfile_serialization=False)


def train(model, train_loader, opt, io):
    LOSS = AverageMeter()
    ACC50 = AverageMeter()

    model.train()
    for i, (data, labels) in enumerate(train_loader):
        # Data Preparation
        data, labels = data.cuda(), labels.cuda()
        data = data.permute(0, 2, 1)

        # Prediction
        pred = model(data)
        loss = MultiClassBCE(pred, labels)

        # Back Propagation
        opt.zero_grad()
        loss.backward()
        opt.step()

        # Calculate Accuracy
        acc = calculate_ACC(pred, labels)

        # Record
        batch_size = data.size()[0]
        LOSS.update(float(loss), batch_size)
        ACC50.update(acc, batch_size)
        print('[{0}/{1}] '
              'Loss: {LOSS.val:.4f} ({LOSS.avg:.4f})  '
              'Acc@0.5: {ACC50.val:.4f} ({ACC50.avg:.4f})'
              .format(i, len(train_loader), LOSS=LOSS, ACC50=ACC50),
              end='    \r')
    print(' ' * 160, end='\r')
    io.cprint(' * Train * Loss: %.6f  Acc@0.5: %.6f' % (LOSS.avg, ACC50.avg))


def test(model, test_loader, io):
    LOSS = AverageMeter()
    ACC50 = AverageMeter()
    model.eval()
    with torch.no_grad():
        for i, (data, labels) in enumerate(test_loader):
            # Data Preparation
            data, labels = data.cuda(), labels.cuda()
            data = data.permute(0, 2, 1)

            # Prediction
            pred = model(data)
            loss = MultiClassBCE(pred, labels)

            # Calculate Accuracy
            acc = calculate_ACC(pred, labels)

            # Record
            batch_size = data.size()[0]
            LOSS.update(float(loss), batch_size)
            ACC50.update(acc, batch_size)
            print('[{0}/{1}] '
                  'Loss: {LOSS.val:.4f} ({LOSS.avg:.4f})  '
                  'Acc@0.5: {ACC50.val:.4f} ({ACC50.avg:.4f})'
                  .format(i, len(test_loader), LOSS=LOSS, ACC50=ACC50),
                  end='    \r')
        print(' ' * 160, end='\r')
        io.cprint(' * Test * Loss: %.6f  Acc@0.5: %.6f' %
                  (LOSS.avg, ACC50.avg))
    return LOSS.avg


if __name__ == "__main__":
    args = _init_()

    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    io.cprint('Using GPU : ' + str(torch.cuda.current_device()) +
              ' from ' + str(torch.cuda.device_count()) + ' devices')

    main(args, io)
