import os
os.environ["CUDA_VISIBLE_DEVICES"] = "8, 9"

from util import cal_loss, IOStream
from src.qdataset import QuaternionFixedDataset, QuaternionTransform, rad_to_deg
from src.pctransforms import OnUnitCube, PointcloudToTensor
from src import sputils
from src import ChamferDistance, FPSSampler, RandomSampler, SampleNet
from torch.utils.data import DataLoader
from models import pcrnet
from data import S3DIS
from model import DGCNN_semseg
import torch
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
import torch.optim as optim
import torch.nn as nn
import argparse
import logging
import sys



# fmt: off
def get_args():
    parser = argparse.ArgumentParser()

    # SampleNet
    parser.add_argument('--sample-points', type=int, default=512,
                        help='Number of point clouds to sample')
    parser.add_argument('--bottleneck-size', type=int, default=1024,
                        help='Dimension of embeddings in SampleNet')
    parser.add_argument('--group-size', type=int, default=8,
                        help='Dimension of embeddings in SampleNet')

    # DGCNN
    parser.add_argument('--test_area', type=str, default='6', metavar='N',
                        choices=['1', '2', '3', '4', '5', '6', 'all'])
    parser.add_argument('--num_points', type=int, default=4096,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')

    parser.add_argument('--workers', default=40, type=int,
                        metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('--batch-size', default=6, type=int,
                        metavar='N', help='mini-batch size (default: 32)')
    parser.add_argument('--epochs', default=400, type=int,
                        metavar='N', help='number of total epochs to run')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')

    args = parser.parse_args()
    return args


def main(args, io):
    train_loader = DataLoader(S3DIS(partition='train', num_points=args.num_points, test_area=args.test_area),
                              num_workers=4, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(S3DIS(partition='test', num_points=args.num_points, test_area=args.test_area),
                             num_workers=4, batch_size=args.batch_size, shuffle=False, drop_last=False)

    print("Let's use", torch.cuda.device_count(), "GPUs!")
    DGCNNModel = DGCNN_semseg(args)

    assert 0, 'import torch==1.4 but I need to use torch==1.7 to load pretrained model'
    DGCNNModel.load_state_dict(torch.load('semseg_6.t7'))

    DGCNNModel.cuda()
    DGCNNModel = nn.DataParallel(DGCNNModel)
    DGCNNModel.eval()

    SampleNetModel = SampleNet(
        num_out_points=args.sample_points,
        bottleneck_size=args.bottleneck_size,
        group_size=args.group_size,
        initial_temperature=1.0,
        input_shape="bnc",
        output_shape="bnc",
    )
    SampleNetModel.cuda()
    SampleNetModel = nn.DataParallel(SampleNetModel)
    opt = optim.SGD(SampleNetModel.parameters(), lr=args.lr,
                    momentum=args.momentum, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=1e-3)

    for i in range(args.epochs):
        print('Epoch [%d]' % (i + 1))
        train(args, io, SampleNetModel, DGCNNModel, train_loader, opt)
        scheduler.step()
        loss = test()


def train(args, io, SampleNetModel, DGCNNModel, train_loader, opt):
    train_loss = 0.0
    count = 0.0
    SampleNetModel.train()
    DGCNNModel.eval()
    train_true_cls = []
    train_pred_cls = []
    train_true_seg = []
    train_pred_seg = []
    train_label_seg = []
    ii = 0
    for data, seg in train_loader:
        print(ii, '/', len(train_loader), end='\r')
        ii += 1

        data, seg = data.cuda(), seg.cuda()
        data = data[:, :, :3]
        # data = data.permute(0, 2, 1)[:, :3]
        data = SampleNetModel(data)
        seg_pred = DGCNNModel(data)

        batch_size = data.size()[0]
        opt.zero_grad()

        seg_pred = seg_pred.permute(0, 2, 1).contiguous()
        loss = criterion(seg_pred.view(-1, 13), seg.view(-1, 1).squeeze())
        loss.backward()
        opt.step()
        pred = seg_pred.max(dim=2)[1]               # (batch_size, num_points)
        count += batch_size
        train_loss += loss.item() * batch_size
        seg_np = seg.cpu().numpy()                  # (batch_size, num_points)
        pred_np = pred.detach().cpu().numpy()       # (batch_size, num_points)
        # (batch_size * num_points)
        train_true_cls.append(seg_np.reshape(-1))
        # (batch_size * num_points)
        train_pred_cls.append(pred_np.reshape(-1))
        train_true_seg.append(seg_np)
        train_pred_seg.append(pred_np)

    train_true_cls = np.concatenate(train_true_cls)
    train_pred_cls = np.concatenate(train_pred_cls)
    train_acc = metrics.accuracy_score(train_true_cls, train_pred_cls)
    avg_per_class_acc = metrics.balanced_accuracy_score(
        train_true_cls, train_pred_cls)
    train_true_seg = np.concatenate(train_true_seg, axis=0)
    train_pred_seg = np.concatenate(train_pred_seg, axis=0)
    train_ious = calculate_sem_IoU(train_pred_seg, train_true_seg)
    print(' * Train * loss: %.6f  '
          'train acc: %.6f  '
          'train avg acc: %.6f  '
          'train iou: %.6f     '
          % (epoch,
             train_loss*1.0/count,
             train_acc,
             avg_per_class_acc,
             np.mean(train_ious)))

    return

# def test(args, io, SampleNetModel, DGCNNModel, train_loader, opt):
#     pass


if __name__ == "__main__":
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    args = get_args()
    io = IOStream('run.log')
    main(args, io)
