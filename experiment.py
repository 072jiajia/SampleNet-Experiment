import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

# custom module
from util import IOStream, AverageMeter, MultiClassBCE, calculate_ACC
from src import SampleNet
from data import S3DIS_cls
from model import DGCNN


def get_args():
    parser = argparse.ArgumentParser()

    # SampleNet
    parser.add_argument('--sample-points', type=int, default=512,
                        help='Number of point clouds to sample')
    parser.add_argument('--bottleneck-size', type=int, default=1024,
                        help='Dimension of embeddings in SampleNet')
    parser.add_argument('--group-size', type=int, default=8,
                        help='Dimension of embeddings in SampleNet')
    parser.add_argument('--path', type=str, help="pretrained model's path",
                        default='checkpoints/pretrained/model_6.t7')

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

    parser.add_argument('--batch-size', default=50, type=int,
                        metavar='N', help='mini-batch size (default: 32)')
    parser.add_argument('--epochs', default=400, type=int,
                        metavar='N', help='number of total epochs to run')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')

    args = parser.parse_args()
    if not os.path.exists('SampleNetCheckPoint/'):
        os.mkdir('SampleNetCheckPoint')
    return args


def main(args, io):
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    DGCNNModel = DGCNN(args).cuda()
    DGCNNModel = nn.DataParallel(DGCNNModel)
    DGCNNModel.load_state_dict(torch.load(args.path))

    SampleNetModel = SampleNet(
        num_out_points=args.sample_points,
        bottleneck_size=args.bottleneck_size,
        group_size=args.group_size,
        initial_temperature=1.0,
        input_shape="bnc",
        output_shape="bnc",
        complete_fps=False
    )
    SampleNetModel.cuda()
    SampleNetModel = nn.DataParallel(SampleNetModel)

    train_loader = DataLoader(S3DIS_cls(partition='train', num_points=args.num_points, test_area=args.test_area),
                              num_workers=0, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(S3DIS_cls(partition='test', num_points=args.num_points, test_area=args.test_area),
                             num_workers=0, batch_size=args.batch_size, shuffle=False, drop_last=False)

    opt = optim.SGD(SampleNetModel.parameters(), lr=args.lr,
                    momentum=0.9, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(opt, args.epochs)

    best_test_loss = 1e10
    for i in range(args.epochs):
        io.cprint('Epoch [%d]' % (i + 1))
        train(SampleNetModel, DGCNNModel, train_loader, opt, io)
        scheduler.step()
        test_loss = test(SampleNetModel, DGCNNModel, test_loader, io)

        torch.save(SampleNetModel, 'SampleNetCheckPoint/checkpoint.t7')
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(SampleNetModel, 'SampleNetCheckPoint/bestsamplenet.t7')


def train(SampleNetModel, DGCNNModel, train_loader, opt, io):
    TASKLOSS = AverageMeter()
    SIMPLIFICATIONLOSS = AverageMeter()
    PROJECTIONLOSS = AverageMeter()
    TOTALLOSS = AverageMeter()
    ACC50 = AverageMeter()

    SampleNetModel.train()
    DGCNNModel.eval()
    for i, (data, labels) in enumerate(train_loader):
        data, labels = data.cuda(), labels.cuda()

        #####################
        ###### Sampling #####
        #####################
        notprojectedpoints, SamplePoints = SampleNetModel(data)

        ############################
        ###### Downstream Task #####
        ############################
        SamplePoints = SamplePoints.permute((0, 2, 1))
        pred = DGCNNModel(SamplePoints)

        #########################
        ###### Compute Loss #####
        #########################
        ALPHA = 0.01
        LMBDA = 1e-4
        # task loss
        task_loss = MultiClassBCE(pred, labels)
        # simplified loss
        simplification_loss = SampleNetModel.module.get_simplification_loss(
            data, SamplePoints, SamplePoints.shape[0], 1, 0
        ) * ALPHA
        # t ** 2
        projection_loss = SampleNetModel.module.get_projection_loss() * LMBDA

        # total loss
        loss = task_loss + simplification_loss + projection_loss

        # Calculate Accuracy
        acc = calculate_ACC(pred, labels)

        ##########################
        ###### Record Losses #####
        ##########################
        batch_size = SamplePoints.shape[0]
        TASKLOSS.update(float(task_loss), batch_size)
        SIMPLIFICATIONLOSS.update(float(simplification_loss), batch_size)
        PROJECTIONLOSS.update(float(projection_loss), batch_size)
        TOTALLOSS.update(float(loss), batch_size)
        ACC50.update(acc, batch_size)

        #############################
        ###### Back Propagation #####
        #############################
        opt.zero_grad()
        loss.backward()
        opt.step()

        #########################
        ###### Print Losses #####
        #########################
        print(' * [{0}/{1}] * '
              'Acc@0.5: {ACC50.val:.4f} ({ACC50.avg:.4f})  '
              'loss {total.val:.4f} ({total.avg:.4f})  '
              'task loss {task.val:.4f} ({task.avg:.4f})  '
              'project loss {project.val:f} ({project.avg:f})  '
              'simplified loss {simplified.val:.4f} ({simplified.avg:.4f})'
              .format(i, len(train_loader),
                      ACC50=ACC50,
                      total=TOTALLOSS,
                      task=TASKLOSS,
                      project=PROJECTIONLOSS,
                      simplified=SIMPLIFICATIONLOSS),
              end='             \r')

    print(' ' * 180, end='\r')
    io.cprint(' * Train * Loss: %.6f  ACC@0.5: %.6f' % (TOTALLOSS.avg, ACC50.avg))

    return


def test(SampleNetModel, DGCNNModel, test_loader, io):
    TOTALLOSS = AverageMeter()
    ACC50 = AverageMeter()

    SampleNetModel.eval()
    DGCNNModel.eval()
    with torch.no_grad():
        for i, (data, labels) in enumerate(test_loader):
            data, labels = data.cuda(), labels.cuda()

            #####################
            ###### Sampling #####
            #####################
            notprojectedpoints, SamplePoints = SampleNetModel(data)

            ############################
            ###### Downstream Task #####
            ############################
            SamplePoints = SamplePoints.permute((0, 2, 1))
            pred = DGCNNModel(SamplePoints)

            #########################
            ###### Compute Loss #####
            #########################
            loss = MultiClassBCE(pred, labels)

            # Calculate Accuracy
            acc = calculate_ACC(pred, labels)

            ##########################
            ###### Record Losses #####
            ##########################
            batch_size = SamplePoints.shape[0]
            TOTALLOSS.update(float(loss), batch_size)
            ACC50.update(acc, batch_size)

            #########################
            ###### Print Losses #####
            #########################
            print(' * [{0}/{1}] * '
                  'Acc@0.5: {ACC50.val:.4f} ({ACC50.avg:.4f})'
                  'loss {total.val:.4f} ({total.avg:.4f})'
                  .format(i, len(test_loader),
                          ACC50=ACC50,
                          total=TOTALLOSS),
                  end='             \r')
        print(' ' * 180, end='\r')
        io.cprint(' * Test * Loss: %.6f  ACC@0.5: %.6f' % (TOTALLOSS.avg, ACC50.avg))
        return TOTALLOSS.avg


if __name__ == "__main__":
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    args = get_args()
    io = IOStream('SampleNetCheckPoint/run.log')
    main(args, io)
