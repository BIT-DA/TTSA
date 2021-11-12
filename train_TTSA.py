# -*- coding: utf-8 -*

import random
import time
import warnings
import sys
import argparse
import copy
import PIL
import torchvision
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.optim import SGD
import torch.utils.data
from torch.utils.data import DataLoader
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torch.nn.functional as F
import os.path as osp
import gc
import numpy as np

from network import ImageClassifier
import backbone as BackboneNetwork
from utils import ContinuousDataloader
from transforms import ResizeImage
from lr_scheduler import LrScheduler
from data_list_index import ImageList
from Loss import *


def get_current_time():
    time_stamp = time.time()
    local_time = time.localtime(time_stamp)
    str_time = time.strftime('%Y-%m-%d_%H-%M-%S', local_time)
    return str_time


def main(args: argparse.Namespace, config):
    torch.multiprocessing.set_sharing_strategy('file_system')
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
    cudnn.benchmark = True

    # load data
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if args.center_crop:
        train_transform = transforms.Compose([
            ResizeImage(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize])
    else:
        train_transform = transforms.Compose([
            ResizeImage(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize])

    val_tranform = transforms.Compose([
        ResizeImage(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize])

    train_source_dataset = ImageList(open(args.s_dset_path).readlines(), transform=train_transform)
    train_source_loader = DataLoader(train_source_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)

    train_target_dataset = ImageList(open(args.t_dset_path).readlines(), transform=train_transform)
    train_target_loader = DataLoader(train_target_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)

    val_dataset = ImageList(open(args.t_dset_path).readlines(), transform=val_tranform)
    if args.dset == "visda":
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=64)
    else:
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    if args.dset == 'domainnet':
        test_dataset = ImageList(open(args.t_test_path).readlines(), transform=val_tranform)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=64)
    else:
        test_loader = val_loader

    train_source_iter = ContinuousDataloader(train_source_loader)
    train_target_iter = ContinuousDataloader(train_target_loader)

    # load model
    print("=> using pre-trained model '{}'".format(args.arch))
    backbone = BackboneNetwork.__dict__[args.arch](pretrained=True)
    if args.dset == "office":
        num_classes = 31
    elif args.dset == "office-home":
        num_classes = 65
    elif args.dset == "visda":
        num_classes = 12
    elif args.dset == "domainnet":
        num_classes = 345
    classifier = ImageClassifier(backbone, num_classes).cuda()
    classifier_feature_dim = classifier.features_dim

    # define optimizer and lr scheduler
    max_iters = args.iters_per_epoch * args.epochs
    all_parameters = classifier.get_parameters()
    optimizer = SGD(all_parameters, args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    lr_sheduler = LrScheduler(optimizer, max_iters, init_lr=args.lr, gamma=10, decay_rate=args.decay_rate)

    # start training
    best_acc1 = 0.
    loss_criterion = Loss_aug_pro(classifier_feature_dim, num_classes, args.alpha).cuda()
    for epoch in range(args.epochs):
        # train for one epoch
        train(train_source_iter, train_target_iter, classifier, optimizer,
              lr_sheduler, epoch, args, loss_criterion, config)

        # evaluate on validation set
        if args.dset == "domainnet":
            if epoch >= 10:
                acc1 = validate(test_loader, classifier, args)
                if acc1 > best_acc1:
                    best_model = copy.deepcopy(classifier.state_dict())
                best_acc1 = max(acc1, best_acc1)
                print(
                    "epoch = {:02d},  acc1={:.3f}, best_acc1 = {:.3f}".format(epoch, acc1, best_acc1))
                config["out_file"].write(
                    "epoch = {:02d},  best_acc1 = {:.3f}, best_acc1 = {:.3f}".format(epoch, acc1, best_acc1) + '\n')
                config["out_file"].flush()
        else:
            if args.dset == "visda":
                acc1 = validate_visda(test_loader, classifier, epoch, config)
            else:
                acc1 = validate(test_loader, classifier, args)
            # remember the best top1 accuracy and checkpoint
            if acc1 > best_acc1:
                best_model = copy.deepcopy(classifier.state_dict())
            best_acc1 = max(acc1, best_acc1)
            print("epoch = {:02d},  acc1={:.3f}, best_acc1 = {:.3f}".format(epoch, acc1, best_acc1))
            config["out_file"].write(
                "epoch = {:02d},  best_acc1 = {:.3f}, best_acc1 = {:.3f}".format(epoch, acc1, best_acc1) + '\n')
            config["out_file"].flush()

    if args.SaveModel:
        torch.save(best_model, osp.join(args.output_dir, "{}_best_model_{:.0f}.pth.tar".format(config["task"], best_acc1)))

    print("best_acc1 = {:.3f}".format(best_acc1))
    config["out_file"].write("best_acc1 = {:.3f}".format(best_acc1) + '\n')
    config["out_file"].flush()

    # evaluate on test set
    classifier.load_state_dict(best_model)
    if args.dset == "visda":
        acc1 = validate_visda(test_loader, classifier, epoch, config)
    else:
        acc1 = validate(test_loader, classifier, args)
    print("test_accuracy = {:.3f}".format(acc1))
    config["out_file"].write("test_accuracy = {:.3f}".format(acc1) + '\n')
    config["out_file"].flush()



def train(train_source_iter: ContinuousDataloader, train_target_iter: ContinuousDataloader, model: ImageClassifier,
          optimizer: SGD, lr_sheduler: LrScheduler, epoch: int, args: argparse.Namespace, loss_criterion, config):
    # switch to train mode
    model.train()
    max_iters = args.iters_per_epoch * args.epochs
    for i in range(args.iters_per_epoch):
        current_iter = i + args.iters_per_epoch * epoch
        Lambda = args.lambda0 * (float(current_iter) / float(max_iters))

        lr_sheduler.step()

        x_s, labels_s, idx_source = next(train_source_iter)
        x_t, _, idx_target = next(train_target_iter)

        x_s = x_s.cuda()
        x_t = x_t.cuda()
        labels_s = labels_s.cuda()

        # get features and logit outputs
        x = torch.cat((x_s, x_t), dim=0)
        y, f = model(x)
        y_s, y_t = y.chunk(2, dim=0)
        f_s, f_t = f.chunk(2, dim=0)

        pseudo_labels_t = y_t.argmax(1)

        # compute loss
        aug_loss, pro_loss = loss_criterion(model.head, f_s, f_t, y_s, labels_s, pseudo_labels_t, Lambda, args.eta)

        MI_loss = MI(y_t)

        total_loss = aug_loss - args.beta * MI_loss + args.gamma * pro_loss

        # compute gradient and do SGD step
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # print training log
        if i % args.print_freq == 0:
            print("Epoch: [{:02d}][{}/{}]	total_loss:{:.3f}	aug_loss:{:.3f}	 pro_loss:{:.3f}  MI_loss:{:.3f}".format( \
                    epoch, i, args.iters_per_epoch, total_loss, aug_loss, pro_loss, MI_loss))
            config["out_file"].write("Epoch: [{:02d}][{}/{}]	total_loss:{:.3f}	aug_loss:{:.3f}	 pro_loss:{:.3f}  MI_loss:{:.3f}\n".format( \
                    epoch, i, args.iters_per_epoch, total_loss, aug_loss, pro_loss, MI_loss))
            config["out_file"].flush()



def validate(val_loader: DataLoader, model: ImageClassifier, args: argparse.Namespace):
    # switch to evaluate mode
    model.eval()
    start_test = True
    with torch.no_grad():
        for i, (images, target, _) in enumerate(val_loader):
            images = images.cuda()
            target = target.cuda()

            # get logit outputs
            output, _ = model(images)
            if start_test:
                all_output = output.float()
                all_label = target.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, output.float()), 0)
                all_label = torch.cat((all_label, target.float()), 0)
        _, predict = torch.max(all_output, 1)
        accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
        accuracy = accuracy * 100.0
        print(' accuracy:{:.3f}'.format(accuracy))
    return accuracy


def validate_visda(val_loader, model, epoch, config):
    dict = {0: "plane", 1: "bcybl", 2: "bus", 3: "car", 4: "horse", 5: "knife", 6: "mcyle", 7: "person", 8: "plant", \
            9: "sktb", 10: "train", 11: "truck"}
    model.eval()
    with torch.no_grad():
        tick = 0
        subclasses_correct = np.zeros(12)
        subclasses_tick = np.zeros(12)
        for i, (imgs, labels, _) in enumerate(val_loader):
            tick += 1
            imgs = imgs.cuda()
            pred, _ = model(imgs)
            pred = nn.Softmax(dim=1)(pred)
            pred = pred.data.cpu().numpy()
            pred = pred.argmax(axis=1)
            labels = labels.numpy()
            for i in range(pred.size):
                subclasses_tick[labels[i]] += 1
                if pred[i] == labels[i]:
                    subclasses_correct[pred[i]] += 1
        subclasses_result = np.divide(subclasses_correct, subclasses_tick)
        print("Epoch [:02d]:".format(epoch))
        for i in range(12):
            log_str1 = '\t{}----------({:.3f})'.format(dict[i], subclasses_result[i] * 100.0)
            print(log_str1)
            config["out_file"].write(log_str1 + "\n")
        avg = subclasses_result.mean()
        avg = avg * 100.0
        log_avg = '\taverage:{:.3f}'.format(avg)
        print(log_avg)
        config["out_file"].write(log_avg + "\n")
        config["out_file"].flush()
    return avg





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Target-oriented Transferable Semantic Augmentation under Prototype Constraint')
    parser.add_argument('--arch', type=str, default='resnet50', choices=['resnet50', 'resnet101'])
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--dset', type=str, default='office', choices=['office', 'visda', 'office-home', 'domainnet'], help="The dataset used")
    parser.add_argument('--s_dset_path', type=str, default='/data1/TL/data/list/office/webcam_31.txt', help="The source dataset path list")
    parser.add_argument('--t_dset_path', type=str, default='/data1/TL/data/list/office/amazon_31.txt', help="The target dataset path list")
    parser.add_argument('--t_test_path', type=str, default='/data1/TL/data/list/office/amazon_31.txt', help="The target test dataset path list")
    parser.add_argument('--output_dir', type=str, default='log/TTSA/office31', help="output directory of logs")
    parser.add_argument('--center_crop', default=False, action='store_true')
    parser.add_argument('--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=40, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--iters-per-epoch', default=500, type=int, help='Number of iterations per epoch')
    parser.add_argument('--print-freq', default=100, type=int, metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--batch-size', default=32, type=int, metavar='N', help='mini-batch size (default: 32)')
    parser.add_argument('--lr', default=0.01, type=float, metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--decay_rate', default=0.75, type=float, dest='decay_rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--weight-decay', default=1e-3, type=float, metavar='W', help='weight decay (default: 1e-3)', dest='weight_decay')
    parser.add_argument('--seed', default=2, type=int, help='seed for initializing training. ')
    parser.add_argument('--alpha', type=float, default=0.1, help="hyper-parameter for covariance estimation")
    parser.add_argument('--lambda0', type=float, default=0.25, help="hyper-parameter for augmentation strength")
    parser.add_argument('--eta', type=float, default=0.5, help="hyper-parameter for angular margin")
    parser.add_argument('--gamma', type=float, default=0.001, help="tradeoff for loss pro")
    parser.add_argument('--beta', type=float, default=0.1, help="tradeoff for loss MI")
    parser.add_argument('--SaveModel', default=False, action='store_true')
    args = parser.parse_args()

    config = {}
    if not osp.exists(args.output_dir):
        os.makedirs(args.output_dir)
    task = args.s_dset_path.split('/')[-1].split('.')[0].split('_')[0] + "-" + \
           args.t_dset_path.split('/')[-1].split('.')[0].split('_')[0]
    config["task"] = task
    config["out_file"] = open(osp.join(args.output_dir, get_current_time() + "_" + task + "_log.txt"), "w")

    config["out_file"].write("file name: train_TTSA\n")
    config["out_file"].write("PTL.version = {}".format(PIL.__version__) + "\n")
    config["out_file"].write("torch.version = {}".format(torch.__version__) + "\n")
    config["out_file"].write("torchvision.version = {}".format(torchvision.__version__) + "\n")

    for arg in vars(args):
        print("{} = {}".format(arg, getattr(args, arg)))
        config["out_file"].write(str("{} = {}".format(arg, getattr(args, arg))) + "\n")
    config["out_file"].flush()
    main(args, config)
