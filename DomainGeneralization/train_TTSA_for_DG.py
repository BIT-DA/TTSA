import argparse
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.optim import SGD
import PIL
import torchvision
import time
import os.path as osp
import warnings
warnings.filterwarnings("ignore")
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

from models.ResNet import resnet18, resnet50
from models.classifier import Classifier
from data import *
from utils.tools import *
from Loss_for_DG import *

import sys
sys.path.append('./..')
from lr_scheduler import *



encoders_map = {
    'resnet18': resnet18,
    'resnet50': resnet50,
}


def get_current_time():
    time_stamp = time.time()
    local_time = time.localtime(time_stamp)
    str_time = time.strftime('%Y-%m-%d_%H-%M-%S', local_time)
    return str_time


class Trainer:
    def __init__(self, args, config):
        self.args = args
        self.config = config
        if args.dset == "PACS":
            self.source_domains = ["photo", "cartoon", "art_painting", "sketch"]
            self.num_classes = 7
        elif args.dset == 'office-home':
            self.source_domains = ["Art", "Clipart", "Product", "RealWorld"]
            self.num_classes = 65
        self.source_domains.remove(args.target_domain)

        self.global_step = 0
        self.source_num = len(self.source_domains)
        self.estimators = []
        self.current_iter = 0

        # networks
        if args.arch == 'resnet18':
            in_dim = 512
        elif args.arch == 'resnet50':
            in_dim = 2048
        self.encoder = encoders_map[args.arch]().cuda()
        self.classifier = Classifier(in_dim=in_dim, num_classes=self.num_classes, bottleneck_dim=args.bottleneck_dim, bias=False).cuda()

        # dataloaders
        self.train_loader, self.len_train = get_train_dataloader(source_list=self.source_domains,
                                                                 batch_size=args.batch_size,
                                                                 image_size=224, crop=True, jitter=0,
                                                                 data_dir=args.data_dir)
        self.val_loader = get_val_dataloader(source_list=self.source_domains, batch_size=args.batch_size,
                                             image_size=224, data_dir=args.data_dir)
        self.test_loader = get_test_loader(target=args.target_domain, batch_size=args.batch_size, image_size=224,
                                           data_dir=args.data_dir)
        self.eval_loader = {'val': self.val_loader, 'test': self.test_loader}

        # optimizers
        max_iters = len(self.train_loader) * args.epochs
        all_parameters = [{'params': self.encoder.parameters(), 'lr_mult': 0.1}, {'params': self.classifier.parameters(), 'lr_mult': 1.0}]
        self.optimizer = SGD(all_parameters, args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
        self.lr_sheduler = LrScheduler(self.optimizer, max_iters, init_lr=args.lr, gamma=args.schdule_gamma, decay_rate=args.decay_rate)

        # critetion
        self.loss_critetion = Loss_aug_pro_for_DG(args.bottleneck_dim, self.num_classes, args.alpha, self.source_num).cuda()


    def _do_epoch(self, epoch):
        # turn on train mode
        self.encoder.train()
        self.classifier.train()

        for it, ((batch, label), domain) in enumerate(self.train_loader):
            self.current_iter += 1
            self.lr_sheduler.step()

            # preprocessing
            batch = batch.cuda()
            label = label.cuda()
            domain = domain.cuda()
            if self.args.dset == 'PACS':
                label = label - 1

            # zero grad
            self.optimizer.zero_grad()

            # forward
            backbone_outputs = self.encoder(batch)
            scores, features = self.classifier(backbone_outputs)

            #--------------------TTSA begin-------------------
            loss_aug = 0.0
            loss_pro = 0.0
            Lambda = self.args.lambda0 * (float(self.current_iter) / float(self.max_iters))
            count = 0
            self.loss_critetion.reset_flags()
            for i in range(self.source_num):
                mask_i = domain == i
                if torch.sum(mask_i.int()) <= 0:
                    continue
                for j in range(self.source_num):
                    if j == i:
                        continue
                    mask_j = domain == j
                    if torch.sum(mask_j.int()) <= 0:
                        continue
                    temp_aug_loss, temp_pro_loss = self.loss_critetion(i, j, self.classifier.weight, features[mask_i],
                                                             features[mask_j],
                                                             scores[mask_i], label[mask_i], label[mask_j], Lambda,
                                                             self.args.eta)
                    loss_aug += temp_aug_loss
                    loss_pro += temp_pro_loss
                    count += 1
            if count != 0:
                loss_aug = loss_aug / count
                loss_pro = loss_pro / count
            # --------------------TTSA end--------------------

            # calculate total loss
            total_loss = loss_aug + self.args.gamma * loss_pro

            # backward
            total_loss.backward()

            # update
            self.optimizer.step()
            self.global_step += 1

            # print training log
            if self.current_iter % self.args.print_freq == 0:
                print(
                    "Epoch: [{:02d}][{}/{}]	total_loss:{:.3f}  loss_aug:{:.3f}    loss_pro:{:.3f}".format(
                        epoch, self.current_iter, self.len_train // 16, total_loss, loss_aug, loss_pro))
                self.config["out_file"].write(
                    "Epoch: [{:02d}][{}/{}]	total_loss:{:.3f}  loss_aug:{:.3f}    loss_pro:{:.3f}\n".format(
                        epoch, self.current_iter, self.len_train // 16, total_loss, loss_aug, loss_pro))
                self.config["out_file"].flush()


        # evaluation
        with torch.no_grad():
            for phase, loader in self.eval_loader.items():
                total = len(loader.dataset)
                class_correct = self.do_eval(loader)
                class_acc = float(class_correct) / total
                class_acc = class_acc * 100
                self.results[phase][self.current_epoch] = class_acc

            # save from best val
            if self.results['val'][self.current_epoch] >= self.best_val_acc:
                self.best_val_acc = self.results['val'][self.current_epoch]
                self.best_val_epoch = self.current_epoch
                # torch.save({
                #     'epoch': self.current_epoch,
                #     'val_acc': self.best_val_acc,
                #     'encoder_state_dict': self.encoder.state_dict(),
                #     'classifier_state_dict': self.classifier.state_dict()
                # }, os.path.join(self.args.output_dir, self.args.target_domain+'_best_model.tar'))

            print("Epoch = {:02d},  current_val_acc={:.3f}, best_val_acc = {:.3f}".format(epoch, self.results['val'][self.current_epoch], self.best_val_acc))
            self.config["out_file"].write("Epoch = {:02d},  current_val_acc={:.3f}, best_val_acc = {:.3f}\n".format(epoch, self.results['val'][self.current_epoch], self.best_val_acc))
            print("Epoch = {:02d},  current_test_acc={:.3f}, best_test_acc = {:.3f}".format(epoch, self.results['test'][self.current_epoch], self.results['test'].max()))
            self.config["out_file"].write("Epoch = {:02d},  current_test_acc={:.3f}, best_test_acc = {:.3f}\n".format(epoch, self.results['test'][self.current_epoch], self.results['test'].max()))
            self.config["out_file"].flush()

    def do_eval(self, loader):
        correct = 0
        # turn on eval mode
        self.encoder.eval()
        self.classifier.eval()
        with torch.no_grad():
            for it, (batch, domain) in enumerate(loader):
                data, labels, domains = batch[0].cuda(), batch[1].cuda(), domain.cuda()
                if self.args.dset == 'PACS':
                    labels = labels - 1
                backbone_outputs = self.encoder(data)
                scores, _ = self.classifier(backbone_outputs)
                correct += calculate_correct(scores, labels)
        return correct


    def do_training(self):
        self.results = {"val": torch.zeros(self.args.epochs), "test": torch.zeros(self.args.epochs)}
        self.best_val_acc = 0
        self.best_val_epoch = 0

        print("\nlen_train={}".format(self.len_train))
        iters_per_epoch = self.len_train // self.args.batch_size
        print("iters_per_epoch={}/{}={}".format(self.len_train, self.args.batch_size, iters_per_epoch))
        self.max_iters = self.args.epochs * iters_per_epoch
        print("max_iters={}*{}={}".format(self.args.epochs, iters_per_epoch, self.max_iters))

        for self.current_epoch in range(self.args.epochs):
            # step schedulers
            self._do_epoch(self.current_epoch)

        # save from best val
        val_res = self.results['val']
        test_res = self.results['test']
        idx_val_best = val_res.argmax()
        idx_test_best = test_res.argmax()
        best_acc_dict = '\nepoch_val_best: {}\nacc_val_best: {:.3f}  acc_val_best_test: {:.3f}\n\nepoch_test_best: {}\nacc_test_best: {:.3f}\n'.format(idx_val_best,
                                                                                                                                                 val_res.max().item(),
                                                                                                                                                 test_res[idx_val_best].item(),
                                                                                                                                                 idx_test_best.item(),
                                                                                                                                                 test_res.max().item())
        print(best_acc_dict)
        self.config["out_file"].write(best_acc_dict)
        self.config["out_file"].flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='resnet50', choices=['resnet18', 'resnet50'])
    parser.add_argument('--batch_size', default=32, type=int, metavar='N', help='mini-batch size (default: 32)')
    parser.add_argument("--data_dir", default='/data1/TL/data/PACS/kfold/', help="The directory of dataset lists")
    parser.add_argument('--dset', type=str, default='PACS', choices=['PACS', 'office-home'], help="The dataset used")
    parser.add_argument("--target_domain", default="sketch", help="Target")
    parser.add_argument("--gpu", default='1', type=str, help="Gpu ID")
    parser.add_argument("--output_dir", default='log/TTSA_for_DG/PACS', help="The directory to save logs and models")
    parser.add_argument('--epochs', default=50, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--print-freq', default=100, type=int, metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--lr', default=0.01, type=float, metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--schdule_gamma', default=10, type=float, help='gamma for lr_sheduler')
    parser.add_argument('--decay_rate', default=0.75, type=float, dest='decay_rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--weight_decay', default=1e-3, type=float, metavar='W', help='weight decay (default: 1e-3)', dest='weight_decay')
    parser.add_argument('--bottleneck_dim', default=256, type=int, help='bottleneck_dim')
    parser.add_argument('--alpha', type=float, default=0.1, help="hyper-parameter for covariance estimation")
    parser.add_argument('--lambda0', type=float, default=0.25, help="hyper-parameter for augmentation strength")
    parser.add_argument('--eta', type=float, default=0.5, help="hyper-parameter for angular margin")
    parser.add_argument('--gamma', type=float, default=0.001, help="tradeoff for loss pro")
    parser.add_argument('--beta', type=float, default=0.1, help="tradeoff for loss MI")
    parser.add_argument('--seed', default=0, type=int, help='seed for initializing training. ')

    args = parser.parse_args()

    torch.multiprocessing.set_sharing_strategy('file_system')
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
    cudnn.benchmark = True

    config = {}
    if not osp.exists(args.output_dir):
        os.makedirs(args.output_dir)
    config["out_file"] = open(osp.join(args.output_dir, get_current_time() + "_" + args.target_domain + "_log.txt"), "w")

    config["out_file"].write("file name: train_TTSA_noMI_lr\n")
    config["out_file"].write("PTL.version = {}".format(PIL.__version__) + "\n")
    config["out_file"].write("torch.version = {}".format(torch.__version__) + "\n")
    config["out_file"].write("torchvision.version = {}".format(torchvision.__version__) + "\n")

    for arg in vars(args):
        print("{} = {}".format(arg, getattr(args, arg)))
        config["out_file"].write(str("{} = {}".format(arg, getattr(args, arg))) + "\n")
    config["out_file"].flush()

    trainer = Trainer(args, config)
    trainer.do_training()


if __name__ == "__main__":
    main()