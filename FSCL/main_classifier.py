from __future__ import print_function

import sys
import argparse
import time
import math

import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets
from dataset import UTKLoader, CelebaLoader

from util import AverageMeter
from util import adjust_learning_rate, warmup_learning_rate, accuracy,val_accuracy
from util import set_optimizer
from networks.resnet_big import FairSupConResNet, LinearClassifier


try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=100,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=5,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of training epochs')
    parser.add_argument('--size', type=int, default=64, help='parameter for RandomResizedCrop')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='60,75,90',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--dataset', type=str, default='celeba',
                        choices=['celeba','utkface'], help='dataset')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument('--name', type=str, default='',help='saved filename')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')

    parser.add_argument('--ckpt', type=str, default='',
                        help='path to pre-trained model')


    # attribute
    parser.add_argument('--target_attribute_1', type=str, default='',help='target attribute')
    parser.add_argument('--target_attribute_2', type=str, default='None',help='target attribute')
    parser.add_argument('--sensitive_attribute_1', type=str, default='',help='sensitive_attribute')
    parser.add_argument('--sensitive_attribute_2', type=str, default='None',help='sensitive_attribute')


    opt = parser.parse_args()

    # set the path according to the environment
    

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_lr_{}_decay_{}_bsz_{}'.\
        format(opt.dataset, opt.model, opt.learning_rate, opt.weight_decay,
               opt.batch_size)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    if opt.target_attribute_2=='None':
        opt.ta_cls = 2
    else:
        opt.ta_cls = 4

    if opt.sensitive_attribute_2=='None':
        opt.sa_cls = 2
    else:
        opt.sa_cls = 4

    return opt


def set_loader(opt):
    # construct data loader
  
    mean = (0.5000, 0.5000, 0.5000)
    std = (0.5000, 0.5000, 0.5000)
  
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=opt.size, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])


    if opt.dataset == 'utkface':
        train_dataset = UTKLoader(0,ta=opt.target_attribute_1,sa=opt.sensitive_attribute_1,data_folder=opt.data_folder,transform=train_transform)
        val_dataset = UTKLoader(2,ta=opt.target_attribute_1,sa=opt.sensitive_attribute_1,data_folder=opt.data_folder,transform=val_transform)

    elif opt.dataset == 'celeba':
        train_dataset = CelebaLoader(0,ta=opt.target_attribute_1,ta2=opt.target_attribute_2,sa=opt.sensitive_attribute_1,sa2=opt.sensitive_attribute_2,data_folder=opt.data_folder,transform=train_transform)
        val_dataset = CelebaLoader(2,ta=opt.target_attribute_1,ta2=opt.target_attribute_2,sa=opt.sensitive_attribute_1,sa2=opt.sensitive_attribute_2,data_folder=opt.data_folder,transform=val_transform)

    else:
        raise ValueError(opt.dataset)

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=opt.batch_size, shuffle=True,
        num_workers=8, pin_memory=True)

    return train_loader, val_loader



def set_model(opt):
    model = FairSupConResNet(name=opt.model)
    
    criterion = torch.nn.CrossEntropyLoss()
    classifier = LinearClassifier(name=opt.model, num_classes=opt.ta_cls)

    ckpt = torch.load(opt.ckpt, map_location='cpu')
    state_dict = ckpt['model']

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        else:
            new_state_dict = {}
            for k, v in state_dict.items():
                k = k.replace("module.", "")
                new_state_dict[k] = v
            state_dict = new_state_dict
        model = model.cuda()
        classifier = classifier.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

        model.load_state_dict(state_dict)

    return model, classifier, criterion


def train(train_loader, model, classifier, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.eval()
    classifier.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    for idx, (images, ta, sa) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        ta = ta.cuda(non_blocking=True)
        bsz = ta.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        with torch.no_grad():
            features = model.encoder(images)
        output = classifier(features.detach())
        loss = criterion(output, ta)

        # update metric
        losses.update(loss.item(), bsz)
        acc1, acc5 = accuracy(output, ta, topk=(1, 1))
        top1.update(acc1[0], bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))
            sys.stdout.flush()

    return losses.avg, top1.avg


def validate(val_loader, model, classifier, criterion, opt):
    """validation"""
    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    groupAcc=[]
    for i in range(opt.ta_cls):
        saGroupAcc=[]
        for j in range(opt.sa_cls):
            saGroupAcc.append(AverageMeter())
        groupAcc.append(saGroupAcc)

    with torch.no_grad():
        end = time.time()
        for idx, (images, ta,sa) in enumerate(val_loader):
            images = images.float().cuda()
            ta = ta.cuda()
            sa=sa.cuda()
            bsz = ta.shape[0]

            # forward
            output = classifier(model.encoder(images))          
            loss = criterion(output, ta)

            # update metric
            losses.update(loss.item(), bsz)
            acc1, group_acc,group_num = val_accuracy(output, ta, sa, opt.ta_cls, opt.sa_cls, topk=(1, 1))
            top1.update(acc1, bsz)

            for i in range(opt.ta_cls):
                for j in range(opt.sa_cls):
                    groupAcc[i][j].update(group_acc[i][j],group_num[i][j])
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time,
                       loss=losses, top1=top1))


    odds=0
    odds_num=0
    for i in range(opt.ta_cls):
        for j in range(opt.sa_cls):
            for k in range(j+1,opt.sa_cls):
                odds_num+=1
                odds+=torch.abs(groupAcc[i][j].avg-groupAcc[i][k].avg)
   
    EO = odds=(odds/odds_num).item()
    print('\n * Acc@1 {top1.avg:.3f}'.format(top1=top1))
    print(' * Equalized Odds {odds:.3f}'.format(odds=(odds/odds_num)))#.item()))
    print(' * Group-wise accuracy')
    for i in range(opt.ta_cls):
        string='    Target class '+str(i)+'\n    '
        for j in range(opt.sa_cls):
            string+= '    Sensitive class '+str(j)+': {groupAcc.avg:.3f}'.format(groupAcc=groupAcc[i][j])
        print(string+'\n') 

    return losses.avg, top1.avg, EO


def main():
    best_acc = 0
    best_EO = 100
    opt = parse_option()
    
    # build data loader
    train_loader, val_loader = set_loader(opt)

    # build model and criterion
    model, classifier, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, classifier)

    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss, acc = train(train_loader, model, classifier, criterion,
                          optimizer, epoch, opt)
        time2 = time.time()
        print('Train epoch {}, total time {:.2f}, accuracy:{:.2f}'.format(
            epoch, time2 - time1, acc))

        # eval for one epoch
        loss, val_acc, EO = validate(val_loader, model, classifier, criterion, opt)
        if val_acc > best_acc:
            best_acc = val_acc
            best_acc_EO = EO
            best_acc_epoch = epoch
        if EO < best_EO:
            best_EO = EO
            best_EO_acc = val_acc
            best_EO_epoch = epoch
        if epoch == 10:
            acc_10 = val_acc
            EO_10 = EO

    print('best accuracy / EO / epoch : {:.2f} / {:.2f} / {}'.format(best_acc, best_acc_EO, best_acc_epoch))
    print('accuracy / best EO / epoch : {:.2f} / {:.2f} / {}'.format(best_EO_acc, best_EO, best_EO_epoch))
    print('10 epoch accuracy / EO / epoch : {:.2f} / {:.2f} / {}'.format(acc_10, EO_10, 10))


if __name__ == '__main__':
    main()
