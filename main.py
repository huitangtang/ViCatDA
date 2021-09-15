##############################################################################
#
# All the codes about the model constructing should be kept in the folder ./models/
# All the codes about the data processing should be kept in the folder ./data/
# The file ./opts.py stores the options
# The file ./trainer.py stores the training and test strategies
# The ./main.py should be simple
#
##############################################################################
import os
import json
import shutil
import torch
import torch.nn as nn
import random
import numpy as np
import torch.backends.cudnn as cudnn
import time
import ipdb

from models.resnet import resnet # construct the baseline model
from trainer import train  # for the training process
from trainer import validate  # for the validation/test process
from opts import opts  # options for the project
from data.prepare_data import generate_dataloader  # prepare data and dataloader
from utils.DomainAdversarialLoss import DomainAdvLoss # domain adversarial loss
from utils.InteractionLossOnTarget import InteractionLossOnTarget # cross-domain weighting loss (target)
from utils.EntropyMinimizationLoss import EMLossForTarget # entropy minimization loss (target)
from utils.consensus_loss import MinEntropyConsensusLoss # consistency loss (target)

best_prec1 = 0

def main():
    global args, best_prec1
    args = opts()
    
    current_epoch = 0
    
    # define base model
    model = resnet(args)
    # define multi-GPU
    model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    criterion_domainAdv = DomainAdvLoss(nClass=args.num_classes).cuda()
    criterion_inter = InteractionLossOnTarget(nClass=args.num_classes).cuda()
    criterion_emp = EMLossForTarget(nClass=args.num_classes).cuda()
    criterion_mec = MinEntropyConsensusLoss(nClass=args.num_classes, div=args.div).cuda()
    
    np.random.seed(2) # may fix data
    random.seed(2)
    torch.manual_seed(2)
    if torch.cuda.device_count() > 1:
        torch.cuda.manual_seed_all(2)
    
    if args.arch == 'resnet50':
        reverse_grad_layer_index = 159
    elif args.arch == 'resnet101':
        reverse_grad_layer_index = 312
    else:
        raise ValueError('Undefined layer index for reversing gradient!')
    
    # apply different learning rates to different layers
    lr_fe = args.lr * 0.1 if args.pretrained else args.lr
    if args.arch.find('resnet') != -1:
        params_list = [
                {'params': model.module.conv1.parameters(), 'lr': lr_fe},
                {'params': model.module.bn1.parameters(), 'lr': lr_fe},
                {'params': model.module.layer1.parameters(), 'lr': lr_fe},
                {'params': model.module.layer2.parameters(), 'lr': lr_fe},
                {'params': model.module.layer3.parameters(), 'lr': lr_fe},
                {'params': model.module.layer4.parameters(), 'lr': lr_fe},
                {'params': model.module.fc.parameters()},
        ]
        if args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(params_list,
                lr=args.lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
                nesterov=args.nesterov)
        if args.lr_scheduler == 'dann':
            lr_lambda = lambda epoch: 1 / pow((1 + 10 * epoch / args.epochs), 0.75)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)
        elif args.lr_scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0, last_epoch=-1)
        elif args.lr_scheduler == 'step':
            lr_lambda = lambda epoch: args.gamma ** (epoch + 1 > args.decay_epoch[1] and 2 or epoch + 1 > args.decay_epoch[0] and 1 or 0)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)
    else:
        raise ValueError('Unavailable model architecture!!!')

    if args.resume:
        print("==> loading checkpoints '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        current_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print("==> loaded checkpoint '{}'(epoch {})"
              .format(args.resume, checkpoint['epoch']))
    if not os.path.isdir(args.log):
        os.makedirs(args.log)
    log = open(os.path.join(args.log, 'log.txt'), 'a')
    state = {k: v for k, v in args._get_kwargs()}
    log.write(json.dumps(state) + '\n')
    log.close()

    # start time
    log = open(os.path.join(args.log, 'log.txt'), 'a')
    log.write('\n-------------------------------------------\n')
    log.write(time.asctime(time.localtime(time.time())))
    log.write('\n-------------------------------------------')
    log.close()

    cudnn.benchmark = True
    # process data and prepare dataloaders
    train_loader_source, train_loader_target, val_loader_target, val_loader_source = generate_dataloader(args)
    train_loader_target.dataset.tgts = list(np.array(torch.LongTensor(train_loader_target.dataset.tgts).fill_(-1)))

    if args.test_only:
        prec1 = validate(val_loader_target, model, criterion, -1, args)
        print('test acc@1: {:.3f}'.format(prec1))
        return
        
    print('begin training')
    train_loader_source_batch = enumerate(train_loader_source)
    train_loader_target_batch = enumerate(train_loader_target)
    batch_number = count_epoch_on_large_dataset(train_loader_target, train_loader_source)
    num_itern_total = args.epochs * batch_number
    test_freq = int(num_itern_total / 200)
    print('test_freq: ', test_freq)
    args.start_epoch = current_epoch
    train_records = {'batch_time': AverageMeter(),
                    'data_time': AverageMeter(),
                    'losses_min': AverageMeter(),
                    'losses_max': AverageMeter(),
                    'top1_source': AverageMeter(),
                    'top1_target': AverageMeter(),}
    for itern in range(args.start_epoch * batch_number, num_itern_total):
        # train for one iteration
        train_loader_source_batch, train_loader_target_batch = train(train_loader_source, train_loader_source_batch, train_loader_target, train_loader_target_batch, model, criterion, criterion_domainAdv, criterion_inter, criterion_emp, criterion_mec, optimizer, itern, current_epoch, reverse_grad_layer_index, train_records, args)
        # evaluate on the val data
        if (itern + 1) % batch_number == 0 or (itern + 1) % test_freq == 0:
            prec1 = validate(val_loader_target, model, criterion, current_epoch, args)
            # record the best prec1
            is_best = prec1 > best_prec1
            if is_best:
                best_prec1 = prec1
                log = open(os.path.join(args.log, 'log.txt'), 'a')
                log.write('\n                                                                         best top-1 acc: %3f' % (best_prec1))
                log.close()
            
            # update learning rate
            if (itern + 1) % batch_number == 0:
                scheduler.step()
                current_epoch += 1
            
            # save checkpoint
            save_checkpoint({
                'epoch': current_epoch,
                'arch': args.arch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_prec1': best_prec1,
            }, is_best, args)
            
            for k in train_records.keys():
                train_records[k].reset()
            
        if current_epoch > args.stop_epoch:
            break
    
    # end time
    log = open(os.path.join(args.log, 'log.txt'), 'a')
    log.write('\n * best_prec1: %3f' % best_prec1)
    log.write('\n-------------------------------------------\n')
    log.write(time.asctime(time.localtime(time.time())))
    log.write('\n-------------------------------------------\n')
    log.close()
    

def count_epoch_on_large_dataset(train_loader_target, train_loader_source):
    batch_number_t = len(train_loader_target)
    batch_number = batch_number_t
    batch_number_s = len(train_loader_source)
    if batch_number_s > batch_number_t:
        batch_number = batch_number_s
    
    return batch_number


def save_checkpoint(state, is_best, args):
    filename = 'final_checkpoint.pth.tar'
    dir_save_file = os.path.join(args.log, filename)
    torch.save(state, dir_save_file)
    if is_best:
        shutil.copyfile(dir_save_file, os.path.join(args.log, 'model_best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        

if __name__ == '__main__':
    main()





