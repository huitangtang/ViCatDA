import time
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import os
import math
import ipdb
import numpy as np


def train(train_loader_source, train_loader_source_batch, train_loader_target, train_loader_target_batch, model, criterion, criterion_domainAdv, criterion_inter, criterion_emp, criterion_mec, optimizer, itern, current_epoch, reverse_grad_layer_index, train_records, args):
    model.train() # turn to training mode

    lam = 2 / (1 + math.exp(-1 * 10 * current_epoch / args.epochs)) - 1 # penalty parameter

    end = time.time()
    # prepare data for model forward and backward
    try:
        (input_source, target_source) = train_loader_source_batch.__next__()[1]
    except StopIteration:
        train_loader_source_batch = enumerate(train_loader_source)
        (input_source, target_source) = train_loader_source_batch.__next__()[1]
    target_source = target_source.cuda(non_blocking=True)
    input_source_var = Variable(input_source)
    target_source_var = Variable(target_source)
    target_source_var2 = Variable(target_source + args.num_classes)

    try:
        data = train_loader_target_batch.__next__()[1]
    except StopIteration:
        train_loader_target_batch = enumerate(train_loader_target)
        data = train_loader_target_batch.__next__()[1]
    
    input_target = data[0]
    input_target_var = Variable(input_target)

    train_records['data_time'].update(time.time() - end)
    
    # model forward for source/target data
    _, output_s = model(input_source_var)
    _, output_t = model(input_target_var)
    
    # loss computation
    if args.vda:
        alpha = np.random.beta(args.alpha, args.alpha) # sample the convex combination coefficient from a Beta distribution
        input_mix_st_var = Variable(alpha * input_source + (1 - alpha) * input_target) # vicinal domain generation
        _, output_mix_st = model(input_mix_st_var) # model forward for vicinal data
        # vicatda objective
        loss_min_temp = alpha * criterion_domainAdv(output_mix_st, classifier='S') + (1 - alpha) * criterion_domainAdv(output_mix_st) + lam * (alpha * criterion(output_mix_st, target_source_var) + (1 - alpha) * criterion_inter(output_mix_st, output_t, args.eps, consistent=args.consistent)) + criterion(output_s[:, :args.num_classes], target_source_var) + criterion(output_s[:, args.num_classes:], target_source_var)
        loss_max_temp = lam * (alpha * criterion_domainAdv(output_mix_st) + (1 - alpha) * criterion_domainAdv(output_mix_st, classifier='S') + alpha * criterion(output_mix_st, target_source_var2) + (1 - alpha) * criterion_inter(output_mix_st, output_t, args.eps, consistent=args.consistent, classifier='S')) + 0.5 * (criterion(output_s[:, :args.num_classes], target_source_var) + criterion(output_s[:, args.num_classes:], target_source_var))
    else:
        # catda objective
        loss_min_temp = criterion_domainAdv(output_s, classifier='S') + criterion_domainAdv(output_t) + lam * (criterion(output_s, target_source_var) + criterion_inter(output_t, output_t, args.eps, consistent=args.consistent)) + criterion(output_s[:, :args.num_classes], target_source_var) + criterion(output_s[:, args.num_classes:], target_source_var)
        loss_max_temp = lam * (criterion_domainAdv(output_t, classifier='S') + criterion_domainAdv(output_s) + criterion(output_s, target_source_var2) + criterion_inter(output_t, output_t, args.eps, consistent=args.consistent, classifier='S')) + 0.5 * (criterion(output_s[:, :args.num_classes], target_source_var) + criterion(output_s[:, args.num_classes:], target_source_var))
    
    loss_min = loss_min_temp + math.log(1 / args.num_classes) - F.softmax(torch.cat([output_t[:, :args.num_classes], output_t[:, args.num_classes:]], dim=0), dim=1).mean(0).log().mean() if args.cls_blc else loss_min_temp # whether adding class balance loss
    loss_max = loss_max_temp + lam * criterion_emp(output_t, args.eps, separate=False) if args.emp else loss_max_temp # whether following entropy minimization principle
        
    if args.aug_tar_agree: # consistency loss
        input_target_dup = data[1]
        input_target_dup_var = Variable(input_target_dup)
        _, output_t_dup = model(input_target_dup_var) # model forward for strongly augmented target data
        if not args.two_consistency: 
            loss_min = loss_min + lam * criterion_mec(output_t, output_t_dup)
            loss_max = loss_max + lam * criterion_mec(output_t, output_t_dup)
        else:
            loss_min = loss_min + lam * (criterion_mec(output_t[:, :args.num_classes], output_t_dup[:, :args.num_classes]) + criterion_mec(output_t[:, args.num_classes:], output_t_dup[:, args.num_classes:]))
            loss_max = loss_max + lam * (criterion_mec(output_t[:, :args.num_classes], output_t_dup[:, :args.num_classes]) + criterion_mec(output_t[:, args.num_classes:], output_t_dup[:, args.num_classes:]))
            
    if args.gray_tar_agree: # consistency loss
        input_target_gray = data[-2]
        input_target_gray_var = Variable(input_target_gray)
        _, output_t_gray = model(input_target_gray_var) # model forward for grayscale target data
        if not args.two_consistency:
            loss_min = loss_min + lam * criterion_mec(output_t, output_t_gray)
            loss_max = loss_max + lam * criterion_mec(output_t, output_t_gray)
        else:
            loss_min = loss_min + lam * (criterion_mec(output_t[:, :args.num_classes], output_t_gray[:, :args.num_classes]) + criterion_mec(output_t[:, args.num_classes:], output_t_gray[:, args.num_classes:]))
            loss_max = loss_max + lam * (criterion_mec(output_t[:, :args.num_classes], output_t_gray[:, :args.num_classes]) + criterion_mec(output_t[:, args.num_classes:], output_t_gray[:, args.num_classes:]))
    
    # record losses and accuracies on source data
    train_records['losses_min'].update(loss_min.item(), input_source.size(0))
    train_records['losses_max'].update(loss_max.item(), input_source.size(0))
    prec1 = accuracy(output_s[:, :args.num_classes], target_source, topk=(1,))[0]
    train_records['top1_source'].update(prec1.item(), input_source.size(0))
    prec1 = accuracy(output_s[:, args.num_classes:], target_source, topk=(1,))[0]
    train_records['top1_target'].update(prec1.item(), input_source.size(0))

    model.zero_grad()
    loss_min.backward(retain_graph=True)
    temp_grad = []
    for param in model.parameters():
        temp_grad.append(param.grad.data.clone())
    grad_for_classifier = temp_grad
    
    model.zero_grad()
    loss_max.backward()
    
    count = 0
    for param in model.parameters():
        if count >= reverse_grad_layer_index:
            temp_grad = param.grad.data.clone()
            temp_grad.zero_()
            temp_grad = temp_grad + grad_for_classifier[count]
            temp_grad = temp_grad
            param.grad.data = temp_grad
        count = count + 1
    
    optimizer.step()
    model.zero_grad()

    train_records['batch_time'].update(time.time() - end)
    if itern % args.print_freq == 0:
        display = 'Train - epoch [{0}/{1}]({2})'.format(current_epoch, args.epochs, itern)
        for k in train_records.keys():
            display += '\t' + k + ': {ph.avg:.3f}'.format(ph=train_records[k])
        print(display)

        log = open(os.path.join(args.log, 'log.txt'), 'a')
        log.write('\n' + display.replace('\t', ', '))
        log.close()
    
    return train_loader_source_batch, train_loader_target_batch


def validate(val_loader_target, model, criterion, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_s = AverageMeter()
    top1_s = AverageMeter()
    losses_t = AverageMeter()
    top1_t = AverageMeter()
    losses_avg = AverageMeter()
    top1_avg = AverageMeter()
    
    mcp_s = MeanClassPrecision(args.num_classes)
    mcp_t = MeanClassPrecision(args.num_classes)
    mcp_avg = MeanClassPrecision(args.num_classes)
    
    model.eval() # turn to eval mode

    end = time.time()
    for i, (input, target) in enumerate(val_loader_target):  # iterarion on the target dataset
        data_time.update(time.time() - end)
        target = target.cuda(non_blocking=True)
        input_var = Variable(input)
        target_var = Variable(target)
        with torch.no_grad():
            _, output = model(input_var)
        
        loss_s = criterion(output[:, :args.num_classes], target_var)
        prec1 = accuracy(output.data[:, :args.num_classes], target, topk=(1,))[0]
        losses_s.update(loss_s.item(), input.size(0))
        top1_s.update(prec1.item(), input.size(0))
        mcp_s.update(output.data[:, :args.num_classes], target)
        
        loss_t = criterion(output[:, args.num_classes:], target_var)
        prec1 = accuracy(output.data[:, args.num_classes:], target, topk=(1,))[0]
        losses_t.update(loss_t.item(), input.size(0))
        top1_t.update(prec1.item(), input.size(0))
        mcp_t.update(output.data[:, args.num_classes:], target)
        
        loss_avg = criterion(output[:, :args.num_classes]+output[:, args.num_classes:], target_var)
        prec1 = accuracy(output.data[:, :args.num_classes]+output.data[:, args.num_classes:], target, topk=(1,))[0]
        losses_avg.update(loss_avg.item(), input.size(0))
        top1_avg.update(prec1.item(), input.size(0))
        mcp_avg.update(output.data[:, :args.num_classes]+output.data[:, args.num_classes:], target)
        
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            print('Eval - epoch [{0}][{1}/{2}]\t'
                  'batch time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'data time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss (F^s) {sc_loss.val:.3f} ({sc_loss.avg:.3f})\t'
                  'acc@1 (F^s) {sc_top1.val:.3f} ({sc_top1.avg:.3f})\t'
                  'loss (F^t) {tc_loss.val:.3f} ({tc_loss.avg:.3f})\t'
                  'acc@1 (F^t) {tc_top1.val:.3f} ({tc_top1.avg:.3f})\t'
                  'loss (F^s+F^t) {avg_loss.val:.3f} ({avg_loss.avg:.3f})\t'
                  'acc@1 (F^s+F^t) {avg_top1.val:.3f} ({avg_top1.avg:.3f})'.format(
                   epoch, i, len(val_loader_target), 
                   batch_time=batch_time, data_time=data_time, 
                   sc_loss=losses_s, sc_top1=top1_s, 
                   tc_loss=losses_t, tc_top1=top1_t, 
                   avg_loss=losses_avg, avg_top1=top1_avg, 
                ))

    print(' * Prec@1 (F^s) {sc_top1.avg:.3f} \n * Prec@1 (F^t) {tc_top1.avg:.3f} \n * Prec@1 (F^s+F^t) {avg_top1.avg:.3f}'
          .format(sc_top1=top1_s, tc_top1=top1_t, avg_top1=top1_avg))
    print('F^s - ' + str(mcp_s) + '\nF^t - ' + str(mcp_t) + '\nF^s+F^t - ' + str(mcp_avg))

    log = open(os.path.join(args.log, 'log.txt'), 'a')
    log.write("\nEval on target data - epoch: %d, loss (F^s): %3f, top1 acc (F^s): %3f, loss (F^t): %3f, top1 acc (F^t): %3f, loss (F^s+F^t): %3f, top1 acc (F^s+F^t): %3f" %\
              (epoch, losses_s.avg, top1_s.avg, losses_t.avg, top1_t.avg, losses_avg.avg, top1_avg.avg))
    log.write('\nF^s - ' + str(mcp_s) + '\nF^t - ' + str(mcp_t) + '\nF^s+F^t - ' + str(mcp_avg))
    
    
    return top1_t.avg if args.src.find('visda') == -1 else mcp_t.mean_class_prec #max(top1_t.avg, top1_s.avg, top1_avg.avg)


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


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class MeanClassPrecision(object):
    """Computes and stores the mean class precision"""
    def __init__(self, num_classes, fmt=':.3f'):
        self.num_classes = num_classes
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.total_vector = torch.zeros(self.num_classes)
        self.correct_vector = torch.zeros(self.num_classes)
        self.per_class_prec = torch.zeros(self.num_classes)
        self.mean_class_prec = 0

    def update(self, output, target):
        pred = output.max(1)[1]
        correct = pred.eq(target).float().cpu()
        for i in range(target.size(0)):
            self.total_vector[target[i]] += 1
            self.correct_vector[target[i]] += correct[i]
        temp = torch.zeros(self.total_vector.size())
        temp[self.total_vector == 0] = 1e-6
        self.per_class_prec = self.correct_vector / (self.total_vector + temp) * 100
        self.mean_class_prec = self.per_class_prec.mean().item()
    
    def __str__(self):
        fmtstr = 'per-class prec: ' + '|'.join([str(i) for i in list(np.around(np.array(self.per_class_prec), int(self.fmt[-2])))])
        fmtstr = 'Mean class prec: {mean_class_prec' + self.fmt + '}, ' + fmtstr
        return fmtstr.format(**self.__dict__)
        
        
