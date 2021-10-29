import argparse
import os
import random
import shutil
import time
import warnings

import registry
import engine

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

parser = argparse.ArgumentParser(description='Data-free Knowledge Distillation')
parser.add_argument('--data_root', default='data')
parser.add_argument('--teacher', default='wrn40_2')
parser.add_argument('--student', default='wrn16_1')
parser.add_argument('--dataset', default='cifar10')

parser.add_argument('--log_tag', default='')

parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--lr_g', default=1e-3, type=float)

parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')

parser.add_argument('--z_dim', default=100, type=float)
parser.add_argument('--oh', default=0.5, type=float)
parser.add_argument('--ie', default=20, type=float)
parser.add_argument('--bn', default=1, type=float)
parser.add_argument('--adv', default=1, type=float)
                    
parser.add_argument('-p', '--print-freq', default=0, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--fp16', action='store_true',
                    help='use fp16')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
best_acc1 = 0

def main():
    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    
    if args.log_tag != '':
        args.log_tag = '-'+args.log_tag
    log_name = 'R%d-%s-%s-%s'%(args.rank, args.dataset, args.teacher, args.student) if args.multiprocessing_distributed else '%s-%s-%s'%(args.dataset, args.teacher, args.student)
    args.logger = engine.utils.logger.get_logger(log_name, output='checkpoints/DatafreeKD/log-%s-%s-%s%s.txt'%(args.dataset, args.teacher, args.student, args.log_tag))
    if args.rank<=0:
        for k, v in engine.utils.flatten_dict( vars(args) ).items(): # print args
            args.logger.info( "%s: %s"%(k,v) )

    ############################################
    # Setup models
    ############################################
    num_classes, train_dataset, val_dataset = registry.get_dataset(name=args.dataset, data_root=args.data_root)
    student = registry.get_model(args.student, num_classes=num_classes)
    teacher = registry.get_model(args.teacher, num_classes=num_classes, pretrained=True).eval()
    normalizer = engine.utils.Normalizer(**registry.NORMALIZE_DICT[args.dataset])
    args.normalizer = normalizer
    aug = transforms.Compose([
        #augmentation.RandomCrop(size=[img_size, img_size], padding=4),
        #augmentation.RandomHorizontalFlip(),
        normalizer,
    ])
    args.aug = aug
    netG = engine.models.generator.Generator(nz=args.z_dim, nc=3, img_size=32)
    teacher.load_state_dict(torch.load('checkpoints/pretrained/%s_%s.pth'%(args.dataset, args.teacher))['state_dict'])

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            student.cuda(args.gpu)
            teacher.cuda(args.gpu)
            netG.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            student = torch.nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu])
            teacher = torch.nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
            netG = torch.nn.parallel.DistributedDataParallel(netG, device_ids=[args.gpu])
        else:
            student.cuda()
            teacher.cuda()
            student = torch.nn.parallel.DistributedDataParallel(student)
            teacher = torch.nn.parallel.DistributedDataParallel(teacher)
            netG = torch.nn.parallel.DistributedDataParallel(netG)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        student = student.cuda(args.gpu)
        teacher = teacher.cuda(args.gpu)
        netG = netG.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        student = torch.nn.DataParallel(student).cuda()
        teacher = torch.nn.DataParallel(teacher).cuda()
        netG = torch.nn.DataParallel(netG).cuda()

    ############################################
    # Setup optimizer
    ############################################
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    optim_g = torch.optim.Adam(netG.parameters(), lr=args.lr_g)
    sched_g = torch.optim.lr_scheduler.CosineAnnealingLR( optim_g, T_max=args.epochs )
    optim_s = torch.optim.Adam(student.parameters(), args.lr, weight_decay=args.weight_decay)
    sched_s = torch.optim.lr_scheduler.CosineAnnealingLR( optim_s, T_max=args.epochs )

    ############################################
    # Resume
    ############################################
    args.current_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)

            try:
                student.module.load_state_dict(checkpoint['state_dict'])
            except:
                student.load_state_dict(checkpoint['state_dict'])
            best_acc1 = checkpoint['best_acc1']
            try: 
                args.start_epoch = checkpoint['epoch']
                optim_g.load_state_dict(checkpoint['optim_g'])
                sched_g.load_state_dict(checkpoint['sched_g'])
                optim_s.load_state_dict(checkpoint['optim_s'])
                sched_s.load_state_dict(checkpoint['sched_s'])
            except: print("Fails to load additional model information")
            print("[!] loaded checkpoint '{}' (epoch {} acc {})"
                  .format(args.resume, checkpoint['epoch'], best_acc1))
        else:
            print("[!] no checkpoint found at '{}'".format(args.resume))
        
    ############################################
    # Setup dataset
    ############################################
    cudnn.benchmark = True
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=False, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)
    
    ############################################
    # Evaluate
    ############################################
    if args.evaluate:
        acc1 = validate(0, 0, val_loader, student, criterion, args)
        return

    ############################################
    # Train Loop
    ############################################
    if args.fp16:
        from torch.cuda.amp import autocast, GradScaler
        args.scaler_s = GradScaler() if args.fp16 else None 
        args.scaler_g = GradScaler() if args.fp16 else None 
        args.autocast = autocast
    else:
        args.autocast = engine.utils.dummy_ctx
    
    hooks = []
    for m in teacher.modules():
        if isinstance(m, nn.BatchNorm2d):
            hooks.append(engine.hooks.FeatureMeanVarHook(m))
    args.hooks = hooks
    args.bn_mean = torch.cat([h.module.running_mean for h in hooks], dim=0).to(args.gpu)
    args.bn_var = torch.cat([h.module.running_var for h in hooks], dim=0).to(args.gpu)

    for epoch in range(args.start_epoch, args.epochs):
        #if args.distributed:
        #    train_sampler.set_epoch(epoch)
        args.current_epoch=epoch
        train( [student, teacher, netG], criterion, [optim_s, optim_g], epoch, epoch_length=len(train_loader), args=args)
        sched_s.step()
        sched_g.step()

        acc1 = validate(epoch, optim_s.param_groups[0]['lr'], val_loader, student, criterion, args)
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        _best_ckpt = 'checkpoints/DatafreeKD/%s_%s_%s.pth'%(args.dataset, args.teacher, args.student)
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.student,
                'state_dict': student.state_dict(),
                'best_acc1': float(best_acc1),
                'optim_s' : optim_s.state_dict(),
                'sched_s': sched_s.state_dict(),
                'optim_g': optim_g.state_dict(),
                'sched_g': sched_g.state_dict(),
            }, is_best, _best_ckpt)
    if args.rank<=0:
        args.logger.info("Best: %.4f"%best_acc1)


def train(model, criterion, optimizer, epoch, epoch_length, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    epoch_length = 200
    progress = ProgressMeter(
        epoch_length,
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))
    student, teacher, netG = model
    optim_s, optim_g = optimizer

    student.train()
    teacher.eval()
    end = time.time()
    for i in range(epoch_length):
        # measure data loading time
        data_time.update(time.time() - end)
        z = torch.randn(size=(args.batch_size, args.z_dim), device=args.gpu)
        # compute output
        with args.autocast():
            images = netG(z)
            images = args.normalizer(images)
            t_out, t_feat = teacher(images, return_features=True)
            s_out, s_feat = student(images, return_features=True)
            pred = t_out.data.max(1)[1]
            loss_one_hot = torch.nn.functional.cross_entropy(t_out, pred)
            softmax_o_T = torch.nn.functional.softmax(t_out, dim = 1).mean(dim = 0)
            loss_information_entropy = (softmax_o_T * torch.log10(softmax_o_T)).sum()
            batch_mean = torch.cat([h.mean for h in args.hooks], dim=0)
            batch_var = torch.cat([h.var for h in args.hooks], dim=0)
            loss_bn = (torch.norm( batch_mean - args.bn_mean, 2 ) + torch.norm( batch_var - args.bn_var, 2 ))
            loss_g = -args.adv * engine.criterions.kldiv(s_out, t_out) + loss_one_hot * args.oh + loss_information_entropy * args.ie + args.bn * loss_bn

        optim_g.zero_grad()
        if args.fp16:
            scaler_g = args.scaler_g
            scaler_g.scale(loss_g).backward()
            scaler_g.step(optim_g)
            scaler_g.update()
        else:
            loss_g.backward()
            optim_g.step()
        
        for _ in range(5):
            with args.autocast():
                with torch.no_grad():
                    z = torch.randn(size=(args.batch_size, args.z_dim), device=args.gpu)
                    images = netG(z)
                    images = args.normalizer(images)
                    t_out, t_feat = teacher(images, return_features=True)
                s_out = student(images.detach())
                loss_s = engine.criterions.kldiv(s_out, t_out.detach())
            optim_s.zero_grad()
            if args.fp16:
                scaler_s = args.scaler_s
                scaler_s.scale(loss_s).backward()
                scaler_s.step(optim_s)
                scaler_s.update()
            else:
                loss_s.backward()
                optim_s.step()

        acc1, acc5 = accuracy(s_out, t_out.max(1)[1], topk=(1, 5))
        #losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        if args.print_freq>0 and i % args.print_freq == 0:
            args.logger.info('Epoch={current_epoch} Iter={i}, loss_s={loss_s:.4f} loss_g={loss_g:.4f} (oh={loss_oh:.4f}, ie={loss_ie:.4f} bn={loss_bn:.4f}) Lr={lr:.4f}'
              .format(current_epoch=epoch, i=i, loss_s=loss_s.item(), loss_g=loss_g.item(), loss_oh=loss_one_hot.item(), loss_ie=loss_information_entropy.item(), loss_bn=loss_bn.item(), lr=optim_s.param_groups[0]['lr']))
    engine.utils.save_image_batch( args.normalizer(images, True), 'checkpoints/DatafreeKD/data-free-%s-%s-gpu-%d.png'%(args.teacher, args.student) )

def validate(current_epoch, current_lr, val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)
            # compute output
            output = model(images)
            loss = criterion(output, target)
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
        if args.rank<=0:
            args.logger.info(' [Eval] Epoch={current_epoch} Acc@1={top1.avg:.4f} Acc@5={top5.avg:.4f} Loss={losses.avg:.4f} Lr={lr:.4f}'
                .format(current_epoch=args.current_epoch, top1=top1, top5=top5, losses=losses, lr=current_lr))
    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    if is_best:
        torch.save(state, filename)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

if __name__ == '__main__':
    main()