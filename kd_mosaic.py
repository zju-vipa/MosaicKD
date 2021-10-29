import argparse
import os
import random
import time
import warnings

import registry
import engine
from tqdm import tqdm
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
import time
from torch.utils.tensorboard import SummaryWriter

from PIL import PngImagePlugin
LARGE_ENOUGH_NUMBER = 100
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)

parser = argparse.ArgumentParser(description='MosaicKD for OOD data')
parser.add_argument('--data_root', default='data')
parser.add_argument('--teacher', default='wrn40_2')
parser.add_argument('--student', default='wrn16_1')
parser.add_argument('--dataset', default='cifar10')
parser.add_argument('--unlabeled', default='cifar10')
parser.add_argument('--log_tag', default='')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--lr_g', default=1e-3, type=float)
parser.add_argument('--T', default=1, type=float)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')

parser.add_argument('--z_dim', default=100, type=int)
parser.add_argument('--output_stride', default=1, type=int)
parser.add_argument('--align', default=0.1, type=float)
parser.add_argument('--local', default=0.1, type=float)
parser.add_argument('--adv', default=1.0, type=float)

parser.add_argument('--balance', default=0.0, type=float)

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
parser.add_argument('--ood_subset', action='store_true',
                    help='use ood subset')
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
    log_name = '%s-%s-%s'%(args.dataset, args.teacher, args.student) if args.multiprocessing_distributed else '%s-%s-%s'%(args.dataset, args.teacher, args.student)
    args.logger = engine.utils.logger.get_logger(log_name, output='checkpoints/MosaicKD/log-%s-%s-%s-%s%s.txt'%(args.dataset, args.unlabeled, args.teacher, args.student, args.log_tag))
    args.tb = SummaryWriter(log_dir=os.path.join( 'tb_log', log_name+'_%s'%(time.asctime().replace(' ', '-')) ))
    if args.rank<=0:
        for k, v in engine.utils.flatten_dict( vars(args) ).items(): # print args
            args.logger.info( "%s: %s"%(k,v) )

    ############################################
    # Setup Dataset
    ############################################
    num_classes, ori_dataset, val_dataset = registry.get_dataset(name=args.dataset, data_root=args.data_root)
    _, train_dataset, _ = registry.get_dataset(name=args.unlabeled, data_root=args.data_root)
    _, ood_dataset, _ = registry.get_dataset(name=args.unlabeled, data_root=args.data_root)
    # see Appendix Sec 2, ood data is also used for training
    ood_dataset.transforms = ood_dataset.transform = train_dataset.transform # w/o augmentation
    train_dataset.transforms = train_dataset.transform = val_dataset.transform # w/ augmentation

    ############################################
    # Setup Models
    ############################################
    student = registry.get_model(args.student, num_classes=num_classes)
    teacher = registry.get_model(args.teacher, num_classes=num_classes, pretrained=True).eval()
    teacher.load_state_dict(torch.load('checkpoints/pretrained/%s_%s.pth'%(args.dataset, args.teacher), map_location='cpu')['state_dict'])
    normalizer = engine.utils.Normalizer(**registry.NORMALIZE_DICT[args.dataset])
    args.normalizer = normalizer

    netG = engine.models.generator.Generator(nz=args.z_dim, nc=3, img_size=32)
    netD = engine.models.generator.PatchDiscriminator(nc=3, ndf=128)

    if args.ood_subset and args.unlabeled in ['imagenet_32x32', 'places365_32x32']:
        ood_index = prepare_ood_data(train_dataset, teacher, ood_size=len(ori_dataset), args=args)
        train_dataset.samples = [ train_dataset.samples[i] for i in ood_index]
        ood_dataset.samples = [ ood_dataset.samples[i] for i in ood_index]

    if args.distributed:
        process_group = torch.distributed.new_group()
        netD = torch.nn.SyncBatchNorm.convert_sync_batchnorm(netD, process_group)   
    
    ############################################
    # Device preparation
    ############################################
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
            netD.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            student = torch.nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu])
            teacher = torch.nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
            netG = torch.nn.parallel.DistributedDataParallel(netG, device_ids=[args.gpu])
            netD = torch.nn.parallel.DistributedDataParallel(netD, device_ids=[args.gpu])
        else:
            student.cuda()
            teacher.cuda()
            student = torch.nn.parallel.DistributedDataParallel(student)
            teacher = torch.nn.parallel.DistributedDataParallel(teacher)
            netG = torch.nn.parallel.DistributedDataParallel(netG)
            netD = torch.nn.parallel.DistributedDataParallel(netD)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        student = student.cuda(args.gpu)
        teacher = teacher.cuda(args.gpu)
        netG = netG.cuda(args.gpu)
        netD = netD.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        student = torch.nn.DataParallel(student).cuda()
        teacher = torch.nn.DataParallel(teacher).cuda()
        netG = torch.nn.DataParallel(netG).cuda()
        netD = torch.nn.DataParallel(netD).cuda()

    ############################################
    # Setup dataset
    ############################################
    #cudnn.benchmark = False
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, sampler=train_sampler)
    ood_loader = torch.utils.data.DataLoader(
        ood_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, sampler=train_sampler)
    ood_iter = engine.utils.DataIter(ood_loader)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers)
    
    ############################################
    # Setup optimizer
    ############################################
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    optim_g = torch.optim.Adam(netG.parameters(), lr=args.lr_g, betas=[0.5, 0.999])
    sched_g = torch.optim.lr_scheduler.CosineAnnealingLR( optim_g, T_max=args.epochs*len(train_loader) )
    optim_d = torch.optim.Adam(netD.parameters(), lr=args.lr_g, betas=[0.5, 0.999])
    sched_d = torch.optim.lr_scheduler.CosineAnnealingLR( optim_d, T_max=args.epochs*len(train_loader) )

    optim_s = torch.optim.SGD(student.parameters(), args.lr, momentum=0.9, weight_decay=args.weight_decay)
    sched_s = torch.optim.lr_scheduler.CosineAnnealingLR( optim_s, T_max=args.epochs*len(train_loader) )

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
                student.module.load_state_dict(checkpoint['s_state_dict'])
                netG.module.load_state_dict(checkpoint['g_state_dict'])
                netD.module.load_state_dict(checkpoint['d_state_dict'])
            except:
                student.load_state_dict(checkpoint['s_state_dict'])
                netG.load_state_dict(checkpoint['g_state_dict'])
                netD.load_state_dict(checkpoint['d_state_dict'])
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
    # Evaluate
    ############################################
    if args.evaluate:
        acc1 = validate(0, val_loader, student, criterion, args)
        return

    ############################################
    # Train Loop
    ############################################
    if args.fp16:
        from torch.cuda.amp import autocast, GradScaler
        args.scaler_s = GradScaler() if args.fp16 else None 
        args.scaler_g = GradScaler() if args.fp16 else None 
        args.scaler_d = GradScaler() if args.fp16 else None 
        args.autocast = autocast
    else:
        args.autocast = engine.utils.dummy_ctx

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        args.current_epoch=epoch
        train( [train_loader,ood_iter], val_loader, [student, teacher, netG, netD], criterion, [optim_s, optim_g, optim_d], [sched_s, sched_g, sched_d], epoch, args)

        acc1 = validate(optim_s.param_groups[0]['lr'], val_loader, student, criterion, args)
        args.tb.add_scalar('acc@1', float(acc1), global_step=epoch)
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        _best_ckpt = 'checkpoints/MosaicKD/%s_%s_%s_%s.pth'%(args.dataset, args.unlabeled, args.teacher, args.student)
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.student,
                's_state_dict': student.state_dict(),
                'g_state_dict': netG.state_dict(),
                'd_state_dict': netD.state_dict(),
                'best_acc1': float(best_acc1),
                'optim_s' : optim_s.state_dict(),
                'sched_s': sched_s.state_dict(),
                'optim_d' : optim_d.state_dict(),
                'sched_d': sched_d.state_dict(),
                'optim_g': optim_g.state_dict(),
                'sched_g': sched_g.state_dict(),
            }, is_best, _best_ckpt)
    if args.rank<=0:
        args.logger.info("Best: %.4f"%best_acc1)


def prepare_ood_data(train_dataset, model, ood_size, args):
    model.eval()
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers)
    if os.path.exists('checkpoints/ood_index/%s-%s-%s-ood-index.pth'%(args.dataset, args.unlabeled, args.teacher)):
        ood_index = torch.load('checkpoints/ood_index/%s-%s-%s-ood-index.pth'%(args.dataset, args.unlabeled, args.teacher))
    else:
        with torch.no_grad():
            entropy_list = []
            model.cuda(args.gpu)
            model.eval()
            for i, (images, target) in enumerate(tqdm(train_loader)):
                if args.gpu is not None:
                    images = images.cuda(args.gpu, non_blocking=True)
                if torch.cuda.is_available():
                    target = target.cuda(args.gpu, non_blocking=True)
                # compute output
                output = model(images)
                p = torch.nn.functional.softmax(output, dim=1)
                ent = -(p*torch.log(p)).sum(dim=1)
                entropy_list.append(ent)
            entropy_list = torch.cat(entropy_list, dim=0)
            ood_index = torch.argsort(entropy_list, descending=True)[:ood_size].cpu().tolist()
            model.cpu()
            os.makedirs('checkpoints/ood_index', exist_ok=True)
            torch.save(ood_index, 'checkpoints/ood_index/%s-%s-%s-ood-index.pth'%(args.dataset, args.unlabeled, args.teacher))
    return ood_index

def train(train_loader, val_loader, model, criterion, optimizer, scheduler, epoch, args):
    global best_acc1
    student, teacher, netG, netD = model
    optim_s, optim_g, optim_d = optimizer
    train_loader, ood_iter = train_loader
    sched_s, sched_g, sched_d = scheduler
    student.train()
    teacher.eval()
    netD.train()
    netG.train()
    for i, (real, _) in enumerate(train_loader):
        if args.gpu is not None:
            real = real.cuda(args.gpu, non_blocking=True)

        ###############################
        # Patch Discrimination
        ###############################
        with args.autocast():
            z = torch.randn(size=(args.batch_size, args.z_dim), device=args.gpu)
            images = netG(z)
            images = args.normalizer(images)
            d_out_fake = netD(images.detach())
            d_out_real = netD(real.detach())
            loss_d = (torch.nn.functional.binary_cross_entropy_with_logits(d_out_fake, torch.zeros_like(d_out_fake), reduction='sum') + \
                torch.nn.functional.binary_cross_entropy_with_logits(d_out_real, torch.ones_like(d_out_real), reduction='sum')) / (2*len(d_out_fake)) * args.local
        optim_d.zero_grad()
        if args.fp16:
            scaler_d = args.scaler_d
            scaler_d.scale(loss_d).backward()
            scaler_d.step(optim_d)
            scaler_d.update()
        else:
            loss_d.backward()
            optim_d.step()

        ###############################
        # Generation
        ###############################
        with args.autocast():
            t_out = teacher(images)
            s_out = student(images)

            pyx = torch.nn.functional.softmax(t_out, dim = 1) # p(y|G(z)
            log_softmax_pyx = torch.nn.functional.log_softmax(t_out, dim=1)
            py = pyx.mean(0) #p(y)

            # Mosaicking to distill
            d_out_fake = netD(images)
            # (Eqn. 3) fool the patch discriminator
            loss_local = torch.nn.functional.binary_cross_entropy_with_logits(d_out_fake, torch.ones_like(d_out_fake), reduction='sum') / len(d_out_fake)
            # (Eqn. 4) label space aligning
            loss_align = -(pyx * log_softmax_pyx).sum(1).mean() #torch.nn.functional.cross_entropy(t_out, t_out.max(1)[1])  #-(pyx * torch.log2(pyx)).sum(1).mean() # or torch.nn.functional.cross_entropy(t_out, t_out.max(1)[1]) 
            # (Eqn. 7) fool the student
            loss_adv = - engine.criterions.kldiv(s_out, t_out)

            # Appendix: Alleviating Mode Collapse for unconditional GAN
            loss_balance = (py * torch.log2(py)).sum() 

            # Final loss: L_align + L_local + L_adv (DRO) + L_balance
            loss_g = args.adv * loss_adv + loss_align * args.align + args.local * loss_local + loss_balance * args.balance

        optim_g.zero_grad()
        if args.fp16:
            scaler_g = args.scaler_g
            scaler_g.scale(loss_g).backward()
            scaler_g.step(optim_g)
            scaler_g.update()
        else:
            loss_g.backward()
            optim_g.step()

        ###############################
        # Knowledge Distillation
        ###############################
        for _ in range(5): 
            with args.autocast():
                with torch.no_grad():
                    z = torch.randn(size=(args.batch_size, args.z_dim), device=args.gpu)
                    vis_images = images = netG(z)
                    images = args.normalizer(images)
                    ood_images = ood_iter.next()[0].to(args.gpu)
                    images = torch.cat([images, ood_images]) # here we use both OOD data and synthetic data for training
                    t_out = teacher(images)
                s_out = student(images.detach())
                loss_s = engine.criterions.kldiv(s_out, t_out.detach(), T=args.T)
            optim_s.zero_grad()
            if args.fp16:
                scaler_s = args.scaler_s
                scaler_s.scale(loss_s).backward()
                scaler_s.step(optim_s)
                scaler_s.update()
            else:
                loss_s.backward()
                optim_s.step()

        sched_s.step()
        sched_d.step()
        sched_g.step()

        if args.print_freq>0 and i % args.print_freq == 0:
            acc1 = validate(optim_s.param_groups[0]['lr'], val_loader, student, criterion, args)
            print('Epoch={current_epoch} Iter={i}/{total_iters}, Acc={acc:.4f} loss_s={loss_s:.4f} loss_d={loss_d:.4f} loss_g={loss_g:.4f} (align={loss_align:.4f}, balance={loss_balance:.4f} adv={loss_adv:.4f}) Lr={lr:.4f}'
              .format(current_epoch=epoch, i=i, total_iters=len(train_loader), acc=float(acc1), loss_s=loss_s.item(), loss_d=loss_d.item(), loss_g=loss_g.item(), loss_align=loss_align.item(), loss_balance=loss_balance.item(), loss_adv=loss_adv.item(), lr=optim_s.param_groups[0]['lr']))
            student.train()
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)
            _best_ckpt = 'checkpoints/MosaicKD/%s_%s_%s_%s.pth'%(args.dataset, args.unlabeled, args.teacher, args.student)
            if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                    and args.rank <= 0):
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.student,
                    's_state_dict': student.state_dict(),
                    'g_state_dict': netG.state_dict(),
                    'd_state_dict': netD.state_dict(),
                    'best_acc1': float(best_acc1),
                    'optim_s' : optim_s.state_dict(),
                    'sched_s': sched_s.state_dict(),
                    'optim_d' : optim_d.state_dict(),
                    'sched_d': sched_d.state_dict(),
                    'optim_g': optim_g.state_dict(),
                    'sched_g': sched_g.state_dict(),
                }, is_best, _best_ckpt)
            with args.autocast(), torch.no_grad():
                predict = t_out[:args.batch_size].max(1)[1]
                idx = torch.argsort(predict)
                vis_images = vis_images[idx]
                engine.utils.save_image_batch( args.normalizer(real, True), 'checkpoints/MosaicKD/%s-%s-%s-%s-ood-data.png'%(args.dataset, args.unlabeled, args.teacher, args.student) )
                engine.utils.save_image_batch( vis_images, 'checkpoints/MosaicKD/%s-%s-%s-%s-mosaic-data.png'%(args.dataset, args.unlabeled, args.teacher, args.student) )

        if i==0:
            with args.autocast(), torch.no_grad():
                predict = t_out[:args.batch_size].max(1)[1]
                idx = torch.argsort(predict)
                vis_images = vis_images[idx]
                engine.utils.save_image_batch( args.normalizer(real, True), 'checkpoints/MosaicKD/%s-%s-%s-%s-ood-data.png'%(args.dataset, args.unlabeled, args.teacher, args.student) )
                engine.utils.save_image_batch( vis_images, 'checkpoints/MosaicKD/%s-%s-%s-%s-mosaic-data.png'%(args.dataset, args.unlabeled, args.teacher, args.student) )

def validate(current_lr, val_loader, model, criterion, args):
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    model.eval()
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)
            output = model(images)
            loss = criterion(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
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