# train.py
#!/usr/bin/env	python3

""" train network using pytorch
    Junde Wu
"""

import os
import sys
import argparse
from datetime import datetime
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, accuracy_score,confusion_matrix
import torchvision
import torchvision.transforms as transforms
from skimage import io
from torch.utils.data import DataLoader
#from dataset import *
from torch.autograd import Variable
from PIL import Image
from tensorboardX import SummaryWriter
#from models.discriminatorlayer import discriminator
from dataset import *
from conf import settings
import time
import cfg
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from utils import *
import function 
import albumentations as A
from albumentations.pytorch import ToTensorV2
from prismalearn.data.benchmarks import EVICANDataModule, CaDISDataModule

import cv2

args = cfg.parse_args()

GPUdevice = torch.device('cuda', args.gpu_device)

net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution = args.distributed)
if args.pretrain:
    weights = torch.load(args.pretrain)
    net.load_state_dict(weights,strict=False)

optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5) #learning rate decay
#scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, epochs=settings.EPOCH, steps_per_epoch=1, verbose=True)

'''load pretrained model'''
if args.weights != 0:
    print(f'=> resuming from {args.weights}')
    assert os.path.exists(args.weights)
    checkpoint_file = os.path.join(args.weights)
    assert os.path.exists(checkpoint_file)
    loc = 'cuda:{}'.format(args.gpu_device)
    checkpoint = torch.load(checkpoint_file, map_location=loc)
    start_epoch = checkpoint['epoch']
    best_tol = checkpoint['best_tol']
    
    net.load_state_dict(checkpoint['state_dict'],strict=False)
    # optimizer.load_state_dict(checkpoint['optimizer'], strict=False)

    args.path_helper = checkpoint['path_helper']
    logger = create_logger(args.path_helper['log_path'])
    print(f'=> loaded checkpoint {checkpoint_file} (epoch {start_epoch})')

args.path_helper = set_log_dir('logs', args.exp_name)
logger = create_logger(args.path_helper['log_path'])
logger.info(args)
vis_path = os.path.join("/home/zozchaab/Medical-SAM-Adapter/vis",args.exp_name,settings.TIME_NOW)


'''segmentation data'''
transform_train = transforms.Compose([
    transforms.Resize((args.image_size,args.image_size)),
    transforms.ToTensor(),
])

transform_train_seg = transforms.Compose([
    transforms.Resize((args.out_size,args.out_size)),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.Resize((args.image_size, args.image_size)),
    transforms.ToTensor(),
])

transform_test_seg = transforms.Compose([
    transforms.Resize((args.out_size,args.out_size)),
    transforms.ToTensor(),
])


if args.dataset == 'isic':
    '''isic data'''
    isic_train_dataset = ISIC2016(args, args.data_path, transform = transform_train, transform_msk= transform_train_seg, mode = 'Training')
    isic_test_dataset = ISIC2016(args, args.data_path, transform = transform_test, transform_msk= transform_test_seg, mode = 'Test')

    nice_train_loader = DataLoader(isic_train_dataset, batch_size=args.b, shuffle=True, num_workers=8, pin_memory=True)
    nice_test_loader = DataLoader(isic_test_dataset, batch_size=args.b, shuffle=False, num_workers=8, pin_memory=True)
    '''end'''

elif args.dataset == 'decathlon':
    nice_train_loader, nice_test_loader, transform_train, transform_val, train_list, val_list = get_decath_loader(args)


elif args.dataset == 'REFUGE':
    '''REFUGE data'''
    refuge_train_dataset = REFUGE(args, args.data_path, transform = transform_train, transform_msk= transform_train_seg, mode = 'Training')
    refuge_test_dataset = REFUGE(args, args.data_path, transform = transform_test, transform_msk= transform_test_seg, mode = 'Test')
    
    nice_train_loader = DataLoader(refuge_train_dataset, batch_size=args.b, shuffle=True, num_workers=8, pin_memory=True)
    nice_test_loader = DataLoader(refuge_test_dataset, batch_size=args.b, shuffle=False, num_workers=8, pin_memory=True)
    '''end'''

elif args.dataset == 'iqs_dv':
    #spilt_data(data_path = '/home/zozchaab/data/deepvision_reduced/iqs_dv'  ,destination_path = '/home/zozchaab/data/deepvision_reduced')
    
    transform_msk_3D = transforms.Compose([
    FillMissingCells(desired_shape=(1,120,120,120)),
    #transforms.functional.crop(crop_size),
    ])

    transform_3D = transforms.Compose([
    FillMissingCells(desired_shape=(1,120,120,120)),
    #transforms.functional.crop(crop_size),
    ])
    transform_2d = transforms.Compose([
    lambda x: x.expand(3, -1, -1),
    transforms.Lambda(lambda x: x / 65535.0),
    
    ])

    train_dataset = iqs_dv(data_path=os.path.join(args.data_path,'iqs_dv_test'),crop_size=args.crop_size, transform_3D=transform_3D, transform_msk_3D=transform_msk_3D,transform_2D=transform_2d)
    val_dataset = iqs_dv(data_path=os.path.join(args.data_path,'iqs_dv_val'),crop_size=args.crop_size, transform_3D=transform_3D, transform_msk_3D=transform_msk_3D,transform_2D=transform_2d)
    nice_train_loader = DataLoader(
        train_dataset,
        batch_size=args.b,
        shuffle=True,
        num_workers=args.w,
    collate_fn=collate_fn
    ) 
    nice_test_loader = DataLoader(
        val_dataset,
        batch_size=args.b,
        shuffle=False,
        num_workers=args.w,
    collate_fn=collate_fn
    ) 



'''checkpoint path and tensorboard'''
# iter_per_epoch = len(Glaucoma_training_loader)
checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)
#use tensorboard
if not os.path.exists(settings.LOG_DIR):
    os.mkdir(settings.LOG_DIR)
writer = SummaryWriter(log_dir=os.path.join(
        settings.LOG_DIR, args.net, settings.TIME_NOW))
# input_tensor = torch.Tensor(args.b, 3, 256, 256).cuda(device = GPUdevice)
# writer.add_graph(net, Variable(input_tensor, requires_grad=True))

#create checkpoint folder to save model
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)
checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

'''begain training'''
best_acc = 0.0
best_tol = 0.0


for epoch in range(settings.EPOCH):
    if args.mod == 'sam_adpt':
        
        """if epoch < 5:
            tol, (eiou, edice) = function.validation_sam(args, nice_test_loader, epoch, net, writer)
            logger.info(f'Total score: {tol}, IOU: {eiou}, DICE: {edice} || @ epoch {epoch}.')"""
            
        net.train()
        time_start = time.time()
        if args.dataset in ['evican', 'oct']:
            loss = function.train_sam_evican(args, net, optimizer, nice_train_loader, epoch, writer, vis = args.vis, schedulers=scheduler)
        elif args.dataset == 'cadis':
            loss = function.train_sam_cadis(args, net, optimizer, nice_train_loader, epoch, writer, vis = args.vis, schedulers=scheduler)
        elif args.dataset == 'iqs_dv':
            loss = function.train_sam_deepvision(args, net, optimizer, nice_train_loader, epoch, writer,vis_path=vis_path, vis = args.vis, schedulers=scheduler)
        # TODO: ADD YOUR CUSTOM TRAINING LOOP HERE
        #elif args.dataset == 'your_dataset':
            # loss = function.train_sam_yourdataset(args, net, optimizer, nice_train_loader, epoch, writer, vis = args.vis, schedulers=scheduler)

        else:
            loss = function.train_sam(args, net, optimizer, nice_train_loader, epoch, writer, vis = args.vis)
        writer.add_scalar('loss', loss, epoch)
        logger.info(f'Train loss: {loss} || @ epoch {epoch}.')
        time_end = time.time()
        print('time_for_training ', time_end - time_start)

        net.eval()
        if epoch and epoch % args.val_freq == 0 or epoch == settings.EPOCH-1:
            # tol, (eiou, edice) = function.validation_sam(args, nice_test_loader, epoch, net, writer)
            if args.dataset in ['evican', 'oct']:
                val_loss, metric_results = function.validation_sam_evican(args, net, nice_test_loader, epoch, writer)
            elif args.dataset == 'cadis':
                val_loss, metric_results = function.validation_sam_cadis(args, net, nice_test_loader, epoch, writer)
            elif args.dataset == 'iqs_dv':
                val_loss, metric_results = function.validation_sam_deepvision(args, net, nice_test_loader, epoch, writer,vis_path=vis_path,vis=args.vis)
            # TODO: Add your dataset here 
            # elif args.dataset == 'yourdataset':
                # val_loss, metric_results = function.validation_sam_yourdataset(args, net, nice_test_loader, epoch, writer)

            tol = metric_results['BinaryJaccardIndex']

            # logger.info(f'Total score: {tol}, IOU: {eiou}, DICE: {edice} || @ epoch {epoch}.')

            if args.distributed != 'none':
                sd = net.module.state_dict()
            else:
                sd = net.state_dict()

            if tol > best_tol:
                best_tol = tol
                is_best = True
            else:
                is_best = False

            save_checkpoint({
                'epoch': epoch + 1,
                'model': args.net,
                'state_dict': sd,
                'optimizer': optimizer.state_dict(),
                'best_tol': best_tol,
                'path_helper': args.path_helper,
            }, is_best, args.path_helper['ckpt_path'], filename=f"checkpoint_last.pth")

writer.close()