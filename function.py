
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
from conf import settings
import time
import cfg
from conf import settings
from tqdm import tqdm
from utils import *
import torch.nn.functional as F
import torch
from einops import rearrange
import pytorch_ssim
import models.sam.utils.transforms as samtrans
import logging
# from lucent.modelzoo.util import get_model_layers
# from lucent.optvis import render, param, transform, objectives
# from lucent.modelzoo import inceptionv1

import shutil
import tempfile

import matplotlib.pyplot as plt
from tqdm import tqdm
from prompter.prompter import *
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
)

from torchmetrics import Accuracy, Precision, Recall, JaccardIndex, F1Score

import torch

logger = logging.getLogger(__name__)

args = cfg.parse_args()

GPUdevice = torch.device('cuda', args.gpu_device)
pos_weight = torch.ones([1]).cuda(device=GPUdevice)*2
criterion_G = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
seed = torch.randint(1,11,(args.b,7))

torch.backends.cudnn.benchmark = True
loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
scaler = torch.cuda.amp.GradScaler()
max_iterations = settings.EPOCH
post_label = AsDiscrete(to_onehot=14)
post_pred = AsDiscrete(argmax=True, to_onehot=14)
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
dice_val_best = 0.0
global_step_best = 0
epoch_loss_values = []
metric_values = []


#### deepvision
def train_sam_deepvision(args, net: nn.Module, optimizer, train_loader,
          epoch, writer,vis_path, schedulers=None, vis = 1):
    epoch_loss = 0
    ind = 0
    # train mode
    net.train()
    optimizer.zero_grad()
    # Freeze all non-adapter layers of SAM
    for n, value in net.image_encoder.named_parameters():
        if "Adapter" not in n:
            value.requires_grad = False

    GPUdevice = torch.device('cuda:' + str(args.gpu_device))
    device = GPUdevice
    
    lossfunc = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # PROMPT SETUP
    N_POINTS_MAX = 10
    N_MAX_ITER_PROMPTS = 3
    # Metrics setup
    metrics = [
        Accuracy(task='binary').to(device), 
        Precision(task='binary').to(device), 
        Recall(task='binary').to(device), 
        F1Score(task='binary').to(device), 
        JaccardIndex(task='binary').to(device)
    ]
    with tqdm(total=len(train_loader), desc=f'Epoch {epoch}', unit='img') as pbar:
        for pack in train_loader:
            preds = []
            prompts = []
            original_preds = []
            imgs = pack['image'].to(dtype = torch.float32, device = GPUdevice)
            targets = pack['label'].to(dtype = torch.float32, device = GPUdevice)
            names = pack['metadata']
            batch_loss = 0.0
            for img, mask in zip(imgs, targets):
                img_emb = net.image_encoder(img.unsqueeze(0))
                # Create Prompts
                
            
                # Randomly sample number of prompts
                #n_points = np.random.randint(1, N_POINTS_MAX)
                #n_pos = np.random.randint(1, n_points) if n_points > 1 else 1
                #n_neg = np.random.randint(0, n_points-n_pos) if (n_points - n_pos) > 0 else 0
                n_neg = 0
                n_pos = 5
                pts, lbls = sample_from_mask(mask.squeeze(0), mode="random", n_pos=n_pos,n_neg = n_neg) 
                

                user_iter = 0 
                # Randomly add pseudo user input 
                #user_iter = np.random.randint(N_MAX_ITER_PROMPTS)
                for i in range(user_iter):
                    # print(f'User interaction {i+1}/{user_iter}')
                    with torch.no_grad():
                        # Set prompt
                        prompt = (pts.unsqueeze(0).to(device), lbls.unsqueeze(0).to(device))
                        se, de = net.prompt_encoder(
                            points=prompt,
                            boxes=None,
                            masks=None,
                        ) # type: ignore
                        
                        # Predict Mask
                        pred, _ = net.mask_decoder(
                            image_embeddings=img_emb,
                            image_pe=net.prompt_encoder.get_dense_pe(),  # type: ignore
                            sparse_prompt_embeddings=se,
                            dense_prompt_embeddings=de, 
                            multimask_output=False,
                        ) # type: ignore
                        # Compare Prediction to GT
                        pred = F.interpolate(pred, mask.shape[-2:]) # This is a bit cumbersome, but the easiest fix for now
                        pred = pred.squeeze() > 0 #check if 0 or 0.5 with david
                        clusters = pred.cpu() != mask
                        # Sample point from largest error cluster 
                        new_prompt = find_best_new_prompt(clusters)
                        new_label = mask[new_prompt[0, 1], new_prompt[0, 0]].to(torch.int64)
                        pts = torch.concatenate([pts, new_prompt])
                        lbls = torch.concatenate([lbls, torch.tensor([new_label])])

                # Final Mask inference
                prompts.append([pts,lbls])
                prompt = (pts.unsqueeze(0).to(device), lbls.unsqueeze(0).to(device))

                # Set Prompt
                with torch.no_grad():
                    se, de = net.prompt_encoder(
                        points=prompt, 
                        boxes=None,
                        masks=None,
                    ) # type: ignore

                # Predict Mask
                pred, _ = net.mask_decoder(
                    image_embeddings=img_emb,
                    image_pe=net.prompt_encoder.get_dense_pe(),  # type: ignore
                    sparse_prompt_embeddings=se,
                    dense_prompt_embeddings=de, 
                    multimask_output=False,
                ) # type: ignore
                original_preds.append((pred.squeeze(0) > 0).float())
                pred = F.interpolate(pred, mask.shape[-2:]).squeeze(0) # This is a bit cumbersome, but the easiest fix for now
                preds.append((pred> 0).float())
                loss = lossfunc(pred, mask)
                batch_loss += loss
                epoch_loss += loss.item()
                for m in metrics:
                    m.update(pred, mask)

            pbar.set_postfix(**{'loss (batch)': batch_loss.item()}) # type: ignore
            batch_loss.backward()

            writer.add_scalar('Batch_loss/train', batch_loss.item(), ind + epoch*len(train_loader))
            # nn.utils.clip_grad_value_(net.parameters(), 0.1)
            optimizer.step()
            optimizer.zero_grad()

            
            
            if vis:
                if ind % vis == 0:
                    visualize_batch(imgs=imgs, masks=targets, pred_masks=preds, names=names, prompts=prompts,original_preds=original_preds,save_path=vis_path)

            ind += 1
            pbar.update()

    for m in metrics:
        out = m.compute().cpu()
        writer.add_scalar(f'{type(m).__name__}/train', out, epoch)
    writer.add_scalar('Epoch_loss/Val', epoch_loss, epoch)
    schedulers.step()
    writer.add_scalar('lr', optimizer.param_groups[0]["lr"], epoch)
    return epoch_loss

def validation_sam_deepvision(args, net: nn.Module,  val_loader, epoch, writer,vis_path, vis = 1):
    net.eval()

    epoch_loss = 0
    ind = 0

    GPUdevice = torch.device('cuda:' + str(args.gpu_device))
    device = GPUdevice
    
    metrics = [
        Accuracy(task='binary').to(device), 
        Precision(task='binary').to(device), 
        Recall(task='binary').to(device), 
        F1Score(task='binary').to(device), 
        JaccardIndex(task='binary').to(device)
    ]
    lossfunc = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    N_POINTS_MAX = 3
    N_MAX_ITER_PROMPTS = 3
    with torch.no_grad():
        with tqdm(total=len(val_loader), desc=f'Epoch {epoch}', unit='img') as pbar:
            for pack in val_loader:
                preds = []
                prompts = []
                original_preds = []
                imgs = pack['image'].to(dtype = torch.float32, device = GPUdevice)
                targets = pack['label'].to(dtype = torch.float32, device = GPUdevice)
                names = pack['metadata']
                batch_loss = 0.0
                for img, mask in zip(imgs, targets):
                    img_emb = net.image_encoder(img.unsqueeze(0))
                    # Create Prompts
                    
                
                    # Randomly sample number of prompts
                    #n_points = np.random.randint(1, N_POINTS_MAX)
                    #n_pos = np.random.randint(1, n_points) if n_points > 1 else 1
                    #n_neg = np.random.randint(0, n_points-n_pos) if (n_points - n_pos) > 0 else 0
                    n_neg = 0
                    n_pos = 5
                    pts, lbls = sample_from_mask(mask.squeeze(0), mode="random", n_pos=n_pos,n_neg = n_neg) 
        

                    user_iter = 0 
                    # Randomly add pseudo user input 
                    #user_iter = np.random.randint(N_MAX_ITER_PROMPTS)
                    for i in range(user_iter):
                        # print(f'User interaction {i+1}/{user_iter}')
                        
                        # Set prompt
                        prompt = (pts.unsqueeze(0).to(device), lbls.unsqueeze(0).to(device))
                        se, de = net.prompt_encoder(
                            points=prompt,
                            boxes=None,
                            masks=None,
                        ) # type: ignore
                        
                        # Predict Mask
                        pred, _ = net.mask_decoder(
                            image_embeddings=img_emb,
                            image_pe=net.prompt_encoder.get_dense_pe(),  # type: ignore
                            sparse_prompt_embeddings=se,
                            dense_prompt_embeddings=de, 
                            multimask_output=False,
                        ) # type: ignore
                        # Compare Prediction to GT
                        pred = F.interpolate(pred, mask.shape[-2:]) # This is a bit cumbersome, but the easiest fix for now
                        pred = pred.squeeze() > 0 #check this!!!
                        clusters = pred.cpu() != mask
                        # Sample point from largest error cluster 
                        new_prompt = find_best_new_prompt(clusters)
                        new_label = mask[new_prompt[0, 1], new_prompt[0, 0]].to(torch.int64)
                        pts = torch.concatenate([pts, new_prompt])
                        lbls = torch.concatenate([lbls, torch.tensor([new_label])])

                    # Final Mask inference
                    prompts.append([pts,lbls])
                    prompt = (pts.unsqueeze(0).to(device), lbls.unsqueeze(0).to(device))

                    # Set Prompt
                    with torch.no_grad():
                        se, de = net.prompt_encoder(
                            points=prompt,
                            boxes=None,
                            masks=None,
                        ) # type: ignore

                    # Predict Mask
                    pred, _ = net.mask_decoder(
                        image_embeddings=img_emb,
                        image_pe=net.prompt_encoder.get_dense_pe(),  # type: ignore
                        sparse_prompt_embeddings=se,
                        dense_prompt_embeddings=de, 
                        multimask_output=False,
                    ) # type: ignore
                    original_preds.append((pred.squeeze(0) > 0).float())
                    pred = F.interpolate(pred, mask.shape[-2:]).squeeze(0) # This is a bit cumbersome, but the easiest fix for now
                    preds.append((pred > 0).float())
                    loss = lossfunc(pred, mask)
                    batch_loss += loss
                    epoch_loss += loss.item()
                    for m in metrics:
                        m.update(pred, mask)
                writer.add_scalar('Batch_loss/Val', batch_loss.item(), ind + epoch*len(val_loader))

                pbar.set_postfix(**{'loss (batch)': batch_loss.item()}) # type: ignore


                '''vis images'''
                if vis:
                    if ind % vis == 0:
                        
                        visualize_batch(imgs=imgs, masks=targets, pred_masks=preds, names=names, prompts=prompts,original_preds=original_preds,save_path= vis_path)

            ind += 1
            pbar.update()

    metric_results = {}
    for m in metrics:
        out = m.compute().cpu()
        writer.add_scalar(f'{type(m).__name__}/val', out, epoch)
        metric_results[f'{type(m).__name__}'] = out
    writer.add_scalar('Epoch_loss/Val', epoch_loss, epoch)

    return epoch_loss, metric_results
def train_sam(args, net: nn.Module, optimizer, train_loader,
          epoch, writer, schedulers=None, vis = 50):
    hard = 0
    epoch_loss = 0
    ind = 0
    # train mode
    net.train()
    optimizer.zero_grad()

    epoch_loss = 0
    GPUdevice = torch.device('cuda:' + str(args.gpu_device))
    device = GPUdevice

    if args.thd:
        lossfunc = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    else:
        lossfunc = criterion_G

    with tqdm(total=len(train_loader), desc=f'Epoch {epoch}', unit='img') as pbar:
        for pack in train_loader:
            imgs = pack['image'].to(dtype = torch.float32, device = GPUdevice)
            masks = pack['label'].to(dtype = torch.float32, device = GPUdevice)
            # for k,v in pack['image_meta_dict'].items():
            #     print(k)
            if 'pt' not in pack:
                imgs, pt, masks = generate_click_prompt(imgs, masks)
                #point_labels, pt, masks = generate_click_prompt_custom(masks)
            else:
                pt = pack['pt']
                point_labels = pack['p_label']
            name = pack['image_meta_dict']['filename_or_obj']

            if args.thd:
                pt = rearrange(pt, 'b n d -> (b d) n')
                imgs = rearrange(imgs, 'b c h w d -> (b d) c h w ')
                masks = rearrange(masks, 'b c h w d -> (b d) c h w ')

                imgs = imgs.repeat(1,3,1,1)
                point_labels = torch.ones(imgs.size(0))

                imgs = torchvision.transforms.Resize((args.image_size,args.image_size))(imgs)
                masks = torchvision.transforms.Resize((args.out_size,args.out_size))(masks)
            
            showp = pt

            mask_type = torch.float32
            ind += 1
            b_size,c,w,h = imgs.size()
            longsize = w if w >=h else h

            if point_labels[0] != -1:
                # point_coords = samtrans.ResizeLongestSide(longsize).apply_coords(pt, (h, w))
                point_coords = pt
                coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=GPUdevice)
                labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=GPUdevice)
                coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
                pt = (coords_torch, labels_torch)

            '''init'''
            if hard:
                true_mask_ave = (true_mask_ave > 0.5).float()
                #true_mask_ave = cons_tensor(true_mask_ave)
            imgs = imgs.to(dtype = mask_type,device = GPUdevice)
            
            '''Train'''
            if args.net == 'sam' or args.net == 'efficient_sam':
                for n, value in net.image_encoder.named_parameters():
                    if "Adapter" not in n:
                        value.requires_grad = False

            imge= net.image_encoder(imgs)

            with torch.no_grad():
                if args.net == 'sam':
                    se, de = net.prompt_encoder(
                        points=pt,
                        boxes=None,
                        masks=None,
                    )
                elif args.net == "efficient_sam":
                    coords_torch,labels_torch = transform_prompt(coords_torch,labels_torch,h,w)
                    se = net.prompt_encoder(
                        coords=coords_torch,
                        labels=labels_torch,
                    )
                    
            if args.net == 'sam':
                pred, _ = net.mask_decoder(
                    image_embeddings=imge,
                    image_pe=net.prompt_encoder.get_dense_pe(), 
                    sparse_prompt_embeddings=se,
                    dense_prompt_embeddings=de, 
                    multimask_output=False,
                )
            elif args.net == "efficient_sam":
                se = se.view(
                    se.shape[0],
                    1,
                    se.shape[1],
                    se.shape[2],
                )

                pred, _ = net.mask_decoder(
                    image_embeddings=imge,
                    image_pe=net.prompt_encoder.get_dense_pe(), 
                    sparse_prompt_embeddings=se,
                    multimask_output=False,
                )
                
            # Resize to the ordered output size
            pred = F.interpolate(pred,size=(args.out_size,args.out_size))

            loss = lossfunc(pred, masks)

            pbar.set_postfix(**{'loss (batch)': loss.item()})
            epoch_loss += loss.item()
            loss.backward()

            # nn.utils.clip_grad_value_(net.parameters(), 0.1)
            optimizer.step()
            optimizer.zero_grad()

            '''vis images'''
            if vis:
                if ind % vis == 0:
                    namecat = 'Train'
                    for na in name:
                        namecat = namecat + na.split('/')[-1].split('.')[0] + '+'
                    vis_image(imgs,pred,masks, os.path.join(args.path_helper['sample_path'], namecat+'epoch+' +str(epoch) + '.jpg'), reverse=False, points=showp)

            pbar.update()

    return loss

def validation_sam(args, val_loader, epoch, net: nn.Module, clean_dir=True):
     # eval mode
    net.eval()

    mask_type = torch.float32
    n_val = len(val_loader)  # the number of batch
    ave_res, mix_res = (0,0,0,0), (0,0,0,0)
    rater_res = [(0,0,0,0) for _ in range(6)]
    tot = 0
    hard = 0
    threshold = (0.1, 0.3, 0.5, 0.7, 0.9)
    GPUdevice = torch.device('cuda:' + str(args.gpu_device))
    device = GPUdevice

    if args.thd:
        lossfunc = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    else:
        lossfunc = criterion_G

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for ind, pack in enumerate(val_loader):
            imgsw = pack['image'].to(dtype = torch.float32, device = GPUdevice)
            masksw = pack['label'].to(dtype = torch.float32, device = GPUdevice)
            # for k,v in pack['image_meta_dict'].items():
            #     print(k)
            if 'pt' not in pack:
                #point_labels, ptw, masksw = generate_click_prompt_custom(masksw)
                imgsw, ptw, masksw = generate_click_prompt(imgsw, masksw)
            else:
                ptw = pack['pt']
                point_labels = pack['p_label']
            name = pack['image_meta_dict']['filename_or_obj']
            
            buoy = 0
            if args.evl_chunk:
                evl_ch = int(args.evl_chunk)
            else:
                evl_ch = int(imgsw.size(-1))

            while (buoy + evl_ch) <= imgsw.size(-1):
                if args.thd:
                    pt = ptw[:,:,buoy: buoy + evl_ch]
                else:
                    pt = ptw

                imgs = imgsw[...,buoy:buoy + evl_ch]
                masks = masksw[...,buoy:buoy + evl_ch]
                buoy += evl_ch

                if args.thd:
                    pt = rearrange(pt, 'b n d -> (b d) n')
                    imgs = rearrange(imgs, 'b c h w d -> (b d) c h w ')
                    masks = rearrange(masks, 'b c h w d -> (b d) c h w ')
                    imgs = imgs.repeat(1,3,1,1)
                    point_labels = torch.ones(imgs.size(0))

                    imgs = torchvision.transforms.Resize((args.image_size,args.image_size))(imgs)
                    masks = torchvision.transforms.Resize((args.out_size,args.out_size))(masks)
                
                showp = pt

                mask_type = torch.float32
                ind += 1
                b_size,c,w,h = imgs.size()
                longsize = w if w >=h else h

                if point_labels[0] != -1:
                    # point_coords = samtrans.ResizeLongestSide(longsize).apply_coords(pt, (h, w))
                    point_coords = pt
                    coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=GPUdevice)
                    labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=GPUdevice)
                    coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
                    pt = (coords_torch, labels_torch)

                '''init'''
                if hard:
                    true_mask_ave = (true_mask_ave > 0.5).float()
                    #true_mask_ave = cons_tensor(true_mask_ave)
                imgs = imgs.to(dtype = mask_type,device = GPUdevice)
                
                '''test'''
                with torch.no_grad():
                    imge= net.image_encoder(imgs)

                    if args.net == 'sam':
                        se, de = net.prompt_encoder(
                            points=pt,
                            boxes=None,
                            masks=None,
                        )
                    elif args.net == "efficient_sam":
                        coords_torch,labels_torch = transform_prompt(coords_torch,labels_torch,h,w)
                        se = net.prompt_encoder(
                            coords=coords_torch,
                            labels=labels_torch,
                        )

                    if args.net == 'sam':
                        pred, _ = net.mask_decoder(
                            image_embeddings=imge,
                            image_pe=net.prompt_encoder.get_dense_pe(), 
                            sparse_prompt_embeddings=se,
                            dense_prompt_embeddings=de, 
                            multimask_output=False,
                        )
                    elif args.net == "efficient_sam":
                        se = se.view(
                            se.shape[0],
                            1,
                            se.shape[1],
                            se.shape[2],
                        )
                        pred, _ = net.mask_decoder(
                            image_embeddings=imge,
                            image_pe=net.prompt_encoder.get_dense_pe(), 
                            sparse_prompt_embeddings=se,
                            multimask_output=False,
                        )

                    # Resize to the ordered output size
                    pred = F.interpolate(pred,size=(args.out_size,args.out_size))
                    tot += lossfunc(pred, masks)
                    if args.vis:
                        '''vis images'''
                        if ind % args.vis == 0:
                            namecat = 'Test'
                            for na in name:
                                img_name = na.split('/')[-1].split('.')[0]
                                namecat = namecat + img_name + '+'
                            vis_image(imgs,pred, masks, os.path.join(args.path_helper['sample_path'], namecat+'epoch+' +str(epoch) + '.jpg'), reverse=False, points=showp)
                        

                    temp = eval_seg(pred, masks, threshold)
                    mix_res = tuple([sum(a) for a in zip(mix_res, temp)])

            pbar.update()

    if args.evl_chunk:
        n_val = n_val * (imgsw.size(-1) // evl_ch)

    return tot/ n_val , tuple([a/n_val for a in mix_res])

def transform_prompt(coord,label,h,w):
    coord = coord.transpose(0,1)
    label = label.transpose(0,1)

    coord = coord.unsqueeze(1)
    label = label.unsqueeze(1)

    batch_size, max_num_queries, num_pts, _ = coord.shape
    num_pts = coord.shape[2]
    rescaled_batched_points = get_rescaled_pts(coord, h, w)

    decoder_max_num_input_points = 6
    if num_pts > decoder_max_num_input_points:
        rescaled_batched_points = rescaled_batched_points[
            :, :, : decoder_max_num_input_points, :
        ]
        label = label[
            :, :, : decoder_max_num_input_points
        ]
    elif num_pts < decoder_max_num_input_points:
        rescaled_batched_points = F.pad(
            rescaled_batched_points,
            (0, 0, 0, decoder_max_num_input_points - num_pts),
            value=-1.0,
        )
        label = F.pad(
            label,
            (0, decoder_max_num_input_points - num_pts),
            value=-1.0,
        )
    
    rescaled_batched_points = rescaled_batched_points.reshape(
        batch_size * max_num_queries, decoder_max_num_input_points, 2
    )
    label = label.reshape(
        batch_size * max_num_queries, decoder_max_num_input_points
    )

    return rescaled_batched_points,label


def get_rescaled_pts(batched_points: torch.Tensor, input_h: int, input_w: int):
        return torch.stack(
            [
                torch.where(
                    batched_points[..., 0] >= 0,
                    batched_points[..., 0] * 1024 / input_w,
                    -1.0,
                ),
                torch.where(
                    batched_points[..., 1] >= 0,
                    batched_points[..., 1] * 1024 / input_h,
                    -1.0,
                ),
            ],
            dim=-1,
        )