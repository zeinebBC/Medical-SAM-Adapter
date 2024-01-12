
import numpy as np
import torch
from torch import Tensor
import cv2
from typing import Union

from prompter.helpers import *
def sample_from_components(mask, mode="random", n_pos: int = 2,
    n_neg: int = 2,
    erosion_factor: float = 0.05,
    dilation_factor: float = 0.01,
    cov_factor: float = 1.5,
    init_cov_factor: float = 0.5,
    plot_points: bool = False): 
    device = mask.device
    mask_numpy = mask.detach().cpu().numpy().astype(np.uint8)

    # Find connected components and label them
    num_labels, labels = cv2.connectedComponents(mask_numpy)
    points_pos = []
    points_neg = []
    if (num_labels-1) == 0:
        pos_points = torch.Tensor(size=(0,2)).to(device)
        n_pos_r = n_pos
    else:
        n_pos_c= n_pos // (num_labels-1)
        n_pos_r = n_pos - n_pos_c * (num_labels-1)

    for label in range(1, num_labels):
        component_mask = torch.from_numpy(np.where(labels == label, 1, 0).astype(np.float32)).to(device)
        
        if mode=="random":
            pos_points, _ = sample_pos_neg_points(component_mask, n_pos = n_pos_c, n_neg = n_neg)
        elif mode =="center":
            pos_points, neg_points = generate_center_for_component(component_mask)
            points_neg.append(neg_points)
        elif mode =="sparse":
            pos_points, _ = sample_sparse_pos_neg_points(component_mask, n_pos=n_pos_c, n_neg=n_neg, cov_factor=cov_factor)
        elif mode =="smart":
            pos_points, _ = sample_points(
        mask=component_mask,
        n_positive=n_pos_c, 
        n_negative=n_neg, 
        cov_factor=cov_factor,
        init_cov_factor=init_cov_factor, 
        erosion_factor=erosion_factor,
        dilation_factor=dilation_factor,
        plot_points=plot_points
        )
        else:
            raise ValueError("mode should be in ['random', 'sparse', 'smart','center']")
        points_pos.append(pos_points)
       
    
    
    if mode=="random":
            pos_points, neg_points = sample_pos_neg_points(mask, n_pos = n_pos_r, n_neg = n_neg)
    elif mode =="center":
            non_empty_list = [tensor for tensor in points_neg if tensor.numel() > 0]
            n_neg_r = n_neg - len(non_empty_list) if (n_neg - len(non_empty_list)>0) else 0
            n_pos_r = 0
            #accepted_centers = (num_labels-1) - len(non_empty_list)
            #n_pos_r= n_pos - accepted_centers if (n_pos - accepted_centers)>0 else 0
            pos_points, neg_points = sample_pos_neg_points(mask, n_pos = n_pos_r, n_neg = n_neg_r)
    elif mode =="sparse":
        pos_points, neg_points = sample_sparse_pos_neg_points(mask, n_pos=n_pos_r, n_neg=n_neg, cov_factor=cov_factor)
    elif mode =="smart":
        pos_points,neg_points = sample_points(
        mask = mask,
        n_positive=n_pos_r, 
        n_negative=n_neg, 
        cov_factor=cov_factor,
        init_cov_factor=init_cov_factor, 
        erosion_factor=erosion_factor,
        dilation_factor=dilation_factor,
        plot_points=plot_points
        )
    points_pos.append(pos_points)
    points_neg.append(neg_points)
        
    stacked_points_pos = torch.cat(points_pos, dim=0)
    stacked_points_neg = torch.cat(points_neg, dim=0)
    points = torch.concatenate([stacked_points_pos, stacked_points_neg])
    p_labels = torch.concatenate([torch.ones((stacked_points_pos.shape[0])), torch.zeros((stacked_points_neg.shape[0]))]).to(torch.int64)
    return points, p_labels

def sample_from_mask(mask, mode="random", n_pos: int = 2,
    n_neg: int = 2,
    erosion_factor: float = 0.05,
    dilation_factor: float = 0.01,
    cov_factor: float = 1.5,
    init_cov_factor: float = 0.5,
    plot_points: bool = False):
    
    
    if mode=="random":
        pos_points, neg_points = sample_pos_neg_points(mask, n_pos = n_pos, n_neg = n_neg)
    elif mode =="sparse":
        pos_points, neg_points = sample_sparse_pos_neg_points(mask, n_pos=n_pos, n_neg=n_neg, cov_factor=cov_factor)
    elif mode =="smart":
        pos_points, neg_points = sample_points(
        mask=mask,
        n_positive=n_pos, 
        n_negative=n_neg, 
        cov_factor=cov_factor,
        init_cov_factor=init_cov_factor, 
        erosion_factor=erosion_factor,
        dilation_factor=dilation_factor,
        plot_points=plot_points
        )
    else:
         raise ValueError("mode should be in ['random', 'sparse', 'smart']")
   
    points = torch.concatenate([pos_points, neg_points])
    p_labels = torch.concatenate([torch.ones((pos_points.shape[0])), torch.zeros((neg_points.shape[0]))]).to(torch.int64)
    return points, p_labels