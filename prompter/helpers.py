import cv2
import numpy as np 
import torch 
from torch import Tensor
from typing import Union


def sample_center(mask):
    return torch.mean((mask == 1).nonzero().to(torch.float), dim=0).int()

def move_center_inside(mask, center):
    center = center.flip(0)  # Flipping for x, y coordinates
    mask_h, mask_w = mask.shape
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Possible movement directions
    cnt=0
    while mask[center[1], center[0]] == 0 and cnt <100:# While center is outside the mask
        cnt+=1
        dx, dy = np.random.choice([d[0] for d in directions]), np.random.choice([d[1] for d in directions])
        new_x = center[0] + dx
        new_y = center[1] + dy
        
        # Check if the new position is within the mask boundaries
        if 0 <= new_x < mask_w and 0 <= new_y < mask_h:
            center = torch.tensor([new_x, new_y], dtype=torch.int)
        else:
            new_x, new_y = center[0], center[1]
        
            
    
    return center.flip(0)     
def generate_center_for_component(mask):  
    device = mask.device
    points_pos = []
    points_neg = []
  
    center = sample_center(mask)
    if mask[center[0], center[1]] == 0:
        center = move_center_inside(mask, center)
    if mask[center[0], center[1]] == 0:
        points_neg.append(center.to(device))
    else:
        points_pos.append(center.to(device))
    if len(points_pos) == 0: 
        points_pos = torch.Tensor(size=(0,2)).to(device)
    else:
        points_pos = torch.stack(points_pos, 0).to(device).flip([1])
    if len(points_neg) == 0: 
        points_neg = torch.Tensor(size=(0,2)).to(device)
    else:
        points_neg = torch.stack(points_neg, 0).to(device).flip([1])
   
    return points_pos, points_neg

def sample_sparse_pos_neg_points(mask, n_pos: int = 5,n_neg: int = 5, cov_factor: float = 1.5):
    device = mask.device
    # Get object mask height and width
    idcs_pos = (mask == 1).nonzero()
    if len(idcs_pos) < n_pos:
        n_pos = len(idcs_pos)
    if len(idcs_pos) == 0:
        mask_h,mask_w = mask.shape[-2:]
        pos_points = torch.Tensor(size=(0,2)).to(device)
    else:
        mask_h = idcs_pos[:,0].max() - idcs_pos[:,0].min()
        mask_w = idcs_pos[:,1].max() - idcs_pos[:,1].min()
        pos_points = sample_sparse_points(mask,idcs_pos,n_pos,cov_factor,mask_h,mask_w).to(device)
    if n_pos == 0: 
        pos_points = torch.Tensor(size=(0,2)).to(device)
            
    if n_neg == 0:
        neg_points = torch.Tensor(size=(0,2)).to(device)
    else:
        idcs_neg = (mask == 0).nonzero()
        mask_neg = -mask + 1
        neg_points = sample_sparse_points(mask_neg,idcs_neg,n_neg,cov_factor,mask_h,mask_w).to(device)
    # Generate first probability map
    
    return pos_points, neg_points
    
def sample_sparse_points(mask,idcs,n_points,cov_factor,mask_h,mask_w):    
    p_map = torch.ones_like(mask)
    points = []
    # Sample points
    for p in range(n_points):
        # sample point from current image
        probs = p_map[mask.to(torch.bool)] 
        probs = torch.nan_to_num(probs)
        
        if probs.sum() <= 0:
            p_map = torch.ones_like(p_map)
            probs = p_map[mask.to(torch.bool)]
            

        probs /= probs.sum()
        if probs.sum() != 1.0:
            probs = None
        else:
            probs = probs.detach().cpu().numpy()
        choice = np.random.choice(range(len(idcs)), p=probs)
        point = idcs[choice]
        points.append(point)
        
        point_img = build_gaussian_image(point[1], point[0], mask_w*cov_factor, mask_h*cov_factor, torch.tensor(0), mask.shape[-2:])

        p_map -= point_img
        p_map[p_map <0] = 0
    points = torch.stack(points, 0).flip([1]).to(mask.device)
    return points

def sample_pos_neg_points(mask, n_pos,n_neg):
    device = mask.device
    idcs_pos = (mask == 1).nonzero()
    idcs_neg = (mask == 0).nonzero()
    if len(idcs_pos) < n_pos:
        n_pos = len(idcs_pos)

    # If mask completely empty or full, return an empty tensor
    if len(idcs_pos) == 0 or n_pos==0 :
        selected_pos_points = torch.Tensor(size=(0, 2)).to(device)
    else:
            # Randomly sample points from the non-zero indices
        selected_pos_indices = np.random.choice(len(idcs_pos), n_pos, replace=False)
        selected_pos_points = idcs_pos[selected_pos_indices].to(torch.int64).flip([1]).to(device)
        
    if len(idcs_neg) == 0 or n_neg==0:
        selected_neg_points = torch.Tensor(size=(0, 2)).to(device)
    else:
             # Randomly sample points from the non-zero indices
        selected_neg_indices = np.random.choice(len(idcs_neg), n_neg, replace=False)
        selected_neg_points = idcs_neg[selected_neg_indices].to(torch.int64).flip([1]).to(device)
    

    return selected_pos_points, selected_neg_points


def find_best_new_prompt(clusters):
    """
    Finds largest cluster and returns center prompt in this cluster.
    """
    contours, _ = cv2.findContours(clusters.numpy().astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    max_idx = -1
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        if area > max_area:
            max_area = area
            max_idx = i
    # cont = contours[max_idx]
    single_cont = np.zeros(clusters.numpy().shape)
    cv2.drawContours(single_cont, contours, max_idx, color=1, thickness=-1)

    return sample_positive_points(torch.from_numpy(single_cont).to(torch.uint8), 1, erosion_factor=0.0)

def sample_positive_points(
        mask, 
        n_points: int = 2,
        cov_factor: float = 1.5,
        init_cov_factor: float = 0.5,
        erosion_factor: float = 0.05,
        plot_points: bool = False
        ):
    """
    Build positive prompt based on distance from center of mass of mask and based on
    distance to each other. The first point is sampled from a Gaussian distribution
    centered at the masks center of mass. All other points are iteratively sampled from
    a uniform distribution, where each previous point has been subtracted with a 
    gaussian.

    Parameters
    ----------
    mask: torch.Tensor
        Input Mask to sample points from
    n_points: int, default 2
        Number of points to sample from the mask
    cov_factor: float, default 1.5
        Multiplication factor for the covariance matrix: Larger factor means points 
        will be further apart.
    init_cov_factor: float, default 0.5
        Multiplication factor for the covariance matrix of the gaussian used to sample
        the first point from. Smaller factor means point will more probably be closer 
        to the center of mass.
    erosion_factor: float, default 0.05
        Factor of erosion that is applied to the mask before sampling.
        0.05 means that the erosion size will be 5% of the masks size.
    plot_points: bool, default False
        Will import matplotlib and plot the points and final distribution
    
    Returns
    -------
    points: torch.Tensor
        Tensor with sampled points in shape [n_points, 2], each point comes in [X, Y].
        points[:,0] are X coordinates, points[:,1] are Y coordinates
    """
    device = mask.device
    if n_points == 0:
        return torch.Tensor(size=(0,2)).to(device)
    # Get object mask height and width
    idcs = (mask > 0).nonzero()
    if len(idcs) < n_points:
        n_points = len(idcs)
    if len(idcs) == 0:
        return torch.Tensor(size=(0,2)).to(device)
    mask_h = idcs[:,0].max() - idcs[:,0].min()
    mask_w = idcs[:,1].max() - idcs[:,1].min()

    # Removes precentage of corner
    mask_er = erode_mask(mask, mask_h, mask_w, erosion_factor) 

    # Choose first point relatively close to center
    p_map = build_first_gaussian_from_mask(mask, init_cov_factor)
    idcs = (mask_er>0).nonzero()
    if len(idcs) < 1:
        print(f'len(idcs) is zero. No space to sample from.')
        # Randomly sample from image
        idcs = (mask_er<1).nonzero()
    probs = p_map[mask_er.to(torch.bool)]
    probs /= probs.sum() # Scale to sum to 1
    probs = torch.nan_to_num(probs)
    probs = probs.detach().cpu().numpy() if probs.sum() == 1.0 else None
    choice = np.random.choice(range(len(idcs)), p=probs)
    points = [idcs[choice]]


    if n_points > 1:
        # Once the first point is sampled, we treat the point probabilities inside the mask 
        # as a uniform distribution, from which we iteratively sampled and remove the 
        # already sampled points
        point_img = build_gaussian_image(points[-1][1], points[-1][0], mask_w*cov_factor, mask_h*cov_factor,torch.tensor( 0), mask.shape[-2:])
        p_map = torch.ones_like(point_img)
        p_map -= point_img
        p_map[p_map < 0] = 0

        for p in range(n_points - 1):
            # sample point from current image
            probs = p_map[mask_er.to(torch.bool)]
            probs = torch.nan_to_num(probs)
            if probs.sum() <= 0:
                # If mask is already full of so many points, s.t. there is no useful 
                # sampling space anymore, we restart sampling from a uniform 
                # distribution
                p_map = torch.ones_like(p_map)
                probs = p_map[mask_er.to(torch.bool)]
    
            probs /= probs.sum()
            if probs.sum() != 1.0:
                probs = None
            else:
                probs = probs.detach().cpu()
                probs = probs.numpy()
            choice = np.random.choice(range(len(idcs)), p=probs)
            points.append(idcs[choice])

            # Update p_map
            point_img = build_gaussian_image(points[-1][1], points[-1][0], mask_w*cov_factor, mask_h*cov_factor,torch.tensor(0), mask.shape[-2:])
            p_map -= point_img
            p_map[p_map < 0] = 0
    
    points = torch.stack(points, 0).to(device).flip([1])
    if plot_points:
        # Debug Plot
        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(mask, cmap='gray')
        #plt.imshow((p_map+0.1)*mask_er, cmap='hot')
        plt.scatter(points[:,0], points[:,1],c='yellow', label='Sampled Points', s=5)
    return points

def sample_negative_points(
        mask, 
        n_points: int = 2, 
        cov_factor: float = 1.5,
        dilation_factor: float = 0.01,
        plot_points: bool = False
        ):
    """
    Build negative prompt based on distance from center of mass of mask and based on
    distance to each other. All points are iteratively sampled from an initial Gaussian 
    distribution centered around the center of mass of the mask. A gaussian will be 
    removed from the distribution map around each sampled point to ensure the area isn't
    oversampled. All points will be sampled from OUTSIDE the mask.

    Parameters
    ----------
    mask: torch.Tensor
        Input Mask to sample points from
    n_points: int, default 2
        Number of points to sample from the mask
    cov_factor: float, default 1.5
        Multiplication factor for the covariance matrix: Larger factor means points 
        will be further apart.
    init_cov_factor: float, default 0.5
        Multiplication factor for the covariance matrix of the gaussian used to sample
        the first point from. Smaller factor means point will more probably be closer 
        to the center of mass.
    dilation_factor: float, default 0.01
        Factor of dilation that is applied to the mask before sampling.
        0.01 means that the dilation size will be 1% of the masks size.
    plot_points: bool, default False
        Will import matplotlib and plot the points and final distribution
    
    Returns
    -------
    points: torch.Tensor
        Tensor with sampled points in shape [n_points, 2], each point comes in [X, Y].
        points[:,0] are X coordinates, points[:,1] are Y coordinates
    """
    device = mask.device
    if n_points == 0:
        return torch.Tensor(size=(0,2)).to(device)
    # Get object mask height and width
    idcs = (mask > 0).nonzero()
    if len(idcs) == 0:
        return torch.Tensor(size=(0,2)).to(device)
    mask_h = idcs[:,0].max() - idcs[:,0].min()
    mask_w = idcs[:,1].max() - idcs[:,1].min()
    
    # Remove percentage of the masks corner
    mask_dil = dilate_mask(mask, mask_h, mask_w, dilation_factor)

    if mask_dil.sum() <= 0:
        mask_dil = mask # Shouldnt happen

    # Generate first probability map
    idcs = (mask_dil <= 0).nonzero()
    p_map = build_first_gaussian_from_mask(mask_dil, 10)
    mask_dil = -mask_dil + 1
    
    points = []
    # Sample points
    for p in range(n_points):
        # sample point from current image
        probs = p_map[mask_dil.to(torch.bool)]
        probs = torch.nan_to_num(probs)
        if probs.sum() <= 0:
            p_map = torch.ones_like(p_map)
            probs = p_map[mask_dil.to(torch.bool)]

        probs /= probs.sum()
        if probs.sum() != 1.0:
            probs = None
        else:
            probs = probs.numpy()
        choice = np.random.choice(range(len(idcs)), p=probs)
        point = idcs[choice]
        points.append(point)
        
        point_img = build_gaussian_image(point[1], point[0], mask_w*cov_factor, mask_h*cov_factor, 0, mask.shape[-2:])

        p_map -= point_img
        p_map[p_map <0] = 0

    points = torch.stack(points, 0).to(device).flip([1])
    if plot_points:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow((p_map+0.1)*mask_dil, cmap='hot')
        plt.scatter(points[:,0], points[:,1],c='red', label='Sampled Points', s=5)
    
    return points

def sample_points(
        mask: Tensor,
        n_positive: int = 2,
        n_negative: int = 2,
        erosion_factor: float = 0.05,
        dilation_factor: float = 0.01,
        cov_factor: float = 1.5,
        init_cov_factor: float = 0.5,
        plot_points: bool = False):
    """
    Sample points from a mask based on masks center of mass and gaussians around the 
    sampled points.

    Parameters
    ----------
    mask: torch.Tensor
        Input Mask to sample points from
    n_positive: int, default 2
        Number of points to sample from within the mask
    n_negative: int, default 2
        Number of points to sample from outside the mask
    cov_factor: float, default 1.5
        Multiplication factor for the covariance matrix: Larger factor means points 
        will be further apart.
    init_cov_factor: float, default 0.5
        Multiplication factor for the covariance matrix of the gaussian used to sample
        the first point from. Smaller factor means point will more probably be closer 
        to the center of mass.
    erosion_factor: float, default 0.05
        Factor of erosion that is applied to the mask before sampling.
        0.05 means that the erosion size will be 5% of the masks size.
    dilation_factor: float, default 0.01
        Factor of dilation that is applied to the mask before sampling.
        0.01 means that the dilation size will be 1% of the masks size.
    plot_points: bool, default False
        Will import matplotlib and plot the points and final distribution
    
    Returns
    -------
    positive_points: torch.Tensor
        Tensor with sampled points from inside the mask in shape [n_points, 2], each 
        point comes in [X, Y]:
        points[:,0] are X coordinates, points[:,1] are Y coordinates
    negative_points: torch.Tensor
        Tensor with sampled points from outside the mask in shape [n_points, 2], each 
        point comes in [X, Y]:
        points[:,0] are X coordinates, points[:,1] are Y coordinates
    """
    positive_points = sample_positive_points(
        mask=mask,
        n_points=n_positive, 
        cov_factor=cov_factor,
        init_cov_factor=init_cov_factor, 
        erosion_factor=erosion_factor,
        plot_points=plot_points
        )
    negative_points = sample_negative_points(
        mask=mask,
        n_points=n_negative,
        cov_factor=cov_factor,
        dilation_factor=dilation_factor,
        plot_points=plot_points
        )
    return positive_points, negative_points


def sample_points_sam(
        mask: Tensor,
        n_positive: int = 2,
        n_negative: int = 2,
        erosion_factor: float = 0.05,
        dilation_factor: float = 0.01,
        cov_factor: float = 1.5,
        init_cov_factor: float = 0.5,
        plot_points: bool = False):
    """
    Sample points from a mask based on masks center of mass and gaussians around the 
    sampled points. Output in format needed for SAM prompting.

    Parameters
    ----------
    mask: torch.Tensor
        Input Mask to sample points from
    n_positive: int, default 2
        Number of points to sample from within the mask
    n_negative: int, default 2
        Number of points to sample from outside the mask
    cov_factor: float, default 1.5
        Multiplication factor for the covariance matrix: Larger factor means points 
        will be further apart.
    init_cov_factor: float, default 0.5
        Multiplication factor for the covariance matrix of the gaussian used to sample
        the first point from. Smaller factor means point will more probably be closer 
        to the center of mass.
    erosion_factor: float, default 0.05
        Factor of erosion that is applied to the mask before sampling.
        0.05 means that the erosion size will be 5% of the masks size.
    dilation_factor: float, default 0.01
        Factor of dilation that is applied to the mask before sampling.
        0.01 means that the dilation size will be 1% of the masks size.
    plot_points: bool, default False
        Will import matplotlib and plot the points and final distribution
    
    Returns
    -------
    positive_points: torch.Tensor
        Tensor with sampled points from inside the mask in shape [n_points, 2], each 
        point comes in [X, Y]:
        points[:,0] are X coordinates, points[:,1] are Y coordinates
    negative_points: torch.Tensor
        Tensor with sampled points from outside the mask in shape [n_points, 2], each 
        point comes in [X, Y]:
        points[:,0] are X coordinates, points[:,1] are Y coordinates
    """
    positive_points, negative_points = sample_points(
        mask=mask,
        n_positive=n_positive, 
        n_negative=n_negative, 
        cov_factor=cov_factor,
        init_cov_factor=init_cov_factor, 
        erosion_factor=erosion_factor,
        dilation_factor=dilation_factor,
        plot_points=plot_points
        )
    device = mask.device 
    points = torch.concatenate([positive_points.to(device), negative_points.to(device)])
    labels = torch.concatenate(
        [torch.ones((n_positive)), torch.zeros((n_negative))]
        ).to(device)
    return points, labels

# ======= Gaussian Building Functions ====== #
def find_rotated_bb(mask: Tensor):
    """
    Find the minimum enclosing rotated bounding box around a mask.
    When mask has multiple clusters, only the biggest is used.
    """
    mask = mask.detach().cpu()
    contour, _ = cv2.findContours(mask.numpy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour = contour[0]
    center, (mask_h, mask_w), angle = cv2.minAreaRect(contour)

    return center, (mask_h, mask_w), angle

def gauss_2d_torch(
        x_grid: Tensor,
        y_grid: Tensor,
        mu_x: Union[float, Tensor],
        mu_y: Union[float, Tensor],
        sigma_x: Union[float, Tensor],
        sigma_y: Union[float, Tensor],
        sigma_xy: Union[float, Tensor]
    ):
    """
    Calculate a 2d Gaussian with mean and covariance matrix over
    fields x and y
    """
    mu_x, mu_y, sigma_x, sigma_y, sigma_xy = mu_x.detach().cpu(), mu_y.detach().cpu(), sigma_x.detach().cpu(), sigma_y.detach().cpu(), sigma_xy.detach().cpu()

    
    div = 2 * np.pi * np.sqrt((sigma_x*sigma_y - sigma_xy**2))

    det = (sigma_x*sigma_y - sigma_xy*sigma_xy)
    term = (x_grid-mu_x)**2 * sigma_y - 2*(x_grid-mu_x)*(y_grid - mu_y)*sigma_xy + \
        (y_grid-mu_y)**2 * sigma_x
    
    return torch.exp(-0.5*term/det)


def build_gaussian_image(
        mu_x: Union[float, Tensor],
        mu_y: Union[float, Tensor],
        sigma_x: Union[float, Tensor],
        sigma_y: Union[float, Tensor],
        sigma_xy: Union[float, Tensor],
        img_size: torch.Size
    ):
    """
    Create a Gaussian Image with specified mu and sigma in spec size
    """
    img_h, img_w = img_size[-2:]
    x = torch.linspace(0, img_w, img_w)
    y = torch.linspace(0, img_h, img_h)
    y_grid, x_grid = torch.meshgrid(y, x)
    return gauss_2d_torch(x_grid.to(mu_x.device), y_grid.to(mu_x.device), mu_x, mu_y, sigma_x, sigma_y, sigma_xy.to(mu_x.device))


def build_first_gaussian_from_mask(
        mask: Tensor, 
        cov_factor: Union[float, Tensor] = 1
    ) -> Tensor:
    """
    Build Gaussian Distribution basd on mask. Used for random choice
    """
    # Find rotated bounding box, center of mass, angle, and box H W
    com = torch.mean((mask>0).nonzero().to(torch.float), dim=0)
    _, (box_H, box_W), angle = find_rotated_bb(mask.to(torch.uint8))
    angle = (angle/180)*np.pi
    
    # Build Covariance Matrix
    cov_m = np.array([[box_W, 0], [0, box_H]])
    # Rotation Matrix from Angle
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    # Rotate matrix to get cov matrix
    cov_m = (rot.transpose() @ cov_m) @ rot
    cov_m *= cov_factor # scale

    return build_gaussian_image(com[1], com[0], torch.tensor(cov_m[0,0]), torch.tensor(cov_m[1,1]), torch.tensor(cov_m[0,1]), mask.shape)

    # return gauss_2d_torch(Xg, Yg, center[0], center[1], cov_m[0,0], cov_m[1,1], cov_m[0,1])

def erode_mask(
        mask: torch.Tensor,
        mask_h: Union[int, Tensor],
        mask_w: Union[int, Tensor],
        erosion_factor: float = 0.05
    ) -> torch.Tensor:
    """
    Erodes a mask based on its size and the erosion factor
    """
    erosion_size = int(np.sqrt(mask_h.detach().cpu()*mask_w.detach().cpu())*erosion_factor)
    if erosion_size < 1:
        return mask
    element = cv2.getStructuringElement(2, (2 * erosion_size + 1, 2 * erosion_size + 1), (erosion_size, erosion_size))
    mask = mask.detach().cpu()
    mask_out = torch.from_numpy(cv2.erode(mask.numpy().astype(np.uint8), element))
    if mask_out.sum() == 0:
        return mask
    return mask_out

def dilate_mask(mask, mask_h, mask_w, dilation_factor = 0.05):
    """
    Dilates a mask based on its size and the dilation factor
    """
    
    erosion_size = int(np.sqrt(mask_h.detach().cpu()*mask_w.detach().cpu())*dilation_factor)
    if erosion_size < 1:
        return mask
    element = cv2.getStructuringElement(2, (2 * erosion_size + 1, 2 * erosion_size + 1), (erosion_size, erosion_size))
    mask = mask.detach().cpu()
    mask = torch.from_numpy(cv2.dilate(mask.numpy().astype(np.uint8), element))
    return mask