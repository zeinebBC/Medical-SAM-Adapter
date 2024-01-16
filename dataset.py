""" train and test dataset

author jundewu
"""
import os
import pickle
import random
import sys
import re
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from monai.transforms import LoadImage, LoadImaged, Randomizable
from PIL import Image
from skimage import io
from skimage.transform import rotate
from torch.utils.data import Dataset

from utils import random_click
import random
from monai.transforms import LoadImaged, Randomizable,LoadImage
import shutil
import json
import h5py

def load_mask_from_file(mask_path):
    with h5py.File(mask_path, 'r') as mask_h5:
        if 'annotations' in mask_h5:
            annotations = mask_h5['annotations']
            if 'proposed' in annotations:
                return torch.from_numpy(annotations['proposed'][:].astype(np.float64)).unsqueeze(0)
            elif 'result' in annotations:
                return torch.from_numpy(annotations['result'][:].astype(np.float64)).unsqueeze(0)
    print("key error in mask loading!")
    return None

class iqs_dv(Dataset):
    def __init__(self, data_path, crop_size, transform_3D=None, transform_msk_3D=None,transform_2D=None, transform_msk_2D=None):
        self.data_path = data_path
        self.crop_size = crop_size
        self.transform_3D = transform_3D
        self.transform_msk_3D = transform_msk_3D
        self.transform_2D = transform_2D
        self.transform_msk_2D = transform_msk_2D
        self.annotations_path = os.path.join(self.data_path, 'zeiss_annotations')
        self.images_path = os.path.join(self.data_path, 'images')
        self.masks_path = os.path.join(self.data_path, 'masks')

        self.image_files = sorted([f for f in os.listdir(self.images_path) if f.endswith('.h5')])
        
        self.num_samples = len(self.image_files)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        img_filename = self.image_files[index]
        match = re.search(r'_dataset_(\d+)\.h5$', img_filename)
        dataset_number = match.group(1)

        # Load JSON annotation
        annotation_file = re.sub(r'_dataset_(\d+\.h5)$', fr'_data_data_dataset_{dataset_number}.json', img_filename)
        annotation_path = os.path.join(self.annotations_path, annotation_file)
        with open(annotation_path, 'r') as f:
            annotation_data = json.load(f)
       
        filepath = annotation_data['filePath'] 
        img_idx = int(filepath.split('/')[-1].split('_')[0])
            

        
        # Load image HDF5 file
        img_path = os.path.join(self.images_path, img_filename)
        
        with h5py.File(img_path, 'r') as img_h5:
            img_tensor = torch.from_numpy(img_h5['data']['data'][:].astype(np.float64)).unsqueeze(0)

        # Load mask HDF5 file
        

        mask_filename = re.sub(r'_dataset_(\d+\.h5)$', fr'_data_data_manual_dataset_{dataset_number}.h5', img_filename)
        #mask_filename = img_filename.split('.h5')[0] + '_data_data_manual.h5'
        mask_path = os.path.join(self.masks_path, mask_filename)
    
        if not os.path.exists(mask_path):
            mask_filename = re.sub(r'_dataset_(\d+\.h5)$', fr'_data_data_generated_dataset_{dataset_number}.h5', img_filename)
            mask_path = os.path.join(self.masks_path, mask_filename)
            
        mask_tensor = load_mask_from_file(mask_path)


        

        if self.transform_3D:
            img_tensor = self.transform_3D(img_tensor)
        if self.transform_msk_3D:
            mask_tensor = self.transform_msk_3D(mask_tensor)

        slices = []
        for d in range(img_tensor.shape[-1]):
            
            img_tensor_slice = img_tensor[..., d]
            mask_tensor_slice = mask_tensor[..., d]
            img_tensor_slice, mask_tensor_slice = crop_image_and_mask(img_tensor_slice.squeeze(0), mask_tensor_slice.squeeze(0), self.crop_size)
            if self.transform_2D:
                img_tensor_slice = self.transform_2D(img_tensor_slice)

            if self.transform_msk_2D:
                mask_tensor_slice = self.transform_msk_2D(mask_tensor_slice)
            
            slices.append({
                'image': img_tensor_slice,
                'label': mask_tensor_slice,
                'metadata': {'img_idx': img_idx, 'slice_idx': d, 'dataset_idx':dataset_number}
            })

        return slices

def collate_fn(batch):
    images = []
    labels = []
    slice_idx = []

    for item in batch:
        if item is not None:
            for slice_data in item:
                images.append(slice_data['image'])
                labels.append(slice_data['label'])
                slice_idx.append(f"{slice_data['metadata']['img_idx']}_{slice_data['metadata']['slice_idx']}_{slice_data['metadata']['dataset_idx']}")

    combined = list(zip(images, labels, slice_idx))  
    random.shuffle(combined)  
    images, labels, slice_idx = zip(*combined)

    return {
        'image':  torch.stack(images, dim=0),
        'label':  torch.stack(labels, dim=0),
        'metadata': slice_idx
    }

class DivideMaskByConstant:
    def __init__(self, constant):
        self.constant = constant

    def __call__(self, mask):
        return mask / self.constant
class FillMissingCells:
    def __init__(self, desired_shape):
        self.desired_shape = desired_shape

    def __call__(self, tensor):
       
        if tensor.shape == self.desired_shape:
            return tensor
        
        padded_tensor = torch.zeros(self.desired_shape, dtype=tensor.dtype)
        min_dims = [min(tensor.shape[i], self.desired_shape[i]) for i in range(tensor.ndim)]
        padded_tensor[tuple(slice(0, min_dim) for min_dim in min_dims)] = tensor[tuple(slice(0, min_dim) for min_dim in min_dims)]

        return padded_tensor


def interpolate(image,crop_size):
    original_height, original_width = image.shape[-2:]
    
    # Use the maximum of original size and crop size for interpolation
    interp_h = max(original_height, crop_size[0])
    interp_w = max(original_width, crop_size[1])

    # Interpolate the image using bilinear interpolation
    image = F.interpolate(image.unsqueeze(0).unsqueeze(0), size=(interp_h, interp_w), mode='bilinear').squeeze(0).squeeze(0)
    return image

def crop(image,crop_size,top_left_x,top_left_y):
    # Crop the image
    return image[top_left_y:top_left_y + crop_size[0], top_left_x:top_left_x + crop_size[1]]

def crop_image_and_mask(image, mask, crop_size):
    
    image = interpolate(image,crop_size)
    mask = interpolate(mask,crop_size)
    # Randomly choose the top-left corner of the crop
    top_left_x = np.random.randint(0, image.shape[1] - crop_size[1] + 1)
    top_left_y = np.random.randint(0, image.shape[0] - crop_size[0] + 1)
    
    cropped_image = crop(image, crop_size,top_left_x,top_left_y)
    cropped_mask = crop(mask, crop_size, top_left_x, top_left_y)

    return cropped_image.unsqueeze(0), cropped_mask.unsqueeze(0)


def spilt_data(data_path = '/home/zozchaab/data/deepvision/iqs_dv_01'  ,destination_path = '/home/zozchaab/data/deepvision',train_ratio = 0.7, val_ratio = 0.2):

    # Create train, val, test folders
    train_folder = os.path.join(destination_path, 'iqs_dv_01_train')
    val_folder = os.path.join(destination_path, 'iqs_dv_01_val')
    test_folder = os.path.join(destination_path, 'iqs_dv_01_test')

    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    # List all images in the images folder
    image_files = os.listdir(os.path.join(data_path, 'images'))

    # Shuffle the image files
    random.shuffle(image_files)

    # Calculate the number of images for each split
    num_images = len(image_files)
    num_train = int(train_ratio * num_images)
    num_val = int(val_ratio * num_images)

    # Split the image files into train, val, and test sets
    train_images = image_files[:num_train]
    val_images = image_files[num_train:num_train + num_val]
    test_images = image_files[num_train + num_val:]

    # Function to copy files to their respective folders
    def copy_files(image_list, source_folder, dest_folder):
        img_folder = os.path.join(dest_folder, 'images')
        mask_folder = os.path.join(dest_folder, 'masks')
        ann_folder = os.path.join(dest_folder, 'zeiss_annotations')
        os.makedirs(img_folder, exist_ok=True)
        os.makedirs(mask_folder, exist_ok=True)
        os.makedirs(ann_folder, exist_ok=True)
        for img_file in image_list:
            base_filename = os.path.splitext(img_file)[0]
            annotation_filename = base_filename + '_data_data.json'
            shutil.copy(os.path.join(source_folder, 'zeiss_annotations', annotation_filename),os.path.join(ann_folder, annotation_filename) )
            shutil.copy(os.path.join(source_folder, 'images', img_file),os.path.join(img_folder,img_file) )
            label_filename_user = base_filename + '_data_data_user_sample.h5'
            label_filename_manual = base_filename + '_data_data_manual.h5'

            source_user_path = os.path.join(source_folder, 'masks', label_filename_user)
            source_manual_path = os.path.join(source_folder, 'masks', label_filename_manual)

            mask_folder_user = os.path.join(mask_folder, label_filename_user)
            mask_folder_manual = os.path.join(mask_folder, label_filename_manual)

            if os.path.exists(source_user_path):
                shutil.copy(source_user_path, mask_folder_user)
            
            shutil.copy(source_manual_path, mask_folder_manual)
            


    copy_files(train_images, data_path, train_folder)
    copy_files(val_images,data_path, val_folder)
    copy_files(test_images, data_path, test_folder)



class ISIC2016(Dataset):
    def __init__(self, args, data_path , transform = None, transform_msk = None, mode = 'Training',prompt = 'click', plane = False):

        df = pd.read_csv(os.path.join(data_path, 'ISBI2016_ISIC_Part1_' + mode + '_GroundTruth.csv'), encoding='gbk')
        self.name_list = df.iloc[:,1].tolist()
        self.label_list = df.iloc[:,2].tolist()
        self.data_path = data_path
        self.mode = mode
        self.prompt = prompt
        self.img_size = args.image_size

        self.transform = transform
        self.transform_msk = transform_msk

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        # if self.mode == 'Training':
        #     point_label = random.randint(0, 1)
        #     inout = random.randint(0, 1)
        # else:
        #     inout = 1
        #     point_label = 1
        inout = 1
        point_label = 1

        """Get the images"""
        name = self.name_list[index]
        img_path = os.path.join(self.data_path, name)
        
        mask_name = self.label_list[index]
        msk_path = os.path.join(self.data_path, mask_name)

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(msk_path).convert('L')

        # if self.mode == 'Training':
        #     label = 0 if self.label_list[index] == 'benign' else 1
        # else:
        #     label = int(self.label_list[index])

        newsize = (self.img_size, self.img_size)
        mask = mask.resize(newsize)

        if self.prompt == 'click':
            pt = random_click(np.array(mask) / 255, point_label, inout)

        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            torch.set_rng_state(state)


            if self.transform_msk:
                mask = self.transform_msk(mask)
                
            # if (inout == 0 and point_label == 1) or (inout == 1 and point_label == 0):
            #     mask = 1 - mask

        name = name.split('/')[-1].split(".jpg")[0]
        image_meta_dict = {'filename_or_obj':name}
        return {
            'image':img,
            'label': mask,
            'p_label':point_label,
            'pt':pt,
            'image_meta_dict':image_meta_dict,
        }


class REFUGE(Dataset):
    def __init__(self, args, data_path , transform = None, transform_msk = None, mode = 'Training',prompt = 'click', plane = False):
        self.data_path = data_path
        self.subfolders = [f.path for f in os.scandir(os.path.join(data_path, mode + '-400')) if f.is_dir()]
        self.mode = mode
        self.prompt = prompt
        self.img_size = args.image_size
        self.mask_size = args.out_size

        self.transform = transform
        self.transform_msk = transform_msk

    def __len__(self):
        return len(self.subfolders)

    def __getitem__(self, index):
        inout = 1
        point_label = 1

        """Get the images"""
        subfolder = self.subfolders[index]
        name = subfolder.split('/')[-1]

        # raw image and raters path
        img_path = os.path.join(subfolder, name + '.jpg')
        multi_rater_cup_path = [os.path.join(subfolder, name + '_seg_cup_' + str(i) + '.png') for i in range(1, 8)]
        multi_rater_disc_path = [os.path.join(subfolder, name + '_seg_disc_' + str(i) + '.png') for i in range(1, 8)]

        # raw image and raters images
        img = Image.open(img_path).convert('RGB')
        multi_rater_cup = [Image.open(path).convert('L') for path in multi_rater_cup_path]
        multi_rater_disc = [Image.open(path).convert('L') for path in multi_rater_disc_path]

        # resize raters images for generating initial point click
        newsize = (self.img_size, self.img_size)
        multi_rater_cup_np = [np.array(single_rater.resize(newsize)) for single_rater in multi_rater_cup]
        multi_rater_disc_np = [np.array(single_rater.resize(newsize)) for single_rater in multi_rater_disc]

        # first click is the target agreement among all raters
        if self.prompt == 'click':
            pt_cup = random_click(np.array(np.mean(np.stack(multi_rater_cup_np), axis=0)) / 255, point_label, inout)
            pt_disc = random_click(np.array(np.mean(np.stack(multi_rater_disc_np), axis=0)) / 255, point_label, inout)

        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            multi_rater_cup = [torch.as_tensor((self.transform(single_rater) >0.5).float(), dtype=torch.float32) for single_rater in multi_rater_cup]
            multi_rater_cup = torch.stack(multi_rater_cup, dim=0)
            # transform to mask size (out_size) for mask define
            mask_cup = F.interpolate(multi_rater_cup, size=(self.mask_size, self.mask_size), mode='bilinear', align_corners=False).mean(dim=0)

            multi_rater_disc = [torch.as_tensor((self.transform(single_rater) >0.5).float(), dtype=torch.float32) for single_rater in multi_rater_disc]
            multi_rater_disc = torch.stack(multi_rater_disc, dim=0)
            mask_disc = F.interpolate(multi_rater_disc, size=(self.mask_size, self.mask_size), mode='bilinear', align_corners=False).mean(dim=0)
            torch.set_rng_state(state)

        image_meta_dict = {'filename_or_obj':name}
        return {
            'image':img,
            'multi_rater_cup': multi_rater_cup,
            'multi_rater_disc': multi_rater_disc,
            'mask_cup': mask_cup,
            'mask_disc': mask_disc,
            'label': mask_disc,
            'p_label':point_label,
            'pt_cup':pt_cup,
            'pt_disc':pt_disc,
            'pt':pt_disc,
            'selected_rater': torch.tensor(np.arange(7)),
            'image_meta_dict':image_meta_dict,
        }
        
