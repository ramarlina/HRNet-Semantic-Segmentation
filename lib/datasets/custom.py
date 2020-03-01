# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import os

import cv2
import numpy as np

import torch
from torch.nn import functional as F

from .base_dataset import BaseDataset

def get_mask_fn(fname):
    return os.path.basename(fname).replace(".jpg", ".png")
    
class ImageFolder(BaseDataset):
    def __init__(self,  
                 images_path, 
                 masks_path, 
                 get_mask_fn=get_mask_fn,
                 num_samples=None, 
                 num_classes=20,
                 multi_scale=True, 
                 flip=True, 
                 ignore_label=-1, 
                 base_size=473, 
                 crop_size=(473, 473), 
                 downsample_rate=1,
                 scale_factor=11,
                 mean=[0.485, 0.456, 0.406], 
                 std=[0.229, 0.224, 0.225],
                 is_test=False):

        super(ImageFolder, self).__init__(ignore_label, base_size,
                crop_size, downsample_rate, scale_factor, mean, std)
 
        self.masks_path = masks_path
        self.get_mask_fn = get_mask_fn
        self.num_classes = num_classes
        self.images_path = images_path
        self.class_weights = None

        self.multi_scale = multi_scale
        self.flip = flip
        self.img_list = [os.path.join(images_path, f) for f in os.listdir(images_path)]

        self.files = self.read_files()
        self.is_test = is_test
        if num_samples:
            self.files = self.files[:num_samples]
    
    def read_files(self):
        files = []
        for image_path in self.img_list:  
            name = os.path.splitext(os.path.basename(image_path))[0]
            label_path = os.path.join(self.masks_path, self.get_mask_fn(image_path))
            sample = {
                "img": image_path,
                "label": label_path, 
                 "name": name
            }  
            files.append(sample) 
        return files

    def resize_image(self, image, label, size): 
        image = cv2.resize(image, size, interpolation = cv2.INTER_LINEAR) 
        label = cv2.resize(label, size, interpolation=cv2.INTER_NEAREST)
        return image, label
     
    def __getitem__(self, index):
        item = self.files[index]
        name = item["name"]
         
        image = cv2.imread(item["img"], 
                    cv2.IMREAD_COLOR)
        label = cv2.imread(item["label"],
                    cv2.IMREAD_GRAYSCALE)
        size = label.shape

        if self.is_test:
            image = cv2.resize(image, self.crop_size, 
                               interpolation = cv2.INTER_LINEAR)
            image = self.input_transform(image)
            image = image.transpose((2, 0, 1))

            return image.copy(), label.copy(), np.array(size), name
        
        image, label = self.resize_image(image, label, self.crop_size)
        image, label = self.gen_sample(image, label, 
                                self.multi_scale, False)

        return image.copy(), label.copy(), np.array(size), name

    def inference(self, model, image, flip):
        size = image.size()
        pred = model(image)
        pred = F.upsample(input=pred, 
                          size=(size[-2], size[-1]), 
                          mode='bilinear')        
        if flip:
            flip_img = image.numpy()[:,:,:,::-1]
            flip_output = model(torch.from_numpy(flip_img.copy()))
            flip_output = F.upsample(input=flip_output, 
                            size=(size[-2], size[-1]), 
                            mode='bilinear')
            flip_pred = flip_output.cpu().numpy().copy()
            flip_pred[:,14,:,:] = flip_output[:,15,:,:]
            flip_pred[:,15,:,:] = flip_output[:,14,:,:]
            flip_pred[:,16,:,:] = flip_output[:,17,:,:]
            flip_pred[:,17,:,:] = flip_output[:,16,:,:]
            flip_pred[:,18,:,:] = flip_output[:,19,:,:]
            flip_pred[:,19,:,:] = flip_output[:,18,:,:]
            flip_pred = torch.from_numpy(flip_pred[:,:,:,::-1].copy()).cuda()
            pred += flip_pred
            pred = pred * 0.5
        return pred.exp()
    
