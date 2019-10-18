#!/usr/bin/python
# encoding: utf-8

import os
import random
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from utils import read_truths_args, read_truths, get_all_files
from image_multi import *

class listDataset(Dataset):

    def __init__(self, root, shape=None, shuffle=True, transform=None, objclass=None, target_transform=None, train=False, seen=0, batch_size=64, num_workers=4, bg_file_names=None): # bg='/cvlabdata1/home/btekin/ope/data/office_bg'
       with open(root, 'r') as file:
           self.lines = file.readlines()
       if shuffle:
           random.shuffle(self.lines)
       self.nSamples         = len(self.lines)
       self.transform        = transform
       self.target_transform = target_transform
       self.train            = train
       self.shape            = shape
       self.seen             = seen
       self.batch_size       = batch_size
       self.num_workers      = num_workers
       # self.bg_file_names    = get_all_files(bg)
       self.bg_file_names    = bg_file_names
       self.objclass         = objclass

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        imgpath = self.lines[index].rstrip()

        if self.train and index % 64== 0:
            if self.seen < 4000*64:
               width = 13*32
               self.shape = (width, width)
            elif self.seen < 8000*64:
               width = (random.randint(0,3) + 13)*32
               self.shape = (width, width)
            elif self.seen < 12000*64:
               width = (random.randint(0,5) + 12)*32
               self.shape = (width, width)
            elif self.seen < 16000*64:
               width = (random.randint(0,7) + 11)*32
               self.shape = (width, width)
            else: # self.seen < 20000*64:
               width = (random.randint(0,9) + 10)*32
               self.shape = (width, width)

        if self.train:
            # jitter = 0.2
            jitter = 0.1
            hue = 0.05
            saturation = 1.5 
            exposure = 1.5

            # Get background image path
            random_bg_index = random.randint(0, len(self.bg_file_names) - 1)
            bgpath = self.bg_file_names[random_bg_index]

            img, label = load_data_detection(imgpath, self.shape, jitter, hue, saturation, exposure, bgpath)
            label = torch.from_numpy(label)
        else:
            img = Image.open(imgpath).convert('RGB')
            if self.shape:
                img = img.resize(self.shape)
            
            labpath = imgpath.replace('benchvise', self.objclass).replace('images', 'labels_occlusion').replace('JPEGImages', 'labels_occlusion').replace('.jpg', '.txt').replace('.png','.txt')
            label = torch.zeros(50*21)
            if os.path.getsize(labpath):
                ow, oh = img.size
                tmp = torch.from_numpy(read_truths_args(labpath, 8.0/ow))
                tmp = tmp.view(-1)
                tsz = tmp.numel()
                if tsz > 50*21:
                    label = tmp[0:50*21]
                elif tsz > 0:
                    label[0:tsz] = tmp

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        self.seen = self.seen + self.num_workers
        return (img, label)
