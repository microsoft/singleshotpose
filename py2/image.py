#!/usr/bin/python
# encoding: utf-8
import random
import os
from PIL import Image, ImageChops, ImageMath
import numpy as np

def scale_image_channel(im, c, v):
    cs = list(im.split())
    cs[c] = cs[c].point(lambda i: i * v)
    out = Image.merge(im.mode, tuple(cs))
    return out

def distort_image(im, hue, sat, val):
    im = im.convert('HSV')
    cs = list(im.split())
    cs[1] = cs[1].point(lambda i: i * sat)
    cs[2] = cs[2].point(lambda i: i * val)
    
    def change_hue(x):
        x += hue*255
        if x > 255:
            x -= 255
        if x < 0:
            x += 255
        return x
    cs[0] = cs[0].point(change_hue)
    im = Image.merge(im.mode, tuple(cs))

    im = im.convert('RGB')
    return im

def rand_scale(s):
    scale = random.uniform(1, s)
    if(random.randint(1,10000)%2): 
        return scale
    return 1./scale

def random_distort_image(im, hue, saturation, exposure):
    dhue = random.uniform(-hue, hue)
    dsat = rand_scale(saturation)
    dexp = rand_scale(exposure)
    res  = distort_image(im, dhue, dsat, dexp)
    return res

def data_augmentation(img, shape, jitter, hue, saturation, exposure):

    ow, oh = img.size
    
    dw =int(ow*jitter)
    dh =int(oh*jitter)

    pleft  = random.randint(-dw, dw)
    pright = random.randint(-dw, dw)
    ptop   = random.randint(-dh, dh)
    pbot   = random.randint(-dh, dh)

    swidth =  ow - pleft - pright
    sheight = oh - ptop - pbot

    sx = float(swidth)  / ow
    sy = float(sheight) / oh
    
    flip = random.randint(1,10000)%2
    cropped = img.crop( (pleft, ptop, pleft + swidth - 1, ptop + sheight - 1))

    dx = (float(pleft)/ow)/sx
    dy = (float(ptop) /oh)/sy

    sized = cropped.resize(shape)

    img = random_distort_image(sized, hue, saturation, exposure)
    
    return img, flip, dx,dy,sx,sy 

def fill_truth_detection(labpath, w, h, flip, dx, dy, sx, sy):
    max_boxes = 50
    label = np.zeros((max_boxes,21))
    if os.path.getsize(labpath):
        bs = np.loadtxt(labpath)
        if bs is None:
            return label
        bs = np.reshape(bs, (-1, 21))
        cc = 0
        for i in range(bs.shape[0]):
            x0 = bs[i][1]
            y0 = bs[i][2]
            x1 = bs[i][3]
            y1 = bs[i][4]
            x2 = bs[i][5]
            y2 = bs[i][6]
            x3 = bs[i][7]
            y3 = bs[i][8]
            x4 = bs[i][9]
            y4 = bs[i][10]
            x5 = bs[i][11]
            y5 = bs[i][12]
            x6 = bs[i][13]
            y6 = bs[i][14]
            x7 = bs[i][15]
            y7 = bs[i][16]
            x8 = bs[i][17]
            y8 = bs[i][18]

            x0 = min(0.999, max(0, x0 * sx - dx)) 
            y0 = min(0.999, max(0, y0 * sy - dy)) 
            x1 = min(0.999, max(0, x1 * sx - dx)) 
            y1 = min(0.999, max(0, y1 * sy - dy)) 
            x2 = min(0.999, max(0, x2 * sx - dx))
            y2 = min(0.999, max(0, y2 * sy - dy))
            x3 = min(0.999, max(0, x3 * sx - dx))
            y3 = min(0.999, max(0, y3 * sy - dy))
            x4 = min(0.999, max(0, x4 * sx - dx))
            y4 = min(0.999, max(0, y4 * sy - dy))
            x5 = min(0.999, max(0, x5 * sx - dx))
            y5 = min(0.999, max(0, y5 * sy - dy))
            x6 = min(0.999, max(0, x6 * sx - dx))
            y6 = min(0.999, max(0, y6 * sy - dy))
            x7 = min(0.999, max(0, x7 * sx - dx))
            y7 = min(0.999, max(0, y7 * sy - dy))
            x8 = min(0.999, max(0, x8 * sx - dx))
            y8 = min(0.999, max(0, y8 * sy - dy))
            
            bs[i][1] = x0
            bs[i][2] = y0
            bs[i][3] = x1
            bs[i][4] = y1
            bs[i][5] = x2
            bs[i][6] = y2
            bs[i][7] = x3
            bs[i][8] = y3
            bs[i][9] = x4
            bs[i][10] = y4
            bs[i][11] = x5
            bs[i][12] = y5
            bs[i][13] = x6
            bs[i][14] = y6
            bs[i][15] = x7
            bs[i][16] = y7
            bs[i][17] = x8
            bs[i][18] = y8
            
            label[cc] = bs[i]
            cc += 1
            if cc >= 50:
                break

    label = np.reshape(label, (-1))
    return label

def change_background(img, mask, bg):
    # oh = img.height  
    # ow = img.width
    ow, oh = img.size
    bg = bg.resize((ow, oh)).convert('RGB')
    
    imcs = list(img.split())
    bgcs = list(bg.split())
    maskcs = list(mask.split())
    fics = list(Image.new(img.mode, img.size).split())
    
    for c in range(len(imcs)):
        negmask = maskcs[c].point(lambda i: 1 - i / 255)
        posmask = maskcs[c].point(lambda i: i / 255)
        fics[c] = ImageMath.eval("a * c + b * d", a=imcs[c], b=bgcs[c], c=posmask, d=negmask).convert('L')
    out = Image.merge(img.mode, tuple(fics))

    return out

def load_data_detection(imgpath, shape, jitter, hue, saturation, exposure, bgpath):
    labpath = imgpath.replace('images', 'labels').replace('JPEGImages', 'labels').replace('.jpg', '.txt').replace('.png','.txt')
    maskpath = imgpath.replace('JPEGImages', 'mask').replace('/00', '/').replace('.jpg', '.png')

    ## data augmentation
    img = Image.open(imgpath).convert('RGB')
    mask = Image.open(maskpath).convert('RGB')
    bg = Image.open(bgpath).convert('RGB')
    
    img = change_background(img, mask, bg)
    img,flip,dx,dy,sx,sy = data_augmentation(img, shape, jitter, hue, saturation, exposure)
    ow, oh = img.size
    label = fill_truth_detection(labpath, ow, oh, flip, dx, dy, 1./sx, 1./sy)
    return img,label

