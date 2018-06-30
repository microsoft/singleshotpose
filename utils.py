import sys
import os
import time
import math
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torch.autograd import Variable
import torch.nn.functional as F
import cv2
from scipy import spatial

import struct 
import imghdr 

def get_all_files(directory):
    files = []

    for f in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, f)):
            files.append(os.path.join(directory, f))
        else:
            files.extend(get_all_files(os.path.join(directory, f)))
    return files

def calcAngularDistance(gt_rot, pr_rot):

    rotDiff = np.dot(gt_rot, np.transpose(pr_rot))
    trace = np.trace(rotDiff) 
    return np.rad2deg(np.arccos((trace-1.0)/2.0))

def get_camera_intrinsic():
    K = np.zeros((3, 3), dtype='float64')
    K[0, 0], K[0, 2] = 572.4114, 325.2611
    K[1, 1], K[1, 2] = 573.5704, 242.0489
    K[2, 2] = 1.
    return K

def compute_projection(points_3D, transformation, internal_calibration):
    projections_2d = np.zeros((2, points_3D.shape[1]), dtype='float32')
    camera_projection = (internal_calibration.dot(transformation)).dot(points_3D)
    projections_2d[0, :] = camera_projection[0, :]/camera_projection[2, :]
    projections_2d[1, :] = camera_projection[1, :]/camera_projection[2, :]
    return projections_2d

def compute_transformation(points_3D, transformation):
    return transformation.dot(points_3D)

def calc_pts_diameter(pts):
    diameter = -1
    for pt_id in range(pts.shape[0]):
        pt_dup = np.tile(np.array([pts[pt_id, :]]), [pts.shape[0] - pt_id, 1])
        pts_diff = pt_dup - pts[pt_id:, :]
        max_dist = math.sqrt((pts_diff * pts_diff).sum(axis=1).max())
        if max_dist > diameter:
            diameter = max_dist
    return diameter

def adi(pts_est, pts_gt):
    nn_index = spatial.cKDTree(pts_est)
    nn_dists, _ = nn_index.query(pts_gt, k=1)
    e = nn_dists.mean()
    return e

def get_3D_corners(vertices):
    
    min_x = np.min(vertices[0,:])
    max_x = np.max(vertices[0,:])
    min_y = np.min(vertices[1,:])
    max_y = np.max(vertices[1,:])
    min_z = np.min(vertices[2,:])
    max_z = np.max(vertices[2,:])

    corners = np.array([[min_x, min_y, min_z],
                        [min_x, min_y, max_z],
                        [min_x, max_y, min_z],
                        [min_x, max_y, max_z],
                        [max_x, min_y, min_z],
                        [max_x, min_y, max_z],
                        [max_x, max_y, min_z],
                        [max_x, max_y, max_z]])

    corners = np.concatenate((np.transpose(corners), np.ones((1,8)) ), axis=0)
    return corners

def pnp(points_3D, points_2D, cameraMatrix):
    try:
        distCoeffs = pnp.distCoeffs
    except:
        distCoeffs = np.zeros((8, 1), dtype='float32')

    assert points_2D.shape[0] == points_2D.shape[0], 'points 3D and points 2D must have same number of vertices'

    _, R_exp, t = cv2.solvePnP(points_3D,
                              # points_2D,
                              np.ascontiguousarray(points_2D[:,:2]).reshape((-1,1,2)),
                              cameraMatrix,
                              distCoeffs)
                              # , None, None, False, cv2.SOLVEPNP_UPNP)
                                
    # R_exp, t, _ = cv2.solvePnPRansac(points_3D,
    #                            points_2D,
    #                            cameraMatrix,
    #                            distCoeffs,
    #                            reprojectionError=12.0)
    # 

    R, _ = cv2.Rodrigues(R_exp)
    # Rt = np.c_[R, t]
    return R, t

def get_2d_bb(box, size):
    x = box[0]
    y = box[1]
    min_x = np.min(np.reshape(box, [9,2])[:,0])
    max_x = np.max(np.reshape(box, [9,2])[:,0])
    min_y = np.min(np.reshape(box, [9,2])[:,1])
    max_y = np.max(np.reshape(box, [9,2])[:,1])
    w = max_x - min_x
    h = max_y - min_y
    new_box = [x*size, y*size, w*size, h*size]
    return new_box

def compute_2d_bb(pts):
    min_x = np.min(pts[0,:])
    max_x = np.max(pts[0,:])
    min_y = np.min(pts[1,:])
    max_y = np.max(pts[1,:])
    w  = max_x - min_x
    h  = max_y - min_y
    cx = (max_x + min_x) / 2.0
    cy = (max_y + min_y) / 2.0
    new_box = [cx, cy, w, h]
    return new_box

def compute_2d_bb_from_orig_pix(pts, size):
    min_x = np.min(pts[0,:]) / 640.0
    max_x = np.max(pts[0,:]) / 640.0
    min_y = np.min(pts[1,:]) / 480.0
    max_y = np.max(pts[1,:]) / 480.0
    w  = max_x - min_x
    h  = max_y - min_y
    cx = (max_x + min_x) / 2.0
    cy = (max_y + min_y) / 2.0
    new_box = [cx*size, cy*size, w*size, h*size]
    return new_box

def bbox_iou(box1, box2, x1y1x2y2=False):
    if x1y1x2y2:
        mx = min(box1[0], box2[0])
        Mx = max(box1[2], box2[2])
        my = min(box1[1], box2[1])
        My = max(box1[3], box2[3])
        w1 = box1[2] - box1[0]
        h1 = box1[3] - box1[1]
        w2 = box2[2] - box2[0]
        h2 = box2[3] - box2[1]
    else:
        mx = min(box1[0]-box1[2]/2.0, box2[0]-box2[2]/2.0)
        Mx = max(box1[0]+box1[2]/2.0, box2[0]+box2[2]/2.0)
        my = min(box1[1]-box1[3]/2.0, box2[1]-box2[3]/2.0)
        My = max(box1[1]+box1[3]/2.0, box2[1]+box2[3]/2.0)
        w1 = box1[2]
        h1 = box1[3]
        w2 = box2[2]
        h2 = box2[3]
    uw = Mx - mx
    uh = My - my
    cw = w1 + w2 - uw
    ch = h1 + h2 - uh
    carea = 0
    if cw <= 0 or ch <= 0:
        return 0.0

    area1 = w1 * h1
    area2 = w2 * h2
    carea = cw * ch
    uarea = area1 + area2 - carea
    return carea/uarea

def corner_confidences(gt_corners, pr_corners, th=30, sharpness=2, im_width=640, im_height=480):
    ''' gt_corners: Ground-truth 2D projections of the 3D bounding box corners, shape: (16 x nA), type: torch.FloatTensor
        pr_corners: Prediction for the 2D projections of the 3D bounding box corners, shape: (16 x nA), type: torch.FloatTensor
        th        : distance threshold, type: int
        sharpness : sharpness of the exponential that assigns a confidence value to the distance
        -----------
        return    : a torch.FloatTensor of shape (nA,) with 8 confidence values 
    '''
    shape = gt_corners.size()
    nA = shape[1]  
    dist = gt_corners - pr_corners
    dist = dist.t().contiguous().view(nA, 8, 2)
    dist[:, :, 0] = dist[:, :, 0] * im_width
    dist[:, :, 1] = dist[:, :, 1] * im_height

    eps = 1e-5
    distthresh = torch.FloatTensor([th]).repeat(nA, 8) 
    dist = torch.sqrt(torch.sum((dist)**2, dim=2)).squeeze() # nA x 8 
    mask = (dist < distthresh).type(torch.FloatTensor)
    conf = torch.exp(sharpness*(1 - dist/distthresh))-1  # mask * (torch.exp(math.log(2) * (1.0 - dist/rrt)) - 1)
    conf0 = torch.exp(sharpness*(1 - torch.zeros(conf.size(0),1))) - 1
    conf = conf / conf0.repeat(1, 8)
    # conf = 1 - dist/distthresh
    conf = mask * conf  # nA x 8
    mean_conf = torch.mean(conf, dim=1)
    return mean_conf

def corner_confidence(gt_corners, pr_corners, th=30, sharpness=2, im_width=640, im_height=480):
    ''' gt_corners: Ground-truth 2D projections of the 3D bounding box corners, shape: (16,) type: list
        pr_corners: Prediction for the 2D projections of the 3D bounding box corners, shape: (16,), type: list
        th        : distance threshold, type: int
        sharpness : sharpness of the exponential that assigns a confidence value to the distance
        -----------
        return    : a list of shape (8,) with 8 confidence values 
    '''
    dist = torch.FloatTensor(gt_corners) - pr_corners
    dist = dist.view(8, 2)
    dist[:, 0] = dist[:, 0] * im_width
    dist[:, 1] = dist[:, 1] * im_height
    eps = 1e-5
    dist  = torch.sqrt(torch.sum((dist)**2, dim=1))
    mask  = (dist < th).type(torch.FloatTensor)
    conf  = torch.exp(sharpness * (1.0 - dist/th)) - 1
    conf0 = torch.exp(torch.FloatTensor([sharpness])) - 1 + eps
    conf  = conf / conf0.repeat(8, 1)
    # conf = 1.0 - dist/th
    conf  = mask * conf 
    return torch.mean(conf)

def corner_confidences9(gt_corners, pr_corners, th=80, sharpness=2, im_width=640, im_height=480):
    ''' gt_corners: Ground-truth 2D projections of the 3D bounding box corners, shape: (16 x nA), type: torch.FloatTensor
        pr_corners: Prediction for the 2D projections of the 3D bounding box corners, shape: (16 x nA), type: torch.FloatTensor
        th        : distance threshold, type: int
        sharpness : sharpness of the exponential that assigns a confidence value to the distance
        -----------
        return    : a torch.FloatTensor of shape (nA,) with 9 confidence values 
    '''
    shape = gt_corners.size()
    nA = shape[1]  
    dist = gt_corners - pr_corners
    dist = dist.t().contiguous().view(nA, 9, 2)
    dist[:, :, 0] = dist[:, :, 0] * im_width
    dist[:, :, 1] = dist[:, :, 1] * im_height

    eps = 1e-5
    distthresh = torch.FloatTensor([th]).repeat(nA, 9) 
    dist = torch.sqrt(torch.sum((dist)**2, dim=2)).squeeze() # nA x 9
    mask = (dist < distthresh).type(torch.FloatTensor)
    conf = torch.exp(sharpness*(1 - dist/distthresh))-1  # mask * (torch.exp(math.log(2) * (1.0 - dist/rrt)) - 1)
    conf0 = torch.exp(sharpness*(1 - torch.zeros(conf.size(0),1))) - 1
    conf = conf / conf0.repeat(1, 9)
    # conf = 1 - dist/distthresh
    conf = mask * conf  # nA x 9
    mean_conf = torch.mean(conf, dim=1)
    return mean_conf

def corner_confidence9(gt_corners, pr_corners, th=80, sharpness=2, im_width=640, im_height=480):
    ''' gt_corners: Ground-truth 2D projections of the 3D bounding box corners, shape: (18,) type: list
        pr_corners: Prediction for the 2D projections of the 3D bounding box corners, shape: (18,), type: list
        th        : distance threshold, type: int
        sharpness : sharpness of the exponential that assigns a confidence value to the distance
        -----------
        return    : a list of shape (9,) with 9 confidence values 
    '''
    dist = torch.FloatTensor(gt_corners) - pr_corners
    dist = dist.view(9, 2)
    dist[:, 0] = dist[:, 0] * im_width
    dist[:, 1] = dist[:, 1] * im_height
    eps = 1e-5
    dist  = torch.sqrt(torch.sum((dist)**2, dim=1))
    mask  = (dist < th).type(torch.FloatTensor)
    conf  = torch.exp(sharpness * (1.0 - dist/th)) - 1
    conf0 = torch.exp(torch.FloatTensor([sharpness])) - 1 + eps
    conf  = conf / conf0.repeat(9, 1)
    # conf = 1.0 - dist/th
    conf  = mask * conf 
    return torch.mean(conf)

def sigmoid(x):
    return 1.0/(math.exp(-x)+1.)

def softmax(x):
    x = torch.exp(x - torch.max(x))
    x = x/x.sum()
    return x

def nms(boxes, nms_thresh):
    if len(boxes) == 0:
        return boxes

    det_confs = torch.zeros(len(boxes))
    for i in range(len(boxes)):
        det_confs[i] = 1-boxes[i][4]                

    _,sortIds = torch.sort(det_confs)
    out_boxes = []
    for i in range(len(boxes)):
        box_i = boxes[sortIds[i]]
        if box_i[4] > 0:
            out_boxes.append(box_i)
            for j in range(i+1, len(boxes)):
                box_j = boxes[sortIds[j]]
                if bbox_iou(box_i, box_j, x1y1x2y2=False) > nms_thresh:
                    #print(box_i, box_j, bbox_iou(box_i, box_j, x1y1x2y2=False))
                    box_j[4] = 0
    return out_boxes

def fix_corner_order(corners2D_gt):
    corners2D_gt_corrected = np.zeros((9, 2), dtype='float32')
    corners2D_gt_corrected[0, :] = corners2D_gt[0, :]
    corners2D_gt_corrected[1, :] = corners2D_gt[1, :]
    corners2D_gt_corrected[2, :] = corners2D_gt[3, :]
    corners2D_gt_corrected[3, :] = corners2D_gt[5, :]
    corners2D_gt_corrected[4, :] = corners2D_gt[7, :]
    corners2D_gt_corrected[5, :] = corners2D_gt[2, :]
    corners2D_gt_corrected[6, :] = corners2D_gt[4, :]
    corners2D_gt_corrected[7, :] = corners2D_gt[6, :]
    corners2D_gt_corrected[8, :] = corners2D_gt[8, :]
    return corners2D_gt_corrected

def convert2cpu(gpu_matrix):
    return torch.FloatTensor(gpu_matrix.size()).copy_(gpu_matrix)

def convert2cpu_long(gpu_matrix):
    return torch.LongTensor(gpu_matrix.size()).copy_(gpu_matrix)

def get_region_boxes(output, conf_thresh, num_classes, only_objectness=1, validation=False):
    
    # Parameters
    anchor_dim = 1 
    if output.dim() == 3:
        output = output.unsqueeze(0)
    batch = output.size(0)
    assert(output.size(1) == (19+num_classes)*anchor_dim)
    h = output.size(2)
    w = output.size(3)

    # Activation
    t0 = time.time()
    all_boxes = []
    max_conf = -100000
    output    = output.view(batch*anchor_dim, 19+num_classes, h*w).transpose(0,1).contiguous().view(19+num_classes, batch*anchor_dim*h*w)
    grid_x    = torch.linspace(0, w-1, w).repeat(h,1).repeat(batch*anchor_dim, 1, 1).view(batch*anchor_dim*h*w).cuda()
    grid_y    = torch.linspace(0, h-1, h).repeat(w,1).t().repeat(batch*anchor_dim, 1, 1).view(batch*anchor_dim*h*w).cuda()
    xs0       = torch.sigmoid(output[0]) + grid_x
    ys0       = torch.sigmoid(output[1]) + grid_y
    xs1       = output[2] + grid_x
    ys1       = output[3] + grid_y
    xs2       = output[4] + grid_x
    ys2       = output[5] + grid_y
    xs3       = output[6] + grid_x
    ys3       = output[7] + grid_y
    xs4       = output[8] + grid_x
    ys4       = output[9] + grid_y
    xs5       = output[10] + grid_x
    ys5       = output[11] + grid_y
    xs6       = output[12] + grid_x
    ys6       = output[13] + grid_y
    xs7       = output[14] + grid_x
    ys7       = output[15] + grid_y
    xs8       = output[16] + grid_x
    ys8       = output[17] + grid_y
    det_confs = torch.sigmoid(output[18])
    cls_confs = torch.nn.Softmax()(Variable(output[19:19+num_classes].transpose(0,1))).data
    cls_max_confs, cls_max_ids = torch.max(cls_confs, 1)
    cls_max_confs = cls_max_confs.view(-1)
    cls_max_ids   = cls_max_ids.view(-1)
    t1 = time.time()
    
    # GPU to CPU
    sz_hw = h*w
    sz_hwa = sz_hw*anchor_dim
    det_confs = convert2cpu(det_confs)
    cls_max_confs = convert2cpu(cls_max_confs)
    cls_max_ids = convert2cpu_long(cls_max_ids)
    xs0 = convert2cpu(xs0)
    ys0 = convert2cpu(ys0)
    xs1 = convert2cpu(xs1)
    ys1 = convert2cpu(ys1)
    xs2 = convert2cpu(xs2)
    ys2 = convert2cpu(ys2)
    xs3 = convert2cpu(xs3)
    ys3 = convert2cpu(ys3)
    xs4 = convert2cpu(xs4)
    ys4 = convert2cpu(ys4)
    xs5 = convert2cpu(xs5)
    ys5 = convert2cpu(ys5)
    xs6 = convert2cpu(xs6)
    ys6 = convert2cpu(ys6)
    xs7 = convert2cpu(xs7)
    ys7 = convert2cpu(ys7)
    xs8 = convert2cpu(xs8)
    ys8 = convert2cpu(ys8)
    if validation:
        cls_confs = convert2cpu(cls_confs.view(-1, num_classes))
    t2 = time.time()

    # Boxes filter
    for b in range(batch):
        boxes = []
        max_conf = -1
        for cy in range(h):
            for cx in range(w):
                for i in range(anchor_dim):
                    ind = b*sz_hwa + i*sz_hw + cy*w + cx
                    det_conf =  det_confs[ind]
                    if only_objectness:
                        conf = det_confs[ind]
                    else:
                        conf = det_confs[ind] * cls_max_confs[ind]
                    
                    if conf > max_conf:
                        max_conf = conf
                        max_ind = ind                  
    
                    if conf > conf_thresh:
                        bcx0 = xs0[ind]
                        bcy0 = ys0[ind]
                        bcx1 = xs1[ind]
                        bcy1 = ys1[ind]
                        bcx2 = xs2[ind]
                        bcy2 = ys2[ind]
                        bcx3 = xs3[ind]
                        bcy3 = ys3[ind]
                        bcx4 = xs4[ind]
                        bcy4 = ys4[ind]
                        bcx5 = xs5[ind]
                        bcy5 = ys5[ind]
                        bcx6 = xs6[ind]
                        bcy6 = ys6[ind]
                        bcx7 = xs7[ind]
                        bcy7 = ys7[ind]
                        bcx8 = xs8[ind]
                        bcy8 = ys8[ind]
                        cls_max_conf = cls_max_confs[ind]
                        cls_max_id = cls_max_ids[ind]
                        box = [bcx0/w, bcy0/h, bcx1/w, bcy1/h, bcx2/w, bcy2/h, bcx3/w, bcy3/h, bcx4/w, bcy4/h, bcx5/w, bcy5/h, bcx6/w, bcy6/h, bcx7/w, bcy7/h, bcx8/w, bcy8/h, det_conf, cls_max_conf, cls_max_id]
                        if (not only_objectness) and validation:
                            for c in range(num_classes):
                                tmp_conf = cls_confs[ind][c]
                                if c != cls_max_id and det_confs[ind]*tmp_conf > conf_thresh:
                                    box.append(tmp_conf)
                                    box.append(c)
                        boxes.append(box)
            if len(boxes) == 0:
                bcx0 = xs0[max_ind]
                bcy0 = ys0[max_ind]
                bcx1 = xs1[max_ind]
                bcy1 = ys1[max_ind]
                bcx2 = xs2[max_ind]
                bcy2 = ys2[max_ind]
                bcx3 = xs3[max_ind]
                bcy3 = ys3[max_ind]
                bcx4 = xs4[max_ind]
                bcy4 = ys4[max_ind]
                bcx5 = xs5[max_ind]
                bcy5 = ys5[max_ind]
                bcx6 = xs6[max_ind]
                bcy6 = ys6[max_ind]
                bcx7 = xs7[max_ind]
                bcy7 = ys7[max_ind]
                bcx8 = xs8[max_ind]
                bcy8 = ys8[max_ind]
                cls_max_conf = cls_max_confs[max_ind]
                cls_max_id = cls_max_ids[max_ind]
                det_conf =  det_confs[max_ind]
                box = [bcx0/w, bcy0/h, bcx1/w, bcy1/h, bcx2/w, bcy2/h, bcx3/w, bcy3/h, bcx4/w, bcy4/h, bcx5/w, bcy5/h, bcx6/w, bcy6/h, bcx7/w, bcy7/h, bcx8/w, bcy8/h, det_conf, cls_max_conf, cls_max_id]
                boxes.append(box)
                all_boxes.append(boxes)
            else:
                all_boxes.append(boxes)

        all_boxes.append(boxes)
    t3 = time.time()
    if False:
        print('---------------------------------')
        print('matrix computation : %f' % (t1-t0))
        print('        gpu to cpu : %f' % (t2-t1))
        print('      boxes filter : %f' % (t3-t2))
        print('---------------------------------')
    return all_boxes


def get_corresponding_region_boxes(output, conf_thresh, num_classes, anchors, num_anchors, correspondingclass, only_objectness=1, validation=False):
    
    # Parameters
    anchor_step = len(anchors)/num_anchors
    if output.dim() == 3:
        output = output.unsqueeze(0)
    batch = output.size(0)
    assert(output.size(1) == (19+num_classes)*num_anchors)
    h = output.size(2)
    w = output.size(3)

    # Activation
    t0 = time.time()
    all_boxes = []
    max_conf = -100000
    max_cls_conf = -100000
    output    = output.view(batch*num_anchors, 19+num_classes, h*w).transpose(0,1).contiguous().view(19+num_classes, batch*num_anchors*h*w)
    grid_x    = torch.linspace(0, w-1, w).repeat(h,1).repeat(batch*num_anchors, 1, 1).view(batch*num_anchors*h*w).cuda()
    grid_y    = torch.linspace(0, h-1, h).repeat(w,1).t().repeat(batch*num_anchors, 1, 1).view(batch*num_anchors*h*w).cuda()
    xs0       = torch.sigmoid(output[0]) + grid_x
    ys0       = torch.sigmoid(output[1]) + grid_y
    xs1       = output[2] + grid_x
    ys1       = output[3] + grid_y
    xs2       = output[4] + grid_x
    ys2       = output[5] + grid_y
    xs3       = output[6] + grid_x
    ys3       = output[7] + grid_y
    xs4       = output[8] + grid_x
    ys4       = output[9] + grid_y
    xs5       = output[10] + grid_x
    ys5       = output[11] + grid_y
    xs6       = output[12] + grid_x
    ys6       = output[13] + grid_y
    xs7       = output[14] + grid_x
    ys7       = output[15] + grid_y
    xs8       = output[16] + grid_x
    ys8       = output[17] + grid_y
    det_confs = torch.sigmoid(output[18])
    cls_confs = torch.nn.Softmax()(Variable(output[19:19+num_classes].transpose(0,1))).data
    cls_max_confs, cls_max_ids = torch.max(cls_confs, 1)
    cls_max_confs = cls_max_confs.view(-1)
    cls_max_ids   = cls_max_ids.view(-1)
    t1 = time.time()
    
    # GPU to CPU
    sz_hw = h*w
    sz_hwa = sz_hw*num_anchors
    det_confs = convert2cpu(det_confs)
    cls_max_confs = convert2cpu(cls_max_confs)
    cls_max_ids = convert2cpu_long(cls_max_ids)
    xs0 = convert2cpu(xs0)
    ys0 = convert2cpu(ys0)
    xs1 = convert2cpu(xs1)
    ys1 = convert2cpu(ys1)
    xs2 = convert2cpu(xs2)
    ys2 = convert2cpu(ys2)
    xs3 = convert2cpu(xs3)
    ys3 = convert2cpu(ys3)
    xs4 = convert2cpu(xs4)
    ys4 = convert2cpu(ys4)
    xs5 = convert2cpu(xs5)
    ys5 = convert2cpu(ys5)
    xs6 = convert2cpu(xs6)
    ys6 = convert2cpu(ys6)
    xs7 = convert2cpu(xs7)
    ys7 = convert2cpu(ys7)
    xs8 = convert2cpu(xs8)
    ys8 = convert2cpu(ys8)
    if validation:
        cls_confs = convert2cpu(cls_confs.view(-1, num_classes))
    t2 = time.time()

    # Boxes filter
    for b in range(batch):
        boxes = []
        max_conf = -1
        for cy in range(h):
            for cx in range(w):
                for i in range(num_anchors):
                    ind = b*sz_hwa + i*sz_hw + cy*w + cx
                    det_conf =  det_confs[ind]
                    if only_objectness:
                        conf = det_confs[ind]
                    else:
                        conf = det_confs[ind] * cls_max_confs[ind]
                    
                    if (det_confs[ind] > max_conf) and (cls_confs[ind, correspondingclass] > max_cls_conf):
                        max_conf = det_confs[ind]
                        max_cls_conf = cls_confs[ind, correspondingclass]
                        max_ind = ind                  
    
                    if conf > conf_thresh:
                        bcx0 = xs0[ind]
                        bcy0 = ys0[ind]
                        bcx1 = xs1[ind]
                        bcy1 = ys1[ind]
                        bcx2 = xs2[ind]
                        bcy2 = ys2[ind]
                        bcx3 = xs3[ind]
                        bcy3 = ys3[ind]
                        bcx4 = xs4[ind]
                        bcy4 = ys4[ind]
                        bcx5 = xs5[ind]
                        bcy5 = ys5[ind]
                        bcx6 = xs6[ind]
                        bcy6 = ys6[ind]
                        bcx7 = xs7[ind]
                        bcy7 = ys7[ind]
                        bcx8 = xs8[ind]
                        bcy8 = ys8[ind]
                        cls_max_conf = cls_max_confs[ind]
                        cls_max_id = cls_max_ids[ind]
                        box = [bcx0/w, bcy0/h, bcx1/w, bcy1/h, bcx2/w, bcy2/h, bcx3/w, bcy3/h, bcx4/w, bcy4/h, bcx5/w, bcy5/h, bcx6/w, bcy6/h, bcx7/w, bcy7/h, bcx8/w, bcy8/h, det_conf, cls_max_conf, cls_max_id]
                        if (not only_objectness) and validation:
                            for c in range(num_classes):
                                tmp_conf = cls_confs[ind][c]
                                if c != cls_max_id and det_confs[ind]*tmp_conf > conf_thresh:
                                    box.append(tmp_conf)
                                    box.append(c)
                        boxes.append(box)
        boxesnp = np.array(boxes)
        if (len(boxes) == 0) or (not (correspondingclass in boxesnp[:,20])):
            bcx0 = xs0[max_ind]
            bcy0 = ys0[max_ind]
            bcx1 = xs1[max_ind]
            bcy1 = ys1[max_ind]
            bcx2 = xs2[max_ind]
            bcy2 = ys2[max_ind]
            bcx3 = xs3[max_ind]
            bcy3 = ys3[max_ind]
            bcx4 = xs4[max_ind]
            bcy4 = ys4[max_ind]
            bcx5 = xs5[max_ind]
            bcy5 = ys5[max_ind]
            bcx6 = xs6[max_ind]
            bcy6 = ys6[max_ind]
            bcx7 = xs7[max_ind]
            bcy7 = ys7[max_ind]
            bcx8 = xs8[max_ind]
            bcy8 = ys8[max_ind]
            cls_max_conf = max_cls_conf # cls_max_confs[max_ind]
            cls_max_id = correspondingclass # cls_max_ids[max_ind]
            det_conf = max_conf # det_confs[max_ind]
            box = [bcx0/w, bcy0/h, bcx1/w, bcy1/h, bcx2/w, bcy2/h, bcx3/w, bcy3/h, bcx4/w, bcy4/h, bcx5/w, bcy5/h, bcx6/w, bcy6/h, bcx7/w, bcy7/h, bcx8/w, bcy8/h, det_conf, cls_max_conf, cls_max_id]
            boxes.append(box)
            # print(boxes)
            all_boxes.append(boxes)
        else:
            all_boxes.append(boxes)

    t3 = time.time()
    if False:
        print('---------------------------------')
        print('matrix computation : %f' % (t1-t0))
        print('        gpu to cpu : %f' % (t2-t1))
        print('      boxes filter : %f' % (t3-t2))
        print('---------------------------------')
    return all_boxes


def get_boxes(output, conf_thresh, num_classes, anchors, num_anchors, correspondingclass, only_objectness=1, validation=False):
    
    # Parameters
    anchor_step = len(anchors)/num_anchors
    if output.dim() == 3:
        output = output.unsqueeze(0)
    batch = output.size(0)
    assert(output.size(1) == (19+num_classes)*num_anchors)
    h = output.size(2)
    w = output.size(3)

    # Activation
    t0 = time.time()
    all_boxes = []
    max_conf = -100000
    max_cls_conf = -100000
    output    = output.view(batch*num_anchors, 19+num_classes, h*w).transpose(0,1).contiguous().view(19+num_classes, batch*num_anchors*h*w)
    grid_x    = torch.linspace(0, w-1, w).repeat(h,1).repeat(batch*num_anchors, 1, 1).view(batch*num_anchors*h*w).cuda()
    grid_y    = torch.linspace(0, h-1, h).repeat(w,1).t().repeat(batch*num_anchors, 1, 1).view(batch*num_anchors*h*w).cuda()
    xs0       = torch.sigmoid(output[0]) + grid_x
    ys0       = torch.sigmoid(output[1]) + grid_y
    xs1       = output[2] + grid_x
    ys1       = output[3] + grid_y
    xs2       = output[4] + grid_x
    ys2       = output[5] + grid_y
    xs3       = output[6] + grid_x
    ys3       = output[7] + grid_y
    xs4       = output[8] + grid_x
    ys4       = output[9] + grid_y
    xs5       = output[10] + grid_x
    ys5       = output[11] + grid_y
    xs6       = output[12] + grid_x
    ys6       = output[13] + grid_y
    xs7       = output[14] + grid_x
    ys7       = output[15] + grid_y
    xs8       = output[16] + grid_x
    ys8       = output[17] + grid_y
    det_confs = torch.sigmoid(output[18])
    cls_confs = torch.nn.Softmax()(Variable(output[19:19+num_classes].transpose(0,1))).data
    cls_max_confs, cls_max_ids = torch.max(cls_confs, 1)
    cls_max_confs = cls_max_confs.view(-1)
    cls_max_ids   = cls_max_ids.view(-1)
    t1 = time.time()
    
    # GPU to CPU
    sz_hw = h*w
    sz_hwa = sz_hw*num_anchors
    det_confs = convert2cpu(det_confs)
    cls_max_confs = convert2cpu(cls_max_confs)
    cls_max_ids = convert2cpu_long(cls_max_ids)
    xs0 = convert2cpu(xs0)
    ys0 = convert2cpu(ys0)
    xs1 = convert2cpu(xs1)
    ys1 = convert2cpu(ys1)
    xs2 = convert2cpu(xs2)
    ys2 = convert2cpu(ys2)
    xs3 = convert2cpu(xs3)
    ys3 = convert2cpu(ys3)
    xs4 = convert2cpu(xs4)
    ys4 = convert2cpu(ys4)
    xs5 = convert2cpu(xs5)
    ys5 = convert2cpu(ys5)
    xs6 = convert2cpu(xs6)
    ys6 = convert2cpu(ys6)
    xs7 = convert2cpu(xs7)
    ys7 = convert2cpu(ys7)
    xs8 = convert2cpu(xs8)
    ys8 = convert2cpu(ys8)
    if validation:
        cls_confs = convert2cpu(cls_confs.view(-1, num_classes))
    t2 = time.time()

    # Boxes filter
    for b in range(batch):
        boxes = []
        max_conf = -1
        for cy in range(h):
            for cx in range(w):
                for i in range(num_anchors):
                    ind = b*sz_hwa + i*sz_hw + cy*w + cx
                    det_conf =  det_confs[ind]
                    if only_objectness:
                        conf = det_confs[ind]
                    else:
                        conf = det_confs[ind] * cls_max_confs[ind]
                    
                    if (conf > max_conf) and (cls_confs[ind, correspondingclass] > max_cls_conf):
                        max_conf = conf
                        max_cls_conf = cls_confs[ind, correspondingclass]
                        max_ind = ind                  
    
                    if conf > conf_thresh:
                        bcx0 = xs0[ind]
                        bcy0 = ys0[ind]
                        bcx1 = xs1[ind]
                        bcy1 = ys1[ind]
                        bcx2 = xs2[ind]
                        bcy2 = ys2[ind]
                        bcx3 = xs3[ind]
                        bcy3 = ys3[ind]
                        bcx4 = xs4[ind]
                        bcy4 = ys4[ind]
                        bcx5 = xs5[ind]
                        bcy5 = ys5[ind]
                        bcx6 = xs6[ind]
                        bcy6 = ys6[ind]
                        bcx7 = xs7[ind]
                        bcy7 = ys7[ind]
                        bcx8 = xs8[ind]
                        bcy8 = ys8[ind]
                        cls_max_conf = cls_max_confs[ind]
                        cls_max_id = cls_max_ids[ind]
                        box = [bcx0/w, bcy0/h, bcx1/w, bcy1/h, bcx2/w, bcy2/h, bcx3/w, bcy3/h, bcx4/w, bcy4/h, bcx5/w, bcy5/h, bcx6/w, bcy6/h, bcx7/w, bcy7/h, bcx8/w, bcy8/h, det_conf, cls_max_conf, cls_max_id]
                        if (not only_objectness) and validation:
                            for c in range(num_classes):
                                tmp_conf = cls_confs[ind][c]
                                if c != cls_max_id and det_confs[ind]*tmp_conf > conf_thresh:
                                    box.append(tmp_conf)
                                    box.append(c)
                        boxes.append(box)
        boxesnp = np.array(boxes)
        if (len(boxes) == 0) or (not (correspondingclass in boxesnp[:,20])):
            bcx0 = xs0[max_ind]
            bcy0 = ys0[max_ind]
            bcx1 = xs1[max_ind]
            bcy1 = ys1[max_ind]
            bcx2 = xs2[max_ind]
            bcy2 = ys2[max_ind]
            bcx3 = xs3[max_ind]
            bcy3 = ys3[max_ind]
            bcx4 = xs4[max_ind]
            bcy4 = ys4[max_ind]
            bcx5 = xs5[max_ind]
            bcy5 = ys5[max_ind]
            bcx6 = xs6[max_ind]
            bcy6 = ys6[max_ind]
            bcx7 = xs7[max_ind]
            bcy7 = ys7[max_ind]
            bcx8 = xs8[max_ind]
            bcy8 = ys8[max_ind]
            cls_max_conf = max_cls_conf # cls_max_confs[max_ind]
            cls_max_id = correspondingclass # cls_max_ids[max_ind]
            det_conf =  det_confs[max_ind]
            box = [bcx0/w, bcy0/h, bcx1/w, bcy1/h, bcx2/w, bcy2/h, bcx3/w, bcy3/h, bcx4/w, bcy4/h, bcx5/w, bcy5/h, bcx6/w, bcy6/h, bcx7/w, bcy7/h, bcx8/w, bcy8/h, det_conf, cls_max_conf, cls_max_id]
            boxes.append(box)
            # print(boxes)
            all_boxes.append(boxes)
        else:
            all_boxes.append(boxes)

    t3 = time.time()
    if False:
        print('---------------------------------')
        print('matrix computation : %f' % (t1-t0))
        print('        gpu to cpu : %f' % (t2-t1))
        print('      boxes filter : %f' % (t3-t2))
        print('---------------------------------')
    return all_boxes


def plot_boxes_cv2(img, boxes, savename=None, class_names=None, color=None):
    import cv2
    colors = torch.FloatTensor([[1,0,1],[0,0,1],[0,1,1],[0,1,0],[1,1,0],[1,0,0]]);
    def get_color(c, x, max_val):
        ratio = float(x)/max_val * 5
        i = int(math.floor(ratio))
        j = int(math.ceil(ratio))
        ratio = ratio - i
        r = (1-ratio) * colors[i][c] + ratio*colors[j][c]
        return int(r*255)

    width = img.shape[1]
    height = img.shape[0]
    for i in range(len(boxes)):
        box = boxes[i]
        x1 = int(round((box[0] - box[2]/2.0) * width))
        y1 = int(round((box[1] - box[3]/2.0) * height))
        x2 = int(round((box[0] + box[2]/2.0) * width))
        y2 = int(round((box[1] + box[3]/2.0) * height))

        if color:
            rgb = color
        else:
            rgb = (255, 0, 0)
        if len(box) >= 7 and class_names:
            cls_conf = box[5]
            cls_id = box[6]
            print('%s: %f' % (class_names[cls_id], cls_conf))
            classes = len(class_names)
            offset = cls_id * 123457 % classes
            red   = get_color(2, offset, classes)
            green = get_color(1, offset, classes)
            blue  = get_color(0, offset, classes)
            if color is None:
                rgb = (red, green, blue)
            img = cv2.putText(img, class_names[cls_id], (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 1.2, rgb, 1)
        img = cv2.rectangle(img, (x1,y1), (x2,y2), rgb, 1)
    if savename:
        print("save plot results to %s" % savename)
        cv2.imwrite(savename, img)
    return img

def plot_boxes(img, boxes, savename=None, class_names=None):
    colors = torch.FloatTensor([[1,0,1],[0,0,1],[0,1,1],[0,1,0],[1,1,0],[1,0,0]]);
    def get_color(c, x, max_val):
        ratio = float(x)/max_val * 5
        i = int(math.floor(ratio))
        j = int(math.ceil(ratio))
        ratio = ratio - i
        r = (1-ratio) * colors[i][c] + ratio*colors[j][c]
        return int(r*255)

    width = img.width
    height = img.height
    draw = ImageDraw.Draw(img)
    for i in range(len(boxes)):
        box = boxes[i]
        x1 = (box[0] - box[2]/2.0) * width
        y1 = (box[1] - box[3]/2.0) * height
        x2 = (box[0] + box[2]/2.0) * width
        y2 = (box[1] + box[3]/2.0) * height

        rgb = (255, 0, 0)
        if len(box) >= 7 and class_names:
            cls_conf = box[5]
            cls_id = box[6]
            print('%s: %f' % (class_names[cls_id], cls_conf))
            classes = len(class_names)
            offset = cls_id * 123457 % classes
            red   = get_color(2, offset, classes)
            green = get_color(1, offset, classes)
            blue  = get_color(0, offset, classes)
            rgb = (red, green, blue)
            draw.text((x1, y1), class_names[cls_id], fill=rgb)
        draw.rectangle([x1, y1, x2, y2], outline = rgb)
    if savename:
        print("save plot results to %s" % savename)
        img.save(savename)
    return img

def read_truths(lab_path):
    if os.path.getsize(lab_path):
        truths = np.loadtxt(lab_path)
        truths = truths.reshape(truths.size/21, 21) # to avoid single truth problem
        return truths
    else:
        return np.array([])

def read_truths_args(lab_path, min_box_scale):
    truths = read_truths(lab_path)
    new_truths = []
    for i in range(truths.shape[0]):
        new_truths.append([truths[i][0], truths[i][1], truths[i][2], truths[i][3], truths[i][4], 
            truths[i][5], truths[i][6], truths[i][7], truths[i][8], truths[i][9], truths[i][10], 
            truths[i][11], truths[i][12], truths[i][13], truths[i][14], truths[i][15], truths[i][16], truths[i][17], truths[i][18]])
    return np.array(new_truths)

def read_pose(lab_path):
    if os.path.getsize(lab_path):
        truths = np.loadtxt(lab_path)
        # truths = truths.reshape(truths.size/21, 21) # to avoid single truth problem
        return truths
    else:
        return np.array([])

def load_class_names(namesfile):
    class_names = []
    with open(namesfile, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.rstrip()
        class_names.append(line)
    return class_names

def image2torch(img):
    width = img.width
    height = img.height
    img = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
    img = img.view(height, width, 3).transpose(0,1).transpose(0,2).contiguous()
    img = img.view(1, 3, height, width)
    img = img.float().div(255.0)
    return img

def do_detect(model, img, conf_thresh, nms_thresh, use_cuda=1):
    model.eval()
    t0 = time.time()

    if isinstance(img, Image.Image):
        width = img.width
        height = img.height
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
        img = img.view(height, width, 3).transpose(0,1).transpose(0,2).contiguous()
        img = img.view(1, 3, height, width)
        img = img.float().div(255.0)
    elif type(img) == np.ndarray: # cv2 image
        img = torch.from_numpy(img.transpose(2,0,1)).float().div(255.0).unsqueeze(0)
    else:
        print("unknow image type")
        exit(-1)

    t1 = time.time()

    if use_cuda:
        img = img.cuda()
    img = torch.autograd.Variable(img)
    t2 = time.time()

    output = model(img)
    output = output.data
    #for j in range(100):
    #    sys.stdout.write('%f ' % (output.storage()[j]))
    #print('')
    t3 = time.time()

    boxes = get_region_boxes(output, conf_thresh, model.num_classes, model.anchors, model.num_anchors)[0]
    #for j in range(len(boxes)):
    #    print(boxes[j])
    t4 = time.time()

    boxes = nms(boxes, nms_thresh)
    t5 = time.time()

    if False:
        print('-----------------------------------')
        print(' image to tensor : %f' % (t1 - t0))
        print('  tensor to cuda : %f' % (t2 - t1))
        print('         predict : %f' % (t3 - t2))
        print('get_region_boxes : %f' % (t4 - t3))
        print('             nms : %f' % (t5 - t4))
        print('           total : %f' % (t5 - t0))
        print('-----------------------------------')
    return boxes

def read_data_cfg(datacfg):
    options = dict()
    options['gpus'] = '0,1,2,3'
    options['num_workers'] = '10'
    with open(datacfg, 'r') as fp:
        lines = fp.readlines()

    for line in lines:
        line = line.strip()
        if line == '':
            continue
        key,value = line.split('=')
        key = key.strip()
        value = value.strip()
        options[key] = value
    return options

def scale_bboxes(bboxes, width, height):
    import copy
    dets = copy.deepcopy(bboxes)
    for i in range(len(dets)):
        dets[i][0] = dets[i][0] * width
        dets[i][1] = dets[i][1] * height
        dets[i][2] = dets[i][2] * width
        dets[i][3] = dets[i][3] * height
    return dets
      
def file_lines(thefilepath):
    count = 0
    thefile = open(thefilepath, 'rb')
    while True:
        buffer = thefile.read(8192*1024)
        if not buffer:
            break
        count += buffer.count('\n')
    thefile.close( )
    return count

def get_image_size(fname):
    '''Determine the image type of fhandle and return its size.
    from draco'''
    with open(fname, 'rb') as fhandle:
        head = fhandle.read(24)
        if len(head) != 24: 
            return
        if imghdr.what(fname) == 'png':
            check = struct.unpack('>i', head[4:8])[0]
            if check != 0x0d0a1a0a:
                return
            width, height = struct.unpack('>ii', head[16:24])
        elif imghdr.what(fname) == 'gif':
            width, height = struct.unpack('<HH', head[6:10])
        elif imghdr.what(fname) == 'jpeg' or imghdr.what(fname) == 'jpg':
            try:
                fhandle.seek(0) # Read 0xff next
                size = 2 
                ftype = 0 
                while not 0xc0 <= ftype <= 0xcf:
                    fhandle.seek(size, 1)
                    byte = fhandle.read(1)
                    while ord(byte) == 0xff:
                        byte = fhandle.read(1)
                    ftype = ord(byte)
                    size = struct.unpack('>H', fhandle.read(2))[0] - 2 
                # We are at a SOFn block
                fhandle.seek(1, 1)  # Skip `precision' byte.
                height, width = struct.unpack('>HH', fhandle.read(4))
            except Exception: #IGNORE:W0703
                return
        else:
            return
        return width, height

def logging(message):
    print('%s %s' % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), message))

def read_pose(lab_path):
    if os.path.getsize(lab_path):
        truths = np.loadtxt(lab_path)
        return truths
    else:
        return np.array([])
