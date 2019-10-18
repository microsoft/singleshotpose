import time
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils_multi import *

def build_targets(pred_corners, target, num_keypoints, anchors, num_anchors, num_classes, nH, nW, noobject_scale, object_scale, sil_thresh, seen):
    nB = target.size(0)
    nA = num_anchors
    nC = num_classes
    anchor_step = len(anchors)//num_anchors
    conf_mask   = torch.ones(nB, nA, nH, nW) * noobject_scale
    coord_mask  = torch.zeros(nB, nA, nH, nW)
    cls_mask    = torch.zeros(nB, nA, nH, nW)
    txs = list()
    tys = list()
    for i in range(num_keypoints):
        txs.append(torch.zeros(nB, nA, nH, nW))
        tys.append(torch.zeros(nB, nA, nH, nW)) 
    tconf       = torch.zeros(nB, nA, nH, nW)
    tcls        = torch.zeros(nB, nA, nH, nW) 

    num_labels = 2 * num_keypoints + 3 # +2 for width, height and +1 for class within label files
    nAnchors = nA*nH*nW
    nPixels  = nH*nW
    for b in range(nB):
        cur_pred_corners = pred_corners[b*nAnchors:(b+1)*nAnchors].t()
        cur_confs = torch.zeros(nAnchors)
        for t in range(50):
            if target[b][t*num_labels+1] == 0:
                break
            g = list()
            for i in range(num_keypoints):
                g.append(target[b][t*num_labels+2*i+1])
                g.append(target[b][t*num_labels+2*i+2])

            cur_gt_corners = torch.FloatTensor(g).repeat(nAnchors,1).t() # 18 x nAnchors
            cur_confs  = torch.max(cur_confs.view_as(conf_mask[b]), corner_confidences(cur_pred_corners, cur_gt_corners).view_as(conf_mask[b])) # some irrelevant areas are filtered, in the same grid multiple anchor boxes might exceed the threshold
        conf_mask[b][cur_confs>sil_thresh] = 0

    nGT = 0
    nCorrect = 0
    for b in range(nB):
        for t in range(50):
            if target[b][t*num_labels+1] == 0:
                break
            nGT = nGT + 1
            best_iou = 0.0
            best_n = -1
            min_dist = sys.maxsize
            gx = list()
            gy = list()
            gt_box = list()
            for i in range(num_keypoints):
                gt_box.extend([target[b][t*num_labels+2*i+1], target[b][t*num_labels+2*i+2]])
                gx.append(target[b][t*num_labels+2*i+1] * nW)
                gy.append(target[b][t*num_labels+2*i+2] * nH)
                if i == 0:
                    gi0  = int(gx[i])
                    gj0  = int(gy[i])
            pred_box = pred_corners[b*nAnchors+best_n*nPixels+gj0*nW+gi0]
            conf = corner_confidence(gt_box, pred_box) 

            # Decide which anchor to use during prediction
            gw  = target[b][t*num_labels+num_labels-2]*nW
            gh  = target[b][t*num_labels+num_labels-1]*nH
            gt_2d_box = [0, 0, gw, gh]
            for n in range(nA):
                aw = anchors[anchor_step*n]
                ah = anchors[anchor_step*n+1]
                anchor_box = [0, 0, aw, ah]
                iou  = bbox_iou(anchor_box, gt_2d_box, x1y1x2y2=False)
                if iou > best_iou:
                    best_iou = iou
                    best_n = n

            coord_mask[b][best_n][gj0][gi0] = 1
            cls_mask[b][best_n][gj0][gi0]   = 1
            conf_mask[b][best_n][gj0][gi0]  = object_scale
            # Update targets
            for i in range(num_keypoints):
                txs[i][b][best_n][gj0][gi0] = gx[i]- gi0
                tys[i][b][best_n][gj0][gi0] = gy[i]- gj0   
            tconf[b][best_n][gj0][gi0]      = conf
            tcls[b][best_n][gj0][gi0]       = target[b][t*num_labels]

            if conf > 0.5:
                nCorrect = nCorrect + 1

    return nGT, nCorrect, coord_mask, conf_mask, cls_mask, txs, tys, tconf, tcls
           
class RegionLoss(nn.Module):
    def __init__(self, num_keypoints=9, num_classes=13, anchors=[], num_anchors=5, pretrain_num_epochs=15):
        super(RegionLoss, self).__init__()
        self.num_classes = num_classes
        self.anchors = anchors
        self.num_anchors = num_anchors
        self.anchor_step = len(anchors)/num_anchors
        self.num_keypoints = num_keypoints
        self.coord_scale = 1
        self.noobject_scale = 1
        self.object_scale = 5
        self.class_scale = 1
        self.thresh = 0.6
        self.seen = 0
        self.pretrain_num_epochs = pretrain_num_epochs

    def forward(self, output, target, epoch):
        # Parameters
        t0 = time.time()
        nB = output.data.size(0)
        nA = self.num_anchors
        nC = self.num_classes
        nH = output.data.size(2)
        nW = output.data.size(3)

        # Activation
        output = output.view(nB, nA, (2*self.num_keypoints+1+nC), nH, nW)
        x = list()
        y = list()
        x.append(torch.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([0]))).view(nB, nA, nH, nW)))
        y.append(torch.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([1]))).view(nB, nA, nH, nW)))
        for i in range(1,self.num_keypoints):
            x.append(output.index_select(2, Variable(torch.cuda.LongTensor([2 * i + 0]))).view(nB, nA, nH, nW))
            y.append(output.index_select(2, Variable(torch.cuda.LongTensor([2 * i + 1]))).view(nB, nA, nH, nW))
        conf   = F.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([2*self.num_keypoints]))).view(nB, nA, nH, nW))
        cls    = output.index_select(2, Variable(torch.linspace(2*self.num_keypoints+1,2*self.num_keypoints+1+nC-1,nC).long().cuda()))
        cls    = cls.view(nB*nA, nC, nH*nW).transpose(1,2).contiguous().view(nB*nA*nH*nW, nC)
        t1     = time.time()

        # Create pred boxes
        pred_corners = torch.cuda.FloatTensor(2*self.num_keypoints, nB*nA*nH*nW)
        grid_x = torch.linspace(0, nW-1, nW).repeat(nH,1).repeat(nB*nA, 1, 1).view(nB*nA*nH*nW).cuda()
        grid_y = torch.linspace(0, nH-1, nH).repeat(nW,1).t().repeat(nB*nA, 1, 1).view(nB*nA*nH*nW).cuda()
        for i in range(self.num_keypoints):
            pred_corners[2 * i + 0]  = (x[i].data.view_as(grid_x) + grid_x) / nW
            pred_corners[2 * i + 1]  = (y[i].data.view_as(grid_y) + grid_y) / nH
        gpu_matrix = pred_corners.transpose(0,1).contiguous().view(-1,2*self.num_keypoints)
        pred_corners = convert2cpu(gpu_matrix)
        t2 = time.time()

        # Build targets
        nGT, nCorrect, coord_mask, conf_mask, cls_mask, txs, tys, tconf, tcls = \
                       build_targets(pred_corners, target.data, self.num_keypoints, self.anchors, nA, nC, nH, nW, self.noobject_scale, self.object_scale, self.thresh, self.seen)
        cls_mask   = (cls_mask == 1)
        nProposals = int((conf > 0.25).sum().data[0])
        for i in range(self.num_keypoints):
            txs[i] = Variable(txs[i].cuda())
            tys[i] = Variable(tys[i].cuda())
        tconf      = Variable(tconf.cuda())
        tcls       = Variable(tcls[cls_mask].long().cuda())
        coord_mask = Variable(coord_mask.cuda())
        conf_mask  = Variable(conf_mask.cuda().sqrt())
        cls_mask   = Variable(cls_mask.view(-1, 1).repeat(1,nC).cuda())
        cls        = cls[cls_mask].view(-1, nC)  
        t3 = time.time()

        # Create loss
        loss_xs   = list()
        loss_ys   = list()
        for i in range(self.num_keypoints):
            loss_xs.append(self.coord_scale * nn.MSELoss(size_average=False)(x[i]*coord_mask, txs[i]*coord_mask)/2.0)
            loss_ys.append(self.coord_scale * nn.MSELoss(size_average=False)(y[i]*coord_mask, tys[i]*coord_mask)/2.0)
        loss_conf = nn.MSELoss(size_average=False)(conf*conf_mask, tconf*conf_mask)/2.0
        loss_x    = np.sum(loss_xs)
        loss_y    = np.sum(loss_ys)
        loss_cls  = self.class_scale * nn.CrossEntropyLoss(size_average=False)(cls, tcls)

        if epoch > self.pretrain_num_epochs:
            loss  = loss_x + loss_y + loss_cls + loss_conf # in single object pose estimation, there is no classification loss
        else:
            # pretrain initially without confidence loss
            # once the coordinate predictions get better, start training for confidence as well
            loss  = loss_x + loss_y + loss_cls
    
        print('%d: nGT %d, recall %d, proposals %d, loss: x %f, y %f, conf %f, cls %f, total %f' % (self.seen, nGT, nCorrect, nProposals, loss_x.data[0], loss_y.data[0], loss_conf.data[0], loss_cls.data[0], loss.data[0]))
        t4 = time.time()

        if False:
            print('-----------------------------------')
            print('          activation : %f' % (t1 - t0))
            print(' create pred_corners : %f' % (t2 - t1))
            print('       build targets : %f' % (t3 - t2))
            print('         create loss : %f' % (t4 - t3))
            print('               total : %f' % (t4 - t0))

        return loss
