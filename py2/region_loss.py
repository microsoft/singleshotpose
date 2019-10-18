import time
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import *

def build_targets(pred_corners, target, anchors, num_anchors, num_classes, nH, nW, noobject_scale, object_scale, sil_thresh, seen):
    nB = target.size(0)
    nA = num_anchors
    nC = num_classes
    anchor_step = len(anchors)/num_anchors
    conf_mask   = torch.ones(nB, nA, nH, nW) * noobject_scale
    coord_mask  = torch.zeros(nB, nA, nH, nW)
    cls_mask    = torch.zeros(nB, nA, nH, nW)
    tx0         = torch.zeros(nB, nA, nH, nW) 
    ty0         = torch.zeros(nB, nA, nH, nW) 
    tx1         = torch.zeros(nB, nA, nH, nW) 
    ty1         = torch.zeros(nB, nA, nH, nW) 
    tx2         = torch.zeros(nB, nA, nH, nW) 
    ty2         = torch.zeros(nB, nA, nH, nW) 
    tx3         = torch.zeros(nB, nA, nH, nW) 
    ty3         = torch.zeros(nB, nA, nH, nW) 
    tx4         = torch.zeros(nB, nA, nH, nW) 
    ty4         = torch.zeros(nB, nA, nH, nW) 
    tx5         = torch.zeros(nB, nA, nH, nW) 
    ty5         = torch.zeros(nB, nA, nH, nW) 
    tx6         = torch.zeros(nB, nA, nH, nW) 
    ty6         = torch.zeros(nB, nA, nH, nW) 
    tx7         = torch.zeros(nB, nA, nH, nW) 
    ty7         = torch.zeros(nB, nA, nH, nW) 
    tx8         = torch.zeros(nB, nA, nH, nW) 
    ty8         = torch.zeros(nB, nA, nH, nW) 
    tconf       = torch.zeros(nB, nA, nH, nW)
    tcls        = torch.zeros(nB, nA, nH, nW) 

    nAnchors = nA*nH*nW
    nPixels  = nH*nW
    for b in range(nB):
        cur_pred_corners = pred_corners[b*nAnchors:(b+1)*nAnchors].t()
        cur_confs = torch.zeros(nAnchors)
        for t in range(50):
            if target[b][t*21+1] == 0:
                break
            gx0 = target[b][t*21+1]*nW
            gy0 = target[b][t*21+2]*nH
            gx1 = target[b][t*21+3]*nW
            gy1 = target[b][t*21+4]*nH
            gx2 = target[b][t*21+5]*nW
            gy2 = target[b][t*21+6]*nH
            gx3 = target[b][t*21+7]*nW
            gy3 = target[b][t*21+8]*nH
            gx4 = target[b][t*21+9]*nW
            gy4 = target[b][t*21+10]*nH
            gx5 = target[b][t*21+11]*nW
            gy5 = target[b][t*21+12]*nH
            gx6 = target[b][t*21+13]*nW
            gy6 = target[b][t*21+14]*nH
            gx7 = target[b][t*21+15]*nW
            gy7 = target[b][t*21+16]*nH
            gx8 = target[b][t*21+17]*nW
            gy8 = target[b][t*21+18]*nH

            cur_gt_corners = torch.FloatTensor([gx0/nW,gy0/nH,gx1/nW,gy1/nH,gx2/nW,gy2/nH,gx3/nW,gy3/nH,gx4/nW,gy4/nH,gx5/nW,gy5/nH,gx6/nW,gy6/nH,gx7/nW,gy7/nH,gx8/nW,gy8/nH]).repeat(nAnchors,1).t() # 16 x nAnchors
            cur_confs  = torch.max(cur_confs, corner_confidences9(cur_pred_corners, cur_gt_corners)) # some irrelevant areas are filtered, in the same grid multiple anchor boxes might exceed the threshold
        conf_mask[b][cur_confs>sil_thresh] = 0
    if seen < -1:#6400:
       tx0.fill_(0.5)
       ty0.fill_(0.5)
       tx1.fill_(0.5)
       ty1.fill_(0.5)
       tx2.fill_(0.5)
       ty2.fill_(0.5)
       tx3.fill_(0.5)
       ty3.fill_(0.5)
       tx4.fill_(0.5)
       ty4.fill_(0.5)
       tx5.fill_(0.5)
       ty5.fill_(0.5)
       tx6.fill_(0.5)
       ty6.fill_(0.5)
       tx7.fill_(0.5)
       ty7.fill_(0.5)
       tx8.fill_(0.5)
       ty8.fill_(0.5)
       coord_mask.fill_(1)

    nGT = 0
    nCorrect = 0
    for b in range(nB):
        for t in range(50):
            if target[b][t*21+1] == 0:
                break
            nGT = nGT + 1
            best_iou = 0.0
            best_n = -1
            min_dist = 10000
            gx0 = target[b][t*21+1] * nW
            gy0 = target[b][t*21+2] * nH
            gi0 = int(gx0)
            gj0 = int(gy0)
            gx1 = target[b][t*21+3] * nW
            gy1 = target[b][t*21+4] * nH
            gx2 = target[b][t*21+5] * nW
            gy2 = target[b][t*21+6] * nH
            gx3 = target[b][t*21+7] * nW
            gy3 = target[b][t*21+8] * nH
            gx4 = target[b][t*21+9] * nW
            gy4 = target[b][t*21+10] * nH
            gx5 = target[b][t*21+11] * nW
            gy5 = target[b][t*21+12] * nH
            gx6 = target[b][t*21+13] * nW
            gy6 = target[b][t*21+14] * nH
            gx7 = target[b][t*21+15] * nW
            gy7 = target[b][t*21+16] * nH
            gx8 = target[b][t*21+17] * nW
            gy8 = target[b][t*21+18] * nH

            best_n = 0 # 1 anchor box
            gt_box = [gx0/nW,gy0/nH,gx1/nW,gy1/nH,gx2/nW,gy2/nH,gx3/nW,gy3/nH,gx4/nW,gy4/nH,gx5/nW,gy5/nH,gx6/nW,gy6/nH,gx7/nW,gy7/nH,gx8/nW,gy8/nH]
            pred_box = pred_corners[b*nAnchors+best_n*nPixels+gj0*nW+gi0]
            conf = corner_confidence9(gt_box, pred_box) 
            coord_mask[b][best_n][gj0][gi0] = 1
            cls_mask[b][best_n][gj0][gi0]   = 1
            conf_mask[b][best_n][gj0][gi0]  = object_scale
            tx0[b][best_n][gj0][gi0]        = target[b][t*21+1] * nW - gi0
            ty0[b][best_n][gj0][gi0]        = target[b][t*21+2] * nH - gj0
            tx1[b][best_n][gj0][gi0]        = target[b][t*21+3] * nW - gi0
            ty1[b][best_n][gj0][gi0]        = target[b][t*21+4] * nH - gj0
            tx2[b][best_n][gj0][gi0]        = target[b][t*21+5] * nW - gi0
            ty2[b][best_n][gj0][gi0]        = target[b][t*21+6] * nH - gj0
            tx3[b][best_n][gj0][gi0]        = target[b][t*21+7] * nW - gi0
            ty3[b][best_n][gj0][gi0]        = target[b][t*21+8] * nH - gj0
            tx4[b][best_n][gj0][gi0]        = target[b][t*21+9] * nW - gi0
            ty4[b][best_n][gj0][gi0]        = target[b][t*21+10] * nH - gj0
            tx5[b][best_n][gj0][gi0]        = target[b][t*21+11] * nW - gi0
            ty5[b][best_n][gj0][gi0]        = target[b][t*21+12] * nH - gj0
            tx6[b][best_n][gj0][gi0]        = target[b][t*21+13] * nW - gi0
            ty6[b][best_n][gj0][gi0]        = target[b][t*21+14] * nH - gj0
            tx7[b][best_n][gj0][gi0]        = target[b][t*21+15] * nW - gi0
            ty7[b][best_n][gj0][gi0]        = target[b][t*21+16] * nH - gj0
            tx8[b][best_n][gj0][gi0]        = target[b][t*21+17] * nW - gi0
            ty8[b][best_n][gj0][gi0]        = target[b][t*21+18] * nH - gj0
            tconf[b][best_n][gj0][gi0]      = conf
            tcls[b][best_n][gj0][gi0]       = target[b][t*21]

            if conf > 0.5:
                nCorrect = nCorrect + 1

    return nGT, nCorrect, coord_mask, conf_mask, cls_mask, tx0, tx1, tx2, tx3, tx4, tx5, tx6, tx7, tx8, ty0, ty1, ty2, ty3, ty4, ty5, ty6, ty7, ty8, tconf, tcls
           
class RegionLoss(nn.Module):
    def __init__(self, num_classes=0, anchors=[], num_anchors=1):
        super(RegionLoss, self).__init__()
        self.num_classes = num_classes
        self.anchors = anchors
        self.num_anchors = num_anchors
        self.anchor_step = len(anchors)/num_anchors
        self.coord_scale = 1
        self.noobject_scale = 1
        self.object_scale = 5
        self.class_scale = 1
        self.thresh = 0.6
        self.seen = 0

    def forward(self, output, target):
        # Parameters
        t0 = time.time()
        nB = output.data.size(0)
        nA = self.num_anchors
        nC = self.num_classes
        nH = output.data.size(2)
        nW = output.data.size(3)

        # Activation
        output = output.view(nB, nA, (19+nC), nH, nW)
        x0     = F.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([0]))).view(nB, nA, nH, nW))
        y0     = F.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([1]))).view(nB, nA, nH, nW))
        x1     = output.index_select(2, Variable(torch.cuda.LongTensor([2]))).view(nB, nA, nH, nW)
        y1     = output.index_select(2, Variable(torch.cuda.LongTensor([3]))).view(nB, nA, nH, nW)
        x2     = output.index_select(2, Variable(torch.cuda.LongTensor([4]))).view(nB, nA, nH, nW)
        y2     = output.index_select(2, Variable(torch.cuda.LongTensor([5]))).view(nB, nA, nH, nW)
        x3     = output.index_select(2, Variable(torch.cuda.LongTensor([6]))).view(nB, nA, nH, nW)
        y3     = output.index_select(2, Variable(torch.cuda.LongTensor([7]))).view(nB, nA, nH, nW)
        x4     = output.index_select(2, Variable(torch.cuda.LongTensor([8]))).view(nB, nA, nH, nW)
        y4     = output.index_select(2, Variable(torch.cuda.LongTensor([9]))).view(nB, nA, nH, nW)
        x5     = output.index_select(2, Variable(torch.cuda.LongTensor([10]))).view(nB, nA, nH, nW)
        y5     = output.index_select(2, Variable(torch.cuda.LongTensor([11]))).view(nB, nA, nH, nW)
        x6     = output.index_select(2, Variable(torch.cuda.LongTensor([12]))).view(nB, nA, nH, nW)
        y6     = output.index_select(2, Variable(torch.cuda.LongTensor([13]))).view(nB, nA, nH, nW)
        x7     = output.index_select(2, Variable(torch.cuda.LongTensor([14]))).view(nB, nA, nH, nW)
        y7     = output.index_select(2, Variable(torch.cuda.LongTensor([15]))).view(nB, nA, nH, nW)
        x8     = output.index_select(2, Variable(torch.cuda.LongTensor([16]))).view(nB, nA, nH, nW)
        y8     = output.index_select(2, Variable(torch.cuda.LongTensor([17]))).view(nB, nA, nH, nW)
        conf   = F.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([18]))).view(nB, nA, nH, nW))
        cls    = output.index_select(2, Variable(torch.linspace(19,19+nC-1,nC).long().cuda()))
        cls    = cls.view(nB*nA, nC, nH*nW).transpose(1,2).contiguous().view(nB*nA*nH*nW, nC)
        t1     = time.time()

        # Create pred boxes
        pred_corners = torch.cuda.FloatTensor(18, nB*nA*nH*nW)
        grid_x = torch.linspace(0, nW-1, nW).repeat(nH,1).repeat(nB*nA, 1, 1).view(nB*nA*nH*nW).cuda()
        grid_y = torch.linspace(0, nH-1, nH).repeat(nW,1).t().repeat(nB*nA, 1, 1).view(nB*nA*nH*nW).cuda()
        pred_corners[0]  = (x0.data + grid_x) / nW
        pred_corners[1]  = (y0.data + grid_y) / nH
        pred_corners[2]  = (x1.data + grid_x) / nW
        pred_corners[3]  = (y1.data + grid_y) / nH
        pred_corners[4]  = (x2.data + grid_x) / nW
        pred_corners[5]  = (y2.data + grid_y) / nH
        pred_corners[6]  = (x3.data + grid_x) / nW
        pred_corners[7]  = (y3.data + grid_y) / nH
        pred_corners[8]  = (x4.data + grid_x) / nW
        pred_corners[9]  = (y4.data + grid_y) / nH
        pred_corners[10]  = (x5.data + grid_x) / nW
        pred_corners[11]  = (y5.data + grid_y) / nH
        pred_corners[12] = (x6.data + grid_x) / nW
        pred_corners[13] = (y6.data + grid_y) / nH
        pred_corners[14] = (x7.data + grid_x) / nW
        pred_corners[15] = (y7.data + grid_y) / nH
        pred_corners[16] = (x8.data + grid_x) / nW
        pred_corners[17] = (y8.data + grid_y) / nH
        gpu_matrix = pred_corners.transpose(0,1).contiguous().view(-1,18)
        pred_corners = convert2cpu(gpu_matrix)
        t2 = time.time()

        # Build targets
        nGT, nCorrect, coord_mask, conf_mask, cls_mask, tx0, tx1, tx2, tx3, tx4, tx5, tx6, tx7, tx8, ty0, ty1, ty2, ty3, ty4, ty5, ty6, ty7, ty8, tconf, tcls = \
                       build_targets(pred_corners, target.data, self.anchors, nA, nC, nH, nW, self.noobject_scale, self.object_scale, self.thresh, self.seen)
        cls_mask   = (cls_mask == 1)
        nProposals = int((conf > 0.25).sum().data[0])
        tx0        = Variable(tx0.cuda())
        ty0        = Variable(ty0.cuda())
        tx1        = Variable(tx1.cuda())
        ty1        = Variable(ty1.cuda())
        tx2        = Variable(tx2.cuda())
        ty2        = Variable(ty2.cuda())
        tx3        = Variable(tx3.cuda())
        ty3        = Variable(ty3.cuda())
        tx4        = Variable(tx4.cuda())
        ty4        = Variable(ty4.cuda())
        tx5        = Variable(tx5.cuda())
        ty5        = Variable(ty5.cuda())
        tx6        = Variable(tx6.cuda())
        ty6        = Variable(ty6.cuda())
        tx7        = Variable(tx7.cuda())
        ty7        = Variable(ty7.cuda())
        tx8        = Variable(tx8.cuda())
        ty8        = Variable(ty8.cuda())
        tconf      = Variable(tconf.cuda())
        tcls       = Variable(tcls.view(-1)[cls_mask].long().cuda())
        coord_mask = Variable(coord_mask.cuda())
        conf_mask  = Variable(conf_mask.cuda().sqrt())
        cls_mask   = Variable(cls_mask.view(-1, 1).repeat(1,nC).cuda())
        cls        = cls[cls_mask].view(-1, nC)  
        t3 = time.time()

        # Create loss
        loss_x0    = self.coord_scale * nn.MSELoss(size_average=False)(x0*coord_mask, tx0*coord_mask)/2.0
        loss_y0    = self.coord_scale * nn.MSELoss(size_average=False)(y0*coord_mask, ty0*coord_mask)/2.0
        loss_x1    = self.coord_scale * nn.MSELoss(size_average=False)(x1*coord_mask, tx1*coord_mask)/2.0
        loss_y1    = self.coord_scale * nn.MSELoss(size_average=False)(y1*coord_mask, ty1*coord_mask)/2.0
        loss_x2    = self.coord_scale * nn.MSELoss(size_average=False)(x2*coord_mask, tx2*coord_mask)/2.0
        loss_y2    = self.coord_scale * nn.MSELoss(size_average=False)(y2*coord_mask, ty2*coord_mask)/2.0
        loss_x3    = self.coord_scale * nn.MSELoss(size_average=False)(x3*coord_mask, tx3*coord_mask)/2.0
        loss_y3    = self.coord_scale * nn.MSELoss(size_average=False)(y3*coord_mask, ty3*coord_mask)/2.0
        loss_x4    = self.coord_scale * nn.MSELoss(size_average=False)(x4*coord_mask, tx4*coord_mask)/2.0
        loss_y4    = self.coord_scale * nn.MSELoss(size_average=False)(y4*coord_mask, ty4*coord_mask)/2.0
        loss_x5    = self.coord_scale * nn.MSELoss(size_average=False)(x5*coord_mask, tx5*coord_mask)/2.0
        loss_y5    = self.coord_scale * nn.MSELoss(size_average=False)(y5*coord_mask, ty5*coord_mask)/2.0
        loss_x6    = self.coord_scale * nn.MSELoss(size_average=False)(x6*coord_mask, tx6*coord_mask)/2.0
        loss_y6    = self.coord_scale * nn.MSELoss(size_average=False)(y6*coord_mask, ty6*coord_mask)/2.0
        loss_x7    = self.coord_scale * nn.MSELoss(size_average=False)(x7*coord_mask, tx7*coord_mask)/2.0
        loss_y7    = self.coord_scale * nn.MSELoss(size_average=False)(y7*coord_mask, ty7*coord_mask)/2.0
        loss_x8    = self.coord_scale * nn.MSELoss(size_average=False)(x8*coord_mask, tx8*coord_mask)/2.0
        loss_y8    = self.coord_scale * nn.MSELoss(size_average=False)(y8*coord_mask, ty8*coord_mask)/2.0
        loss_conf  = nn.MSELoss(size_average=False)(conf*conf_mask, tconf*conf_mask)/2.0
        # loss_cls   = self.class_scale * nn.CrossEntropyLoss(size_average=False)(cls, tcls)
        loss_cls = 0
        loss_x     = loss_x0 + loss_x1 + loss_x2 + loss_x3 + loss_x4 + loss_x5 + loss_x6 + loss_x7 + loss_x8 
        loss_y     = loss_y0 + loss_y1 + loss_y2 + loss_y3 + loss_y4 + loss_y5 + loss_y6 + loss_y7 + loss_y8 
        if False:
            loss   = loss_x + loss_y + loss_conf + loss_cls
        else:
            loss   = loss_x + loss_y + loss_conf
        t4 = time.time()

        if False:
            print('-----------------------------------')
            print('        activation : %f' % (t1 - t0))
            print(' create pred_corners : %f' % (t2 - t1))
            print('     build targets : %f' % (t3 - t2))
            print('       create loss : %f' % (t4 - t3))
            print('             total : %f' % (t4 - t0))

        if False:
            print('%d: nGT %d, recall %d, proposals %d, loss: x %f, y %f, conf %f, cls %f, total %f' % (self.seen, nGT, nCorrect, nProposals, loss_x.data[0], loss_y.data[0], loss_conf.data[0], loss_cls.data[0], loss.data[0]))
        else:
            print('%d: nGT %d, recall %d, proposals %d, loss: x %f, y %f, conf %f, total %f' % (self.seen, nGT, nCorrect, nProposals, loss_x.data[0], loss_y.data[0], loss_conf.data[0], loss.data[0]))
        
        return loss
