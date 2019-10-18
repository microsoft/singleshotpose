from __future__ import print_function
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
import os
import random
import math
import shutil
from torchvision import datasets, transforms
from torch.autograd import Variable # Useful info about autograd: http://pytorch.org/docs/master/notes/autograd.html

import dataset
from utils import *    
from cfg import parse_cfg
from region_loss import RegionLoss
from darknet import Darknet
from MeshPly import MeshPly

# Create new directory
def makedirs(path):
    if not os.path.exists( path ):
        os.makedirs( path )

# Adjust learning rate during training, learning schedule can be changed in network config file
def adjust_learning_rate(optimizer, batch):
    lr = learning_rate
    for i in range(len(steps)):
        scale = scales[i] if i < len(scales) else 1
        if batch >= steps[i]:
            lr = lr * scale
            if batch == steps[i]:
                break
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr/batch_size
    return lr

def train(epoch):

    global processed_batches
    
    # Initialize timer
    t0 = time.time()

    # Get the dataloader for training dataset
    train_loader = torch.utils.data.DataLoader(dataset.listDataset(trainlist, shape=(init_width, init_height),
                                                            shuffle=True,
                                                            transform=transforms.Compose([transforms.ToTensor(),]), 
                                                            train=True, 
                                                            seen=model.seen,
                                                            batch_size=batch_size,
                                                            num_workers=num_workers, bg_file_names=bg_file_names),
                                                batch_size=batch_size, shuffle=False, **kwargs)

    # TRAINING
    lr = adjust_learning_rate(optimizer, processed_batches)
    logging('epoch %d, processed %d samples, lr %f' % (epoch, epoch * len(train_loader.dataset), lr))
    # Start training
    model.train()
    t1 = time.time()
    avg_time = torch.zeros(9)
    niter = 0
    # Iterate through batches
    for batch_idx, (data, target) in enumerate(train_loader):
        t2 = time.time()
        # adjust learning rate
        adjust_learning_rate(optimizer, processed_batches)
        processed_batches = processed_batches + 1
        # Pass the data to GPU
        if use_cuda:
            data = data.cuda()
        t3 = time.time()
        # Wrap tensors in Variable class for automatic differentiation
        data, target = Variable(data), Variable(target)
        t4 = time.time()
        # Zero the gradients before running the backward pass
        optimizer.zero_grad()
        t5 = time.time()
        # Forward pass
        output = model(data)
        t6 = time.time()
        model.seen = model.seen + data.data.size(0)
        region_loss.seen = region_loss.seen + data.data.size(0)
        # Compute loss, grow an array of losses for saving later on
        loss = region_loss(output, target)
        training_iters.append(epoch * math.ceil(len(train_loader.dataset) / float(batch_size) ) + niter)
        training_losses.append(convert2cpu(loss.data))
        niter += 1
        t7 = time.time()
        # Backprop: compute gradient of the loss with respect to model parameters
        loss.backward()
        t8 = time.time()
        # Update weights
        optimizer.step()
        t9 = time.time()
        # Print time statistics
        if False and batch_idx > 1:
            avg_time[0] = avg_time[0] + (t2-t1)
            avg_time[1] = avg_time[1] + (t3-t2)
            avg_time[2] = avg_time[2] + (t4-t3)
            avg_time[3] = avg_time[3] + (t5-t4)
            avg_time[4] = avg_time[4] + (t6-t5)
            avg_time[5] = avg_time[5] + (t7-t6)
            avg_time[6] = avg_time[6] + (t8-t7)
            avg_time[7] = avg_time[7] + (t9-t8)
            avg_time[8] = avg_time[8] + (t9-t1)
            print('-------------------------------')
            print('       load data : %f' % (avg_time[0]/(batch_idx)))
            print('     cpu to cuda : %f' % (avg_time[1]/(batch_idx)))
            print('cuda to variable : %f' % (avg_time[2]/(batch_idx)))
            print('       zero_grad : %f' % (avg_time[3]/(batch_idx)))
            print(' forward feature : %f' % (avg_time[4]/(batch_idx)))
            print('    forward loss : %f' % (avg_time[5]/(batch_idx)))
            print('        backward : %f' % (avg_time[6]/(batch_idx)))
            print('            step : %f' % (avg_time[7]/(batch_idx)))
            print('           total : %f' % (avg_time[8]/(batch_idx)))
        t1 = time.time()
    t1 = time.time()
    return epoch * math.ceil(len(train_loader.dataset) / float(batch_size) ) + niter - 1 

def test(epoch, niter):
    def truths_length(truths):
        for i in range(50):
            if truths[i][1] == 0:
                return i

    # Set the module in evaluation mode (turn off dropout, batch normalization etc.)        
    model.eval()

    # Parameters
    num_classes          = model.num_classes
    anchors              = model.anchors
    num_anchors          = model.num_anchors
    testtime             = True
    testing_error_trans  = 0.0
    testing_error_angle  = 0.0
    testing_error_pixel  = 0.0
    testing_samples      = 0.0
    errs_2d              = []
    errs_3d              = []
    errs_trans           = []
    errs_angle           = []
    errs_corner2D        = []

    logging("   Testing...")
    logging("   Number of test samples: %d" % len(test_loader.dataset))
    notpredicted = 0
    # Iterate through test examples 
    for batch_idx, (data, target) in enumerate(test_loader):
        t1 = time.time()
        # Pass the data to GPU
        if use_cuda:
            data = data.cuda()
            target = target.cuda()
        # Wrap tensors in Variable class, set volatile=True for inference mode and to use minimal memory during inference
        data = Variable(data, volatile=True)
        t2 = time.time()
        # Formward pass
        output = model(data).data  
        t3 = time.time()
        # Using confidence threshold, eliminate low-confidence predictions
        all_boxes = get_region_boxes(output, conf_thresh, num_classes, anchors, num_anchors)        
        t4 = time.time()
        # Iterate through all batch elements
        for i in range(output.size(0)):
            # For each image, get all the predictions
            boxes   = all_boxes[i]
            # For each image, get all the targets (for multiple object pose estimation, there might be more than 1 target per image)
            truths  = target[i].view(-1, 21)
            # Get how many object are present in the scene
            num_gts = truths_length(truths)

            # Iterate through each ground-truth object
            for k in range(num_gts):
                box_gt        = [truths[k][1], truths[k][2], truths[k][3], truths[k][4], truths[k][5], truths[k][6], 
                                truths[k][7], truths[k][8], truths[k][9], truths[k][10], truths[k][11], truths[k][12], 
                                truths[k][13], truths[k][14], truths[k][15], truths[k][16], truths[k][17], truths[k][18], 1.0, 1.0, truths[k][0]]
                best_conf_est = -1

                # If the prediction has the highest confidence, choose it as our prediction
                for j in range(len(boxes)):
                    if boxes[j][18] > best_conf_est:
                        best_conf_est = boxes[j][18]
                        box_pr        = boxes[j]
                        match         = corner_confidence9(box_gt[:18], torch.FloatTensor(boxes[j][:18]))

                # Denormalize the corner predictions 
                corners2D_gt = np.array(np.reshape(box_gt[:18], [9, 2]), dtype='float32')
                corners2D_pr = np.array(np.reshape(box_pr[:18], [9, 2]), dtype='float32')
                corners2D_gt[:, 0] = corners2D_gt[:, 0] * im_width
                corners2D_gt[:, 1] = corners2D_gt[:, 1] * im_height               
                corners2D_pr[:, 0] = corners2D_pr[:, 0] * im_width
                corners2D_pr[:, 1] = corners2D_pr[:, 1] * im_height

                # Compute corner prediction error
                corner_norm = np.linalg.norm(corners2D_gt - corners2D_pr, axis=1)
                corner_dist = np.mean(corner_norm)
                errs_corner2D.append(corner_dist)

                # Compute [R|t] by pnp
                R_gt, t_gt = pnp(np.array(np.transpose(np.concatenate((np.zeros((3, 1)), corners3D[:3, :]), axis=1)), dtype='float32'),  corners2D_gt, np.array(internal_calibration, dtype='float32'))
                R_pr, t_pr = pnp(np.array(np.transpose(np.concatenate((np.zeros((3, 1)), corners3D[:3, :]), axis=1)), dtype='float32'),  corners2D_pr, np.array(internal_calibration, dtype='float32'))

                # Compute errors

                # Compute translation error
                trans_dist   = np.sqrt(np.sum(np.square(t_gt - t_pr)))
                errs_trans.append(trans_dist)

                # Compute angle error
                angle_dist   = calcAngularDistance(R_gt, R_pr)
                errs_angle.append(angle_dist)

                # Compute pixel error
                Rt_gt        = np.concatenate((R_gt, t_gt), axis=1)
                Rt_pr        = np.concatenate((R_pr, t_pr), axis=1)
                proj_2d_gt   = compute_projection(vertices, Rt_gt, internal_calibration) 
                proj_2d_pred = compute_projection(vertices, Rt_pr, internal_calibration) 
                norm         = np.linalg.norm(proj_2d_gt - proj_2d_pred, axis=0)
                pixel_dist   = np.mean(norm)
                errs_2d.append(pixel_dist)

                # Compute 3D distances
                transform_3d_gt   = compute_transformation(vertices, Rt_gt) 
                transform_3d_pred = compute_transformation(vertices, Rt_pr)  
                norm3d            = np.linalg.norm(transform_3d_gt - transform_3d_pred, axis=0)
                vertex_dist       = np.mean(norm3d)    
                errs_3d.append(vertex_dist)  

                # Sum errors
                testing_error_trans  += trans_dist
                testing_error_angle  += angle_dist
                testing_error_pixel  += pixel_dist
                testing_samples      += 1

        t5 = time.time()

    # Compute 2D projection, 6D pose and 5cm5degree scores
    px_threshold = 5
    acc = len(np.where(np.array(errs_2d) <= px_threshold)[0]) * 100. / (len(errs_2d)+eps)
    acc3d = len(np.where(np.array(errs_3d) <= vx_threshold)[0]) * 100. / (len(errs_3d)+eps)
    acc5cm5deg = len(np.where((np.array(errs_trans) <= 0.05) & (np.array(errs_angle) <= 5))[0]) * 100. / (len(errs_trans)+eps)
    corner_acc = len(np.where(np.array(errs_corner2D) <= px_threshold)[0]) * 100. / (len(errs_corner2D)+eps)
    mean_err_2d = np.mean(errs_2d)
    mean_corner_err_2d = np.mean(errs_corner2D)
    nts = float(testing_samples)
    
    if testtime:
        print('-----------------------------------')
        print('  tensor to cuda : %f' % (t2 - t1))
        print('         predict : %f' % (t3 - t2))
        print('get_region_boxes : %f' % (t4 - t3))
        print('            eval : %f' % (t5 - t4))
        print('           total : %f' % (t5 - t1))
        print('-----------------------------------')

    # Print test statistics
    logging("   Mean corner error is %f" % (mean_corner_err_2d))
    logging('   Acc using {} px 2D Projection = {:.2f}%'.format(px_threshold, acc))
    logging('   Acc using {} vx 3D Transformation = {:.2f}%'.format(vx_threshold, acc3d))
    logging('   Acc using 5 cm 5 degree metric = {:.2f}%'.format(acc5cm5deg))
    logging('   Translation error: %f, angle error: %f' % (testing_error_trans/(nts+eps), testing_error_angle/(nts+eps)) )

    # Register losses and errors for saving later on
    testing_iters.append(niter)
    testing_errors_trans.append(testing_error_trans/(nts+eps))
    testing_errors_angle.append(testing_error_angle/(nts+eps))
    testing_errors_pixel.append(testing_error_pixel/(nts+eps))
    testing_accuracies.append(acc)

if __name__ == "__main__":

    # Training settings
    datacfg       = sys.argv[1]
    cfgfile       = sys.argv[2]
    weightfile    = sys.argv[3]

    # Parse configuration files
    data_options  = read_data_cfg(datacfg)
    net_options   = parse_cfg(cfgfile)[0]
    trainlist     = data_options['train']
    testlist      = data_options['valid']
    nsamples      = file_lines(trainlist)
    gpus          = data_options['gpus']  # e.g. 0,1,2,3
    gpus = '0'
    meshname      = data_options['mesh']
    num_workers   = int(data_options['num_workers'])
    backupdir     = data_options['backup']
    diam          = float(data_options['diam'])
    vx_threshold  = diam * 0.1
    if not os.path.exists(backupdir):
        makedirs(backupdir)
    batch_size    = int(net_options['batch'])
    max_batches   = int(net_options['max_batches'])
    learning_rate = float(net_options['learning_rate'])
    momentum      = float(net_options['momentum'])
    decay         = float(net_options['decay'])
    steps         = [float(step) for step in net_options['steps'].split(',')]
    scales        = [float(scale) for scale in net_options['scales'].split(',')]
    bg_file_names = get_all_files('VOCdevkit/VOC2012/JPEGImages')

    # Train parameters
    max_epochs    = 700 # max_batches*batch_size/nsamples+1
    use_cuda      = True
    seed          = int(time.time())
    eps           = 1e-5
    save_interval = 10 # epoches
    dot_interval  = 70 # batches
    best_acc      = -1 

    # Test parameters
    conf_thresh   = 0.1
    nms_thresh    = 0.4
    iou_thresh    = 0.5
    im_width      = 640
    im_height     = 480 

    # Specify which gpus to use
    torch.manual_seed(seed)
    if use_cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpus
        torch.cuda.manual_seed(seed)

    # Specifiy the model and the loss
    model       = Darknet(cfgfile)
    region_loss = model.loss

    # Model settings
    # model.load_weights(weightfile)
    model.load_weights_until_last(weightfile) 
    model.print_network()
    model.seen = 0
    region_loss.iter  = model.iter
    region_loss.seen  = model.seen
    processed_batches = model.seen/batch_size
    init_width        = model.width
    init_height       = model.height
    test_width        = 672
    test_height       = 672
    init_epoch        = model.seen/nsamples 

    # Variable to save
    training_iters          = []
    training_losses         = []
    testing_iters           = []
    testing_losses          = []
    testing_errors_trans    = []
    testing_errors_angle    = []
    testing_errors_pixel    = []
    testing_accuracies      = []

    # Get the intrinsic camerea matrix, mesh, vertices and corners of the model
    mesh                 = MeshPly(meshname)
    vertices             = np.c_[np.array(mesh.vertices), np.ones((len(mesh.vertices), 1))].transpose()
    corners3D            = get_3D_corners(vertices)
    internal_calibration = get_camera_intrinsic()

    print("vertex unit")
    print(np.max(vertices))
    print(np.min(vertices))

    # Specify the number of workers
    kwargs = {'num_workers': num_workers, 'pin_memory': True} if use_cuda else {}

    # Get the dataloader for test data
    test_loader = torch.utils.data.DataLoader(dataset.listDataset(testlist, shape=(test_width, test_height),
                                                                           shuffle=False,
                                                                           transform=transforms.Compose([transforms.ToTensor(),]), 
                                                                           train=False),
                                             batch_size=1, shuffle=False, **kwargs)

    # Pass the model to GPU
    if use_cuda:
        model = model.cuda() # model = torch.nn.DataParallel(model, device_ids=[0]).cuda() # Multiple GPU parallelism

    # Get the optimizer
    params_dict = dict(model.named_parameters())
    params = []
    for key, value in params_dict.items():
        if key.find('.bn') >= 0 or key.find('.bias') >= 0:
            params += [{'params': [value], 'weight_decay': 0.0}]
        else:
            params += [{'params': [value], 'weight_decay': decay*batch_size}]
    optimizer = optim.SGD(model.parameters(), lr=learning_rate/batch_size, momentum=momentum, dampening=0, weight_decay=decay*batch_size)
    # optimizer = optim.Adam(model.parameters(), lr=0.001) # Adam optimization

    evaluate = False
    if evaluate:
        logging('evaluating ...')
        test(0, 0)
    else:
        for epoch in range(init_epoch, max_epochs): 
            # TRAIN
            niter = train(epoch)
            # TEST and SAVE
            if (epoch % 10 == 0) and (epoch is not 0): 
                test(epoch, niter)
                logging('save training stats to %s/costs.npz' % (backupdir))
                np.savez(os.path.join(backupdir, "costs.npz"),
                    training_iters=training_iters,
                    training_losses=training_losses,
                    testing_iters=testing_iters,
                    testing_accuracies=testing_accuracies,
                    testing_errors_pixel=testing_errors_pixel,
                    testing_errors_angle=testing_errors_angle) 
                if (testing_accuracies[-1] > best_acc ):
                    best_acc = testing_accuracies[-1]
                    logging('best model so far!')
                    logging('save weights to %s/model.weights' % (backupdir))
                    model.save_weights('%s/model.weights' % (backupdir))
        shutil.copy2('%s/model.weights' % (backupdir), '%s/model_backup.weights' % (backupdir))
