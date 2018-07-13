import os
os.sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch.autograd import Variable
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import scipy.misc
import warnings
warnings.filterwarnings("ignore")

from darknet_multi import Darknet
from utils import *
import dataset_multi
from MeshPly import MeshPly

def valid(datacfg, cfgfile, weightfile, conf_th):
    def truths_length(truths):
        for i in range(50):
            if truths[i][1] == 0:
                return i

    # Parse configuration files
    options       = read_data_cfg(datacfg)
    valid_images  = options['valid']
    meshname      = options['mesh']
    name          = options['name']
    prefix        = 'results'
    # Read object model information, get 3D bounding box corners
    mesh          = MeshPly(meshname)
    vertices      = np.c_[np.array(mesh.vertices), np.ones((len(mesh.vertices), 1))].transpose()
    corners3D     = get_3D_corners(vertices)
    diam          = float(options['diam'])

    # Read intrinsic camera parameters
    internal_calibration = get_camera_intrinsic()

    # Get validation file names
    with open(valid_images) as fp:
        tmp_files = fp.readlines()
        valid_files = [item.rstrip() for item in tmp_files]
    
    # Specicy model, load pretrained weights, pass to GPU and set the module in evaluation mode
    model = Darknet(cfgfile)
    model.load_weights(weightfile)
    model.cuda()
    model.eval()

    # Get the parser for the test dataset
    valid_dataset = dataset_multi.listDataset(valid_images, shape=(model.width, model.height),
                       shuffle=False,
                       objclass=name,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                       ]))
    valid_batchsize = 1

    # Specify the number of workers for multiple processing, get the dataloader for the test dataset
    kwargs = {'num_workers': 4, 'pin_memory': True}
    test_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=valid_batchsize, shuffle=False, **kwargs) 

    # Parameters
    use_cuda        = True
    num_classes     = 13
    anchors         = [1.4820, 2.2412, 2.0501, 3.1265, 2.3946, 4.6891, 3.1018, 3.9910, 3.4879, 5.8851] 
    num_anchors     = 5
    eps             = 1e-5
    conf_thresh     = conf_th
    iou_thresh      = 0.5

    # Parameters to save
    errs_2d             = []
    edges = [[1, 2], [1, 3], [1, 5], [2, 4], [2, 6], [3, 4], [3, 7], [4, 8], [5, 6], [5, 7], [6, 8], [7, 8]]
    edges_corners = [[0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3], [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]]

    # Iterate through test batches (Batch size for test data is 1)
    logging('Testing {}...'.format(name))
    for batch_idx, (data, target) in enumerate(test_loader):
        
        t1 = time.time()
        # Pass data to GPU
        if use_cuda:
            data = data.cuda()
            # target = target.cuda()
        
        # Wrap tensors in Variable class, set volatile=True for inference mode and to use minimal memory during inference
        data = Variable(data, volatile=True)
        t2 = time.time()
        
        # Forward pass
        output = model(data).data  
        t3 = time.time()
        
        # Using confidence threshold, eliminate low-confidence predictions
        trgt = target[0].view(-1, 21)
        all_boxes = get_corresponding_region_boxes(output, conf_thresh, num_classes, anchors, num_anchors, int(trgt[0][0]), only_objectness=0)        
        t4 = time.time()
        
        # Iterate through all images in the batch
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
                    if (boxes[j][18] > best_conf_est) and (boxes[j][20] == int(truths[k][0])):
                        best_conf_est = boxes[j][18]
                        box_pr        = boxes[j]
                        bb2d_gt       = get_2d_bb(box_gt[:18], output.size(3))
                        bb2d_pr       = get_2d_bb(box_pr[:18], output.size(3))
                        iou           = bbox_iou(bb2d_gt, bb2d_pr)
                        match         = corner_confidence9(box_gt[:18], torch.FloatTensor(boxes[j][:18]))
                    
                # Denormalize the corner predictions 
                corners2D_gt = np.array(np.reshape(box_gt[:18], [9, 2]), dtype='float32')
                corners2D_pr = np.array(np.reshape(box_pr[:18], [9, 2]), dtype='float32')
                corners2D_gt[:, 0] = corners2D_gt[:, 0] * 640
                corners2D_gt[:, 1] = corners2D_gt[:, 1] * 480               
                corners2D_pr[:, 0] = corners2D_pr[:, 0] * 640
                corners2D_pr[:, 1] = corners2D_pr[:, 1] * 480
                corners2D_gt_corrected = fix_corner_order(corners2D_gt) # Fix the order of corners
                
                # Compute [R|t] by pnp
                objpoints3D = np.array(np.transpose(np.concatenate((np.zeros((3, 1)), corners3D[:3, :]), axis=1)), dtype='float32')
                K = np.array(internal_calibration, dtype='float32')
                R_gt, t_gt = pnp(objpoints3D,  corners2D_gt_corrected, K)
                R_pr, t_pr = pnp(objpoints3D,  corners2D_pr, K)
                
                # Compute pixel error
                Rt_gt        = np.concatenate((R_gt, t_gt), axis=1)
                Rt_pr        = np.concatenate((R_pr, t_pr), axis=1)
                proj_2d_gt   = compute_projection(vertices, Rt_gt, internal_calibration) 
                proj_2d_pred = compute_projection(vertices, Rt_pr, internal_calibration) 
                proj_corners_gt = np.transpose(compute_projection(corners3D, Rt_gt, internal_calibration)) 
                proj_corners_pr = np.transpose(compute_projection(corners3D, Rt_pr, internal_calibration)) 
                norm         = np.linalg.norm(proj_2d_gt - proj_2d_pred, axis=0)
                pixel_dist   = np.mean(norm)
                errs_2d.append(pixel_dist)

        t5 = time.time()

    # Compute 2D projection score
    for px_threshold in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]:
        acc = len(np.where(np.array(errs_2d) <= px_threshold)[0]) * 100. / (len(errs_2d)+eps)
        # Print test statistics
        logging('   Acc using {} px 2D Projection = {:.2f}%'.format(px_threshold, acc))

if __name__ == '__main__' and __package__ is None:
    import sys
    if len(sys.argv) == 3:
        conf_th = 0.05
        cfgfile = sys.argv[1]
        weightfile = sys.argv[2]
        datacfg = 'cfg/ape_occlusion.data'
        valid(datacfg, cfgfile, weightfile, conf_th)
        datacfg = 'cfg/can_occlusion.data'
        valid(datacfg, cfgfile, weightfile, conf_th)
        datacfg = 'cfg/cat_occlusion.data'
        valid(datacfg, cfgfile, weightfile, conf_th)
        datacfg = 'cfg/duck_occlusion.data'
        valid(datacfg, cfgfile, weightfile, conf_th)
        datacfg = 'cfg/glue_occlusion.data'
        valid(datacfg, cfgfile, weightfile, conf_th)
        datacfg = 'cfg/holepuncher_occlusion.data'
        valid(datacfg, cfgfile, weightfile, conf_th)
    else:
        print('Usage:')
        print(' python valid.py cfgfile weightfile')
