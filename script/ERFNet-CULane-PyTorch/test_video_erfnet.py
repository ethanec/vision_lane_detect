import os
import os.path as ops
import time
import shutil
import torch
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import cv2
import utils.transforms as tf
from utils import cluster
import numpy as np
import models
from models import sync_bn
import dataset as ds
from options.options import parser
import torch.nn.functional as F

best_mIoU = 0


def main():
    global args, best_mIoU
    args = parser.parse_args()
    #print(args)
    if args.video_path is None:
        print('No video found!')
    
    else:
        assert ops.exists(args.video_path), '{:s} not exist'.format(args.video_path)
        
        #os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(gpu) for gpu in args.gpus)
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        #args.gpus = len(args.gpus)
        args.gpus = 1

        if args.no_partialbn:
            sync_bn.Synchronize.init(args.gpus)

        if args.dataset == 'VOCAug' or args.dataset == 'VOC2012' or args.dataset == 'COCO':
            num_class = 21
            ignore_label = 255
            scale_series = [10, 20, 30, 60]
        elif args.dataset == 'Cityscapes':
            num_class = 19
            ignore_label = 255 
            scale_series = [15, 30, 45, 90]
        elif args.dataset == 'ApolloScape':
            num_class = 37 
            ignore_label = 255 
        elif args.dataset == 'CULane':
            num_class = 5
            ignore_label = 255
        else:
            raise ValueError('Unknown dataset ' + args.dataset)

        model = models.ERFNet(num_class, partial_bn = not args.no_partialbn)
        input_mean = model.input_mean
        input_std = model.input_std
        policies = model.get_optim_policies()
        model = torch.nn.DataParallel(model, device_ids=range(args.gpus)).cuda()

        if args.resume:
            if os.path.isfile(args.resume):
                print(("=> loading checkpoint '{}'".format(args.resume)))
                checkpoint = torch.load(args.resume)
                args.start_epoch = checkpoint['epoch']
                best_mIoU = checkpoint['best_mIoU']
                torch.nn.Module.load_state_dict(model, checkpoint['state_dict'])
                print(("=> loaded checkpoint '{}' (epoch {})".format(args.evaluate, checkpoint['epoch'])))
            else:
                print(("=> no checkpoint found at '{}'".format(args.resume)))


        cudnn.benchmark = True
        cudnn.fastest = True

        # Data loading code
        '''
        transform = torchvision.transforms.Compose([
            tf.GroupRandomScaleNew(size=(args.img_width, args.img_height), interpolation=[cv2.INTER_LINEAR]),
            tf.GroupNormalize(mean=(input_mean, (0, )), std=(input_std, (1, ))),
        ])
        '''
        '''
        test_loader = torch.utils.data.DataLoader(
            getattr(ds, args.dataset.replace("CULane", "VOCAug") + 'DataSet')(data_list=args.val_list, transform=torchvision.transforms.Compose([
                tf.GroupRandomScaleNew(size=(args.img_width, args.img_height), interpolation=(cv2.INTER_LINEAR, cv2.INTER_NEAREST)),
                tf.GroupNormalize(mean=(input_mean, (0, )), std=(input_std, (1, ))),
            ])), batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=False)
        '''
        # define loss function (criterion) optimizer and evaluator
        weights = [1.0 for _ in range(5)]
        weights[0] = 0.4
        class_weights = torch.FloatTensor(weights).cuda()
        criterion = torch.nn.NLLLoss(ignore_index=ignore_label, weight=class_weights).cuda()
        for group in policies:
            print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))
        optimizer = torch.optim.SGD(policies, args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        evaluator = EvalSegmentation(num_class, ignore_label)

        ### evaluate ###
        #validate(test_loader, model, criterion, 0, evaluator)
        inference(args.video_path, model, criterion, 0, evaluator, (input_mean, (0, )), (input_std, (1, )))
    return
'''
def ComputeMinLevel(hist, pnum):
    index = np.add.accumulate(hist)
    return np.argwhere(index>pnum * 8.3 * 0.01)[0][0]

def ComputeMaxLevel(hist, pnum):
    hist_0 = hist[::-1]
    Iter_sum = np.add.accumulate(hist_0)
    index = np.argwhere(Iter_sum > (pnum * 2.2 * 0.01))[0][0]
    return 255-index

def LinearMap(minlevel, maxlevel):
    if (minlevel >= maxlevel):
        return []
    else:
        index = np.array(list(range(256)))
        screenNum = np.where(index<minlevel,0,index)
        screenNum = np.where(screenNum> maxlevel,255,screenNum)
        for i in range(len(screenNum)):
            if screenNum[i]> 0 and screenNum[i] < 255:
                screenNum[i] = (i - minlevel) / (maxlevel - minlevel) * 255
        return screenNum

def CreateNewImg(img):
    h, w, d = img.shape
    newimg = np.zeros([h, w, d])
    for i in range(d):
        imghist = np.bincount(img[:, :, i].reshape(1, -1)[0])
        minlevel = ComputeMinLevel(imghist,  h * w)
        maxlevel = ComputeMaxLevel(imghist, h * w)
        screenNum = LinearMap(minlevel, maxlevel)
        if (screenNum.size == 0):
            continue
        for j in range(h):
            newimg[j, :, i] = screenNum[img[j, :, i]]
    return newimg
'''
def inference(video_path, model, criterion, iter, evaluator, mean, std, logger=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    IoU = AverageMeter()
    mIoU = 0
    erf_cluster = cluster.ERFNetCluster()
    # switch to evaluate mode
    model.eval()

    cap = cv2.VideoCapture(video_path)
    if(cap.isOpened() == False): 
        print("Error opening video stream or file")
    else:
        while(cap.isOpened()):
            tt_start = time.time()
            ret, frame = cap.read()
            
            if ret == True:
                
                #frame_vis = cv2.resize(frame, (976, 208))
                frame_vis = frame
                h = frame_vis.shape[0]
                w = frame_vis.shape[1]
                frame = frame[240:, :, :] # (448, 448, 3) -> (208, 448, 3)
                o_h = frame.shape[0]
                o_w = frame.shape[1]

                #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                frame = cv2.resize(frame, (976, 208), interpolation = cv2.INTER_LINEAR)
                
                '''
                a = 1.2
                b = -25
                frame = frame * float(a) + float(b)
                frame[frame > 255] = 255
                frame = np.round(frame)
                frame = frame.astype(np.uint8)
                '''

                #cv2.imshow('Result', cv2.resize(frame, (1024, 512)))
                if cv2.waitKey(1) == 27: # Important!!!!!!!!!
                    break
                
                # GroupNormalize
                img_group = [frame]
                out_images = list()
                for img, m, s in zip(img_group, mean, std):
                    if len(m) == 1:
                        img = img - np.array(m)  # single channel image
                        img = img / np.array(s)
                    else:
                        img = img - np.array(m)[np.newaxis, np.newaxis, ...]
                        img = img / np.array(s)[np.newaxis, np.newaxis, ...]
                    out_images.append(img)

                frame = out_images[0]
                frame = torch.from_numpy(frame).permute(2, 0, 1).contiguous().float()
                #print(frame.shape)
                frame = frame.reshape(1, 3, 208, 976)
                #print(frame.shape)

                input_var = torch.autograd.Variable(frame, volatile=True)

                # compute output
                output, output_exist = model(input_var)

                # measure accuracy and record loss
                output = F.softmax(output, dim=1)
                pred = output.data.cpu().numpy() # BxCxHxW (1, 5, 208, 976)
                pred_exist = output_exist.data.cpu().numpy() # BxO
                
                pred_input = cv2.resize(pred[0].transpose((1, 2, 0)), (448, 208), interpolation = cv2.INTER_LINEAR) # (208, 976, 3) -> (208, 448, 3)
                '''
                merge = ((pred_input[:, :, 1] * 255).astype(np.uint8) + \
                        (pred_input[:, :, 2] * 255).astype(np.uint8) + \
                        (pred_input[:, :, 3] * 255).astype(np.uint8) + \
                        (pred_input[:, :, 4] * 255).astype(np.uint8)) / 4
                '''
                #mask_list = erf_cluster.get_lane_mask(merge, pred_list)
                mask_list = erf_cluster.get_lane_mask(pred_input[:, :, 1:])
                
                #ttt_start = time.time()
                for m in range(4):
                    tmp = mask_list[m]
                    mask_list[m] = []
                    mask_list[m] = cv2.resize(tmp, (o_w, o_h), interpolation = cv2.INTER_LINEAR) # (208, 448) -> (o_h, o_w)
                #ttt_cost = time.time() - ttt_start
                #print(('Resize: {} s'.format(ttt_cost)))

                mask_output = np.zeros([o_h, o_w, 3]).astype(np.uint8) # (208, 448, 3)
                mask_output[:, :, 0] = np.zeros([o_h, o_w]).astype(np.uint8) # B
                mask_output[:, :, 1] = mask_list[0].astype(np.uint8) + mask_list[1].astype(np.uint8) # G
                #mask_output[:, :, 1] = mask_list[1].astype(np.uint8) # G
                mask_output[:, :, 2] = mask_list[2].astype(np.uint8) + mask_list[3].astype(np.uint8) # R
                #mask_output[:, :, 2] = mask_list[2].astype(np.uint8) # R
                
                mask_frame = np.zeros([h, w, 3]).astype(np.uint8) # (448, 448, 3)
                for i in range(0, 3):
                    compensate_zero = np.zeros([240, o_w]).astype(np.uint8)
                    mask_frame[:, :, i] = np.concatenate((compensate_zero, mask_output[:, :, i]), axis = 0)
                
                output_frame = (cv2.addWeighted(frame_vis, 1, mask_frame, 1, 0))
                
                #output_frame = frame_vis
                cv2.imshow('Result', cv2.resize(output_frame, (1024, 512)))
            else:
                break
                
            tt_cost = time.time() - tt_start
            print(('FPS: {:.5f}'.format(1 / tt_cost)))

    cap.release()
    cv2.destroyAllWindows()
    return mIoU

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def update(self, val, n=1):
        if self.val is None:
            self.val = val
            self.sum = val * n
            self.count = n
            self.avg = self.sum / self.count
        else:
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count


class EvalSegmentation(object):
    def __init__(self, num_class, ignore_label=None):
        self.num_class = num_class
        self.ignore_label = ignore_label

    def __call__(self, pred, gt):
        assert (pred.shape == gt.shape)
        gt = gt.flatten().astype(int)
        pred = pred.flatten().astype(int)
        locs = (gt != self.ignore_label)
        sumim = gt + pred * self.num_class
        hs = np.bincount(sumim[locs], minlength=self.num_class**2).reshape(self.num_class, self.num_class)
        return hs


def adjust_learning_rate(optimizer, epoch, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    # decay = 0.1**(sum(epoch >= np.array(lr_steps)))
    decay = ((1 - float(epoch) / args.epochs)**(0.9))
    lr = args.lr * decay
    decay = args.weight_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = decay * param_group['decay_mult']


if __name__ == '__main__':
    main()
