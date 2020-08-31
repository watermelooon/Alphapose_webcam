import os
import torch
from torch.autograd import Variable
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
from SPPE.src.utils.img import load_image, cropBox, im_to_torch
from opt import opt
from yolo.preprocess import prep_image, prep_frame, inp_to_image
from pPose_nms import pose_nms, write_json
from SPPE.src.utils.eval import getPrediction
from yolo.util import write_results, dynamic_write_results
from yolo.darknet import Darknet
from SPPE.src.main_fast_inference import *
from tqdm import tqdm
import cv2
import json
import numpy as np
import sys
import time
import torch.multiprocessing as mp
from multiprocessing import Process
from multiprocessing import Queue as pQueue
from threading import Thread
from queue import Queue, LifoQueue, PriorityQueue
import csv

from fn import vis_frame_fast as vis_frame
from fn import mqMsg, getTime
from ctypes import *

class WebcamLoader:
    def __init__(self, webcam):
        # initialize the file video stream along with the boolean
        # used to indicate if the thread should be stopped or not
        self.stream = cv2.VideoCapture(webcam)
        assert self.stream.isOpened(), 'Cannot capture source'
        self.stopped = False
        # initialize the queue used to store frames read from
        # the video file

    def forward(self, Q_load):
        # keep looping infinitely
        i = 0
        while True:
            time.sleep(0.02)
            while Q_load.qsize()>14:
                Q_load.get()
            img = []
            orig_img = []
            im_dim_list = []
            
            (grabbed, frame) = self.stream.read()

            # if the `grabbed` boolean is `False`, then we have
            # reached the end of the video file
            if not grabbed:
                return 0
            
            frame = frame[: ,250: ]
            #frame = cv2.resize(frame,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)

            inp_dim = int(opt.inp_dim)
            img_k, orig_img_k, im_dim_list_k = prep_frame(frame, inp_dim)
            img.append(img_k)
            orig_img.append(orig_img_k)
            im_dim_list.append(im_dim_list_k)

            
            img = torch.cat(img)
            im_dim_list = torch.FloatTensor(im_dim_list).repeat(1,2)
            Q_load.put((img, orig_img, im_dim_list))
                    


class DetectionLoader:
    def __init__(self, dataloder, batchSize=1):
        # initialize the file video stream along with the boolean
        # used to indicate if the thread should be stopped or not
        self.det_model = Darknet("yolo/cfg/yolov3-spp.cfg")
        self.det_model.load_weights('models/yolo/yolov3-spp.weights')
        self.det_model.net_info['height'] = opt.inp_dim
        self.det_inp_dim = int(self.det_model.net_info['height'])
        assert self.det_inp_dim % 32 == 0
        assert self.det_inp_dim > 32
        self.det_model.cuda()
        self.det_model.eval()
        self.dataloader=dataloder
        self.stopped = False
        self.batchSize = batchSize
        # initialize the queue used to store frames read from
        # the video file

    def forward(self,Q_load, Q_det):
        # keep looping the whole dataset
        
        while True:
            #print(Q_load.qsize(), Q_det.qsize())
            img, orig_img, im_dim_list = Q_load.get()

            with torch.no_grad():
                # Human Detection
                img = img.cuda()
                
                prediction = self.det_model(img, CUDA=True)
                # NMS process
                dets = dynamic_write_results(prediction, opt.confidence,
                                    opt.num_classes, nms=True, nms_conf=opt.nms_thesh)
                if isinstance(dets, int) or dets.shape[0] == 0:
                    
                    for k in range(len(orig_img)):
                        if Q_det.full():
                            time.sleep(0.1)
                            #print("detectionloaderQ1 full ")
                        #Q_det.put((orig_img[k],  None, None, None, None, None))
                        Q_det.put((None, orig_img[k],  None, None, None, None))
                    continue
                dets = dets.cpu()
                im_dim_list = torch.index_select(im_dim_list,0, dets[:, 0].long())
                scaling_factor = torch.min(self.det_inp_dim / im_dim_list, 1)[0].view(-1, 1)

                # coordinate transfer
                dets[:, [1, 3]] -= (self.det_inp_dim - scaling_factor * im_dim_list[:, 0].view(-1, 1)) / 2
                dets[:, [2, 4]] -= (self.det_inp_dim - scaling_factor * im_dim_list[:, 1].view(-1, 1)) / 2

                
                dets[:, 1:5] /= scaling_factor
                for j in range(dets.shape[0]):
                    dets[j, [1, 3]] = torch.clamp(dets[j, [1, 3]], 0.0, im_dim_list[j, 0])
                    dets[j, [2, 4]] = torch.clamp(dets[j, [2, 4]], 0.0, im_dim_list[j, 1])
                boxes = dets[:, 1:5]
                scores = dets[:, 5:6]

            for k in range(len(orig_img)):
                boxes_k = boxes[dets[:,0]==k]
                inps = torch.zeros(boxes_k.size(0), 3, opt.inputResH, opt.inputResW)
                pt1 = torch.zeros(boxes_k.size(0), 2)
                pt2 = torch.zeros(boxes_k.size(0), 2)
            
                inp = im_to_torch(cv2.cvtColor(orig_img[k], cv2.COLOR_BGR2RGB))
                inps, pt1, pt2 = crop_from_dets(inp, boxes_k, inps, pt1, pt2)
                
                if Q_det.full():
                    time.sleep(0.1)
                    #print("detectionloaderQ3 full ")
                #Q_det.put((orig_img[k],  boxes_k, scores[dets[:,0]==k], inps, pt1, pt2))
                Q_det.put((inps, orig_img[k],  boxes_k, scores[dets[:,0]==k], pt1, pt2))




class pose_processing:
    def __init__(self):
        print('Loading Alphapose model..')
        
        self.pose_model = InferenNet_fast(4 * 1 + 1, Mscoco())
        self.pose_model.cuda()
        self.pose_model.eval()

        
        self.mq=mqMsg("192.168.1.131")
        self.dll=cdll.LoadLibrary("./libmjpg_send.so") 
        self.dll.mjpg_init.argtypes=()
        self.dll.mjpg_write.argtypes=(c_char_p,c_int)
        self.dll.mjpg_init(opt.mjpg_port)
        
        
    def forward(self,Q_det):
        clock=0
        while 1:
          with torch.no_grad():
            start_time = getTime()
            (inps, orig_img, boxes, scores, pt1, pt2)=Q_det.get()
            ckpt_time, det_time = getTime(start_time)
            img=orig_img
            
            if boxes is not None:
                
                hm_data = self.pose_model(inps.cuda())
                ckpt_time, pose_time = getTime(ckpt_time)
                
                hm_data = hm_data.cpu()
                preds_hm, preds_img, preds_scores = getPrediction(
                            hm_data, pt1, pt2, opt.inputResH, opt.inputResW, opt.outputResH, opt.outputResW)
                result = pose_nms(boxes, scores, preds_img, preds_scores)
                result={'result': result}
                img, datas, valid= vis_frame(img, result, opt.webcam_num)
                ckpt_time, post_time = getTime(ckpt_time)
                #print("det_time={:.3f}, pose_time={:.3f}, post_time={:.3f}, fps={:.1f}".format(det_time,pose_time,post_time, 1/(det_time+pose_time+post_time)))
                if valid:
                    for data in datas:
                        clock=0
                        self.mq.sendMsg(data)
                        print(data)
                else:
                    clock=clock+1
                    if clock>500:
                        self.mq.sendMsg('{{ "source": "Behavior", "deviceId": {}, "hand": "{},{}", "body": {}, "state": {} }}'.format(0, 0, 0, 0, 0))
                        clock=0
            else:
                time.sleep(0.05)
                clock=clock+1
                if clock>500:
                    self.mq.sendMsg('{{ "source": "Behavior", "deviceId": {}, "hand": "{},{}", "body": {}, "state": {} }}'.format(0, 0, 0, 0, 0))
                    clock=0
                ''''
                self.mq.sendMsg('{ "source": "Behavior", "deviceId": 0, "hand": "0,0", "body": 0 }')
                print('{ "source": "Behavior", "deviceId": 0, "hand": "0,0", "body": 0 }')
                '''
    

            if opt.vis:
                cv2.imshow("AlphaPose Demo", img)
                cv2.waitKey(10)
                
            
            temp=cv2.imencode(".jpg",img)[1]
            icode=np.array(temp)
            se=icode.tostring()      
            n=len(se)
            self.dll.mjpg_write(c_char_p(se),n)
            
            


class Mscoco(data.Dataset):
    def __init__(self, train=True, sigma=1,
                 scale_factor=(0.2, 0.3), rot_factor=40, label_type='Gaussian'):
        self.img_folder = '../data/coco/images'    # root image folders
        self.is_train = train           # training set or test set
        self.inputResH = opt.inputResH
        self.inputResW = opt.inputResW
        self.outputResH = opt.outputResH
        self.outputResW = opt.outputResW
        self.sigma = sigma
        self.scale_factor = scale_factor
        self.rot_factor = rot_factor
        self.label_type = label_type

        self.nJoints_coco = 17
        self.nJoints_mpii = 16
        self.nJoints = 33

        self.accIdxs = (1, 2, 3, 4, 5, 6, 7, 8,
                        9, 10, 11, 12, 13, 14, 15, 16, 17)
        self.flipRef = ((2, 3), (4, 5), (6, 7),
                        (8, 9), (10, 11), (12, 13),
                        (14, 15), (16, 17))

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass


def crop_from_dets(img, boxes, inps, pt1, pt2):
    '''
    Crop human from origin image according to Dectecion Results
    '''

    imght = img.size(1)
    imgwidth = img.size(2)
    tmp_img = img
    tmp_img[0].add_(-0.406)
    tmp_img[1].add_(-0.457)
    tmp_img[2].add_(-0.480)
    for i, box in enumerate(boxes):
        upLeft = torch.Tensor(
            (float(box[0]), float(box[1])))
        bottomRight = torch.Tensor(
            (float(box[2]), float(box[3])))

        ht = bottomRight[1] - upLeft[1]
        width = bottomRight[0] - upLeft[0]
        if width > 100:
            scaleRate = 0.2
        else:
            scaleRate = 0.3

        upLeft[0] = max(0, upLeft[0] - width * scaleRate / 2)
        upLeft[1] = max(0, upLeft[1] - ht * scaleRate / 2)
        bottomRight[0] = max(
            min(imgwidth - 1, bottomRight[0] + width * scaleRate / 2), upLeft[0] + 5)
        bottomRight[1] = max(
            min(imght - 1, bottomRight[1] + ht * scaleRate / 2), upLeft[1] + 5)

        inps[i] = cropBox(tmp_img.clone(), upLeft, bottomRight, opt.inputResH, opt.inputResW)
        pt1[i] = upLeft
        pt2[i] = bottomRight

    return inps, pt1, pt2
