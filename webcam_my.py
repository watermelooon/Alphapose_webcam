import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms

import torch.nn as nn
import torch.utils.data
import numpy as np
from opt import opt

from dataloader_my import WebcamLoader, DetectionLoader,crop_from_dets, Mscoco, pose_processing
from yolo.darknet import Darknet
from SPPE.src.main_fast_inference import *
import os
import sys
from tqdm import tqdm
from fn import getTime
import cv2
from threading import Thread
from queue import Queue, LifoQueue, PriorityQueue

args = opt
args.dataset = 'coco'

if __name__ == "__main__":
    webcam = args.webcam

    Q_load = Queue(maxsize=16)
    Q_det = Queue(maxsize=10)

    print('Loading YOLO model..')
    data_loader = WebcamLoader(webcam)
    det_loader = DetectionLoader(data_loader, batchSize=args.detbatch)

    poseP=pose_processing()

    thread_list = [Thread(target=data_loader.forward, args=(Q_load,)),
               Thread(target=det_loader.forward, args=(Q_load, Q_det,)),
               Thread(target=poseP.forward, args=(Q_det,))]
    
    for t in thread_list:
        t.start()

    


    print('===========================> Finish Model Running.')
