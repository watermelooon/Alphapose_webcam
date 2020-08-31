import torch
import re
import os
import collections
from torch._six import string_classes, int_classes
import cv2
from opt import opt
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math
import copy
import pika
from queue import Queue
from threading import Thread

numpy_type_map = {
    'float64': torch.DoubleTensor,
    'float32': torch.FloatTensor,
    'float16': torch.HalfTensor,
    'int64': torch.LongTensor,
    'int32': torch.IntTensor,
    'int16': torch.ShortTensor,
    'int8': torch.CharTensor,
    'uint8': torch.ByteTensor,
}

_use_shared_memory = True

def getTime(time1=0):
    if not time1:
        return time.time()
    else:
        interval = time.time() - time1
        return time.time(), interval








class kankantest():
    def start(self):
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self
    def update(self):
        while Q.qsize()>10:
            Q.get()
            
        (grabbed, frame) = stream.read()
        frame = cv2.resize(frame[350:,:3000],(1280,720))
        with torch.no_grad():
            Q.put(frame)



class mqMsg():
    def __init__(self, mqip):
    
        self.credentials = pika.PlainCredentials('admin', 'hh@mq!')
        # 虚拟队列需要指定参数 virtual_host，如果是默认的可以不填。
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(host = mqip,port = 5672,virtual_host = '/',credentials = self.credentials))
        self.channel=self.connection.channel()
        # 声明exchange，由exchange指定消息在哪个队列传递，如不存在，则创建。durable = True 代表exchange持久化存储，False 非持久化存储
        self.channel.exchange_declare(exchange = 'NonPay-MainSend1',durable = True, exchange_type='direct')
        # 向队列插入数值 routing_key是队列名。delivery_mode = 2 声明消息在队列中持久化，delivery_mod = 1 消息非持久化。routing_key 不需要配置
    def sendMsg(self, msgData):
        self.channel.basic_publish(exchange = 'NonPay-MainSend1',routing_key = 'behavior' ,body = msgData)
        #W      properties=pika.BasicProperties(delivery_mode = 2))
    
    def closeMq(self):
        self.connection.close()
        
        
        
def vis_frame_fast(frame, im_res, webcamID):
    '''
    frame: frame image
    im_res: im_res of predictions
    format: coco or mpii

    return rendered image
    '''

    l_pair = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            (17, 11), (17, 12),  # Body
            (11, 13), (12, 14), (13, 15), (14, 16)
        ]
    p_color = [(0, 255, 255), (0, 191, 255),(0, 255, 102),(0, 77, 255), (0, 255, 0), 
                    #Nose, LEye, REye, LEar, REar
                    (77,255,255), (77, 255, 204), (77,204,255), (191, 255, 77), (77,191,255), (191, 255, 77), 
                    #LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist
                    (204,77,255), (77,255,204), (191,77,255), (77,255,191), (127,77,255), (77,255,127), (0, 255, 255)] 
                    #LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck
    line_color = [(0, 215, 255), (0, 255, 204), (0, 134, 255), (0, 255, 50), 
                    (77,255,222), (77,196,255), (77,135,255), (191,255,77), (77,255,77), 
                    (77,222,255), (255,156,127), 
                    (0,127,255), (255,127,77), (0,77,255), (255,77,36)]
    
    threshold_y = {2:320, 3:370, 4:230, 5:260}#threshold to send MQ message
    threshold_x = {2:280, 3:250, 4:280, 5:200}
    
    IP2=[
        (0.456,0.006),
        (0.417,0.319),
        (0.544,0.864),
        (0.679,0.796),
        (0.725,0.567) 
    ]
    IP3=[
        (0.484,0.456),
        (0.508,0.981),
        (0.894,0.983),
        (0.710,0.383),
        (0.529,0.372)
    ]
    IP4=[
        (0.443,0.317),
        (0.515,0.860),
        (0.823,0.692),
        (0.591,0.229),
        (0.453,0.260)
    ]
    IP5=[
        (0.459,0.303),
        (0.459,0.992),
        (0.825,0.964),
        (0.610,0.258),
        (0.473,0.260)
    ]  
    area={2:IP2, 3:IP3, 4:IP4, 5:IP5}
    
    
    
    
    img = frame
    hand_x=0
    hand_y=0
    body_y=0
    msg=[]
    #msg=['{{ "source": "Behavior", "deviceId": {}, "hand": "{},{}", "body": {}, "state": {} }}'.format(webcamID, hand_x, hand_y, body_y, 0)]
    msg_valid=0

    
    for human in im_res['result']:
        
        part_line = {}
        kp_preds = human['keypoints']
        kp_scores = human['kp_score']
        kp_preds = torch.cat((kp_preds, torch.unsqueeze((kp_preds[5,:]+kp_preds[6,:])/2,0)))
        kp_scores = torch.cat((kp_scores, torch.unsqueeze((kp_scores[5,:]+kp_scores[6,:])/2,0)))
        
        kp_preds[9]=kp_preds[9]+(kp_preds[9]-kp_preds[7])*0.4
        kp_preds[10]=kp_preds[10]+(kp_preds[10]-kp_preds[8])*0.4
        
        
        
        # Draw keypoints
        for n in range(kp_scores.shape[0]):
            if kp_scores[n] <= 0.05:
                continue
            cor_x, cor_y = int(kp_preds[n, 0]), int(kp_preds[n, 1])
            part_line[n] = (cor_x, cor_y)
            cv2.circle(img, (cor_x, cor_y), 4, p_color[n], -1)
        # Draw limbs
        for i, (start_p, end_p) in enumerate(l_pair):
            if start_p in part_line and end_p in part_line:
                start_xy = part_line[start_p]
                end_xy = part_line[end_p]
                cv2.line(img, start_xy, end_xy, line_color[i], 2*(kp_scores[start_p] + kp_scores[end_p]) + 1)
        
        #draw area
        points=area[opt.webcam_num]
        for i in range(4):
            st=(int(points[i][0]*1030), int(points[i][1]*720))
            ed=(int(points[i+1][0]*1030), int(points[i+1][1]*720))
            cv2.line(img, st, ed, (0,0,255), 2)
        st=(int(points[4][0]*1030), int(points[4][1]*720))
        ed=(int(points[0][0]*1030), int(points[0][1]*720))
        cv2.line(img, st, ed, (0,0,255), 2)
                    
        
        hand_index= 9 if ( kp_preds[9,0] >= kp_preds[10,0]) else 10
        if kp_preds[hand_index,0]>threshold_x[opt.webcam_num]  and kp_preds[12,0]<400 and  kp_preds[12,1]>threshold_y[opt.webcam_num] :
            hand_x=int(kp_preds[hand_index,0])/1030 
            hand_y=int(kp_preds[hand_index,1])/720
            body_y=int(kp_preds[12,1])/720
            
            state=0
            for i in range(0,3):
                for j in range(i+1,4):
                    for k in range(j+1,5):
                        A=np.array(points[i])
                        B=np.array(points[j])
                        C=np.array(points[k])
                        P=np.array([hand_x,hand_y])
                        if np.cross(B-A,P-A)<0 and np.cross(C-B,P-B)<0 and np.cross(A-C,P-C)<0:
                            state=1
                        
            if(state==1):
                cv2.putText(img, "in",(int(kp_preds[4,0]),int(kp_preds[4,1])),cv2.FONT_HERSHEY_COMPLEX,3,(0,0,255),6)
                msg.append('{{ "source": "Behavior", "deviceId": {}, "hand": "{:.3f},{:.3f}", "body": {:.3f}, "state": {} }}'.format(webcamID, hand_x, hand_y, body_y, 1))
            elif(state == 0):
            ##
                state2=0
                for i in range(0,3):
                    for j in range(i+1,4):
                        for k in range(j+1,5):
                            A=np.array(points[i])
                            A[0]=A[0]-0.03
                            B=np.array(points[j])
                            B[0]=B[0]-0.03
                            C=np.array(points[k])
                            C[0]=C[0]-0.03
                            P=np.array([hand_x,hand_y])
                            if np.cross(B-A,P-A)<0 and np.cross(C-B,P-B)<0 and np.cross(A-C,P-C)<0:
                                state2=1
            ##
                if (state2 == 0):
                    cv2.putText(img, "out",(int(kp_preds[4,0]),int(kp_preds[4,1])),cv2.FONT_HERSHEY_COMPLEX,3,(0,0,255),6)
                    msg.append('{{ "source": "Behavior", "deviceId": {}, "hand": "{:.3f},{:.3f}", "body": {:.3f}, "state": {} }}'.format(webcamID, hand_x, hand_y, body_y, 0))
            #msg.append('{{ "source": "Behavior", "deviceId": {}, "hand": "{:.3f},{:.3f}", "body": {:.3f}, "state": {} }}'.format(webcamID, hand_x, hand_y, body_y, 0))
            msg_valid=1
            
            
            
            x=0.1162*math.log(body_y) + 0.9899
            y=(-11.0 / 36.0) *x +  7.0 / 18.0
            x=x*1030
            #y=y*720
            cv2.circle(img, (int(x), 20), 4, p_color[n], -1)
        
        
        
        

        '''
        if kp_preds[9,0]>threshold_x[opt.webcam_num] and kp_preds[12,0]<400 and kp_preds[12,1]>threshold_y[opt.webcam_num] :
            hand_x=int(kp_preds[9,0])/1030
            hand_y=int(kp_preds[9,1])/720
            body_y=int(kp_preds[12,1])/720
            msg.append('{{ "source": "Behavior", "deviceId": {}, "hand": "{:.3f},{:.3f}", "body": {:.3f} }}'.format(webcamID, hand_x, hand_y, body_y))
            msg_valid=1
            
        elif kp_preds[10,0]>threshold_x[opt.webcam_num] and kp_preds[12,0]<400 and kp_preds[12,1]>threshold_y[opt.webcam_num] :
            hand_x=int(kp_preds[10,0])/1030
            hand_y=int(kp_preds[10,1])/720
            body_y=int(kp_preds[12,1])/720
            msg.append('{{ "source": "Behavior", "deviceId": {}, "hand": "{:.3f},{:.3f}", "body": {:.3f} }}'.format(webcamID, hand_x, hand_y, body_y))
            msg_valid=1
        '''
    return img, msg, msg_valid



