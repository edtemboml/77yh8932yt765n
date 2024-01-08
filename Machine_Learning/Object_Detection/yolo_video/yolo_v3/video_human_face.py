###################################################################################
### By Edwin Tembo
### Created in 2020 - Based on a template provided in a Bootcamp class
### Modified 01-08-2023 for use in Google Colab###
### Uses Weights from a Custom Trained Yolo V3 Model for Facial Anatomy Detection
### For Research or Educational Purposes
###################################################################################


from __future__ import division
import time
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2 
from util import *
from darknet import Darknet
from preprocess import prep_image, inp_to_image, letterbox_image
from google.colab.patches import cv2_imshow
import pandas as pd
import random 
import pickle as pkl
import argparse
from PIL import Image as im
import matplotlib.pyplot as plt
import matplotlib.image as mpimg



def get_test_input(input_dim,test_img, CUDA):
    img = cv2.imread(test_img)
    img = cv2.resize(img, (input_dim, input_dim)) 
    img_ =  img[:,:,::-1].transpose((2,0,1))
    img_ = img_[np.newaxis,:,:,:]/255.0
    img_ = torch.from_numpy(img_).float()
    img_ = Variable(img_)
    
    if CUDA:
        img_ = img_.cuda()
    
    return img_

def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network. 
    
    Returns a Variable 
    """

    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]
    img = (letterbox_image(orig_im, (inp_dim, inp_dim)))
    img_ = img[:,:,::-1].transpose((2,0,1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim

def write(x, img):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    ## converting int tensor to integer
    c1 = int(c1[0].item()), int(c1[1].item()) 
    c2 = int(c2[0].item()), int(c2[1].item())

    cls = int(x[-1])
    label = "{0}".format(classes[cls])
    color = random.choice(colors)

    cv2.rectangle(img, c1, c2,color, 3)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 1.5 , 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2,color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_DUPLEX, 1.5, [225,255,255], 1);
    return img

def arg_parse():
    """
    Parse arguements to the detect module
    
    """
    
    
    parser = argparse.ArgumentParser(description='YOLO v3 Video Facial Anatomy Detection')
    parser.add_argument("--working_dir", dest= "working_dir", help="The directory_path")
    parser.add_argument('--test_img', dest= 'test_img', help = "Path to test image")
    parser.add_argument("--video", dest = 'video', help = 
                        "Video to run detection upon",
                        default = "video.avi", type = str)
    parser.add_argument("--dataset", dest = "dataset", help = "Dataset on which the network has been trained", default = "pascal")
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.5)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
    parser.add_argument("--cfg", dest = 'cfgfile', help = 
                        "Config file",
                        default = "model.cfg", type = str)
    parser.add_argument("--weights", dest = 'weightsfile', help = 
                        "weightsfile",
                        default = "model.weights", type = str)
    parser.add_argument("--reso", dest = 'reso', help = 
                        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default = "416", type = str)
    return parser.parse_args()


if __name__ == '__main__':
    args = arg_parse()
    confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)
    start = 0
    working_dir = args.working_dir

    CUDA = torch.cuda.is_available()

    num_classes = 6

    CUDA = torch.cuda.is_available()
    
    bbox_attrs = 5 + num_classes
    
    print("Loading network.....")
    model = Darknet(args.cfgfile)
    model.load_weights(args.weightsfile)
    print("Network successfully loaded")

    model.net_info["height"] = args.reso
    inp_dim = int(model.net_info["height"])
    assert inp_dim % 32 == 0 
    assert inp_dim > 32

    

    if CUDA:
        model.cuda()
        
    model(get_test_input(inp_dim, args.test_img, CUDA), CUDA)

    model.eval()
    
    videofile = args.video
    
    cap = cv2.VideoCapture(videofile)
    
    assert cap.isOpened(), 'Cannot capture source'
    img_counter = 0 
    frames = 0
    start  = time.time()    
    while cap.isOpened():
        
        ret, frame = cap.read()
        if ret:
            

            img, orig_im, dim = prep_image(frame, inp_dim)
            
            im_dim = torch.FloatTensor(dim).repeat(1,2)                        
            
            
            if CUDA:
                im_dim = im_dim.cuda()
                img = img.cuda()
            
            with torch.no_grad():   
                output = model(Variable(img), CUDA)

            output = write_results(output, confidence, num_classes, nms = True, nms_conf = nms_thesh)

            if type(output) == int:
                frames += 1

                if frames%10 == 0 :

                  print(f"FPS of the video is aprox: {frames / (time.time() - start)}")
  
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break
                continue
            
            im_dim = im_dim.repeat(output.size(0), 1)
            scaling_factor = torch.min(inp_dim/im_dim,1)[0].view(-1,1)
            
            output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim[:,0].view(-1,1))/2
            output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim[:,1].view(-1,1))/2
            
            output[:,1:5] /= scaling_factor
    
            for i in range(output.shape[0]):
                output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim[i,0])
                output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim[i,1])
            
            classes = load_classes(f"{working_dir}/data/coco.names")
            colors  = pkl.load(open(f"{working_dir}/pallete", "rb"))
            
            list(map(lambda x: write(x, orig_im), output))
            
            data = im.fromarray(orig_im)
            img_counter +=1 
            ### saving locally  in Colab : Change to appropriate directory##
            data.save(f"/content/bbox_images/inf_im{img_counter}.png")

            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            frames += 1
            if frames%10 ==0:
              print(f"FPS of the video is {frames / (time.time() - start)}")
            
        else:
            break
    

    
    

