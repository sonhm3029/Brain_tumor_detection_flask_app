from __future__ import print_function, division, absolute_import
import torch.utils.model_zoo as model_zoo
import sys
import urllib.request
import ssl
import torch
import torchvision
from torch.optim import lr_scheduler
import torch.nn as nn 
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF
import numpy as np
import cv2
import os
from src.model.model import InceptionResNetV2
from PIL import Image
import imutils
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

SAVED_MODEL = "src/model/brain_tumor_inceptionresnetv1_98_acc_2_8.pth"
MEAN = [0.23740229, 0.23729787, 0.23700129]
STD = [0.23173477, 0.23151317, 0.23122775]


class ImageEnhanced(object):
    """_summary_
    transform to enhanced image quality for prediction 
    """
    def __init__(self):
        pass
    def __call__(self, img ,add_pixels_value = 0):
        
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        # threshold the image, then perform a series of erosions +
        # dilations to remove any small regions of noise
        thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)
        # find contours in thresholded image, then grab the largest one
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)
        # find the extreme points
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])
        ADD_PIXELS = add_pixels_value
        new_img = img[extTop[1]-ADD_PIXELS:extBot[1]+ADD_PIXELS, extLeft[0]-ADD_PIXELS:extRight[0]+ADD_PIXELS].copy()
        return Image.fromarray(new_img)

transforms = transforms.Compose([
    ImageEnhanced(),
    transforms.ToTensor(),
    transforms.Resize((299, 299), interpolation=InterpolationMode.BICUBIC),
    transforms.Normalize(MEAN, STD)
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = InceptionResNetV2()
num_inputs = model.last_linear.in_features
model.last_linear = nn.Sequential(
    nn.Dropout(0.1),
    nn.Linear(num_inputs, 1000),
    nn.ReLU(),
    nn.Linear(1000, 512),
    nn.ReLU(),
    nn.Linear(512,448),
    nn.ReLU(),
    nn.Linear(448, 320),
    nn.ReLU(),
    nn.Linear(320, 2)
).to(device)

model.load_state_dict(torch.load(SAVED_MODEL,map_location=device))
model.eval()

def img_transform(image):
    return transforms(image).unsqueeze(0)

def get_prediction(image_tensor):
    # forward
    output = model(image_tensor)
    _, output = torch.max(output, 1)
    return output