
# coding=utf-8 

import sys

import torchvision
import torch.nn as nn
from os.path import exists, join
from torchvision.transforms import Lambda,Compose, CenterCrop, Grayscale,RandomRotation,ToPILImage,RandomAffine,ToTensor,ColorJitter, RandomVerticalFlip,Normalize, Scale,RandomCrop,Pad,RandomHorizontalFlip,RandomCrop,CenterCrop

import torchvision.transforms.functional as TF

from torchvision.transforms.functional import rotate
import numpy.random as random
from torchvision.datasets.folder import IMG_EXTENSIONS
import matplotlib.pyplot as plt
import torch.utils.data as data
from os import listdir
import nibabel as nib
from skimage.transform import rescale, resize, downscale_local_mean

import os
import PIL
from PIL import Image
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import pdb
import torch
import collections
from scipy.ndimage.filters import gaussian_filter
import torch.nn.functional as F
import cv2
from scipy import ndimage
from skimage.filters import threshold_otsu

def AddPaddingAndCrop(image,segmentation,ImgSize):
    
    image=np.float32(image)
    segmentation=np.float32(segmentation)

    
    w=image.shape[0]
    h=image.shape[1]
    
    
    if w>ImgSize and h>ImgSize:
        i=int((w-ImgSize)/2.)
        j=int((h-ImgSize)/2.)


        if image.ndim==3:
            imageNew=image[i:i+ImgSize, j:j+ImgSize,:]
            segmentationNew=segmentation[i:i+ImgSize, j:j+ImgSize,:]
        elif image.ndim==2:

            imageNew=image[i:i+ImgSize, j:j+ImgSize]
            segmentationNew=segmentation[i:i+ImgSize, j:j+ImgSize]

    elif h<ImgSize and w>ImgSize:
        i=int((w-ImgSize)/2.)
        j=int((-h+ImgSize)/2.)


        if image.ndim==3:
            imageNew=np.zeros((ImgSize,ImgSize,image.shape[-1]))
            segmentationNew=np.zeros((ImgSize,ImgSize,image.shape[-1]))

            imageNew[:,j:j+h,:]=image[i:i+ImgSize, :,:]
            segmentationNew[:,j:j+h,:]=segmentation[i:i+ImgSize, :,:]
        elif image.ndim==2:
            imageNew=np.zeros((ImgSize,ImgSize))
            #pdb.set_trace()
            segmentationNew=np.zeros((ImgSize,ImgSize))
            
            imageNew[:,j:j+h]=image[i:i+ImgSize, :]
            segmentationNew[:,j:j+h]=segmentation[i:i+ImgSize, :]
        

            
    elif h>ImgSize and w<ImgSize:
        i=int((-w+ImgSize)/2.)
        j=int((+h-ImgSize)/2.)


        if image.ndim==3:
            imageNew=np.zeros((ImgSize,ImgSize,image.shape[-1]))
            segmentationNew=np.zeros((ImgSize,ImgSize,image.shape[-1]))

            imageNew[i:i+w,:,:]=image[:,j:j+ImgSize, :]
            segmentationNew[i:i+w,:,:]=segmentation[:,j:j+ImgSize, :]
        elif image.ndim==2:
            imageNew=np.zeros((ImgSize,ImgSize))
            #pdb.set_trace()
            segmentationNew=np.zeros((ImgSize,ImgSize))
            
            imageNew[i:i+w,:]=image[:,j:j+ImgSize]
            segmentationNew[i:i+w,:]=segmentation[:,j:j+ImgSize]
        
           
    elif h<ImgSize and w<ImgSize:
        i=int((-w+ImgSize)/2.)
        j=int((-h+ImgSize)/2.)


        if image.ndim==3:
            imageNew=np.zeros((ImgSize,ImgSize,image.shape[-1]))
            segmentationNew=np.zeros((ImgSize,ImgSize,image.shape[-1]))

            imageNew[i:i+w,j:j+h,:]=image#[:,:, :]
            segmentationNew[i:i+w,j:j+h,:]=segmentation#[i:i+ImgSize,j:j+ImgSize, :]
        elif image.ndim==2:
            imageNew=np.zeros((ImgSize,ImgSize))
            #pdb.set_trace()
            segmentationNew=np.zeros((ImgSize,ImgSize))
            
            imageNew[i:i+w,j:j+h]=image#[i:i+ImgSize,j:j+ImgSize]
            segmentationNew[i:i+w,j:j+h]=segmentation#[i:i+ImgSize,j:j+ImgSize]
            
    elif h==ImgSize and w==ImgSize:
        
            imageNew=image#[i:i+ImgSize,j:j+ImgSize]
            segmentationNew=segmentation#[i:i+ImgSize,j:j+ImgSize]
    
    
    return imageNew,segmentationNew


import argparse


from skimage.transform import resize

def get_data(opt,dataPath,DataloaderType='training'): #validation, 'test'
   
 
    return ReadTheDataset(opt,join(dataPath, DataloaderType))
    
    
def CropCentre(image,segmentation,ImgSize):
    image=np.float32(image)
    segmentation=np.float32(segmentation)

    w=image.shape[0]
    h=image.shape[1]
    #print(w,h)
   
    i=int((w-ImgSize)/2.)
    j=int((h-ImgSize)/2.)
    
    if image.ndim==3:
    
        imageNew=image[i:i+ImgSize, j:j+ImgSize,:]
        segmentationNew=segmentation[i:i+ImgSize, j:j+ImgSize,:]
    elif image.ndim==2:
        
        imageNew=image[i:i+ImgSize, j:j+ImgSize]
        segmentationNew=segmentation[i:i+ImgSize, j:j+ImgSize]
 
            
    return imageNew,segmentationNew

def ImageTransforms(image, segmentation,ImgSize=128, training=True):
    
        
    image=Image.fromarray((255*np.array(image)).astype(np.uint8)) 
    segmentation=Image.fromarray(255*np.array(segmentation))

    
    if training:
        
        
        aa=np.random.randint(-30,10,1)
        if aa>0:
            scale=np.random.randint(101,120,1)*0.01
        elif aa<-15:
            scale=np.random.randint(80,99,1)*0.01
        else:
            scale= 1                
                
        
        
        #scale=np.random.randint(80,120,1)*0.01
        
        aa=np.random.randint(-20,20,1)
        shear=0
        if aa>10:
            shear=(np.random.randint(-10,10,1),np.random.randint(-10,10,1))

       
        angle=45*(2*np.random.rand(1)[0]-1)
        translate=[np.random.randint(-30,30,1),np.random.randint(-30,30,1)]
          


        image=TF.affine(image, angle, translate, scale, shear, interpolation=TF.InterpolationMode.BILINEAR, fillcolor=None)     
        segmentation=TF.affine(segmentation, angle, translate, scale, shear, interpolation=TF.InterpolationMode.BILINEAR, fillcolor=None)



       
        aa=np.random.randint(-100,100,5)
        Threshold=0         
        if aa[0]<Threshold:
            gamma=np.random.randint(60,150,1)[0]*0.01
            image=TF.adjust_gamma(image, gamma, gain=1)

        if aa[2]<Threshold:
            saturation_factor=np.random.randint(60,150,1)*0.01
            image=TF.adjust_saturation(image, saturation_factor)
        if aa[3]<Threshold:
            contrast_factor=np.random.randint(80,120,1)*0.01
            image=TF.adjust_contrast(image, contrast_factor)              
            
        
       
        #contrast_factor=np.random.randint(80,120,1)*0.01
        #image=TF.adjust_contrast(image, contrast_factor)   
         
    
    segmentation=np.array(segmentation)
    image=np.array(image)
        
    image,segmentation=AddPaddingAndCrop(image,segmentation,ImgSize)
    #image,segmentation= CropCentre(image,segmentation,ImgSize)
   

    if image.ndim==2:
        image=image[:,:,np.newaxis]
        
    if segmentation.ndim==2:
        segmentation=segmentation[:,:,np.newaxis]  

    
    image= Normalise(image)

    segmentation= Normalise(segmentation)

    
    image=TF.to_tensor(image)    
    segmentation=TF.to_tensor(segmentation)
    

    return image, segmentation

def Normalise(Image):
    if isinstance(Image,(np.ndarray)): 
        return (Image-np.min(Image))/(np.max(Image)-np.min(Image)+0.0000001)
    elif isinstance(Image,(torch.Tensor)):
        return (Image-torch.min(Image))/(torch.max(Image)-torch.min(Image)+0.0000001)
    

    
    
    
class ReadTheDataset(data.Dataset):
    def __init__(self,opt, image_dir):
        super(ReadTheDataset, self).__init__()
        
        ############ test set ######################

        self.Images_3d = [x for x in sorted(listdir(join(image_dir,"images") )) if is_np_file(x) ] 
        self.GTs_3d = [x for x in sorted(listdir(join(image_dir,"1st_manual") )) if is_np_file(x) ] 
        
        self.Images = [x for x in sorted(listdir(join(image_dir,"images") )) if is_png_image_file(x) ] 
        self.GTs = [x for x in sorted(listdir(join(image_dir,"1st_manual") )) if is_png_image_file(x) ] 

        
       

        self.opt=opt


        self.image_dir = image_dir
        


    def __getitem__(self, index):
        

        #print(self.image_dir)
        if self.image_dir.endswith('test'):

            
            self.GTs_3d.sort()
            self.Images_3d.sort()  
            
            targets= np.load(join(self.image_dir,'1st_manual/',self.GTs_3d[index]))
            inputIms=np.load(join(self.image_dir,'images/',self.Images_3d[index])) 
                    
            
            #targets=Normalise(targets)
            #inputIms=Normalise(inputIms)
            assert(np.max(targets)<1.001, 'error 1')
            assert(np.max(inputIms)<1.001, 'error 2')
            
                
            inputIms,targets= CropCentre(inputIms,targets, self.opt.InputSize)
 

            if inputIms.ndim==2:
                inputIms=inputIms[:,:,None]
            if targets.ndim==2:
                targets=targets[:,:,None]


            inputIms=TF.to_tensor(inputIms)
            targets=TF.to_tensor(targets)

    
            view=[]    
        elif self.image_dir.endswith('training'):
            
            self.GTs.sort()
            self.Images.sort()
            

            targets= load_img(join(self.image_dir,'1st_manual/',self.GTs[index]))
            inputIms=load_img(join(self.image_dir,'images/',self.Images[index])) 
            
            targets=Normalise(targets)
            inputIms=Normalise(inputIms)              

            
                
            inputIms,targets=ImageTransforms(inputIms, targets,self.opt.InputSize, training=True)
            #print(inputIm.type())
            
            #inputIm,targets=AddPadding(inputIm,targets,self.opt.InputSize)
            

            
            Names=self.GTs[index]
            cc=Names.split(".",2)[0]
            view=cc.split("_",3)[1][0]
  
  
        elif self.image_dir.endswith('validation'):

            
            self.GTs.sort()
            self.Images.sort()      
            
            targets= load_img(join(self.image_dir,'1st_manual/',self.GTs[index]))
            inputIms=load_img(join(self.image_dir,'images/',self.Images[index])) 
            
            targets=Normalise(targets)
            inputIms=Normalise(inputIms)  
            
            inputIm2=torch.zeros((self.opt.InputSize,self.opt.InputSize))
            targets2=torch.zeros((self.opt.InputSize,self.opt.InputSize))
                
            inputIms,targets=ImageTransforms(inputIms, targets,self.opt.InputSize, training=False)
            
            #inputIm,targets=AddPadding(inputIm,targets,self.opt.InputSize)



            
            
            Names=self.GTs[index]
            cc=Names.split(".",2)[0]
            view=cc.split("_",3)[1][0]
  
 
        #print(inputIm.type())
        torch._assert( torch.max(targets)<1.01 , "target should be normalized")
        torch._assert( torch.max(inputIms)<1.01 , "input images should be normalized")
        

        return inputIms, targets,view
    
    def __len__(self):
        if self.image_dir.endswith('test'):
            return len(self.GTs_3d)
            
        else:
            return len(self.GTs)
    
                

  

def is_np_file(filename):
    return any(filename.endswith(extension) for extension in [".npy"])

def is_png_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png"])

def load_img(filepath,colordim=1):
    if colordim==1:
        #img=PIL.ImageOps.grayscale(Image.open(filepath))
        #img = Image.open(filepath).convert('I')
        img = Image.open(filepath).convert('L')
    else:
        img = Image.open(filepath).convert('RGB')
    #y, _, _ = img.split()
    return np.array(img)
