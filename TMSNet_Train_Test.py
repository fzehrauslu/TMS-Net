#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn 
import numpy as np
import torchvision
import os
import pdb
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import PIL
from skimage.filters import threshold_otsu
from medpy.metric.binary import hd, assd, dc,jc

crossEntropy=nn.BCEWithLogitsLoss() 
cosSimilarity=nn.CosineSimilarity(dim=1)
crossEntropy2=nn.BCEWithLogitsLoss(reduce=False)





def train(TranPara,opt,model,epoch,optimizer):
    

    model.train()

    trainLoss=0

    valLoss=0
        
     
    
    DataloadersCell=[TranPara.training_data_loader_All,TranPara.training_data_loader_A, TranPara.training_data_loader_C,TranPara.training_data_loader_S]
    TrainedParts=["encoder ",'decoder A','decoder C','decoder S' ]
    
    print("training starts")
    it=0
    for training_data_loader in DataloadersCell:
        
        for _, batch in enumerate(training_data_loader):


            #pdb.set_trace()
            input=batch[0].type(torch.FloatTensor).to(TranPara.device)
            
            target=batch[1].type(torch.FloatTensor).to(TranPara.device)

            ########################## 
            if training_data_loader==TranPara.training_data_loader_All:
                if epoch ==0:
                    model.Encoder.requires_grad = True
                    model.Decoder1.requires_grad = True
                    model.Decoder2.requires_grad = True
                    model.Decoder3.requires_grad = True


                else:
                    model.Encoder.requires_grad = True
                    model.Decoder1.requires_grad = False
                    model.Decoder2.requires_grad = False
                    model.Decoder3.requires_grad = False
            elif training_data_loader==TranPara.training_data_loader_A:
                model.Encoder.requires_grad = False
                model.Decoder1.requires_grad = True
                model.Decoder2.requires_grad = False
                model.Decoder3.requires_grad = False

                
            elif training_data_loader==TranPara.training_data_loader_C:
                
                model.Encoder.requires_grad = False
                model.Decoder1.requires_grad = False
                model.Decoder2.requires_grad = True
                model.Decoder3.requires_grad = False                
                
            elif training_data_loader==TranPara.training_data_loader_S:
                
                model.Encoder.requires_grad = False
                model.Decoder1.requires_grad = False
                model.Decoder2.requires_grad = False
                model.Decoder3.requires_grad = True                

                
                
            #######################
            if training_data_loader==TranPara.training_data_loader_All:
                #pdb.set_trace()
                prediction1,prediction2,prediction3 =model(input)
                
                if epoch ==0:
                    loss=(crossEntropy(prediction1,target)+crossEntropy(prediction2,target)+crossEntropy(prediction3,target))/3.

                else:

                    losses= crossEntropy2(prediction1,target)

                   



                    for view in range(len(batch[-1])):
                        if batch[-1][view]=="C":
                            losses[view,:,:,:]=crossEntropy2(prediction2[view,:,:,:],target[view,:,:,:])

                        elif batch[-1][view]=="S":
                            losses[view,:,:,:]=crossEntropy2(prediction3[view,:,:,:],target[view,:,:,:])



                    loss=torch.mean(losses)

                
                
                
                
                
                del prediction2
                del prediction3
                torch.cuda.empty_cache()
            
            elif training_data_loader==TranPara.training_data_loader_A:
            
                prediction1=model(input)[0]
                loss= crossEntropy(prediction1,target)
            elif training_data_loader==TranPara.training_data_loader_C:
            
                prediction1=model(input)[1]                
                loss= crossEntropy(prediction1,target)
            elif training_data_loader==TranPara.training_data_loader_S:
            
                prediction1=model(input)[2]
                loss= crossEntropy(prediction1,target)
                
                
         
            del target       
            del input             
            torch.cuda.empty_cache()


            trainLoss += loss.item()
            model.zero_grad()
            loss.backward()
            optimizer.step()

            np.random.seed(TranPara.initial_seed + epoch)

        print("                                ")        
        print(TrainedParts[it]+' was trained')        
        print("                                ") 
        it+=1
        
        
        
        
    print("validation starts")
    
    DataloadersCell=[TranPara.val_data_loader_All,TranPara.val_data_loader_A,TranPara.val_data_loader_C, TranPara.val_data_loader_S]
    TrainedParts=["encoder ",'decoder A','decoder C','decoder S' ]

    with torch.no_grad():
        model.eval()
        
        for val_data_loader in DataloadersCell:

            for _, batch in enumerate(val_data_loader):

                

                input=batch[0].type(torch.FloatTensor).to(TranPara.device)
                target=batch[1].type(torch.FloatTensor).to(TranPara.device)
                

            
                if val_data_loader==TranPara.val_data_loader_All:
                    prediction1,prediction2,prediction3 =model(input)
                    losses= crossEntropy2(prediction1,target)

                    for view in range(len(batch[-1])):
                        if batch[-1][view]=="C":
                            losses[view,:,:,:]=crossEntropy2(prediction2[view,:,:,:],target[view,:,:,:])

                        elif batch[-1][view]=="S":
                            losses[view,:,:,:]=crossEntropy2(prediction3[view,:,:,:],target[view,:,:,:])



                    loss=torch.mean(losses)

                    del prediction2
                    del prediction3
                    torch.cuda.empty_cache()

                elif val_data_loader==TranPara.val_data_loader_A:

                    prediction1=model(input)[0]
                    loss= crossEntropy(prediction1,target)
                elif val_data_loader==TranPara.val_data_loader_C:

                    prediction1=model(input)[1]                
                    loss= crossEntropy(prediction1,target)
                elif val_data_loader==TranPara.val_data_loader_S:

                    prediction1=model(input)[2]
                    loss= crossEntropy(prediction1,target)
               
            
                valLoss +=loss.item()
            
            ###########################################################

            #ImagesToSave=torch.cat((prediction1,input,target ),1)
            #torchvision.utils.save_image(ImagesToSave,os.path.join(opt.PathToTrainingFolder,"Estimations_"+str(epoch) +'.jpg'),normalize=False)




    trainLoss=trainLoss/(len(TranPara.training_data_loader_All)+ len(TranPara.training_data_loader_A)+len(TranPara.training_data_loader_C)+len(TranPara.training_data_loader_S))
    valLoss=valLoss/(len(TranPara.val_data_loader_All)+len(TranPara.val_data_loader_A)+ len(TranPara.val_data_loader_C)+len(TranPara.val_data_loader_S))



    print("===> Epoch {} Complete: Avg. Unet Loss: {:.4f}".format(epoch, trainLoss ))
    print("===> Epoch {} Complete: Avg. Unet Validation Loss: {:.4f}".format(epoch, valLoss ))

    return model, optimizer,trainLoss, valLoss


def test(TranPara,opt,model,device,dice,hfDistance,ASSD,Jaccard,CosineSimilarities,epsilon=0):
    model.eval()
    totalloss = 0
    ImgNo=0

    for batch in TranPara.testing_data_loader_S:
        batch[0]=batch[0].permute(1,0,2,3)
        batch[1]=batch[1].permute(1,0,2,3)
        
        #pdb.set_trace()
        ImgNo=ImgNo+1
        CosineSimilarity=[]
        
        #print(batch[0].shape)
        model.eval()

        batch[1]=batch[1].to(device)
        batch[0]=batch[0].type(torch.FloatTensor).to(device)

        b,c,w,h=batch[0].shape
        if w%32 !=0 or h%32 !=0:
            batch0_AA=torch.zeros(b,c,opt.InputSize,opt.InputSize).to(device)
            batch0_AA[:,:,:w,:h]=batch[0].clone()
            batch1_AA=torch.zeros(b,c,opt.InputSize,opt.InputSize).to(device)
            batch1_AA[:,:,:w,:h]=batch[1].clone()                

        else:
            batch0_AA=batch[0].clone()
            batch1_AA=batch[1].clone()

            
        
            
        with torch.no_grad(): 
            vect_S= PredictVolume(opt,batch0_AA,model,ImgNo,TranPara.device,DecoderNo=3)
            


        ################################ reslicing along the axial view 


        b,c,w,h=batch[0].permute(2,1,0,3).shape
        if w%32 !=0 or h%32 !=0:
            batch0_AA=torch.zeros(b,c,opt.InputSize,opt.InputSize).to(device)
            batch0_AA[:,:,:w,:h]=batch[0].permute(2,1,0,3).clone()
            batch1_AA=torch.zeros(b,c,opt.InputSize,opt.InputSize).to(device)
            batch1_AA[:,:,:w,:h]=batch[1].permute(2,1,0,3).clone()                    

        else:
            batch0_AA=batch[0].permute(2,1,0,3).clone()
            batch1_AA=batch[1].permute(2,1,0,3).clone()


        
        with torch.no_grad(): 
            vect_A= PredictVolume(opt,batch0_AA.permute(2,1,0,3),model,ImgNo,TranPara.device,DecoderNo=1)


        ################################ reslicing along the coronal view 

            b,c,w,h=batch[0].permute(3,1,0,2).shape
            if w%32 !=0 or h%32 !=0:
                batch0_AA=torch.zeros(b,c,opt.InputSize,opt.InputSize).to(device)
                batch0_AA[:,:,:w,:h]=batch[0].permute(3,1,0,2).clone()
                batch1_AA=torch.zeros(b,c,opt.InputSize,opt.InputSize).to(device)
                batch1_AA[:,:,:w,:h]=batch[1].permute(3,1,0,2).clone()                     

            else:
                batch0_AA=batch[0].permute(3,1,0,2).clone()
                batch1_AA=batch[1].permute(3,1,0,2).clone()
                

        
        vect_C= PredictVolume(opt,batch0_AA.permute(3,1,0,2),model,ImgNo,TranPara.device,DecoderNo=2) 
                


        #######################  calculate cosine similarities between decoder outputs ################       
        batch[1]=batch[1].cpu()
        batch[0]=batch[0].cpu()

        del batch0_AA            
        torch.cuda.empty_cache()
        
        
        CosineSimilarity=np.max([np.array([torch.nn.functional.cosine_similarity(vect_S.flatten(),
                vect_A.permute(2,1,0,3).flatten(),dim=0).cpu(),torch.nn.functional.cosine_similarity(
            vect_S.flatten(),vect_C.permute(2,1,3,0).flatten(),dim=0).cpu(),
        torch.nn.functional.cosine_similarity(vect_C.permute(2,1,3,0).flatten(),vect_A.permute(2,1,0,3).flatten(),dim=0).cpu()])])
        CosineSimilarities.append(CosineSimilarity)
        


        #######################  calculate mean decoder outputs ####################### 

        vect=torch.zeros_like(vect_S)
        for nnn in range(vect_S.shape[0]):
            imS=TF.to_tensor((TF.to_pil_image(vect_S[nnn,0,:,:].cpu())).filter(PIL.ImageFilter.GaussianBlur(2)))
            imC=TF.to_tensor((TF.to_pil_image(vect_C.permute(2,1,3,0)[nnn,0,:,:].cpu())).filter(PIL.ImageFilter.GaussianBlur(2)))
            imA=TF.to_tensor((TF.to_pil_image(vect_A.permute(2,1,0,3)[nnn,0,:,:].cpu())).filter(PIL.ImageFilter.GaussianBlur(2)))

            vect[nnn,0,:,:]=  (imS+imC+imA)/3.

        #######################


        predictions,dice,Jaccard,hfDistance,ASSD=CalculatePerformanceMetrics(opt,vect,batch,
                                                        dice,Jaccard,hfDistance,ASSD,ImgNo)
        print('CosineSimilarity',CosineSimilarity)
        #np.savez(os.path.join(opt.PathToSaveTrainedModels,str(ImgNo)+'.npz'), name1=vect.cpu().numpy(), name2=batch[1].cpu().numpy(),name3=CosineSimilarity)
        torchvision.utils.save_image( torch.cat((vect.cpu(),batch[0],batch[1]),1),os.path.join(opt.PathToTrainingFolder,"TestEstimations_"+str(ImgNo) +'.jpg'),normalize=False)

    return dice,hfDistance,ASSD,Jaccard,CosineSimilarities  

def testWithAdversarialNoise(TranPara,opt,model,device,dice,hfDistance,ASSD,Jaccard,CosineSimilarities,epsilon=0):
    model.eval()
    totalloss = 0
    ImgNo=0

    for batch in TranPara.testing_data_loader_S:
        batch[0]=batch[0].permute(1,0,2,3)
        batch[1]=batch[1].permute(1,0,2,3)
        
        #pdb.set_trace()
        ImgNo=ImgNo+1
        CosineSimilarity=[]
        
        #print(batch[0].shape)
        model.eval()

        batch[1]=batch[1].to(device)
        batch[0]=batch[0].type(torch.FloatTensor).to(device)

        b,c,w,h=batch[0].shape
        if w%32 !=0 or h%32 !=0:
            batch0_AA=torch.zeros(b,c,opt.InputSize,opt.InputSize).to(device)
            batch0_AA[:,:,:w,:h]=batch[0].clone()
            batch1_AA=torch.zeros(b,c,opt.InputSize,opt.InputSize).to(device)
            batch1_AA[:,:,:w,:h]=batch[1].clone()                

        else:
            batch0_AA=batch[0].clone()
            batch1_AA=batch[1].clone()

            
          
        ################# add adversarial noise to input images ###############


        perturbed_data=NoisyInputGeneration(TranPara,opt,model, batch0_AA,batch1_AA,epsilon,output=2,w=w,h=h)


        with torch.no_grad(): 
            vect_S= PredictVolume(opt,perturbed_data.clone(),model,ImgNo,TranPara.device,DecoderNo=3) 
            
        del batch1_AA
        del batch0_AA
        torch.cuda.empty_cache()


        ################################ reslicing along the axial view 


        b,c,w,h=batch[0].permute(2,1,0,3).shape
        if w%32 !=0 or h%32 !=0:
            batch0_AA=torch.zeros(b,c,opt.InputSize,opt.InputSize).to(device)
            batch0_AA[:,:,:w,:h]=batch[0].permute(2,1,0,3).clone()
            batch1_AA=torch.zeros(b,c,opt.InputSize,opt.InputSize).to(device)
            batch1_AA[:,:,:w,:h]=batch[1].permute(2,1,0,3).clone()                    

        else:
            batch0_AA=batch[0].permute(2,1,0,3).clone()
            batch1_AA=batch[1].permute(2,1,0,3).clone()


            
  
        if opt.AttackNumber==3:               

            #######################  attack to axial view ################# 

            perturbed_data=NoisyInputGeneration(TranPara,opt,model, batch0_AA,batch1_AA,epsilon,output=0,w=w,h=h)

            with torch.no_grad(): 

                vect_A= PredictVolume(opt,perturbed_data.detach(),model,ImgNo,TranPara.device,DecoderNo=1)

            del perturbed_data  
            del batch1_AA
            del batch0_AA
            torch.cuda.empty_cache()            
        ################################ reslicing along the coronal view 

            b,c,w,h=batch[0].permute(3,1,0,2).shape
            if w%32 !=0 or h%32 !=0:
                batch0_AA=torch.zeros(b,c,opt.InputSize,opt.InputSize).to(device)
                batch0_AA[:,:,:w,:h]=batch[0].permute(3,1,0,2).clone()
                batch1_AA=torch.zeros(b,c,opt.InputSize,opt.InputSize).to(device)
                batch1_AA[:,:,:w,:h]=batch[1].permute(3,1,0,2).clone()                     

            else:
                batch0_AA=batch[0].permute(3,1,0,2).clone()
                batch1_AA=batch[1].permute(3,1,0,2).clone()
                


                
            #######################  attack to coronal view ################# 
        if opt.AttackNumber==3:               

            perturbed_data=NoisyInputGeneration(TranPara,opt,model, batch0_AA,batch1_AA,epsilon,output=1,w=w,h=h)


            with torch.no_grad(): 
                vect_C= PredictVolume(opt,perturbed_data.detach(),model,ImgNo,TranPara.device,DecoderNo=2) 


            del perturbed_data        
            del batch1_AA
            del batch0_AA
            torch.cuda.empty_cache()
        
        if opt.AttackNumber==1:
            ### in the case of single attack, generate outputs for other decoders, using the original images
            with torch.no_grad(): 
                vect_A= PredictVolume(opt,perturbed_data.clone().detach().permute(2,1,0,3),model,ImgNo,TranPara.device,DecoderNo=1)
                vect_C= PredictVolume(opt,perturbed_data.clone().detach().permute(3,1,0,2),model,ImgNo,TranPara.device,DecoderNo=2) 




        #######################  calculate cosine similarities between decoder outputs ################       
        batch[1]=batch[1].cpu()
        batch[0]=batch[0].cpu()

        
        CosineSimilarity=np.max([np.array([torch.nn.functional.cosine_similarity(vect_S.flatten(),
                vect_A.permute(2,1,0,3).flatten(),dim=0).cpu(),torch.nn.functional.cosine_similarity(
            vect_S.flatten(),vect_C.permute(2,1,3,0).flatten(),dim=0).cpu(),
        torch.nn.functional.cosine_similarity(vect_C.permute(2,1,3,0).flatten(),vect_A.permute(2,1,0,3).flatten(),dim=0).cpu()])])
        CosineSimilarities.append(CosineSimilarity)
        


        #######################  calculate mean decoder outputs ####################### 

        vect=torch.zeros_like(vect_S)
        for nnn in range(vect_S.shape[0]):
            imS=TF.to_tensor((TF.to_pil_image(vect_S[nnn,0,:,:].cpu())).filter(PIL.ImageFilter.GaussianBlur(2)))
            imC=TF.to_tensor((TF.to_pil_image(vect_C.permute(2,1,3,0)[nnn,0,:,:].cpu())).filter(PIL.ImageFilter.GaussianBlur(2)))
            imA=TF.to_tensor((TF.to_pil_image(vect_A.permute(2,1,0,3)[nnn,0,:,:].cpu())).filter(PIL.ImageFilter.GaussianBlur(2)))

            vect[nnn,0,:,:]=  (imS+imC+imA)/3.

        #######################


        '''
        if opt.AttackNumber==3:

            perturbed_data=perturbed_data.permute(2,1,3,0) ## transform perturbed data of coronal view into the sagittal view
        '''


        predictions,dice,Jaccard,hfDistance,ASSD=CalculatePerformanceMetrics(opt,vect,batch,
                                                        dice,Jaccard,hfDistance,ASSD,ImgNo)
        print('CosineSimilarity',CosineSimilarity)
        #np.savez(os.path.join(opt.PathToSaveTrainedModels,str(ImgNo)+'.npz'), name1=vect.cpu().numpy(), name2=batch[1].cpu().numpy(),name3=CosineSimilarity)
        torchvision.utils.save_image( torch.cat((vect.cpu(),batch[0],batch[1]),1),os.path.join(opt.PathToTrainingFolder,"TestEstimations_"+str(ImgNo) +'.jpg'),normalize=False)

    return dice,hfDistance,ASSD,Jaccard,CosineSimilarities  
    
    
def NoisyInputGeneration(TranPara, opt,model, batch0_AA,batch1_AA,epsilon,output,w,h ):

    perturbed_data=torch.zeros_like(batch0_AA).to(TranPara.device)

    if  opt.AttackType=='Rician' :
        perturbed_data= AddGaussianNoise(batch0_AA,TranPara.device,std=epsilon,mean=0)

    else:

        for k in range(batch0_AA.shape[0]):
            input =batch0_AA[k,None,:,:,:].to(TranPara.device)
            target=batch1_AA[k,None,:,:,:]
            if opt.AttackType=='FGSM':

                perturbed_data[k,:,:,:]=fgsm_attack(input,target,model,epsilon)[0,:,:,:].detach()
            elif opt.AttackType=='IterativeFGSM':
                perturbed_data[k,:,:,:]=IFGSM(input, target,model,TranPara.device,eps=epsilon,output=output)[0,:,:,:].detach()

                    
                
    return perturbed_data[:,:,:w,:h]
    
def PredictVolume(opt,inputN,model,ImgNo,device,DecoderNo=1,NoSigmoid=False):
    #pdb.set_trace()
    b,c,w,h=inputN.shape
    
    
    #print(inputN.shape)
    
    
    if w%32 !=0 or h%32 !=0:
        input=torch.zeros((b,c,opt.InputSize,opt.InputSize)).to(device)
        input[:,:,:w,:h]=inputN
    else:
        input=inputN
     
    del inputN
    torch.cuda.empty_cache()
     
    ChunkSize=10   
    for ChunckNo in range(ChunkSize,input.shape[0]+ChunkSize,ChunkSize):

        if  ChunckNo==ChunkSize:  

            if DecoderNo==1:
                vect=model(input[:ChunckNo,:,:,:])[0] 
            elif DecoderNo==2:
                vect=model(input[:ChunckNo,:,:,:])[1] 
            elif DecoderNo==3:
                vect=model(input[:ChunckNo,:,:,:])[2] 
            if NoSigmoid==False:
                vect=torch.sigmoid(vect)
        else:


            if NoSigmoid==False:
                if DecoderNo==1:          
                    vect=torch.cat((vect,torch.sigmoid(model(input[ChunckNo-ChunkSize:ChunckNo,:,:,:])[0])),0)
                elif DecoderNo==2:
                    vect=torch.cat((vect,torch.sigmoid(model(input[ChunckNo-ChunkSize:ChunckNo,:,:,:])[1])),0)
                elif DecoderNo==3:
                    vect=torch.cat((vect,torch.sigmoid(model(input[ChunckNo-ChunkSize:ChunckNo,:,:,:])[2])),0)  
            else:
                if DecoderNo==1:          
                    vect=torch.cat((vect,model(input[ChunckNo-ChunkSize:ChunckNo,:,:,:])[0]),0)
                elif DecoderNo==2:
                    vect=torch.cat((vect,model(input[ChunckNo-ChunkSize:ChunckNo,:,:,:])[1]),0)
                elif DecoderNo==3:
                    vect=torch.cat((vect,model(input[ChunckNo-ChunkSize:ChunckNo,:,:,:])[2]),0)  

              
    if w%32 !=0 or h%32 !=0:
        vectR = vect[:,:,:w,:h]
    else:
        vectR = vect
    del vect
    torch.cuda.empty_cache()        
      
    return vectR
    
def CalculatePerformanceMetrics(opt,vect,batch,dice,Jaccard,hfDistance,ASSD,ImgNo):
    NuOfSlices=vect.shape[0]
    w=vect.shape[2]
    
    vectBinarised=np.reshape(np.array(vect.cpu().numpy()),(-1))
    GT=batch[1]

    thresh=threshold_otsu(vectBinarised)
    vectBinarised=np.int8(vectBinarised>thresh)
    predictions=np.reshape(vectBinarised,(NuOfSlices,w,w))


    vectBinarised=np.reshape(predictions,(-1))
    GTvect=np.reshape(GT.numpy(),(-1))
    

    dice.append(dc(GTvect,vectBinarised))
    Jaccard.append(jc(GTvect,vectBinarised))
    


    if opt.Dataset=='Atrium2013':

        voxelspacing=(2.7,1.25,1.25)             
                
    elif opt.Dataset=='Atrium2018':
        voxelspacing=(0.625,0.625*2,0.625*2)
        
        
    hfDistance.append(hd(predictions,np.reshape(GTvect,(NuOfSlices,w,w)),voxelspacing=voxelspacing))
    ASSD.append(assd(predictions,np.reshape(GTvect,(NuOfSlices,w,w)),voxelspacing=voxelspacing))



    print("dice",dice[-1])
    print("Jaccard",Jaccard[-1])
    print("assd",ASSD[-1])
    print("hd",hfDistance[-1])

    return predictions,dice,Jaccard,hfDistance,ASSD






# FGSM attack code
def fgsm_attack(image, labels,model,epsilon):
    lossFunc = nn.BCEWithLogitsLoss()
    
    image.requires_grad=True 
    loss=lossFunc(model(image)[-1],labels)
    model.zero_grad()
    loss.backward()
    data_grad=image.grad.data

                
                
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image



def  IFGSM(images, labels,model,device,steps=5,alpha=0.01,eps=8/255.,output=0):

    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)

    loss = nn.BCEWithLogitsLoss()

    ori_images = images.clone().detach()

    for _ in range(steps):
        
        images.requires_grad = True
        outputs = model(images)
        cost = loss(outputs[output], labels)

        # Update adversarial images
        grad = torch.autograd.grad(cost, images,
                                   retain_graph=False, create_graph=False)[0]
      
        
        adv_images = images + alpha*grad.sign()
        a = torch.clamp(ori_images - eps, min=0)
        b = (adv_images >= a).float()*adv_images \
            + (adv_images < a).float()*a
        c = (b > ori_images+eps).float()*(ori_images+eps) \
            + (b <= ori_images + eps).float()*b
        images = torch.clamp(c, max=1).detach() 


    return images    