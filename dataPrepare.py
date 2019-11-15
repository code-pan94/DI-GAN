from scipy.misc import imresize
import numpy as np
def downsampling_images(I_MS,I_PAN):
    I_PAN_LR = imresize(I_PAN,[int(I_PAN.shape[0]/4),int(I_PAN.shape[1]/4)],'bicubic',mode='F')
    channel = I_MS.shape[0]
    I_MS_LR = np.zeros((I_MS.shape[0],int(I_MS.shape[1]/4),int(I_MS.shape[2]/4)))
    I_MS_UP = np.zeros(I_MS.shape)
    for i in range(channel):
        I_MS_LR[i,:,:]= imresize(I_MS[i,:,:],1/4,'bicubic',mode='F')
        I_MS_UP[i,:,:]= imresize(I_MS_LR[i,:,:],[I_MS.shape[1],I_MS.shape[2]],'bicubic',mode='F')
    return (I_MS_UP,I_PAN_LR)
def interp(I_MS):
    I_MS_UP = np.zeros((I_MS.shape[0],I_MS.shape[1]*4,I_MS.shape[2]*4))
    for i in range(channel):
        I_MS_UP[i,:,:]= imresize(I_MS[i,:,:],[I_MS.shape[1]*4,I_MS.shape[2]*4],'bicubic',mode='F') 
    return (I_MS_UP)
def Convolution_opMS(Image, size, strides):
    start_x = 0
    start_y = 0
    end_x = Image.shape[1] - size[0]
    end_y = Image.shape[2] - size[1]

    n_rows = (end_x//strides[0]) + 1
    n_columns = (end_y//strides[1]) + 1
    small_images = []
    for i in range(n_rows):
        for j in range(n_columns):
            new_start_x = start_x+i*strides[0]
            new_start_y= start_y+j*strides[1]
            small_images.append(Image[:,new_start_x:new_start_x+size[0],new_start_y:new_start_y+size[1]])        
    small_images=np.asanyarray(small_images)   
    return small_images
def prepare_training_data(I_MS_LR,I_PAN_LR,I_MS):
    (X_train1)=Convolution_opMS(I_MS_LR,(128,128),(60,60))
    (X_train2)=Convolution_opMS(I_PAN_LR,(128,128),(60,60))
    (Y_train)=Convolution_opMS(I_MS,(128,128),(60,60))
    X_train1 = X_train1.transpose(0,2,3,1)
    X_train2 = X_train2.transpose(0,2,3,1)
    Y_train = Y_train.transpose(0,2,3,1)
    
    return (X_train1,X_train2,Y_train)
def normalize(X,max_value):
    #between -1 and 1
    #X = (X / ((max_value)/2)) - 1
    return (X/max_value)
def denormalize(X,max_value):
    #X = (X + 1) *  ((max_value)/2)
    return (X*max_value)