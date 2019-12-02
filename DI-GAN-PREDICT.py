#!/usr/bin/env python
# coding: utf-8

# In[1]:



# coding: utf-8

# In[1]:


from keras.layers.core import Activation
from keras.layers import Input
from keras.layers.convolutional import Conv2D, Conv2DTranspose,Convolution2D
from keras.models import Model
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers import add,concatenate,Add,Concatenate
import scipy.io as sio
from keras.optimizers import Adam
from keras.losses import mse
from keras.layers import Dense
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras import initializers 
import numpy as np
import scipy.io as sio
from tqdm import tqdm
from sklearn.model_selection import train_test_split





from dataPrepare import downsampling_images,interp,prepare_training_data,normalize,denormalize
import sys

mode = sys.argv[1]
data_directory = sys.argv[2]


inputImg=sio.loadmat(data_directory)
I_MS = np.array(inputImg['I_MS'],dtype='double').transpose(2,0,1) 
I_PAN = np.array(inputImg['I_PAN'],dtype='double')
sensor = np.array(inputImg['sensor'],dtype='str')[0]

L=11
max_value= 2**L

(channel,l,c)=I_MS.shape
(l_pan,c_pan)=I_PAN.shape
del(inputImg)

print()


if (mode=='reduced'):
    (I_IN_MS,I_IN_PAN) = downsampling_images(I_MS,I_PAN)
    I_IN_PAN = np.expand_dims(I_IN_PAN,axis=0)
else: 
    I_IN_MS = interp(I_MS)
    I_IN_PAN = np.expand_dims(I_PAN,axis=0)
I_IN_MS = normalize(I_IN_MS,max_value)
I_IN_PAN = normalize(I_IN_PAN,max_value)



from models import Generator





image_shape_gen=(128,128,channel+1)

model_save_dir = 'Models/'
I_IN_MS = np.expand_dims(I_IN_MS.transpose(1,2,0),axis=0)
I_IN_PAN = np.expand_dims(I_IN_PAN,axis=3) 


generator = Generator(image_shape_gen).generator()





generator.load_weights(model_save_dir + sensor+'_gen_model.h5')



I_fused = generator.predict([I_IN_MS,I_IN_PAN])
I_fused = I_fused[0,:]
I_fused = denormalize(I_fused,max_value)





result ={'I_fused':I_fused,'sensor':sensor}
sio.savemat('QB_fused_'+mode,result)







