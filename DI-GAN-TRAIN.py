
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


# In[24]:


from dataPrepare import downsampling_images,interp,prepare_training_data,normalize,denormalize


# In[19]:


## data_load
sensor = 'QB'
# Change with dataset directory
data_directory='training/'+sensor
inputImg=sio.loadmat(data_directory+' MS PAN.mat')
I_MS = np.array(inputImg['I_MS'],dtype='double').transpose(2,0,1) 
I_PAN = np.array(inputImg['I_PAN'],dtype='double')
L=11
max_value= 2**L
#max_g=2.5
(channel,l,c)=I_MS.shape
(l_pan,c_pan)=I_PAN.shape
del(inputImg)


# In[11]:


(I_MS_LR,I_PAN_LR) = downsampling_images(I_MS,I_PAN)
I_PAN_LR = np.expand_dims(I_PAN_LR,axis=0)
I_MS_LR = normalize(I_MS_LR,max_value)
I_PAN_LR = normalize(I_PAN_LR,max_value)
I_MS = normalize(I_MS,max_value)


# In[16]:


from tqdm import tqdm_notebook as tqdm


# In[17]:


from sklearn.model_selection import train_test_split


# In[18]:


#from sklearn.utils import shuffle
#X_train1,X_train2,Y_train=shuffle(X_train1,X_train2,Y_train,random_state=0)


# In[20]:


from models import Generator,Discriminator,get_gan_network


# In[23]:


image_shape_gen=(128,128,channel+1)
image_shape_dec=(128,128,channel)
batch_size=32
epochs = 500
model_save_dir = 'Models/'

(X_train1,X_train2,Y_train) = prepare_training_data(I_MS_LR,I_PAN_LR,I_MS) 
 

generator = Generator(image_shape_gen).generator()
discriminator = Discriminator(image_shape_dec).discriminator()

optimizer = Adam(lr=1e-5)
generator.compile(loss=mse, optimizer=optimizer)
discriminator.compile(loss="binary_crossentropy", metrics=['accuracy'],optimizer=optimizer)

gan = get_gan_network(discriminator, image_shape_gen, generator, optimizer)

loss_file = open(model_save_dir + 'losses.txt' , 'w+')
loss_file.close()
batch_count = int(X_train1.shape[0] / batch_size)


# # Train Only Generator

# In[35]:


MSE_loss=[]
generator_loss=[]
discriminator_history=[]


# In[36]:


min_val_loss= float("inf")
for e in range(1, 240+1):
        print ('-'*15, 'Epoch %d' % e, '-'*15)
        #X_train1,X_train2,Y_train = shuffle (X_train1,X_train2,Y_train)
        for i in tqdm(range(batch_count)):
            
            #rand_nums = np.random.randint(0, x_train_hr.shape[0], size=batch_size)
            rand_nums = np.random.randint(0,X_train1.shape[0], size=batch_size)
            image_batch_hr = Y_train[rand_nums]
            image_MS_lr = X_train1[rand_nums]
            image_PAN_lr = X_train2[rand_nums]
            generated_images_PanSharpened = generator.predict([image_MS_lr,image_PAN_lr])
            

            real_data_Y = np.ones(batch_size) #- np.random.random_sample(batch_size)*0.2
            fake_data_Y = np.zeros(batch_size)#np.random.random_sample(batch_size)*0.2
            
            discriminator.trainable = True
            
            d_loss_real = discriminator.train_on_batch(image_batch_hr, real_data_Y)
            d_loss_fake = discriminator.train_on_batch(generated_images_PanSharpened, fake_data_Y)
            discriminator_loss = 0.5 * np.add(d_loss_fake, d_loss_real)
            discriminator_history.append(discriminator_loss[0])
            #rand_nums = np.random.randint(0, x_train_hr.shape[0], size=batch_size)
            #image_batch_hr = x_train_hr[rand_nums]
            #image_batch_lr = x_train_lr[rand_nums]
            rand_nums = np.random.randint(0,X_train1.shape[0], size=batch_size)
            image_batch_hr = Y_train[rand_nums]
            image_MS_lr = X_train1[rand_nums]
            image_PAN_lr = X_train2[rand_nums]
            gan_Y = np.ones(batch_size) #- np.random.random_sample(batch_size)*0.2
            discriminator.trainable = False
            # je peux faire une modification image_batch_hr avec 3 dimension only qui est la moyenne des 3 dupliqué
            #for i in range(10):
            gan_loss = gan.train_on_batch(x=[image_MS_lr,image_PAN_lr], y=[image_batch_hr,gan_Y])
            generator_loss.append(gan_loss[0])
            MSE_loss.append(gan_loss[1])
            loss_file = open(model_save_dir + 'losses.txt' , 'a')
            loss_file.write('epoch%d_%d : gan_loss = %s ; discriminator_loss = %f\n' %(e,i, gan_loss, discriminator_loss[0]) )
            loss_file.close()
            
            
                #discriminator.save('Loss2/'  +sensor +'_dis_model_2stream_newest3by3.h5')
        
            #gan_loss = str(gan_loss)

         
        print("discriminator_loss_training : %f" % discriminator_loss[0])
        print("gan_loss_training :", gan_loss)
        val_loss = generator.evaluate([X_train1_val,X_train2_val],[Y_train_val],verbose=0)
        if (e % 50 == 0):
            generator.save(model_save_dir + sensor+'_gen_model__%d.h5' % e)
            discriminator.save(model_save_dir +sensor +'_dis__%d.h5' % e)

