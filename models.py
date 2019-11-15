from keras.layers.core import Activation
from keras.layers import Input
from keras.layers.convolutional import Conv2D, Conv2DTranspose,Convolution2D
from keras.models import Model
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers import add,concatenate,Add,Concatenate
from keras.optimizers import Adam
from keras.losses import mse
from keras.layers import Dense
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras import initializers 
import numpy as np

class Generator(object):
    def __init__(self, noise_shape):
        
        self.noise_shape = noise_shape
        self.noise_shape = (None,None,noise_shape[2])
    def generator(self):
        
        #gen_input = Input(shape = self.noise_shape)

        
        init_MS=Input((None,None,self.noise_shape[2]-1))
        init_PAN=Input((None,None,1))
        filter_size=3
        x1=Convolution2D(32, (filter_size, filter_size),padding='same',activation='relu',kernel_initializer=initializers.random_normal(stddev=np.sqrt(2/9)))(init_MS)
        x1=Convolution2D(64, (filter_size, filter_size),padding='same',activation='relu',kernel_initializer=initializers.random_normal(stddev=np.sqrt(2/9/64)))(x1)
        x2=Convolution2D(32, (filter_size, filter_size),padding='same',activation='relu',kernel_initializer=initializers.random_normal(stddev=np.sqrt(2/9)))(init_PAN)
        x2=Convolution2D(64, (filter_size, filter_size),padding='same',activation='relu',kernel_initializer=initializers.random_normal(stddev=np.sqrt(2/9/64)))(x2)
        x = Concatenate()([x1,x2])
#fused_features = x

        for i in range(10):
            x=Convolution2D(64, (filter_size, filter_size), padding='same', activation='relu',kernel_initializer=initializers.random_normal(stddev=np.sqrt(2/9/64)))(x)
        di= Convolution2D(self.noise_shape[2]-1, (filter_size, filter_size),padding='same',kernel_initializer=initializers.random_normal(stddev=np.sqrt(2/9/64)))(x)

        output= Add()([di,init_MS])

        generator_model = Model(inputs=[init_MS,init_PAN], outputs=output)
        
        return generator_model
def discriminator_block(model, filters, kernel_size, strides):
    
    model = Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = "same")(model)
    model = LeakyReLU(alpha = 0.2)(model)
    
    return model
  
class Discriminator(object):

    def __init__(self, image_shape):
        
        self.image_shape = image_shape
    
    def discriminator(self):
        
        dis_input = Input(shape = self.image_shape)
        
        model = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = "same")(dis_input)
        model = LeakyReLU(alpha = 0.2)(model)
        
        model = discriminator_block(model, 64, 3, 2)
        model = discriminator_block(model, 128, 3, 1)
        model = discriminator_block(model, 128, 3, 2)
        model = discriminator_block(model, 256, 3, 1)
        model = discriminator_block(model, 256, 3, 2)
        model = discriminator_block(model, 512, 3, 1)
        model = discriminator_block(model, 512, 3, 2)
        
        model = Flatten()(model)
        model = Dense(1024)(model)
        model = LeakyReLU(alpha = 0.2)(model)
       
        model = Dense(1)(model)
        model = Activation('sigmoid')(model) 
        
        discriminator_model = Model(inputs = dis_input, outputs = model)
        
        return discriminator_model
    
def get_gan_network(discriminator, shape, generator, optimizer):
    discriminator.trainable = False
    gan_input_ms = Input(shape=(shape[0],shape[1],shape[2]-1))
    gan_input_pan = Input(shape=(shape[0],shape[1],1))
    x = generator([gan_input_ms,gan_input_pan])
    gan_output = discriminator(x)
    gan = Model(inputs=[gan_input_ms,gan_input_pan], outputs=[x,gan_output])
    gan.compile(loss=["mse","binary_crossentropy"], loss_weights=[1., 1e-3], optimizer=optimizer)

    return gan

