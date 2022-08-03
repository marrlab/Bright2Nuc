'''
Bright2Nuc
2D and 3D UNet models
Written by Dominik Waibel
The original publication is: arXiv:1505.04597
'''

from keras.models import *
from keras.layers import Input, Conv2D, Conv3D, LeakyReLU, BatchNormalization, Dropout, MaxPooling2D, concatenate, MaxPooling3D, UpSampling2D, UpSampling3D, Conv3DTranspose
from keras.optimizers import Adam, SGD, RMSprop
import numpy as np
from keras.losses import *
import logging
import os

class UNetBuilder(object):
   '''
   Builds a 2D or 3D U-Net for image segmentation following the publication from Ronneberger et al.:
   https://arxiv.org/abs/1505.04597
   The layers consist of a convolution followed by a LeackyReLu activation and a Batch Normalization
   Dropout is integrated with a dropout rate of 20% during training to increase generalizability
   Padding has bee set to "same"
   '''

   def unet3D_3SliceInput(pretrained_weights, loss_function, Dropout_On, base_n_filters=16):
      print("started UNet")
      inputs = Input(shape=(3, None, None, 1))
      conv1 = Conv3D(base_n_filters, (2,3,3), padding='same', kernel_initializer='he_normal')(inputs)
      conv1 = BatchNormalization()(conv1)
      conv1 = LeakyReLU(alpha=0.2)(conv1)
      print("finished 1. convolution")
      conv1 = Conv3D(base_n_filters, (2,3,3), padding='same', kernel_initializer='he_normal')(conv1)
      conv1 = BatchNormalization()(conv1)
      conv1 = LeakyReLU(alpha=0.2)(conv1)
      drop1 = Dropout(rate=0.3)(conv1, training = Dropout_On)
      pool1 = MaxPooling3D(pool_size=(1, 2, 2))(drop1)

      conv2 = Conv3D(base_n_filters * 2, (2,3,3), padding='same', kernel_initializer='he_normal')(pool1)
      conv2 = BatchNormalization()(conv2)
      conv2 = LeakyReLU(alpha=0.2)(conv2)
      conv2 = Conv3D(base_n_filters * 2, (2,3,3), padding='same', kernel_initializer='he_normal')(conv2)
      conv2 = BatchNormalization()(conv2)
      conv2 = LeakyReLU(alpha=0.2)(conv2)
      drop2 = Dropout(rate=0.3)(conv2, training = Dropout_On)
      pool2 = MaxPooling3D(pool_size=(1, 2, 2))(drop2)

      conv3 = Conv3D(base_n_filters * 4, (2,3,3), padding='same', kernel_initializer='he_normal')(pool2)
      conv3 = BatchNormalization()(conv3)
      conv3 = LeakyReLU(alpha=0.2)(conv3)
      conv3 = Conv3D(base_n_filters * 4, (2,3,3), padding='same', kernel_initializer='he_normal')(conv3)
      conv3 = BatchNormalization()(conv3)
      conv3 = LeakyReLU(alpha=0.2)(conv3)
      drop3 = Dropout(rate=0.3)(conv3, training = Dropout_On)
      pool3 = MaxPooling3D(pool_size=(1, 2, 2))(drop3)

      conv4 = Conv3D(base_n_filters * 8, (2,3,3), padding='same', kernel_initializer='he_normal')(pool3)
      conv4 = BatchNormalization()(conv4)
      conv4 = LeakyReLU(alpha=0.2)(conv4)
      conv4 = Conv3D(base_n_filters * 8, (2,3,3), padding='same', kernel_initializer='he_normal')(conv4)
      conv4 = BatchNormalization()(conv4)
      conv4 = LeakyReLU(alpha=0.2)(conv4)
      drop4 = Dropout(rate=0.3)(conv4, training = Dropout_On)
      pool4 = MaxPooling3D(pool_size=(1, 2, 2))(drop4)

      conv5 = Conv3D(base_n_filters * 16, (2,3,3), padding='same', kernel_initializer='he_normal')(pool4)
      conv5 = BatchNormalization()(conv5)
      conv5 = LeakyReLU(alpha=0.2)(conv5)
      conv5 = Conv3D(base_n_filters * 16, (2,3,3), padding='same', kernel_initializer='he_normal')(conv5)
      conv5 = BatchNormalization()(conv5)
      conv5 = LeakyReLU(alpha=0.2)(conv5)
      drop5 = Dropout(rate=0.3)(conv5, training = Dropout_On)
      up6 = Conv3DTranspose(base_n_filters * 8, (2, 3, 3), strides=(1, 2, 2), activation='relu', padding='same')(drop5)


      merge6 = concatenate([drop4, up6], axis=4)
      conv6 = Conv3D(base_n_filters * 8, (2,3,3), activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
      conv6 = BatchNormalization()(conv6)
      conv6 = LeakyReLU(alpha=0.1)(conv6)
      conv6 = Conv3D(base_n_filters * 8, (2,3,3), activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
      conv6 = BatchNormalization()(conv6)
      conv6 = LeakyReLU(alpha=0.2)(conv6)

      up7 = Conv3DTranspose(base_n_filters * 4, (2, 3, 3), strides=(1, 2, 2), activation='relu', padding='same')(
         conv6)
         
      merge7 = concatenate([drop3, up7], axis=4)
      conv7 = Conv3D(base_n_filters * 4, (2,3,3), padding='same', kernel_initializer='he_normal')(merge7)
      conv7 = BatchNormalization()(conv7)
      conv7 = LeakyReLU(alpha=0.1)(conv7)
      conv7 = Conv3D(base_n_filters * 4, (2,3,3), padding='same', kernel_initializer='he_normal')(conv7)
      conv7 = BatchNormalization()(conv7)
      conv7 = LeakyReLU(alpha=0.1)(conv7)

      up8 = Conv3DTranspose(base_n_filters * 2, (2, 3, 3), strides=(1, 2, 2), activation='relu', padding='same')(
         conv7)


      merge8 = concatenate([drop2, up8], axis=4)
      conv8 = Conv3D(base_n_filters * 2, (2,3,3), padding='same', kernel_initializer='he_normal')(merge8)
      conv8 = BatchNormalization()(conv8)
      conv8 = LeakyReLU(alpha=0.1)(conv8)
      conv8 = Conv3D(base_n_filters * 2, (2,3,3), padding='same', kernel_initializer='he_normal')(conv8)
      conv8 = BatchNormalization()(conv8)
      conv8 = LeakyReLU(alpha=0.1)(conv8)

      up9 = Conv3DTranspose(base_n_filters, (2, 3, 3), strides=(1, 2, 2), activation='relu', padding='same')(
         conv8)


      merge9 = concatenate([drop1, up9], axis=4)
      conv9 = Conv3D(base_n_filters, (2,3,3), padding='same', kernel_initializer='he_normal')(merge9)
      conv9 = BatchNormalization()(conv9)
      conv9 = LeakyReLU(alpha=0.1)(conv9)
      conv9 = Conv3D(base_n_filters, (2,3,3), padding='same', kernel_initializer='he_normal')(conv9)
      conv9 = BatchNormalization()(conv9)
      conv9 = LeakyReLU(alpha=0.1)(conv9)

      conv9 = Conv3D(3, 3, padding='same', kernel_initializer='he_normal')(conv9)
      conv9 = BatchNormalization()(conv9)
      conv9 = LeakyReLU(alpha=0.1)(conv9)
      conv10 = Conv3D(1, 1, activation='sigmoid')(conv9)  # Changed activiation from Relu to linear

      model2D = Model(inputs=inputs, outputs=conv10)
      print("shape input UNet:", np.shape(inputs))
      print("shape output UNet:", np.shape(conv10))
      model2D.compile(optimizer="adam", loss=loss_function, metrics=['mse'])

      if (pretrained_weights):
         if os.path.isfile(pretrained_weights):
            model2D.load_weights(pretrained_weights, by_name=True, skip_mismatch=True)
            for layer, pre in zip(model2D.layers, pretrained_weights):
                weights = layer.get_weights()
                if weights:
                    if np.array_equal(weights[0], pre[0]):
                        print('not loaded', layer.name)
                    else:
                        print('loaded', layer.name)
      return model2D
