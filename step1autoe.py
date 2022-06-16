# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 12:41:45 2022

@author: AlessandroDeCarlo
"""

#%% loading libs
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model

#%% set seeds
tf.random.set_seed(12345)
np.random.seed(12345)
#%% test wheater GPU is present
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
    print('GPU device not found')
else:
    print('Found GPU at: {}'.format(device_name))
#%% image generator

IMAGE_SIZE = 256
BATCH_SIZE = 4#64

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(
    'training/',
    target_size=(IMAGE_SIZE,IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    shuffle = True,
    color_mode='rgb',
    class_mode ='input'
)

validation_generator = validation_datagen.flow_from_directory(
    'validation/',
    target_size=(IMAGE_SIZE,IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    shuffle = True,
    color_mode='rgb',
    class_mode ='input'   
)

#%% define base model

base_model = tf.keras.Sequential()

base_model.add(layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))
base_model.add(layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=2))
base_model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', strides=2))
base_model.add(layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same', strides=2))
base_model.add(layers.Conv2DTranspose(16, (3, 3), activation='relu', padding='same', strides=2))
base_model.add(layers.Conv2DTranspose(3, kernel_size=(3, 3), activation='relu', padding='same'))

base_model.summary()


base_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss=losses.MeanSquaredError())
#%% train base model

NEPOCHS = 20#1000

checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath='basemodel.hdf5', monitor='val_loss', verbose=1, 
                             save_best_only=True, mode='min', save_weights = False, save_format="tf")


history_base = base_model.fit(train_generator,
            epochs=NEPOCHS,
            shuffle=True,
            validation_data = validation_generator,
            callbacks=checkpoint)
#%% plot training results

vals = np.array(history_base.history["val_loss"])

best_epoch = vals.argmin()+1
best_v = vals.min()
allv = list()
allv.append(history_base.history["loss"])
allv.append(history_base.history["val_loss"])

sup_lim = np.array(allv).max()

plt.rcParams.update({'font.size': 18})
fig,(ax1,ax2) = plt.subplots(1,2,figsize=(16,16),sharex=True)
fig.suptitle('Training of Base model')
ax1.plot(np.arange(1,NEPOCHS+1),history_base.history["loss"], 
         label="Training Loss",linewidth=3)
ax1.plot(np.arange(1,NEPOCHS+1),history_base.history["val_loss"], 
         label="Validation Loss",linewidth=3)
ax1.vlines(best_epoch,0,sup_lim,linestyles='dotted',color='black')
ax1.plot(best_epoch,best_v,'og',label='Best',markersize=10)
ax1.grid(visible=True)
ax1.set_title('Natural Scale')
ax1.legend(framealpha=1)

ax2.semilogy(np.arange(1,NEPOCHS+1),history_base.history["loss"], 
             label="Training Loss",linewidth=3)
ax2.semilogy(np.arange(1,NEPOCHS+1),history_base.history["val_loss"], 
             label="Validation Loss",linewidth=3)
ax2.vlines(best_epoch,0,sup_lim,linestyles='dotted',color='black')
ax2.semilogy(best_epoch,best_v,'og',label='Best',markersize=10)
ax2.grid(visible=True)
ax2.set_title('Semilogy Scale')
ax2.legend(framealpha=1)

fig.supxlabel('Epochs')
fig.savefig('basemodeltraining.png')
