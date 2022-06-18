# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 15:22:14 2022

@author: AlessandroDeCarlo

the content of _f variables will be written into the log file which summarizes 
the performed attempts
"""

#%% loading libs and setting working dir
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
os.chdir('C:\\Users\\AlessandroDeCarlo\\Desktop\\DT\projs\\QSARdeep')
#%% set seeds
tf.random.set_seed(12345)
np.random.seed(12345)
#%% useful const and objs

IMAGE_SIZE = 256
BATCH_SIZE = 4

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

#%% load best model and set values to write within the log file
nameMod_f = 'basemodel' #name of the model
nameBest_f = 'basemodel.hdf5' #name of the file with the best model (training)
nameFigTr_f = '-' #name of the figure with training output
nameFolderTest_f = 'baseModel' 
#name of the folder with the images obtained applying the model on the test set

nepochs_f = str(150) #number of epochs used to train the model



model = tf.keras.models.load_model(''.join(['bestModels/',nameBest_f]))
model.summary()

# describe model structure 
#C2D: convolutional 2d layer
#C2DT: transpose convolutional 2d layer
#X_t: trained layer
#X_f: freezed layer
description_f = 'C2D_t-C2D_t-C2DT_t-C2DT_t-C2DT_t' 
#add eventually some comments
comments_f = 'Base starting model 24h of training approx. Figure training not avail. Update matplotlib on cluster'
#%% generator for images 
# test set is very huge, make some plots of the predictions
test_generator_img = test_datagen.flow_from_directory(
    'test/',
    target_size=(IMAGE_SIZE,IMAGE_SIZE),
    batch_size=4,
    shuffle = False,
    color_mode='rgb',
    class_mode ='input'   
)


test_generator_img.reset()
#%% plot some images
pred=model.predict(test_generator_img)

counter = 0
for j in range(int(np.ceil(test_generator_img.n/4))):

    bt = test_generator_img.next()
    imbt = bt[0]
    Nimg = 4
    
    plt.figure(figsize=(40, 8))
    for i in range(Nimg):
      # display original
      ax = plt.subplot(2, Nimg, i + 1)
      plt.imshow(imbt[i])
      plt.title("original")
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)
    
      # display reconstruction
      ax = plt.subplot(2, Nimg, i + 1 + Nimg)
      plt.imshow(pred[counter+i])
      plt.title("reconstructed")
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)
    counter += (Nimg)
    name_fig = ''.join(['outputTest/',nameFolderTest_f,'/fig',str(j+1),'.png'])
    plt.savefig(name_fig)
    
#%% generator for performances
test_generator_perf = test_datagen.flow_from_directory(
    'C:/Users/AlessandroDeCarlo/Desktop/DT/CLUSTER/imgQsarCnn/test/',
    target_size=(IMAGE_SIZE,IMAGE_SIZE),
    batch_size=4,
    shuffle = False,
    color_mode='rgb',
    class_mode ='input'   
)


test_generator_perf.reset()

#%% evaluate performances
eval_test = model.evaluate(test_generator_perf)
#%% save info in the log.txt to keep track of training events

tloss_f = str(eval_test)


newline = ['\n',nameMod_f,'\t',description_f,'\t',tloss_f,'\t',nepochs_f,'\t',comments_f,
           '\t',nameBest_f,'\t',nameFigTr_f,'\t',nameFolderTest_f]
with open('log/log.txt','a') as f:
    f.writelines(''.join(newline))

f.close()
