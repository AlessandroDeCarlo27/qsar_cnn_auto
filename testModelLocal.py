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
import pandas as pd
os.chdir('C:\\Users\\AlessandroDeCarlo\\Desktop\\DT\projs\\QSARdeep')
#%% set seeds
tf.random.set_seed(12345)
np.random.seed(12345)
#%% useful const and objs

IMAGE_SIZE = 256
BATCH_SIZE = 4

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

#%% load best model and set values to write within the log file
nameMod_f = 'step4modelRed256FT' #name of the model
nameBest_f = 'step4modelRed256FT.hdf5' #name of the file with the best model (training)
nameFigTr_f = 'step4modelRed256TrainFT' #name of the figure with training output
nameFolderTest_f = 'step4Red256FT'
#name of the folder with the images obtained applying the model on the test set

nameLogTrainingFile = 'log_step4modelRed256FT.csv' #name of the csv with the log of the training procedure
nepochs_f = str(200) #number of epochs used to train the model



model = tf.keras.models.load_model(''.join(['bestModels/',nameBest_f]))
model.summary()

# describe model structure 
#C2D: convolutional 2d layer
#C2DT: transpose convolutional 2d layer
#X_t: trained layer
#X_f: freezed layer
description_f = 'C2D_ft-C2D_ft-C2D_ft-C2D_ft-C2D_ft-C2DT_ft-C2DT_ft-C2DT_ft-C2DT_ft-C2DT_ft-C2DT_ft'
#add eventually some comments
comments_f = '6.5e-4 @100 ep. of fine tune. Idea: train the model on a dataset with more molecules with CL, P, S.'
#%% training output

training_log = pd.read_csv(''.join(['log/',nameLogTrainingFile]))
epoch_log = training_log['epoch'].to_numpy()+1
tloss_log = training_log['loss'].to_numpy()
vloss_log = training_log['val_loss'].to_numpy()

best_epoch = vloss_log.argmin()+1
best_v = vloss_log.min()
allv = np.concatenate([tloss_log,vloss_log])

sup_lim = allv.max()


plt.rcParams.update({'font.size': 18})
fig,(ax1,ax2) = plt.subplots(1,2,figsize=(16,16),sharex=True)
fig.suptitle('Training of Step3 model')
ax1.plot(epoch_log,tloss_log,
         label="Training Loss",linewidth=3)
ax1.plot(epoch_log,vloss_log,
         label="Validation Loss",linewidth=3)
ax1.vlines(best_epoch,0,sup_lim,linestyles='dotted',color='black')
ax1.plot(best_epoch,best_v,'og',label='Best',markersize=10)
ax1.grid(visible=True)
ax1.set_title('Natural Scale')
ax1.legend(framealpha=1)

ax2.semilogy(epoch_log,tloss_log,
             label="Training Loss",linewidth=3)
ax2.semilogy(epoch_log,vloss_log,
             label="Validation Loss",linewidth=3)
ax2.vlines(best_epoch,0,sup_lim,linestyles='dotted',color='black')
ax2.semilogy(best_epoch,best_v,'og',label='Best',markersize=10)
ax2.grid(visible=True)
ax2.set_title('Semilogy Scale')
ax2.legend(framealpha=1)

fig.supxlabel('Epochs')
fig.savefig(''.join(['log/',nameFigTr_f]))
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
