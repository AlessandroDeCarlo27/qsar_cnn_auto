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
#%% load base model

step2_model = tf.keras.models.load_model('bestModels/step2modelFT.hdf5') #change
step2_model.summary()

#%% define step3 model

model3 = tf.keras.Sequential()
model3.add(layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))
#conv1 ereditato
model3.add(step2_model.layers[0])
model3.layers[0].trainable = False
#conv2 ereditato
model3.add(step2_model.layers[1])
model3.layers[1].trainable = False
#conv3 ereditato
model3.add(step2_model.layers[2])
model3.layers[2].trainable = False
#conv3 conv reduction
model3.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same', strides=2,name='conv2D_m3'))
model3.add(layers.Conv2DTranspose(128, (3, 3), activation='relu', padding='same', strides=2,name='conv2DT_m3_1'))
model3.add(layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same', strides=2,name='conv2DT_m3_2'))
model3.add(step2_model.layers[4])
model3.layers[6].trainable = False
model3.add(step2_model.layers[5])
model3.layers[7].trainable = False
model3.add(step2_model.layers[6])
model3.layers[8].trainable = False

model3.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss=losses.MeanSquaredError())
model3.summary()

NEPOCHS = 5#300

checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath='step3modelRed.hdf5', monitor='val_loss', verbose=1,
                             save_best_only=True, mode='min', save_weights = False, save_format="tf")

history =   model3.fit(train_generator,
            epochs=NEPOCHS,
            shuffle=True,
            validation_data = validation_generator,#,
            callbacks=[checkpoint])


vals = np.array(history.history["val_loss"])

best_epoch = vals.argmin()+1
best_v = vals.min()
allv = list()
allv.append(history.history["loss"])
allv.append(history.history["val_loss"])

sup_lim = np.array(allv).max()

plt.rcParams.update({'font.size': 18})
fig,(ax1,ax2) = plt.subplots(1,2,figsize=(16,16),sharex=True)
fig.suptitle('Training of Step2 model')
ax1.plot(np.arange(1,NEPOCHS+1),history.history["loss"],
         label="Training Loss",linewidth=3)
ax1.plot(np.arange(1,NEPOCHS+1),history.history["val_loss"],
         label="Validation Loss",linewidth=3)
ax1.vlines(best_epoch,0,sup_lim,linestyles='dotted',color='black')
ax1.plot(best_epoch,best_v,'og',label='Best',markersize=10)
ax1.grid(visible=True)
ax1.set_title('Natural Scale')
ax1.legend(framealpha=1)

ax2.semilogy(np.arange(1,NEPOCHS+1),history.history["loss"],
             label="Training Loss",linewidth=3)
ax2.semilogy(np.arange(1,NEPOCHS+1),history.history["val_loss"],
             label="Validation Loss",linewidth=3)
ax2.vlines(best_epoch,0,sup_lim,linestyles='dotted',color='black')
ax2.semilogy(best_epoch,best_v,'og',label='Best',markersize=10)
ax2.grid(visible=True)
ax2.set_title('Semilogy Scale')
ax2.legend(framealpha=1)

fig.supxlabel('Epochs')
fig.savefig('step3training.png')

