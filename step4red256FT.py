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
#%% load step4 model

model4 = tf.keras.models.load_model('bestModels/step4modelRed256.hdf5')

for layer in model4.layers:
    layer.trainable = True

model4.summary()


NEPOCHS = 10#300

checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath='step4modelRed256FT.hdf5', monitor='val_loss', verbose=1,
                             save_best_only=True, mode='min', save_weights = False, save_format="tf")
bckpTraining = tf.keras.callbacks.CSVLogger('log_step4modelRed256FT.csv',separator=",",append=True)

history =   model4.fit(train_generator,
            epochs=NEPOCHS,
            shuffle=True,
            validation_data = validation_generator,#,
            callbacks=[checkpoint,bckpTraining])

