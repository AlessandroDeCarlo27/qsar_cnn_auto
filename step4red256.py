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
#%% load step3 model

step3_model = tf.keras.models.load_model('bestModels/step3modelRed.hdf5') #change
step3_model.summary()

#%% define step 4 model

model4 = tf.keras.Sequential()
model4.add(layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))
#conv1 ereditato
model4.add(step3_model.layers[0])
model4.layers[0].trainable = False
#conv2 ereditato
model4.add(step3_model.layers[1])
model4.layers[1].trainable = False
#conv3 ereditato
model4.add(step3_model.layers[2])
model4.layers[2].trainable = False
#conv4 ereditato
model4.add(step3_model.layers[3])
model4.layers[3].trainable = False
#new conv layer
model4.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same', strides=2,name='conv2D_m4')) #4
model4.add(layers.Conv2DTranspose(256, (3, 3), activation='relu', padding='same', strides=2,name='conv2DT_m4_1')) #5
model4.add(layers.Conv2DTranspose(128, (3, 3), activation='relu', padding='same', strides=2,name='conv2DT_m4_2')) #6
model4.add(step3_model.layers[5])
model4.layers[7].trainable = False
model4.add(step3_model.layers[6])
model4.layers[8].trainable = False
model4.add(step3_model.layers[7])
model4.layers[9].trainable = False
model4.add(step3_model.layers[8])
model4.layers[10].trainable = False
model4.summary()

model4.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss=losses.MeanSquaredError())
model4.summary()

NEPOCHS = 50#300

checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath='step4modelRed256.hdf5', monitor='val_loss', verbose=1,
                             save_best_only=True, mode='min', save_weights = False, save_format="tf")
bckpTraining = tf.keras.callbacks.CSVLogger('log_step4modelRed256.csv',separator=",",append=True)

history =   model4.fit(train_generator,
            epochs=NEPOCHS,
            shuffle=True,
            validation_data = validation_generator,#,
            callbacks=[checkpoint,bckpTraining])