import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from numpy import expand_dims
from sklearn.model_selection import StratifiedKFold, KFold
from keras import Sequential
from keras.layers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam,SGD
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.applications import EfficientNetB4, EfficientNetB2
import datetime
from time import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Flatten, BatchNormalization, Dense, Activation
from tensorflow.python.keras.applications.efficientnet import EfficientNetB7

SEED = 66
IMAGE_SIZE = (128,128,3)
EPOCH = 50
OPTIMIZER =Adam(learning_rate= 1e-3)

#data load
x = np.load("C:/LPD_competition/npy/150project_x.npy",allow_pickle=True)
y = np.load("C:/LPD_competition/npy/150project_y.npy",allow_pickle=True)
x_pred = np.load('C:/LPD_competition/npy/150test.npy',allow_pickle=True)


print(x_pred.shape)

x = preprocess_input(x) 
x_pred = preprocess_input(x_pred)   

idg = ImageDataGenerator(
    width_shift_range=(-1,1),   
    height_shift_range=(-1,1),  
    shear_range=0.2,
    zoom_range = 0.1) 
   
idg2 = ImageDataGenerator()

idg3 = ImageDataGenerator(
    width_shift_range=(-1,1),   
    height_shift_range=(-1,1),
    shear_range=0.2,
    zoom_range = 0.1
    ) 

# y = np.argmax(y, axis=1)

x_train, x_valid, y_train, y_valid = train_test_split(x,y, train_size = 0.8, shuffle = True, random_state=SEED)

train_generator = idg.flow(x_train,y_train,batch_size=8)
valid_generator = idg2.flow(x_valid,y_valid)
test_generator = idg3.flow(x_pred,shuffle = False)

from keras.utils.generic_utils import get_custom_objects
from tensorflow.python.keras.activations import swish
from keras.layers import Activation
get_custom_objects().update({'swish': Activation(swish)})

md = EfficientNetB2(input_shape = IMAGE_SIZE, weights = "imagenet", include_top = False)
for layer in md.layers:
    layer.trainable = True
c = md.output
c = GlobalAveragePooling2D()(c)
c = Flatten()(c)
c = Dense(6048, activation= 'swish') (c)
c = Dropout(0.2) (c)
c = Dense(1000, activation="softmax")(c)
model = Model(inputs=md.input, outputs = c)


from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
cp = ModelCheckpoint('C:/LPD_competition/lotte_projcet0323.h5',monitor='val_acc',save_best_only=True, verbose=1)
early_stopping = EarlyStopping(monitor='val_acc',patience= 8,verbose=1)
lr = ReduceLROnPlateau(monitor='val_loss',patience= 4, factor=0.4,verbose=1)

model.compile(loss='categorical_crossentropy', optimizer=OPTIMIZER,
                metrics=['acc'])

t1 = time()       
history = model.fit_generator(train_generator,
    validation_data=valid_generator, epochs=EPOCH, steps_per_epoch=len(train_generator), 
                    validation_steps=len(valid_generator),callbacks=[early_stopping,lr,cp])

t2 = time()
print("execution time: ", t2 - t1)
# predict
model.load_weights('C:/LPD_competition/lotte_projcet0323.h5')
# # result = model.predict(x_pred,verbose=True)


from math import ceil
from tqdm import tqdm

preds_tta = []
tta_steps = 10
for i in tqdm(range(tta_steps)):
    test_generator.reset()
    preds = model.predict_generator(
        generator=test_generator,
        steps =ceil(len(x_pred)/32)
    )
    preds_tta.append(preds)

preds_mean = np.mean(preds_tta, axis=0)
sub = pd.read_csv('C:/LPD_competition/sample.csv')
sub['prediction'] = np.argmax(preds_mean, axis=1)
sub.to_csv('C:/LPD_competition/answer0323.csv',index=False)

# 데이터 변경