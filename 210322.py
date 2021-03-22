import numpy as np
from tensorflow.keras.applications import EfficientNetB4, EfficientNetB2
from tensorflow.keras.applications.efficientnet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import GlobalAveragePooling2D, Flatten, BatchNormalization, Dense, Activation, Dropout
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import VGG19, MobileNet, ResNet101, EfficientNetB7, EfficientNetB2
from tensorflow.keras.optimizers import Adam, SGD
from tqdm import tqdm

#데이터 지정 및 전처리
x = np.load("C:/LPD_competition/npy/P_project_x4.npy",allow_pickle=True)
x_pred = np.load('C:/LPD_competition/npy/test.npy',allow_pickle=True)
y = np.load("C:/LPD_competition/npy/P_project_y4.npy",allow_pickle=True)
# print(x.shape, x_pred.shape, y.shape)   #(48000, 128, 128, 3) (72000, 128, 128, 3) (48000, 1000)

x = preprocess_input(x) # (48000, 255, 255, 3)
x_pred = preprocess_input(x_pred)   # 

idg = ImageDataGenerator(
    # rotation_range=10, acc 하락
    width_shift_range=(-1,1),  
    height_shift_range=(-1,1), 
    rotation_range=40, 
    # shear_range=0.2)    # 현상유지
    zoom_range=0.2,
    horizontal_flip=False,
    fill_mode='nearest')

idg2 = ImageDataGenerator()

x_train, x_valid, y_train, y_valid = train_test_split(x,y, train_size = 0.9, shuffle = True, random_state=66)

train_generator = idg.flow(x_train,y_train,batch_size=28, seed = 2048)
# seed => random_state
valid_generator = idg2.flow(x_valid,y_valid)
test_generator = idg2.flow(x_pred)

mc = ModelCheckpoint('C:/LPD_competition/lotte_0322_2.h5',save_best_only=True, verbose=1)
mobile = EfficientNetB2(include_top=False,weights='imagenet',input_shape=x_train.shape[1:])
mobile.trainable = True
a = mobile.output
a = GlobalAveragePooling2D() (a)
a = Flatten() (a)
a = Dense(4048, activation= 'swish') (a)
a = Dropout(0.2) (a)
a = Dense(1000, activation= 'softmax') (a)

model = Model(inputs = mobile.input, outputs = a)

# early_stopping = EarlyStopping(patience= 30)
# lr = ReduceLROnPlateau(patience= 15, factor=0.5)

# model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.1), metrics=['acc'])
# learning_history = model.fit_generator(train_generator,epochs=200, steps_per_epoch= len(x_train) / 28,
#     validation_data=valid_generator, callbacks=[early_stopping,lr,mc])

# predict
model.load_weights('C:/LPD_competition/lotte_0322_1.h5')
# tta_steps = 1
# predictions = []

# for i in tqdm(range(tta_steps)):
# 	# generator 초기화
#     test_generator.reset()
#     preds = model.predict_generator(generator = test_generator, steps = len(x_pred) // 28, verbose = 1)
#     print(preds)
#     print(preds.shape)

#     predictions.append(preds)
#     print(predictions)
#     # print(predictions.shape)

# # 평균을 통한 final prediction
# pred = np.mean(predictions, axis=0)
# print(pred)
# print(pred.shape)

# # argmax for submission
# np.mean(np.equal(np.argmax(y_val, axis=-1), np.argmax(pred, axis=-1)))

result = model.predict(x_pred,verbose=True)

# 제출생성
sub = pd.read_csv('C:/LPD_competition/sample.csv')
sub['prediction'] = np.argmax(pred,axis = 1)
sub.to_csv('C:/LPD_competition/lotte0322_2.csv',index=False)