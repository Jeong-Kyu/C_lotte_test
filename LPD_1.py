# ImageDataGenerator fit_generator

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, MaxPooling2D,Flatten
from sklearn.decomposition import PCA
import pandas as pd

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.5,
    height_shift_range=0.5,
    rotation_range=20,
    zoom_range=1.5,
    shear_range=0.9,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)
xy_train = train_datagen.flow_from_directory(
    # 첫번 째 인자는 디렉토리 폴더를 받음
    'C:/LPD_competition/train', # (160,256,256,3)
    # 폴더 안에 있는 전체 이미지 데이터를 사이즈를 바꿈
    target_size=(150,150), # (160,150,150,3)
    batch_size=5, 
    class_mode='categorical'
)

xy_test = test_datagen.flow_from_directory(
    # 첫번 째 인자는 디렉토리 폴더를 받음
    'C:/LPD_competition/test', # (120,256,256,3)
    # 폴더 안에 있는 전체 이미지 데이터를 사이즈를 바꿈
    target_size=(150,150), # (120,150,150,3)
    batch_size=5, 
    shuffle=False
)

model = Sequential()
model.add(Conv2D(128,(3,3), input_shape=(150,150,3)))
model.add(Dropout(0.2))
model.add(Conv2D(64,(3,3)))
model.add(Dropout(0.2))
model.add(Conv2D(32,(3,3)))
model.add(Dropout(0.2))
model.add(Conv2D(16,(3,3)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(32, activation = 'relu'))
model.add(Dense(1000, activation = 'softmax'))

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
es = EarlyStopping(monitor='loss', patience=10, mode='auto')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, verbose=1)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
# history = model.fit_generator(
#     xy_train, steps_per_epoch=32, epochs=200, validation_data=xy_test, validation_steps=4, callbacks=[es,reduce_lr]
# )
history = model.fit(xy_train, epochs=200, validation_split=0.2, callbacks=[es,reduce_lr])

# fit_generator -> xy together
# # step_per_epoch -> data / batch_size
# loss, acc = model.evaluate_generator(xy_test)
# print("loss : ", loss)
# print("acc : ", acc)

result = model.predict(xy_test)

sub = pd.read_csv('C:/LPD_competition/sample.csv')
sub['prediction'] = result.argmax(1) # y값 index 2번째에 저장
sub.to_csv('C:/LPD_competition/sample_1.csv',index=False)
