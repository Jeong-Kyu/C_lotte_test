import os
import os, glob, numpy as np
from PIL import Image
import numpy as np
from numpy import asarray
from tensorflow.keras.models import Model, Sequential,load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.layers import Activation
import tensorflow as tf
import scipy.signal as signal
from keras.applications.resnet50 import ResNet50,preprocess_input,decode_predictions
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.callbacks import ReduceLROnPlateau
#########데이터 로드

caltech_dir =  'C:/LPD_competition/train/'
categories = []
for i in range(0,1000) :
    i = "%d"%i
    categories.append(i)

nb_classes = len(categories)

image_w = 140
image_h = 140

pixels = image_h * image_w * 3

X = []
y = []

for idx, cat in enumerate(categories):
    
    #one-hot 돌리기.
    label = [0 for i in range(nb_classes)]
    label[idx] = 1

    image_dir = caltech_dir + "/" + cat
    files = glob.glob(image_dir+"/*.jpg")
    print(cat, " 파일 길이 : ", len(files))
    for i, f in enumerate(files):
        img = Image.open(f)
        img = img.convert("RGB")
        img = img.resize((image_w, image_h))
        data = np.asarray(img)

        X.append(data) 
        y.append(label)

        if i % 700 == 0:
            print(cat, " : ", f)

X = np.array(X)
y = np.array(y)

np.save("C:/LPD_competition/npy/140project_x.npy", arr=X)
np.save("C:/LPD_competition/npy/140project_y.npy", arr=y)
# x_pred = np.load("../data/npy/P_project_test.npy",allow_pickle=True)
x = np.load("C:/LPD_competition/npy/140project_x.npy",allow_pickle=True)
y = np.load("C:/LPD_competition/npy/140project_y.npy",allow_pickle=True)

print(x.shape)
print(y.shape)


img1=[]
for i in range(0,72000):
    filepath='C:/LPD_competition/t/test/%d.jpg'%i
    image2=Image.open(filepath)
    image2 = image2.convert('RGB')
    image2 = image2.resize((140,140))
    image_data2=asarray(image2)
    # image_data2 = signal.medfilt2d(np.array(image_data2), kernel_size=3)
    img1.append(image_data2)    

# np.save('../data/csv/Dacon3/train4.npy', arr=img)
np.save('C:/LPD_competition/npy/140test.npy', arr=img1)
# alphabets = string.ascii_lowercase
# alphabets = list(alphabets)


# x = np.load('../data/csv/Dacon3/train4.npy')
x_pred = np.load('C:/LPD_competition/npy/140test.npy',allow_pickle=True)

print(x_pred.shape)