from sklearn.model_selection import train_test_split
import numpy as np
from glob import glob
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
from tqdm import tqdm

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)

image_datas = glob('D:\\code\\data\\garbage2/*/*/*.jpg')
class_name = ["can", "glass", "paper", "pet", "plastic", "styrofoam", "vinyl"]
dic = {"can":0, "glass":1, "paper":2, "pet":3, "plastic":4, "styrofoam":5, "vinyl":6}

X = []
Y = []
for imagename in tqdm(image_datas):
    image = Image.open(imagename)
    image = image.resize((128, 128))
    image = np.array(image)
    X.append(image)
    label = imagename.split('\\')[4]
    label = dic[label]
    Y.append(label)

X = np.array(X)
Y = np.array(Y)

train_images, test_images, train_labels, test_labels = train_test_split(
    X, Y,test_size=0.1, shuffle=True, random_state=44)

train_labels = train_labels[..., tf.newaxis]
test_labels = test_labels[..., tf.newaxis]

print(train_images.shape, train_labels.shape, test_images.shape, test_labels.shape)

## training set의 각 class 별 image 수 확인
unique, counts = np.unique(np.reshape(train_labels, (15754,)), axis=-1, return_counts=True)
print(dict(zip(unique, counts)))

## test set의 각 class 별 images 수 확인
unique, counts = np.unique(np.reshape(test_labels, (1751,)), axis=-1, return_counts=True)
print(dict(zip(unique, counts)))

N_TRAIN = train_images.shape[0]
N_TEST = test_images.shape[0]

# pixel 값을 0~1 사이 범위로 조정
train_images = train_images.astype(np.float32) / 255.
test_images = test_images.astype(np.float32) / 255.

# label을 onahot-encoding
train_labels = tf.keras.utils.to_categorical(train_labels)
test_labels = tf.keras.utils.to_categorical(test_labels)

print(train_images.shape, train_labels.shape)
print(test_images.shape, test_labels.shape)

learning_rate = 0.0001
N_EPOCHS = 100
N_BATCH = 1
N_CLASS = 7

## dataset 구성
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(buffer_size=15754).batch(N_BATCH).repeat()
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(N_BATCH)

new_model = tf.keras.models.load_model('D:\\code\\model\\Xception.h5')

test_loss ,test_acc = new_model.evaluate(test_dataset)

print("test_loss : ", test_loss)
print("test_acc : ", test_acc)