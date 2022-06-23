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

tf.keras.backend.clear_session()
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)

# image_datas = glob('D:\\code\\data\\final/*/*/*.jpg')
# class_name = ["can01", "can02", "can03", "can04", "can05", 
#             "glass01", "glass02", "glass03", "glass04", "glass05", "glass06",
#             "paper01", "paper02", "paper03", "paper04",
#             "pet01", "pet02",
#             "plastic",
#             "styrofoam01", "styrofoam02", "styrofoam03",
#             "vinyl"]
# dic = {"can01":0, "can02":1, "can03":3, "can04":4, "can05":5, 
#             "glass01":6, "glass02":7, "glass03":8, "glass04":9, "glass05":10, "glass06":11,
#             "paper01":12, "paper02":13, "paper03":14, "paper04":15,
#             "pet01":16, "pet02":17,
#             "plastic":18,
#             "styrofoam01":19, "styrofoam02":20, "styrofoam03":21,
#             "vinyl":22}
image_datas = glob('D:\\code\\data\\final_2/*/*/*.jpg')
class_name = ["can01", "glass01", "paper01", "pet01", "plastic", "styrofoam01", "vinyl"]
dic = {"can01":0, "glass01":1, "paper01":2, "pet01":3, "plastic":4, "styrofoam01":5, "vinyl":6}

X = []
Y = []
for imagename in tqdm(image_datas):
    image = Image.open(imagename)
    image = image.resize((71, 71))
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
unique, counts = np.unique(np.reshape(train_labels, (29223,)), axis=-1, return_counts=True)
print(dict(zip(unique, counts)))

## test set의 각 class 별 images 수 확인
unique, counts = np.unique(np.reshape(test_labels, (3247,)), axis=-1, return_counts=True)
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

learning_rate = 0.001
N_EPOCHS = 100
N_BATCH = 4
N_CLASS = 23

## dataset 구성
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(buffer_size=15754).batch(N_BATCH).repeat()
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(N_BATCH)

# model1 = tf.keras.models.load_model('D:\\code\\model\\lenet5.h5')

# test_loss1 ,test_acc1 = model1.evaluate(test_dataset)

# print("lenet5_loss : ", test_loss1)
# print("lenet5_acc : ", test_acc1)

# model2 = tf.keras.models.load_model('D:\\code\\model\\ResNet50V2.h5')

# test_loss2 ,test_acc2 = model2.evaluate(test_dataset)

# print("ResNet50V2_loss : ", test_loss2)
# print("ResNet50V2_acc : ", test_acc2)


# model3 = tf.keras.models.load_model('D:\\code\\model\\VGG19.h5')

# test_loss3 ,test_acc3 = model3.evaluate(test_dataset)

# print("VGG19_loss : ", test_loss3)
# print("VGG19_acc : ", test_acc3)

# model4 = tf.keras.models.load_model('D:\\code\\model\\final_model.h5')

# test_loss4 ,test_acc4 = model4.evaluate(test_dataset)

# print("final_model_loss : ", test_loss4)
# print("final_model_acc : ", test_acc4)


model5 = tf.keras.models.load_model('D:\\code\\model\\Xception_final.h5')

test_loss5 ,test_acc5 = model5.evaluate(test_dataset)

print("Xception_loss : ", test_loss5)
print("Xception_acc : ", test_acc5)