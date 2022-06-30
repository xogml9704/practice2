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

image_datas = glob('D:\\code\\data\\bottle/*/*/*.jpg')
class_name = ["a", "b", "c", "d"]
dic = {"a":0, "b":1, "c":2, "d":3}

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
unique, counts = np.unique(np.reshape(train_labels, (9346,)), axis=-1, return_counts=True)
print(dict(zip(unique, counts)))

## test set의 각 class 별 images 수 확인
unique, counts = np.unique(np.reshape(test_labels, (1039,)), axis=-1, return_counts=True)
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

learning_rate = 0.00001
N_EPOCHS = 100
N_BATCH = 2
N_CLASS = 7

## dataset 구성
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(buffer_size=15754).batch(N_BATCH).repeat()
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(N_BATCH)


## Create model, compile & summary
Xception = tf.keras.applications.Xception(
    include_top=False,
    weights=None,
    input_shape=(64, 64, 3),
    pooling=max
)

model = tf.keras.models.Sequential()
model.add(Xception)
model.add(tf.keras.layers.GlobalAveragePooling2D())
model.add(tf.keras.layers.Dense(4, activation='softmax'))

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

## parameters for training
steps_per_epoch = N_TRAIN//N_BATCH
validation_steps = N_TEST//N_BATCH
print(steps_per_epoch, validation_steps)

## Training
history = model.fit(train_dataset, epochs=N_EPOCHS, steps_per_epoch=steps_per_epoch, validation_data=test_dataset, validation_steps=validation_steps)

model.evaluate(test_dataset)

model.save('D:\\code\\model\\test_ResNet50V2.h5')