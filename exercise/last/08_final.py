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

image_data = glob('D:\\code\\data\\ex/*.png')
# dic = {"can01":0, "can02":1, "can03":3, "can04":4, "can05":5, 
#             "glass01":6, "glass02":7, "glass03":8, "glass04":9, "glass05":10, "glass06":11,
#             "paper01":12, "paper02":13, "paper03":14, "paper04":15,
#             "pet01":16, "pet02":17,
#             "plastic":18,
#             "styrofoam01":19, "styrofoam02":20, "styrofoam03":21,
#             "vinyl":22}
X = []

for imagename in tqdm(image_data):
    image = Image.open(imagename)
    image = image.resize((71, 71))
    image = np.array(image)
    X.append(image)

print(image.shape)
image = (np.expand_dims(image, 0))

probability_model = tf.keras.models.load_model('D:\\code\\model\\Xception_load.h5')

print(image.shape)

predictions_single = probability_model.predict(image)

value = (np.argmax(predictions_single))

print(type(value))

if 0 <= value < 6:
    message = 1
    print("can")
elif 6 <= value < 12:
    message = 2
    print("glass")
elif 12 <= value < 16:
    message = 3
    print("paper")
elif 16 <= value < 18:
    message = 4
    print("pat")
elif value == 18:
    message = 5
    print("plastic")
elif 19 <= value < 22:
    message = 6
    print("styrofoam")
elif 22 == value:
    message = 7
    print("vinyl")

print(message)