# 학습이 잘 되는 모델
#  사용할 레이어
#   tf.keras.layers.BatchNormailzation()
#   tf.keras.layers.Activation('swish')
 
#  데이터
#   보스턴 집값 예측
#   아이리스 품종 분류

# 라이브러리 사용
import tensorflow as tf
import pandas as pd

# 1. 과거의 데이터를 준비합니다.
파일경로 = 'D:\\code\\vs\\Tensorflow_practice2\\boston.csv'
보스턴 = pd.read_csv(파일경로)
print(보스턴.columns)

# 종속변수, 독립변수
독립 = 보스턴[['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'b', 'lstat']]
종속 = 보스턴[['medv']]
print(독립.shape, 종속.shape)

# 2. 모델의 구조를 만듭니다.
X = tf.keras.layers.Input(shape=[13])

H = tf.keras.layers.Dense(8)(X)
H = tf.keras.layers.BatchNormalization()(H)
H = tf.keras.layers.Activation('swish')(H)

H = tf.keras.layers.Dense(8)(H)
H = tf.keras.layers.BatchNormalization()(H)
H = tf.keras.layers.Activation('swish')(H)

H = tf.keras.layers.Dense(8)(H)
H = tf.keras.layers.BatchNormalization()(H)
H = tf.keras.layers.Activation('swish')(H)

Y = tf.keras.layers.Dense(1)(H)
model = tf.keras.models.Model(X, Y)
model.compile(loss='mse')

# 3. 데이터로 모델을 학습(FIT)합니다.
model.fit(독립, 종속, epochs=1000)

# 1. 과거의 데이터를 준비합니다.
파일경로 = 'D:\\code\\vs\\Tensorflow_practice2\\iris.csv'
아이리스 = pd.read_csv(파일경로)

# 원핫인코딩
아이리스 = pd.get_dummies(아이리스)

# 종속변수, 독립변수
독립 = 아이리스[['꽃잎길이', '꽃잎폭', '꽃받침길이', '꽃받침폭']]
종속 = 아이리스[['품종_setosa', '품종_versicolor', '품종_virginica']]
print(독립.shape, 종속.shape)

# 2. 모델의 구조를 만듭니다.
X = tf.keras.layers.Input(shape=[4])

H = tf.keras.layers.Dense(8)(X)
H = tf.keras.layers.BatchNormalization()(H)
H = tf.keras.layers.Activation('swish')(H)

H = tf.keras.layers.Dense(8)(H)
H = tf.keras.layers.BatchNormalization()(H)
H = tf.keras.layers.Activation('swish')(H)

H = tf.keras.layers.Dense(8)(H)
H = tf.keras.layers.BatchNormalization()(H)
H = tf.keras.layers.Activation('swish')(H)
Y = tf.keras.layers.Dense(3, activation='softmax')(H)
model = tf.keras.models.Model(X, Y)
model.compile(loss='categorical_crossentropy', metrics='accuracy')

# 3. 데이터로 모델을 학습(FTI)합니다.
model.fit(독립, 종속, epochs=1000, batch_size = 150)