from tensorflow import keras

model = keras.models.load_model('D:\\code\\model\\Xception_final.h5', compile=False)

export_path = 'D:\\code\\model'
model.save(export_path, save_format="tf")