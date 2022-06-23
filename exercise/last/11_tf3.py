from tensorflow import keras

model = keras.models.load_model('D:\\code\\model\\Xception_final.h5', compile=False)

model.export(export_dir='.', with_metadata=False)