from tensorflow.keras.models import load_model
import cv2
import numpy as np

cnn_model = load_model('protovgg19.h5')

# print(cnn_model.summary())
input_img = cv2.imread('test1.jpg')
input_img = cv2.resize(input_img, (100, 100))
input_img = input_img.reshape(1, 100, 100, 3)
print(np.argmax(cnn_model.predict(input_img)))