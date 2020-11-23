import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from tensorflow.keras.preprocessing import image

image_path = '/home/notoboto/Downloads/destino/train/African_hunting_dog/n02116738_9762.jpg'

loaded_model = './model'
classes_path = './class_indices'
classes = []

preprocessed_image = image.load_img(image_path, target_size=(224, 224))
preprocessed_image = np.array([image.img_to_array(preprocessed_image)])/255

with open(classes_path, 'rb') as class_file:
    classes = pickle.load(class_file)
    classes = list(classes.keys())

print(classes)
model = tf.keras.models.load_model(loaded_model)

prediction = model.predict(preprocessed_image)
prediction = np.asarray(prediction)[0]
print(prediction)
argsorted_prediction = (-prediction).argsort()

top3 = []
for i in argsorted_prediction[:3]:
   top3.append([str(classes[i]).replace('_', ' ').capitalize(), prediction[i]])

print(top3)
