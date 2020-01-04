import tensorflow as tf 
from tensorflow import keras
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images = train_images/255
test_images = test_images/255

input1 = keras.Input(shape=(28,28))
input2 = keras.Input(shape=(28,28))
x1 = keras.layers.Flatten()(input1)
x2 = keras.layers.Flatten()(input2)
x = keras.layers.concatenate([x1,x2])
x = keras.layers.Dense(32, activation='relu')(x)
x = keras.layers.Dropout(0.5)(x)
x = keras.layers.Dense(64, activation='relu')(x)
output = keras.layers.Dense(10, activation='softmax')(x)


model = keras.Model(inputs=[input1, input2], outputs=output)

# 编译方式跟其他的类似




