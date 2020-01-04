# 在tf.keras里，对于多分类问题我们使用
# categorical_crossentropy和
# sparse_categorical_crossentropy来计算交叉熵

import tensorflow as tf
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np

# 加载fashion mnist
# 如果报错将 已经下载好的 fashion_mnist文件夹放到 /Users/chenjianglin/.keras/datasets/fashion-mnist
(train_image, train_label), (test_image, test_label) = tf.keras.datasets.fashion_mnist.load_data()

# plt.imshow(train_image[0])
# plt.show()

# np.max(train_image[0])
# 对数几率回归解决的是二分类的问题
# 对于多分类的问题，我们可以使用softmax函数
# 它是对说几率回归在N个可能不同的值上的推广

print(train_image.shape)

print(train_label)

# 先要做一个归一化的处理
train_image = train_image/255
test_image = test_image/255

model = tf.keras.Sequential()
# 内部自动处理成 28*28
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

# 当我们使用数字编码的时候 结果是 0，1，2，3等的时候 
# 损失函数是 sparse_categorical_crossentropy  
# 、当label进行独热编码的时候我们使用categorical_crossentropy作为我们的损失函数
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['acc']
)

model.fit(train_image, train_label, epochs=6)


# 在test上进行评价
model.evaluate(test_image, test_label)

# model.predict(test_image[0])

train_label_onehot = tf.keras.utils.to_categorical(train_label)
print(train_label_onehot)
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['acc']
)

model.fit(train_image, train_label_onehot, epochs=6)


predict = model.predict(test_image)

predict[0]

print(np.argmax(predict[0]))


