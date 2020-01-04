# 1. 多层感知器
# 输入层，隐含层，输出层，输出

#1. 如果要预测一个连续的值，这时候我们直接输出，不进行激活
#2.如果想要预测的是一个二分类的是或否  使用sigmoild激活
#3.如果要预测是一个多分类的问题，这时候要使用softmax激活

# 多层感知器使用的是一个梯度下降算法


# 学习速率
# 梯度就是表明损失函数相对参数的变化率
# 对梯度进行缩放的参数被称为学习速率（learning rate）
# 学习速率是一种超参数或是对模型的一种手工配置的设置
# 如果学习速率太小，则找到损失函数极小值时可能需要许多论迭代；如果太大，
# 则算法可能会跳过极小值点并且因周期性的跳跃二永远无法找到极小值点
# 具体实践中，可以通过查看损失函数值随时间的变化曲线来判断学习速率的选取合适
# 合适的学习速率，损失函数的值会随着时间下降，直到底部
# 不合适的学习速率，损失函数可能会发生震荡

# 常见的与优化器
# 1.adam
# 2.SGD 随机梯度下降优化器 和 min-batch 是一个意思，抽取m个小批量（独立同分布样本），
# 通过计算他们的平均梯度值
# 3. RMSprop  处理序列问题比较多


import pandas as pd 
import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt
 
(train_image, train_label), (test_image, test_label) = tf.keras.datasets.fashion_mnist.load_data()

# 先做一个归一化 这样梯度下降比较快

train_image = train_image/255
test_image = test_image/255
 
model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss='sparse_categorical_crossentropy',
    metrics=['acc']
)

model.fit(train_image, train_label, epochs=6)

# test上进行评估
model.evaluate(test_image, test_label)






