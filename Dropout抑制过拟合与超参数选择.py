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
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))

model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['acc']
)

history = model.fit(train_image, train_label, 
                    epochs=10,
                    validation_data=(test_image, test_label))

# history.history.keys()
# dict_keys(['loss', 'acc', 'val_loss', 'val_acc'])

plt.plot(history.epoch, history.history.get('loss'), label='loss')
plt.plot(history.epoch, history.history.get('val_loss'), label='val_loss')
plt.legend()
plt.show()


# test上进行评估
model.evaluate(test_image, test_label)

# 过拟合： 在训练数据集上得分比较高，在测试数据上得分相对比较低
# 欠拟合：在训练数据上得分比较低，在测试数据上得分相对比较低

# 为什么说Dropout可以解决过拟合
# 1，取平均的作用
# 2.减少神经元之间的复杂的共适应关系
# 3.Dropout类似于性别在生物进化中的角色
# 理想的模型是刚好在欠拟合和过拟合的界线上，也就是正好拟合数据

# 参数选择原则
# 首先开发一个过拟合的模型
# 1.添加更多的层
# 2.让每一层变得更大
# 3.训练更多的轮次

# 然后抑制过拟合
# 1.dropout
# 2.正则化
# 3.图像增强

# 抑制过拟合的最好的办法是增加训练数据

# 最后调节超参数
# 1.学习效率
# 2.隐藏层单元数
# 3.训练轮次

# 构建网络的总原则
# 保证神经网络容量足够拟合数据
# 一、增大网络容量，直到过拟合
# 二、采取措施抑制过拟合
# 三、继续增大网络容量，直到过拟合