'''
单变量线性回归算法（比如， x代表学历， f(x)代表收入）
f(x) = ax+b 我们使用f(x)这个函数来映射输入特征和输入值
'''

import tensorflow as tf
import pandas as pd 
import matplotlib.pyplot as plt


data = pd.read_csv('Source/Income1.csv')

print(data.head())

# plt.scatter(data.Education, data.Income)
# plt.show()

# 均方差作为损失函数
# 也就是预测值和真实值中间差的平方去均值

x = data.Education 
y = data.Income 

model = tf.keras.Sequential()
# Dense(1,) 输出维度 input_shape=(1,)输入数据的维度
model.add(tf.keras.layers.Dense(1, input_shape=(1,)))
# 自己新建出两个参数 ax+b
# print(model.summary())

# 编译 optimizer优化 默认的优化速率     loss损失 mse 均方差
model.compile(optimizer='adam',
loss='mse'
)

history = model.fit(x, y, epochs=5000)
# 预测
model.predict(x)

result = model.predict(pd.Series([20]))
print(result)



