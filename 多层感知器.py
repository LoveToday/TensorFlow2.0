
# 线性回归模型是单个的神经元 （二分类）
# 神经元的缺陷

# 神经元要求数据必须是线性可分的
# 异或问题无法找到一条直线分割两个类  
# 这时候在输入和输出之间插入更多的神经元  多层感知器

# 激活函数
# 1. relu  
# 2. sigmoid
# 3. Leak relu
# 4.tanh

import pandas as pd 
import tensorflow as tf 
import matplotlib.pyplot as plt

data = pd.read_csv('Source/Advertising.csv')

# print(data.head())

x = data.iloc[:,1:-1]
y = data.iloc[:,-1]



model = tf.keras.Sequential([tf.keras.layers.Dense(10, input_shape=(3,), activation='relu'),
                             tf.keras.layers.Dense(1)
                            ])
print(model.summary())


model.compile(optimizer='adam',
loss='mse')

model.fit(x, y, epochs=20)

test = data.iloc[:10,1:-1]
result = model.predict(test)
print(result)

print(data.iloc[:10,-1])

