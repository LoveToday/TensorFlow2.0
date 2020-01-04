import tensorflow as tf 
import pandas as pd 
import matplotlib.pyplot as plt

# 对数几率回归解决是二分类的问题
# 对于多个选项的问题，我们可以使用softmax函数
# sigmoid 与交叉熵  信用卡欺诈数据

data = pd.read_csv('Source/credit-a.csv',header=None)
print(data.head())
y = data.iloc[:,-1]
print(y.value_counts())
#  1    357
# -1    296
# 只有两个值 可以肯定的是是一个二分类的问题

x = data.iloc[:,1:-1]

# 将其中的-1替换成0
y = data.iloc[:,-1].replace(-1, 0)

print(y.value_counts())

# 顺序模型
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(4, input_shape=(14,), activation='relu'))
model.add(tf.keras.layers.Dense(4, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

print(model.summary())

# 编译 adam 优化  binary_crossentropy 交叉熵
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['acc']
)

history = model.fit(x, y, epochs=150)

print(history.history.keys())


# plt.plot(history.epoch, history.history.get('loss'))
plt.plot(history.epoch, history.history.get('acc'))
plt.show()




