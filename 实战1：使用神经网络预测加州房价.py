# 使用神经网络预测加州房价
# 1.数据加载
# 导入库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# 加载加州房价数据集
data = fetch_california_housing()
# 将特征数据转换为 DataFrame 格式，方便后续处理
df = pd.DataFrame(data.data, columns=data.feature_names)
df['Price'] = data.target
# 查看数据分布
# print(df.head())
# print(df.describe())
# 可视化特征与房价关系（示例
plt.scatter(df['MedInc'],df['Price'],alpha=0.3)
plt.xlabel('Median Income')
plt.ylabel('House Price')
plt.show()

# 2.数据预处理
# 处理缺失值（本次数据集无缺失，此处示例）
df.fillna(df.mean(),inplace=True)
# 特征与标签分离
X = df.drop('Price',axis=1).values
y = df['Price'].values
# 特征标准化
scaler = StandardScaler()
X_scaler = scaler.fit_transform(X)
# 划分训练集和测试集
X_train,X_test,y_train,y_test = train_test_split(X_scaler,y,test_size=0.2,random_state=1)
# 解释：标准化处理：使特征均值为0，方差为1，加速神经网络收敛。

# 3.构建神经网络
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout

model = Sequential([
    Dense(64,activation='relu',input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(32,activation='relu'),
    Dense(1) # 输出层（回归问题，无激活函数）
])
# 编译模型
model.compile(
    optimizer='adam',  # 优化器，adam是自适应学习率的优化算法
    loss='mse',        # 损失函数，mse均方误差，常用于回归任务。它的计算方法是预测值与真实值之差的平方的平均值
    metrics=['mae']    # 评估指标，平均绝对误差。计算方式为预测值与真实值之差的绝对值的平均值
)
# 训练模型
history = model.fit(
    X_train,y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,  # 从训练集划分20% 作为验证集
    verbose=1
)

# 绘制训练曲线
plt.plot(history.history['loss'],label='Training loss')
plt.plot(history.history['val_loss'],label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 评估模型
loss,mae = model.evaluate(X_test,y_test)
print(f'Test loss:{loss:.4f}, Test MAE:{mae:,.4f}')
# 预测示例
sample = X_test[0].reshape(1,-1)
predicted_price = model.predict(sample)
print(f'Predicted Price:{predicted_price[0][0]:.2f}, Actual Price:{y_test[0]:.2f}')