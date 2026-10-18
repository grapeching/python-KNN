#1.导包

from sklearn.neighbors import KNeighborsRegressor

from KNN算法_分类 import x_train

#2.准备数据集（测试集和训练集）

x_train=[[0,0,1],[1,1,0],[3,10,10],[4,11,12]]
y_train=[0.1,0.2,0.3,0.4]
x_test=[[3,11,10]]

#3.创建（KNN回归模型）模型对象
estimator = KNeighborsRegressor(n_neighbors=3)
#4.模型训练

estimator.fit(x_train,y_train)
#5.模型预测

y_pred = estimator.predict(x_test)
#6.打印预测结果

print(f"预测值为:{y_pred}")