#分类问题
#1.导包

from sklearn.neighbors import KNeighborsClassifier



#2.准备数据集（训练集和测试集）

x_train=[[1],[2],[3],[4]]
y_train=[0,0,1,1]
x_test=[[5]]

#3.创建模型对象
estimator = KNeighborsClassifier(n_neighbors=2)

#4.模型训练
#传入：训练集特征数据和标签数据
estimator.fit(x_train,y_train)

#5.模型预测
#传入：训练集的特征数据，获取到预测结果(测试集的标签，y_test)
y_pred=estimator.predict(x_test)

#6.打印预测结果
print(f"预测值为：{y_pred}")
