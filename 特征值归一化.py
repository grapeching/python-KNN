#导包
from sklearn.preprocessing import MinMaxScaler

#1.准备数据集
x_train=[[90,2,10,40],[60,4,15,45],[75,3,13,46]]

#2.创建归一化对象
transfer=MinMaxScaler()

#3.对原数据集进行归一化操作
x_train_new=transfer.fit_transform(x_train)

#4.打印处理后数据
print("归一化后的数据集为：\n")
print(x_train_new)