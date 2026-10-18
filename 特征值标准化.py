#都是为了解决单位问题导致模型评估较低问题
#导包

from sklearn.preprocessing import StandardScaler

#1.准备数据集
x_train=[[90,2,10,40],[60,4,15,45],[75,3,13,46]]

#2.创建归一化对象
transfer=StandardScaler()

#3.对原数据集进行归一化操作
x_train_new=transfer.fit_transform(x_train)

#4.打印处理后数据
print("标准化后的数据集为：\n")
print(x_train_new)
print(f"数据集均值为：{transfer.mean_}")
print(f"数据集方差为：{transfer.var_}")
print(f"数据集标准差为：{transfer.scale_}")