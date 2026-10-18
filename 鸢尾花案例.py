#1.加载数据
#2.数据预处理
#3.特征工程
#4.模型训练
#5.模型评估
#6.模型预测

#导包
from sklearn.datasets import load_iris
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from KNN算法_分类 import x_train, estimator


#1.定义函数，加载鸢尾花数据集，并且查看

def dm01_load_iris():
    #1.加载鸢尾花数据集
    iris_data = load_iris()
    #2.查看数据集
    print(f"数据集：{iris_data}")
    print(f"数据集类型：{type(iris_data)}")

    #3.查看数据集的键
    print(f"数据集所有的键：{iris_data.keys()}")
    #4.查看数据集的键对应的数据
    print(f"具体数据：{iris_data.data[:5]}")
    print(f"具体标签：{iris_data.target[:5]}")
    print(f"具体标签的名称：{iris_data.target_names}")
    print(f"具体特征的名称：{iris_data.feature_names}")

    #print(f"数据集的描述：{iris_data.DESCR}")
    print(f"数据集的框架：{iris_data.frame}")

#2.定义函数，绘制数据集散点图
def dm02_show_iris():
    #1.加载数据集
    iris_data = load_iris()
    #2.把 鸢尾花数据集封装成DataFrame对象
    iris_df = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
    #3.给df新对象新增1列→标签列
    iris_df['label'] = iris_data.target
    print(iris_df)

    #4.通过seaborn绘制散点图
    sns.lmplot(data=iris_df, x="sepal length (cm)", y="sepal width (cm)",hue="label",fit_reg=True)
    #5.设置标题，显示
    plt.title('iris data')
    plt.tight_layout() #自动调节子图格式
    plt.show()

#3.定义函数，切分训练集和测试集
def dm03_split_train_test():
    #1.加载数据集
    iris_data = load_iris()
    #2.数据预处理：从150个特征标签中，按照8：2的比例，切分训练集和测试集
    #参1：特征数据，参2：标签数据，参3：数据集比例，参4：随机种子
    x_train,x_test,y_train,y_test=train_test_split(iris_data.data,iris_data.target,test_size=0.2,random_state=23)
    #3.打印切割后的数据
    print(f"训练集的特征：{x_train},个数:{len(x_train)}")
    print(f"训练集的标签：{y_train},个数:{len(y_train)}")
    print(f"测试集的特征：{x_test},个数:{len(x_test)}")
    print(f"测试集的标签：{y_test},个数:{len(y_test)}")

#4.定义函数，实现鸢尾花完整案例

def dm04_iris_evaluate_test():
    #1.加载数据
    iris_data = load_iris()
    #2.数据预处理
    x_train, x_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, test_size=0.2,random_state=23)
    #3.特征工程(提取，预处理...)
    #3.1 创建标准化对象
    transfer=StandardScaler()
    #3.2标准化 transfer.fit.transform(兼具训练与转化，一般用于第一次进行标准化时候使用
    x_train=transfer.fit_transform(x_train)
    #transform只有转换
    x_test=transfer.transform(x_test)

    #4.模型训练
    #创建模型对象
    estimator=KNeighborsClassifier(n_neighbors=3)
    #具体训练模型动作
    estimator.fit(x_train,y_train)

    #5.模型评估
    #对切分的30条测试集进行测试
    #5.1直接预测，获取预测结果
    y_pre=estimator.predict(x_test)
    print(f"预测值为{y_pre}")

    #新150条测试集进行测试
    #5.1自定义测试数据集
    my_data=[[7.8,2.1,3.9,1.6]]
    #5.2标准化处理
    my_data= transfer.transform(my_data)
    #5.3模型预测
    y_pre_new=estimator.predict(my_data)
    print(f'预测值为：{y_pre_new}')
    #5.4查看上述数据集，每种分类预测概率
    y_pre_proba=estimator.predict_proba(my_data)
    print(f'（各分类）预测概率为：{y_pre_proba}')

    #6.模型评估
    #方式1.直接评分，基于：训练集的特征 和 训练集的标签
    print(f'正确率：{estimator.score(x_train,y_train)}')

    #方式2.基于测试集的标签和预测结果评分
    print(f'正确率：{accuracy_score(y_test, y_pre)}')



#5.测试
if __name__ == "__main__":
    #dm01_load_iris()
    #dm02_show_iris()
    #dm03_split_train_test()
    dm04_iris_evaluate_test()