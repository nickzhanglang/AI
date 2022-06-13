# 导入pandas库
import pandas as pd
import numpy as np
# 导入数据集拆分工具
from sklearn.model_selection import train_test_split
from sklearn import tree, datasets

'''1.打开文件'''
# 用pandas打开csv文件
data = pd.read_csv(r'D:\python\python_project\pythonProject\adult.csv', header=None, index_col=False,
                   names=['年龄', '单位性质', '权重', '学历', '受教育时长', '婚姻状况', '职业', '家庭情况', '种族', '性别', '资产所得', '资产损失', '周工作时长',
                          '原籍', '收入'])
# 为了方便展示,我们选取其中一部分数据
data_lite = data[['年龄', '单位性质', '受教育时长', '婚姻状况', '种族', '性别', '周工作时长', '收入']]
# 下面看一下数据的前五行是不是我们想要的结果
print(data_lite.head())
print(type(data_lite),'\n')


"验证集和数据集分割"
data_lite1=data_lite.iloc[29800:-1]
data_lite=data_lite.iloc[0:29800]


"训练集数据处理"
print("-------训练集数据处理----------")
#使用get_dummies将文本数据转化为数值
data_dummies = pd.get_dummies(data_lite)
#对比样本原始特征和虚拟变量特征
#虚拟变量特征是
print('样本原始特征:\n',list(data_lite.columns),'\n')
print('虚拟变量特征:\n',list(data_dummies.columns))
#显示数据集中的前5行
print(data_dummies.head())
# 定义数据集的特征
features = data_dummies.loc[:, '年龄':'性别_ Male']
# 将特征的数值赋值为X
X = features.values
# print("特征值",features)
# "print(X)"
# 将收入大于50K作为预测目标
y = data_dummies['收入_ >50K'].values
# print("标签值",data_dummies['收入_ >50K'])
# "print(y)"
# 打印数据形态
# print('特征形态:{} 标签形态:{}'.format(X.shape, y.shape))
# print("-------训练集数据处理完毕----------",'\n')


"验证集数据处理"
print("-------验证集数据处理----------")
#使用get_dummies将文本数据转化为数值
data_dummies1 = pd.get_dummies(data_lite1)
print('样本原始特征:\n',list(data_lite1.columns),'\n')
print('虚拟变量特征:\n',list(data_dummies1.columns))
#显示数据集中的前5行
print(data_dummies1.head())
# 定义数据集的特征
features1 = data_dummies1.loc[:, '年龄':'性别_ Male']
# 将特征的数值赋值为X
X1 = features1.values
# print("特征值",features)
print("验证集特征值格式",type(X))
print(X[0])
# 将收入大于50K作为预测目标
y1 = data_dummies1['收入_ >50K'].values
# print("标签值",data_dummies['收入_ >50K'])
print("验证集标签值格式",type(y1))
# 打印数据形态
# print('特征形态:{} 标签形态:{}'.format(X1.shape, y1.shape))
print("-------验证集数据处理完毕----------",'\n')


"验证集格式处理"
print("-------验证集格式处理----------")
X1 =X1.tolist()
y1 =y1.tolist()
print("最终验证集的数据格式：",type(X1))
print("--------验证集格式处理完毕---------",'\n')
print(y1)


X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0)
#用最大深度为5的随机森林拟合数据
go_dating_tree = tree.DecisionTreeClassifier(max_depth=5)
go_dating_tree.fit(X_train,y_train)

print('代码运行结果')
print('====================================\n')
#打印数据形态
print('模型得分:{:.2f}'.format(go_dating_tree.score(X_test,y_test)))
print('\n====================================')


'''5.使用模型预测'''

c=[]

for i in iter(X1):
    j = []    #每次循环都定义成空的
    j.append(i)
    dating_dec = go_dating_tree.predict(j)
    dating_dec1 = dating_dec[0]
    c.append(dating_dec1)
print("预测值集合：",c,'\n')

p=0
for l, j in zip(c, y1):
    if l==j:
        p+=1
print("使用随机森林 在验证集上的准确率为：",p/len(c)*100,"%")
# #将Mr Z先生的数据输入给模型（这个大兄弟的特征数值，可以自己设定）
# Mr_Z = [[37, 40,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0]]
#使用模型做出预测
# i=X1[0]
# print(i)
# i_all = []
# i_all.append(i)
# dating_dec = go_dating_tree.predict(i_all)
# print('代码运行结果')
# print('====================================\n')
# if dating_dec == 1:
#     print("这哥们月薪过5万了")
# else:
#     print("小辣鸡月薪不过五万")
# print('\n====================================')