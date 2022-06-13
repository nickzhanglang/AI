# import pandas as pd
# import os
import numpy as np
# data = pd.read_csv('adult1.csv', encoding='utf-8')
# with open('adult2.txt', 'a+', encoding='utf-8') as f:
#     for line in data.values:
#         f.write((str(line[0]) + '\t' + str(line[1]) + '\n'))
#
# data = pd.read_csv(r'D:\python\python_project\pythonProject\adult1.csv', header=None, index_col=False,
#                    names=['年龄', '单位性质', '权重', '学历', '受教育时长', '婚姻状况', '职业', '家庭情况', '种族', '性别', '资产所得', '资产损失', '周工作时长',
#                           '原籍', '收入'])
# # 为了方便展示,我们选取其中一部分数据
# data_lite = data[['年龄', '单位性质', '受教育时长', '婚姻状况', '种族', '性别', '周工作时长', '收入']]
# # 下面看一下数据的前五行是不是我们想要的结果
# print(data_lite.head())
# print(type(data_lite),'\n')
# data_lite.to_csv('adult3.csv', index = False, header = False )
"实验1"
# a = 5
# b = np.array(range(1,5))
# c = list(b)
# print("三个值：",a,b,c,'\n')
# print("三个值的数据类型：",type(a),type(b),type(c),'\n')
# # print("测试是否能直接Int：",int(a),int(b),int(c))
"实验2"
class car():
    def __init__(self,model,year):
        self.model = model
        self.year = year
    def print(self):
        print(self.model+"\tis\t"+str(self.year)+"\tproduction")


class e_car():
    def __init__(self,model,year):
        super().__init__(self,model,year)

tesla = car('model_s', 2018)
print(tesla.print())
