import numpy as np
import pandas as pd

data_lite = pd.read_csv(r'D:\python\python_project\pythonProject\adult3.csv', header=None, index_col=False,
                   names=['年龄', '单位性质', '受教育时长', '婚姻状况', '种族', '性别', '周工作时长', '收入'])

# 下面看一下数据的前五行是不是我们想要的结果
print(data_lite.head())
print("数据类型",type(data_lite),'\n')

# data_lite=data_lite.astype(int)

def get_Xy(data):
    data.insert(0, 'ones', 1)
    X_ = data.iloc[0:30000, 0:-1]
    X = X_.values
    X = np.array(X)
    X = X.tolist()
    print("数据类型", type(X), '\n')
    X_new = []
    for i in X:
        i_new = list(map(int, i))
        X_new.append(i_new)
    print("数据类型", type(X_new), '\n')

    y_ = data.iloc[0:30000, -1]
    y = y_.values.reshape(len(y_), 1)
    print("y的维度",np.array(y).shape)

    return X_new, y

X,y = get_Xy(data_lite)
# print("特征值：",X,"标签值：",y)
print("X的维度",np.array(X).shape)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def costFunction(X, y, theta):

    A = sigmoid(X @ theta)

    first = y * np.log(A)
    # print(np.array(A).shape)
    second = (1-y) * np.log(1-A+ 1e-6)

    return -np.sum(first + second) / len(X)

theta = np.zeros((8,1))
print("theta的维度：",theta.shape)

cost_init = costFunction(X,y,theta)
print('\n')
print("使用BP神经网络 在验证集上的准确率为：",cost_init*100,"%",'\n')

def gradientDescent(X, y, theta, iters, alpha):
    m = len(X)
    costs = []

    for i in range(iters):
        A = sigmoid(X @ theta)
        theta = theta - (alpha / m) * np.transpose(X) @ (A - y)
        cost = costFunction(X, y, theta)
        costs.append(cost)
        # if i % 1000 == 0:
            # print(cost)
    return costs, theta

alpha = 0.004
iters=200000

costs,theta_final =gradientDescent(X,y,theta,iters,alpha)

"验证集的切片"
def get_Xy1(data):
    data.insert(0, 'ones', 1)
    X_ = data.iloc[30001:-1, 0:-1]
    X = X_.values
    X = np.array(X)
    X = X.tolist()
    print("数据类型", type(X), '\n')
    X_new = []
    for i in X:
        i_new = list(map(int, i))
        X_new.append(i_new)
    print("数据类型", type(X_new), '\n')

    y_ = data.iloc[30001:-1, -1]
    y = y_.values.reshape(len(y_), 1)
    print("y的维度",np.array(y).shape)

    return X_new, y

X1,y1 = get_Xy1(data_lite)

def predict(X, theta):
    prob = sigmoid(X @ theta)
    return [1 if x >= 0.5 else 0 for x in prob]

y1_ = np.array(predict(X1,theta_final))
y1_pre = y1_.reshape(len(y1_),1)

acc  = np.mean(y1_pre == y1)

# print(acc)
coef1 = - theta_final[0,0] / theta_final[2,0]
coef2 = - theta_final[1,0] / theta_final[2,0]

x = np.linspace(20,100,100)
f = coef1 + coef2 * x
