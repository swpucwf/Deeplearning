import  random
from sklearn import ensemble,neighbors,datasets,preprocessing
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score, r2_score
import matplotlib.pyplot as plt
import time
np.random.RandomState(0)
scores = []
plt.rcParams['font.sans-serif']=['SimHei'] #显示中文标签
plt.rcParams['axes.unicode_minus']=False

for _ in range(100):
    # 笑容渐渐起步
    data_s = []
    with open("label.txt","r") as f:
        data = f.readline()
        while data:
            data_s.append(np.array([float(i) for i in data.split()]))
            data = f.readline()

    data_s = np.array(data_s)
    # print(data_s.shape)
    #
    x,y = data_s[:,:-1],data_s[:,-1]
    # wine = datasets.load_wine()
    # x,y = wine.data,wine.target
    # x_train,y_train = x,y

    # 待判
    x_train,x_test,y_train,y_test =  train_test_split(x,y,test_size=0.2,random_state=0)

    x_2,y = x_train,y_train 
    # 数据标准化
    #
    scaler = preprocessing.StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    # # print(x_train)
    #
    # time = time.time()


    # scaler = preprocessing.MinMaxScaler().fit(x_train)
    #
    # x_train,x_test = scaler.transform(x_train),scaler.transform(x_test)
    clf = ensemble.AdaBoostRegressor(
        DecisionTreeRegressor(max_depth=6),
        n_estimators=50,
    )

    clf.fit(x_train,y_train)

    # 待判数据
    y_pred = clf.predict(x_test)
    x = np.zeros_like(y_test)
    x = [i for i in range(len(x))]
    plt.axis([min(y_test[:,]),max(y_test[:,]),min(y_test[:,]),max(y_test[:,])])
    plt.plot(y_test[:,],y_test[:,],c="red", label="岩心分析孔隙结构指数", linewidth=2)
    plt.scatter(y_pred[:,],y_test[:,],color="blue",linewidth=2,edgecolor="black", label="data")
    score = r2_score(y_test,y_pred)
    test_loss = (abs(y_pred-y_test)/y_test).mean()

    plt.xlabel("岩心分析孔隙结构指数,相对误差率={:.2f}%".format(test_loss*100))
    plt.ylabel(f"预测孔隙结构指数")

    plt.savefig("./images_test/{:.4f}and{:.2f}.jpg".format(score, test_loss))
    plt.show()
    print("相对误差率", (abs((y_pred - y_test) / y_test)).mean())
    with open("./result/{:.4f}and{:.2f}.txt".format(score, test_loss), "w") as f:
        f.write(str((x_2, y)))
        f.write(str((x_test, y_test)))
        f.write(str(score))
    # # 回判数据
    # y_pred = clf.predict(x_train)
    # x = np.zeros_like(y_train)
    # x = [i for i in range(len(x))]
    # plt.axis([min(y_train[:,]),max(y_train[:,]),min(y_train[:,]),max(y_train[:,])])
    # plt.plot(y_train[:,],y_train[:,],c="darkorange", label="岩心分析孔隙结构指数", linewidth=2)
    # plt.scatter(y_pred[:,],y_train[:,],color="blue",linewidth=2,edgecolor="black", label="data")
    # score = r2_score(y_train,y_pred)
    # test_loss = ((abs((y_pred-y_train))/y_train)*abs(y_pred)/y_train).mean()
    #
    # plt.xlabel("岩心分析孔隙结构指数,相对误差率={:.2f}%".format(test_loss
    #                                              *100))
    # plt.ylabel(f"预测孔隙结构指数")
    #
    # plt.savefig("./images/{:.4f}and{:.2f}.jpg".format(score,test_loss))
    # plt.show()
    # print("相对误差率",(abs((y_pred-y_train)/y_train)).mean())
    # with open("./result/{:.4f}and{:.2f}.txt".format(score,test_loss),"w") as f:
    #     f.write(str((x_2,y)))
    #     # f.write(str((x_test, y_test)))
    #     f.write(str(score))
    # # print(y_pred,y_train)
    plt.show()
    scores.append(score)

print(max(scores))

#
# plt.scatter(x, y, s=20, edgecolor="black", c="darkorange", label="data")
# plt.plot(x_test, y_1, color="cornflowerblue", label="max_depth=1", linewidth=2)
# plt.plot(x_test, y_2, color="yellowgreen", label="max_depth=3", linewidth=2)
# plt.plot(x_test, y_3, color='red', label='liner regression', linewidth=2)