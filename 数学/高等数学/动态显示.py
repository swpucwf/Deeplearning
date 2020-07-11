import random
import matplotlib.pyplot as plt

_x = [i/100 for i in range(100)]
print(_x)
_y = [3*e+4+random.random() for e in _x]

w = random.random()
b = random.random()

plt.ion()#开启会话

for i in range(30):
    for x, y in zip(_x, _y):
        z = w * x + b#将数据x输入到构建好的线性模型中，得到输出z（前向过程，推理过程）
        o = z - y
        loss = o ** 2#根据前向输出z和标签y定义损失函数，得到损失

        dw = -2 * o * x
        db = -2 * o

        w = w + 0.1 * dw#更新w
        b = b + 0.1 * db#更新b
        print("w=", w, ";b=", b, ";loss=", loss)
        v = [w * e + b for e in _x]
        plt.cla()#擦除
        plt.plot(_x, _y, ".")
        plt.plot(_x, v)
        plt.title(loss)#增加标题
        plt.pause(0.01)#睡眠0.01秒

plt.ioff()#结束会话
plt.show()