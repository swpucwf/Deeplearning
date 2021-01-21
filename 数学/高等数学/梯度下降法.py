import random
import matplotlib.pyplot as plt


if __name__ == '__main__':

    _x = [i/100. for i in range(100)]
    _y = [4*i+4+random.random()/100. for i in _x]
    w = random.random()
    b = random.random()
    for i in range(1000):
        for x,y in zip(_x,_y):
            z = w*x+b
            o = z-y
            loss = o**2
            dw = -2*o*x
            db = -2*o

            w = w+0.01*dw
            b = b+0.01*db
            print("w=",w,"b=",b,"loss=",loss)
    plt.plot(_x,_y,".")
    v = [w*e+b for e in _x]
    plt.plot(_x,v)
    plt.show()



