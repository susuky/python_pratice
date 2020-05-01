import numpy as np
import matplotlib.pyplot as plt
import time


interval = 100

# data to plot
x = np.arange(-15, 30, 0.1)
def f(x): return np.sin(x) * x - np.cos(x) * x


if __name__ == "__main__":
    
    for i in range(len(x)):
        plt.ylim((-40, 40))
        if i > interval:
            plt.plot(x[i-interval:i], f(x[i-interval:i]), 'C1')
        else:  
            plt.plot(x[:i], f(x[:i]), 'C1')
        plt.pause(0.05)
        plt.clf()

    plt.show()