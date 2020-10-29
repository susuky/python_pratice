import numpy as np
import matplotlib.pyplot as plt

def relu6(x): return np.minimum(np.maximum(x, 0), 6)

def hardswish(x): return x * (relu6(x + 3)) / 6


if __name__ == '__main__':
    x = np.arange(-10, 10, 0.1)
    plt.subplot(211)
    plt.plot(relu6(x))

    plt.subplot(212)
    plt.plot(x, hardswish(x))
    plt.show()