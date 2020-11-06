

def hello(func):
    def warp_1():
        print('hello')
        func()
    return warp_1


@hello
def aaa():
    print(123)


if __name__ == '__main__':
    aaa()