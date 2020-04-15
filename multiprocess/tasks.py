def job(q):
    res = 0
    for i in range(1000):
        res += i+i**2+i**3
    q.put(res)  # queue
