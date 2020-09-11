#!/usr/bin/env python
# coding: utf-8

# Given a 32-bit signed integer, reverse digits of an integer.

# In[1]:


class Solution(object):
    def reverse(self, x):
        """
        :type x: int
        :rtype: int
        """
        if x == 0: return 0
        if x < 0:
            x *= -1
            sign = -1
        else:
            sign = 1

        tmp = []
        while x / 10 != 0:
            tmp.append(x % 10)
            x = x // 10

        a = 0
        for i, j in zip(range(len(tmp), 0, -1), range(len(tmp))):
            a += 10**(i - 1) * tmp[j]
        if a > 2 **31 -1:
            return 0
        else:
            return a * sign


Solution.reverse(None, -123), Solution.reverse(None, 120)


# In[2]:


class Solution(object):
    def reverse(self, x):
        """
        :type x: int
        :rtype: int
        """
        if x == 0: return 0
        if x < 0:
            x *= -1
            sign = -1
        else:
            sign = 1

        tmp = [int(i) for i in reversed(str(x))]

        a = 0
        for i, j in zip(range(len(tmp), 0, -1), range(len(tmp))):
            a += 10**(i - 1) * tmp[j]
        if a > 2 **31 -1:
            return 0
        else:
            return a * sign

Solution.reverse(None, -123), Solution.reverse(None, 120)


# In[3]:


class Solution(object):
    def reverse(self, x):
        """
        :type x: int
        :rtype: int
        """
        if x > 0:
            a = int(str(x)[::-1])
        else:
            a = -int(str(-x)[::-1])

            
        if a > 2**31 - 1 or a < -2**31:
            return 0
        else:
            return a


Solution.reverse(None, -123), Solution.reverse(None, 120)


# In[ ]:




