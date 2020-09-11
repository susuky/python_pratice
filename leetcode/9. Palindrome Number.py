#!/usr/bin/env python
# coding: utf-8

# Determine whether an integer is a palindrome. An integer is a palindrome when it reads the same backward as forward.

# In[1]:


class Solution:
    def isPalindrome(self, x: int) -> bool:
        if x < 0: return False
        
        tmp = [int(i) for i in str(x)]
        tmp_r = [int(i) for i in reversed(str(x))]
        
        return tmp==tmp_r
    
Solution.isPalindrome(None, -121), Solution.isPalindrome(None, 121)


# In[2]:


class Solution:
    def isPalindrome(self, x: int) -> bool:
        return str(x) == str(x)[::-1]
    
Solution.isPalindrome(None, -121), Solution.isPalindrome(None, 121)


# In[3]:


class Solution:
    def isPalindrome(self, x: int) -> bool:
        if x < 0: return False

        tmp = []
        while x != 0:
            x, y = divmod(x, 10)
            tmp.append(y)
            
        tmp_ = tmp.copy()
        tmp.reverse()
        return tmp == tmp_


Solution.isPalindrome(None, -121), Solution.isPalindrome(None, 121)


# In[4]:


class Solution:
    def isPalindrome(self, x: int) -> bool:
        if x < 0: return False

        tmp = []
        while x != 0:
            x, y = divmod(x, 10)
            tmp.append(y)
            
        return tmp == list(reversed(tmp))


Solution.isPalindrome(None, -121), Solution.isPalindrome(None, 121)


# In[5]:


class Solution:
    def isPalindrome(self, x: int) -> bool:
        if x < 0: return False

        tmp = []
        while x != 0:
            x, y = divmod(x, 10)
            tmp.append(y)
            
        return tmp == tmp[::-1]


Solution.isPalindrome(None, -121), Solution.isPalindrome(None, 121)

