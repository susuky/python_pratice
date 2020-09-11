#!/usr/bin/env python
# coding: utf-8

# Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.
# 
# You may assume that each input would have ***exactly one solution***, and you may ***not use the same element twice***.
# 
# You can return the answer in any order.

# In[8]:


class Solution(object):
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """

        lens = len(nums)

        for i in range(lens):
            for j in range(i + 1, lens):
                if nums[i] + nums[j] == target:
                    return [i, j]


Solution.twoSum(None, [2, 11, 7, 15], 9)


# In[95]:


from typing import List


class Solution(object):
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """

        tmp = []
        for n in nums:
            sub = target - n

            if n in tmp:
                if nums.count(n) == 1:
                    return (nums.index(sub), nums.index(n))
                else:
                    idx = nums.index(sub)
                    return (idx, nums.index(n, idx+1))

            tmp.append(sub)


Solution.twoSum(None, [3, 3], 6), Solution.twoSum(None, [2, 11, 7, 15], 9)


# In[117]:


from typing import List


class Solution(object):
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """

        tmp = {}
        for i, n in enumerate(nums):
            sub = target - n
            if n in tmp:
                return([tmp[n], i])

            tmp[sub] = i


Solution.twoSum(None, [3, 3], 6), Solution.twoSum(None, [2, 11, 7, 15], 9)


# In[ ]:




