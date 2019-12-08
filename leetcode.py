class Solution(object):
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        output = []
        length = len(nums)
        for i,num in enumerate(nums):
            for i1 in range(i,length):
                if num+nums[i1] == target:
                    output.append(i)
                    output.append(i1)
                    print(output)



class MyCircularQueue(object):

    def __init__(self, k):
        """
        Initialize your data structure here. Set the size of the queue to be k.
        :type k: int
        """
        self.queue = [None] * k
        self.head = -1
        self.tail = -1
        self.size = k

    def enQueue(self, value):
        """
        Insert an element into the circular queue. Return true if the operation is successful.
        :type value: int
        :rtype: bool
        """
        if self.isFull():
            return False
        if self.isEmpty():
            self.head = 0
            print('?')
        self.tail = (self.tail + 1) % self.size
        self.queue[self.tail] = value
        return True

    def deQueue(self):
        """
        Delete an element from the circular queue. Return true if the operation is successful.
        :rtype: bool
        """
        if self.isEmpty():
            return False
        else:
            self.queue[self.head]=None
            self.head = (self.head + 1) % self.size
            return True

    def Front(self):
        """
        Get the front item from the queue.
        :rtype: int
        """
        if self.isEmpty():
            return -1
        else:
            return self.queue[self.head]

    def Rear(self):
        """
        Get the last item from the queue.
        :rtype: int
        """
        if self.isEmpty():
            return -1
        else:
            return self.queue[self.tail]

    def isEmpty(self):
        """
        Checks whether the circular queue is empty or not.
        :rtype: bool
        """
        for i in self.queue:
            if i:
                return False
        else:
            self.head=-1
            self.tail=-1
            return True

    def isFull(self):
        """
        Checks whether the circular queue is full or not.
        :rtype: bool
        """
        for i in self.queue:
            if not i:
                return False
        return True

class MovingAverage(object):

    def __init__(self, size):
        """
        Initialize your data structure here.
        :type size: int
        """
        self.size = size
        self.queue = [None] * self.size
        self.tail = -1

    def next(self, val):
        self.tail = (self.tail + 1) % self.size
        self.queue[self.tail] = val
        sum = 0
        float(sum)
        cut = 0
        for i in self.queue:
            if i != None:
                cut += 1
                sum += i
        return sum/cut


class WallAndDoor(object):
    def way(self,rooms):
        m = len(room)
        n = len(room[0])
        for mi in range(0,m):
            for ni in range(0,n):
                step = 0
                if a[mi][ni] == 0:
