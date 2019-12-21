class Solution1(object):
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

class Solution622(object):

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

class Solution346(object):

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

class Solution286(object):
    def wallsAndGates(self, rooms):
        """
        :type rooms: List[List[int]]
        :rtype: None Do not return anything, modify rooms in-place instead.
        """
        from collections import deque
        if not rooms or not rooms[0]:
            return rooms
        m, n = len(rooms), len(rooms[0])
        dx = [1, -1, 0, 0]
        dy = [0, 0, 1, -1]
        INF = 2147483647
        queue = deque()

        def bfs(queue):
            while queue:
                pos = queue.popleft()
                x0, y0 = pos
                x0, y0, rooms[x0][y0]
                for k in range(4):
                    x = x0 + dx[k]
                    y = y0 + dy[k]
                    if 0 <= x < m and 0 <= y < n and rooms[x][y] == INF:
                        rooms[x][y] = rooms[x0][y0] + 1
                        queue.append((x, y))

        for i in range(m):
            for j in range(n):
                if rooms[i][j] == 0:  # 现在从每扇门出发
                    queue.append((i, j))
        bfs(queue)
        return rooms

class Solution155(object):

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.a = []
    def push(self, x):
        """
        :type x: int
        :rtype: None
        """

        self.a.append(x)

    def pop(self):
        """
        :rtype: None
        """
        return self.a.pop()
    def top(self):
        """
        :rtype: int
        """
        return self.a[-1]
    def getMin(self):
        """
        :rtype: int
        """
        return min(self.a)

class Solution200(object):
    def numIslands(self,grid):
        from collections import deque
        if not grid or not grid[0]:
            return 0
        m,n = len(grid),len(grid[0])
        dx = [1,-1,0,0]
        dy = [0,0,-1,1]
        visted = set()
        def flood(i,j):
            q = deque()
            grid[i][j] = 0
            q.append((i,j))
            visted.add((i,j))
            while q:
                pos = q.popleft()
                i,j = pos
                for k in range(4):
                    x = i+dx[k]
                    y = j+dy[k]
                    if 0<=x<m and 0<=y<n and (x,y) not in visted and grid[x][y] == '1':
                        grid[x][y] = 0
                        q.append((x,y))
                        visted.add((x,y))

        result = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] == '1':
                    flood(i,j)
                    result += 1
        return result

'''
class Solution752(object):
    def openlock(self,deadends,target):

class MinSquare(object):
    def numSquares(self,n):
        x = int(pow(n,0.5))
        for i in x:
            find(i)
        def find(x):
            for i in range(x):
'''

class Solution20():
    def isValid(self,s):
        a = {')': '(', ']': '[', '}': '{'}
        l = [None]
        for i in s:
            if i in a and a[i] == l[-1]:
                l.pop()
            else:
                l.append(i)
            print(l)
        return len(l) == 1

class Solution150(object):
    def evalRPN(self, tokens):
        """
        :type tokens: List[str]
        :rtype: int
        """
        stack = []
        for i in tokens:
            if i not in ['+','-','*','/']:
                stack.append(int(i))
            if i in ['+','-','*','/']:
                if i == '+':
                    a = stack[-1]+stack[-2]
                    stack.pop()
                    stack.pop()
                    stack.append(a)
                if i == '-':
                    a = stack[-1]-stack[-2]
                    stack.pop()
                    stack.pop()
                    stack.append(a)
                if i == '*':
                    a = stack[-1]*stack[-2]
                    stack.pop()
                    stack.pop()
                    stack.append(a)
                if i == '/':
                    a = stack[-2]/stack[-1]
                    stack.pop()
                    stack.pop()
                    stack.append(int(a))
        return stack[0]

class Solution739(object):
    def dailyTemperatures(self, T):
        """
        :type T: List[int]
        :rtype: List[int]
        """
        results = [0] * len(T)
        stack = []
        for index,tep in enumerate(T):
            while stack and tep>T[stack[-1]]:
                results[stack[-1]] = index - stack[-1]
                stack.pop()
            stack.append(index)
        return results

class Solution2():
    def addTwoNumbers(self,l1,l2):
        jin = 0

        head = ListNode(0)#返回的是头节点的next，而不是用于操作的p的next
        p = head
        while l1 and l2:
            p.next = ListNode((l1.val + l2.val + jin) % 10)
            jin = int((l1.val + l2.val + jin) / 10)
            l1 = l1.next
            l2 = l2.next
            p = p.next

        while l1:
            p.next = ListNode((l1.val + jin) % 10)
            jin = (l1.val + jin) // 10
            l1 = l1.next
            p = p.next

        while l2:
            p.next = ListNode((l2.val + jin) % 10)
            jin = (l2.val + jin) // 10
            l2 = l2.next
            p = p.next

        if jin == 1:
            p.next = ListNode(1)
        return head.next