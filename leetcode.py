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

class Solution136():#一个数异或同一个数两次还是自身
    def singlenumber(self,nums):
        a = 0
        for i in nums:
            a ^= i
        return  a

class Solution7():
    def reverse(self,x):
        mark = 0
        if x<0:mark=1
        x=str(abs(x))
        x=x[::-1]
        x=int(x)
        if -2 ** 31 < x < 2 ** 31 - 1:
            return (x * ((-1) ** mark))
        else:
            return 0

class Solution9():
    def isPalindrome(self,x):
        x=str(x)
        for i in range(len(x)//2):
            if not x[i] == x[-i-1]:
                return False
        return True

class Solution14():
    def longestCommonPrefix(self,strs):
        pre = []

class Solution994():
    def orangesRotting(self,grid):
        from collections import deque
        quque = deque()
        m,n = len(grid),len(grid[0])
        dx = [1,-1,0,0]
        dy = [0,0,1,-1]
        def bfs(queue):
            time = 0
            while quque:
                pos = quque.popleft()
                i,j,time=pos
                for k in range(4):
                    i1=i+dx[k]
                    j1=j+dy[k]
                    if 0<=i1<m and 0<=j1<n:
                        if grid[i1][j1]==1:
                            grid[i1][j1]=2
                            quque.append((i1,j1,time+1))
            return time

        for i in range(m):
            for j in range(n):
                if grid[i][j] == 2:
                    quque.append((i,j,0))
        result = bfs(quque)
        if any(1 in row for row in grid):
            return -1
        return result

class Solution53():
    def brute(self,nums):
        max_num = nums[0]
        for i in range(len(nums)):
            sums = 0
            for j in range(i,len(nums)):
                sums += nums[j]
                if sums>max_num:
                    max = sums
        return max_num
    def dp(self,nums):
        n = len(nums)
        max_num = nums[0]
        for i in range(1,n):
            if nums[i-1]>0:
                nums[i] += nums[i-1]
            max_num = max(max_num,nums[i])
        return max_num
    def greedy(self,nums):
        max_num = nums[0]
        sums = 0
        for i in range(len(nums)):
            sums += nums[i]
            max_num = max(max_num,sums)
            if sums<0:
                sums = 0
        return max_num
    def divide(self,nums):
        return self.helper(nums,0,len(nums)-1)
    def helper(self,nums,left,right):
        if left==right:
            return nums[left]
        p = (left+right)//2
        left_sum = self.helper(nums,left,p)
        right_sum = self.helper(nums,p+1,right)
        cross_sum = self.cross_sum(nums,left,right,p)
        return max(left_sum,right_sum,cross_sum)
    def cross_sum(self,nums,left,right,p):
        if left == right:
            return nums[left]
        left_sums = 0
        left_max = float('-inf')
        for i in range(p,left-1,-1):
            left_sums += nums[i]
            left_max = max(left_max,left_sums)
        right_sums = 0
        right_max = float('-inf')
        for i in range(p+1,right+1):
            right_sums += nums[i]
            right_max = max(right_max, right_sums)
        return left_max+right_max
class Solution1103():
    def distributeCandies(self,candies,num_people):
        ans = [0]*num_people
        i = 0
        while candies:
            ans[i%num_people] += min(i+1,candies)
            candies -= min(i+1,candies)
            i += 1
        return ans
    def anotherway(self,candies,num_people):
        p = int((2*candies+0.25)**0.5-0.5)
        ans = [0]*num_people
        remain = int(candies-p*(p+1)*0.5)
        turns = p//num_people
        cols = p%num_people
        for i in range(num_people):
            ans[i] =(i + 1) * turns + int(turns * (turns - 1) * 0.5) * num_people
            if i<cols:
                ans[i]+=i+1+turns*num_people
        ans[cols] += remain
        return ans

class Solution57():
    def findContinuousSequence(self,target):
        ans = []
        i=1
        j=2
        while i<target/2:
            if (i+j)*(j-i+1)/2 == target:
                ans.append(list(range(i,j+1)))
                i +=1
            elif (i+j)*(j-i+1)/2 < target:
                j+=1
            elif (i+j)*(j-i+1)/2 > target:
                i+=1
        return ans

class Solution543():
    def diameterOfBinaryTree(self, root):
        ans = 1
        def depth(node):
            if not node:
                return 0
            L = depth(node.left)
            R = depth(node.right)
            ans = max(ans,L+R+1)
            return max(L,R)+1
        depth(root)
        return ans-1

class Solution206():
    def reverse(self,head):
        pre = None
        cur = head
        while cur:
            tep = cur.next
            cur.next = pre
            pre = cur
            cur = temp
        return pre

class Solution1013():
    def canThreePartsEqualSum(self, A):
        sums = int(sum(A) / 3)
        if not sum(A) % 3 == 0:
            return False
        i = 1
        a = A[0]

        while not (a == sums) and i<len(A)-1:
            a += A[i]
            i += 1
        j = -2
        c = A[-1]
        while not (c == sums) and j>-len(A):
            c += A[j]
            j -= 1
        if not (i <=(j+len(A))):
            return False
        b=A[i]
        while i<j+len(A):
            i+=1
            b+=A[i]
        if b==sums:
            return True
        else:
            return False

class Solution1071():
    def gcdOfStrings(self, str1, str2):
        for i in range(min(len(str1),len(str2)),0,-1):
            if len(str1)%i==0 and len(str2)%i==0:
                if str1[:i]*(int(len(str1)/i))==str1 and str1[:i]*(int(len(str2)/i))==str2:
                    return str1[:i]
        return ''

class Solution912():
    def selection_sort(self,nums):
        n=len(nums)
        for i in range(n):
            for j in range(i,n):
                if nums[i] > nums[j]:
                    nums[i],nums[j] = nums[j],nums[i]
        return nums
    def bubble_sort(self,nums):
        n=len(nums)
        for i in range(n):
            for j in range(1,n-i):
                if nums[j-1]>nums[j]:
                    nums[j-1],nums[j]=nums[j],nums[j-1]
        return nums
    def insection_sort(self,nums):
        pass
    def shell_sort(self,nums):
        pass
    def merge_sort(self,nums):
        pass
    def quick_sort(self,nums):
        n=len(nums)
        def quick(left,right):
            if left>=right:
                return nums
            pivot = left
            i=left
            j=right
            while i<j:
                while i<j and nums[j]>nums[pivot]:
                    j-=1
                while i<j and nums[i]<=nums[pivot]:
                    i+=1
                nums[i],nums[j]=nums[j],nums[i]
            nums[pivot],nums[j]=nums[j],nums[pivot]
            quick(left,j-1)
            quick(j+1,right)
            return nums
        return quick(0,n-1)
    def heap_sort(self,nums):
        pass
    def counting_sort(self,nums):
        pass
    def bucket_sort(self,nums):
        pass
    def radix_sort(self,nums):
        pass

class Solution5():
    def longestPalindrome(self,str):
        size = len(str)
        max_pa = 1
        start = 0
        if size<2:
            return str
        dp = [[False for _ in range(size)] for _ in range(size)]
        for i in range(size):
            dp[i][i] = True
        for j in range(1,size):
            for i in range(0,j):
                if str[i]==str[j]:
                    if j-i<3:
                        dp[i][j]=True
                    else:dp[i][j]=dp[i+1][j-1]
                else:
                    dp[i][j]=False
                if dp[i][j]==True:
                    cur_pa = j-i+1
                    if cur_pa>max_pa:
                        max_pa = cur_pa
                        start = i
        return str[start:start+max_pa]
    def mid_spread(self,s):
        size = len(s)
        if size<2:
            return s
        max_len = 1
        res = s[0]
        for i in range(size):
            p_odd,len_odd = self.center_spread(s,size,i,i)
            p_even,len_even = self.center_spread(s,size,i,i+1)
            cur_p = p_odd if len_odd >= len_even else p_even
            if len(cur_p)>max_len:
                max_len = len(cur_p)
                res = cur_p
        return res

    def center_spread(self,s,size,left,right):
        i=left
        j=right
        while i>=0 and j<size and s[i]==s[j]:
            i-=1
            j+=1
        return s[i+1:j],j-i+1

sol = Solution5()
print(sol.mid_spread(''))