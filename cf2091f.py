import sys 
# sys.setrecursionlimit(1000000000)
# sys.set_int_max_str_digits(1000000)
import math
import heapq
import time 
import random
import bisect
from collections import deque,Counter
from collections import defaultdict
from itertools import combinations, permutations
from itertools import combinations_with_replacement
from bisect import bisect_left, bisect_right
# from sortedcontainers import SortedList

#lcm = (a*b)//math.gcd(a,b)
#nCr combinatorics qn use math.comb(n,k)
#https://codeforces.com/problemset/page/12?tags=greedy&order=BY_RATING_ASC
#https://codeforces.com/problemset/page/1?tags=binary+search%2Ccombine-tags-by-or%2Cconstructive+algorithms%2Cdata+structures%2Cgreedy%2Cnumber+theory%2Ctwo+pointers%2C1400-&order=BY_RATING_ASC
# 12 50
# MOD = 10**9 + 7

# Author: nazeef, cf 2091f from codeforces round 1013 div3, a really nice prefix sum dp with a lot of preprocessing involved, im surprised i actually got this , the 1.6k solves 
# is hella fishy tho cos i really dont think this is that easy but as it turns out they literally livestreamed the answer on youtube and telegram so what can you do lol 
# ok so at every row u MUST take one  OR two holds, and the euclidean distance between any 2 consecutive holds <= r. this qn is difficult to do top down
# because u need to compute thte prefix sum of ur prev dp levels , quite dynamic actually, so dp[i][j][1] means how many ways if there is a hold at a[i][j], 
# and its the first hold im taking in the ith row, this means u basically find the leftmost and rightmost columns that have euclid distance <= d, this is easily 
# precomputed in mlogm time by binary search. so how do we count the number of ways from dp[i-1][l][1] to dp[i-1][r][1] ? just prefix sum! also take not u can also 
# transition from dp[i-1][l:r][2] because u cld also have taken 2 holds in the previous row. very nice stuff, but how to compute dp[i][j][2] then? notice that 
# in order to take 2 holds, u must have taken one hold from ur current row already, which means u MUST compute dp[i][j][1] for all j in the entire row FIRST, 
# compute the new prefix sum of dp[i][j][1], and then apply the same idea, find left and right bounds, then take prefix sum, so dp[i][j][2] = dp[i][r][1] - dp[i][l][1]
# and dont forget to modulo, very nice dp prefix sum precomp, very nice transitions also 
mod = 998244353
def solve(n,m,d,a):

    dp = [[[0]*3 for i in range(m)] for i in range(n)]
    bound = [[0,0] for i in range(m)]
    p1 = [0]*(m+1)
    p2 = [0]*(m+1)
    for i in range(m):
        if a[n-1][i] == "X":
            dp[n-1][i][1] = 1
        left,right = i,m-1 
        r = left 
        while left <= right: 
            mid = left + (right - left) // 2
            if (mid - i) ** 2 <= d**2 - 1:
                r = mid 
                left = mid+1
            else:
                right = mid - 1
        left,right = 0,i 
        while left <= right: 
            mid = left + (right - left) // 2
            if (mid - i) ** 2 <= d**2 - 1:
                l = mid 
                right = mid-1 
            else:
                left = mid+1 
        bound[i] = [l,r]
    for i in range(m):
        p1[i+1] = p1[i] + dp[n-1][i][1] 
    
    for i in range(m):
        if a[n-1][i] == "X":
            l,r = max(i-d,0), min(i+d,m-1)
            dp[n-1][i][2] = p1[r+1] - p1[l] - dp[n-1][i][1]
        p2[i+1] = dp[n-1][i][2] + p2[i]

    for i in range(n-2,-1,-1):
        for j in range(m):
            if a[i][j] == "X":
                l,r = bound[j][0], bound[j][1]
                dp[i][j][1] = (p1[r+1] - p1[l] + p2[r+1] - p2[l])%mod
        
        for j in range(m):
            p1[j+1] = (p1[j] + dp[i][j][1])%mod
        for j in range(m):
            if a[i][j] == "X":
                l,r = max(j-d,0), min(j+d,m-1)
                dp[i][j][2] = (p1[r+1] - p1[l] - dp[i][j][1])%mod
            p2[j+1] = (dp[i][j][2] + p2[j])%mod
    res = 0
    for i in range(m):
        res = (res + dp[0][i][1] + dp[0][i][2])%mod
    print(res)


    
    
# count number of valid paths basically 

# in one level we MUST take a hold, and we can take at most 2 holds 

# ok just travserse the array if theres any row with no holds just return 0

# okay so we cant skip a level also, 

# ok dp[i][j] just means no. of valid paths ending at a[i][j], i dont need to store all n rows 
# just need to store i-1th row 

# okay so when im at some cell, im either the 1st hold in that ith row to be used, or im the 2nd hold 

# if im the 2nd hold, then i must go to the next row already 
# if im the first hold, i can either take another hold in the same row where k - j >= d or j - k >= d 

# actly for every cell i can sort of precompute the left and right boundary of the row below 
# using bin serach, can do it in nmlogm should pass 

# dp[i][j][1] means if im taking a[i][j] as the first hold in ith row, means i just take sum of dp[i-1][leftmost: rightmost][can be from both 1 or 2 holds]
# which i can use prefix sum to calculate 

# how to calculate dp[i][j][2] then? 
# dp[i][j][2] MUST be from dp[i][j][1]
if __name__ == "__main__":
    q = int(input())
    for _ in range(q):
        # n,m= map(int,input().split())
        n,m,k = map(int,input().split())
        # n = int(input())
        # r = int(input())
        # # # #     # b = input()
        # a = list(map(int,input().split()))
        # b = list(map(int,input().split()))
            #     # b = input()
            # # c = input()
        a = []
        # a = input()
        # b = list(map(int,input().split()))
        # # # m = int(input())
        # b = []

        for i in range(n):
        # a = input()
            c = input()
            # c = int(input())
            # c = list(map(int,input().split()))
            a.append(c)
                # d = [b,a,c] 
        res2 = solve(n,m,k,a)
     
  
    # for j in res2:
    #     print(j)
        # print(f"Case #{_+1}: {res2[0]} {res2[1]} ")  
        
# # # # # # Remove or comment out the file reading part before submitting to codeforces
# with open('codeforcesinput.txt', 'r') as f:
#     lines = f.read().splitlines()
#     q = int(lines[0])
#     index = 1
#     #index += no.of lines per test c ase
#     #for strings instead of a =input() it is a = lines[index]
#     # n,m = map(int,lines[index].split())
   
#     for _ in range(q):
#         # a = lines[index]
#         # n = int(lines[index])

#         # s = lines[index]
#         n,m,k =map(int,lines[index].split())
#         # b = lines[index].split()
#         # a,k = b[0], int(b[1])
#         # print(a)
#         # k = int(lines[index+1])
#         # r = int(lines[index+2])
#         # a = lines[index]
#         # m = int(lines[index+1])
#         # a = lines[index+1]
#         # c = lines[index+3]
#         # d = [a,b,c]
#         # # n = lines[index+1]
#         # a = list(map(int,lines[index+1].split()))
#         # # b = lines[index+2]
#         # b = list(map(int,lines[index+2].split()))
#         a = []
#         # b = []
#         for i in range(n):
#             # b = list(map(int,lines[index+1+i].split()))
#             d = lines[index+i+1]
#             # c = int(lines[index+2+i])
#             a.append(d)
#         # a = []
#         # # # m = int(lines[index+2])
#         # # # a = []
#         # for i in range(n):
#         #     d = list(map(int,lines[index+1+i].split()))
#         #     # c = lines[index+i+1]
#         #     a.append(d)
#         # c = list(map(int,lines[index+3].split()))

#         # b = []
#         # for j in range(n):
#         #     c = lines[index+2+j]
#         #     b.append(c)
#         # b = list(map(int,lines[index+5].split()))
#         res2 = solve(n,m,k,a)
#         index += n+1
    #         # print(f"Case #{_+1}: {res2}")  
            
        


# # #     MAX = 200000  # Adjust as needed depending on your problem constraints

# # #     # Precompute factorials
# # #     factorial = [1] * (MAX + 1)
# # #     for j in range(2, MAX + 1):
# # #         factorial[j] = (factorial[j - 1] * j)%mod

  
# #     # def mod_inv(x, mod):
# #     #     return pow(x, mod - 2, mod)

# #     # def comb(n, k):
# #     #     if k > n:
# #     #         return 0
# #     #     numerator = factorial[n] % mod
# #     #     denominator = (factorial[k] * factorial[n - k]) % mod
# #     #     return (numerator * mod_inv(denominator, mod)) % mod
# #     # def mod_exponentiation(base, exponent, mod):
# #     #     result = 1
# #     #     base = base % mod  # Ensure base is within mod
        
# #     #     while exponent > 0:
# #     #         # If exponent is odd, multiply the base with the result
# #     #         if (exponent % 2) == 1:
# #     #             result = (result * base) % mod
            
# #     #         # Now exponent must be even, square the base
# #     #         base = (base * base) % mod
            
# #     #         # Divide the exponent by 2
# #     #         exponent = exponent // 2
# #     #     return result

# # # # # # class DisjointSetUnion:
# # # # # # #     def __init__(self, n):
# # # # # # #         self.parent = list(range(n))
# # # # # # #         self.rank = [1] * n
    
# # # # #     def find(self, x):
# # # # #         if self.parent[x] != x:
# # # #             self.parent[x] = self.find(self.parent[x])  # Path compression
# # # #         return self.parent[x]
    
# # # #     def union(self, x, y):
# # # #         rootX = self.find(x)
# # # #         rootY = self.find(y)
        
# # # #         if rootX != rootY:
# # # # #             # Union by rank
# # # # #             if self.rank[rootX] > self.rank[rootY]:
# # # # #                 self.parent[rootY] = rootX
# # # # #             elif self.rank[rootX] < self.rank[rootY]:
# # # # #                 self.parent[rootX] = rootY
# # # #             else:
# # #                 self.parent[rootY] = rootX
# # # #                 self.rank[rootX] += 1
# a = 5000000
# p = [True]*(a+1)
# p[0] = p[1] = False 
# for j in range(2,int(a**0.5)+1):
#     if p[j] == True:
#         p[j] = j
#         # print(j)
#         for j in range(j**2,a+1,j):
#             # print(j,j)
#             p[j] = j

# for j in range(int(a**0.5)+1,len(p)):
#     if p[j] == True:
#         p[j] = j
# # print(p[:100])
# pref = [0]*(a+1)
# for j in range(2,a+1):
#     c = j 
#     cnt = 0
#     while c>1:
        
#         d = p[c]
#         c = c//d 
#         cnt+=1 
#     pref[j] = pref[j-1]+cnt
