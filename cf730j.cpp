#include <iostream>
#include <vector>
#include <fstream>  // For file input
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <cmath> // Include cmath for pow
#include <algorithm> //this is for sorting etc 
#include <queue> //this one is for heap and deque 
#include <bitset>
#include <bits/stdc++.h>
#include <numeric> 
#include <map>
#include <stack>
#include <functional>  // For std::greater (min-heap)
#include <string>
#include <ext/pb_ds/assoc_container.hpp> 
typedef long long ll;
#define pb push_back 
#define mp make_pair
#define FOR(i, n) for(int i = 0; i < (n); i++)
using namespace std;

// fun fact, using fermats little theorem, we can compute the nCk table in o(n) time, and k up to n also, so yeah, rmb to call initFactorials tho lolol 
int MOD = 1e9+7;  // or use 0 if no modulo needed

const int MAXN = 1e6+5;

long long fact[MAXN], invFact[MAXN];

ll modpow(ll base, ll exp, ll mod) {
    ll res = 1;
    while (exp > 0) {
        if (exp % 2) res = (res * base) % mod;
        base = (base * base) % mod;
        exp /= 2;
    }
    return res;
}

void initFactorials() {
    fact[0] = invFact[0] = 1;
    for (int i = 1; i < MAXN; i++) {
        fact[i] = fact[i - 1] * i % MOD;
        invFact[i] = modpow(fact[i], MOD - 2, MOD);
    }
}

long long nCk(int n, int k) {
    if (k < 0 || k > n) return 0;
    return fact[n] * invFact[k] % MOD * invFact[n - k] % MOD;
}

int findp(int x, vector<int>& parent) {
    if (x != parent[x]) {
        parent[x] = findp(parent[x], parent); // Path compession
    }
    return parent[x];
}

// Union function with rank optimization
void unite(int x, int y, vector<int>& parent, vector<int>& rank) {
    int rootX = findp(x, parent);
    int rootY = findp(y, parent);

    if (rootX != rootY) {
        if (rank[rootX] > rank[rootY]) {
            parent[rootY] = rootX;
        } else if (rank[rootX] < rank[rootY]) {
            parent[rootX] = rootY;
        } else {
            parent[rootY] = rootX;
            rank[rootX]++;
        }
    }
}



bool connected(int x, int y, vector<int>& parent) {
    return findp(x,parent) == findp(y,parent);
}

vector<int> prime(MAXN, 1), spf(MAXN, -1); // smallest prime factor

// Sieve of Eratosthenes + SPF
void build_spf() {
    prime[0] = prime[1] = 0;
    spf[0] = 0;
    spf[1] = 1;

    for (int i = 2; i * i < MAXN; ++i) {
        if (prime[i]) {
            spf[i] = i;
            for (int j = i * i; j < MAXN; j += i) {
                if (spf[j] == -1) spf[j] = i;
                prime[j] = 0;
            }
        }
    }

    for (int i = 2; i < MAXN; ++i) {
        if (spf[i] == -1) {
            spf[i] = i; // mark itself if prime
        }
    }
}

// prime factorization for each number from 1 to MAXN-1
vector<map<int, int>> pf(MAXN); // pf[i] = map of {prime -> exponent}

void build_pf() {
    for (int i = 2; i < MAXN; ++i) {
        int x = i;
        while (x > 1) {
            int p = spf[x];
            int cnt = 0;
            while (x % p == 0) {
                x /= p;
                cnt++;
            }
            pf[i][p] = cnt;
        }
    }
}

map<int,int> get_pf_of_product(const vector<int>& c) {
    map<int,int> res;
    for (int x : c) {
        map<int,int> pf_x = pf[x];
        for (auto &[p,e] : pf_x) {
            res[p] += e;
        }
    }
    return res;
}

bool is_divisible(const map<int,int>& f, const map<int,int>& pf_prod) {
    for (auto &[p,e] : pf_prod) {
        if (f.find(p) == f.end() || f.at(p) < e) return false;
    }
    return true;
}

// basic tips:
// always check for overflow especially if it fails on some high petest like >= 7 lol , ll(a[i]*x) can overflow, shld do ll(a[i]) * x
// for tree poblems, sometimes u might not even need to reroot due to some symmetry poperty, so sometiems we can jsut give an answer by fixing 0 as root 
// for question involving formulas, can try to rearrange formulas like basic algebra
// sometimes the poblem is simpler if we try to solve for the converse of what the qn is asking 
// subarray counting poblems, we can do some method like for each index i, find the left boundary and right boundary that satisfies poperty, then take 
// poduct of i-l * r-i 
// never ever use unordered map, cfm will get hacked lol 
// for offlien query questions, if ure stuck, can always try some big small query split eg for queries with some poepty x <= sqrt(n)
// we do some pecomp, and for x > sqrt(n) just brute force simulate, 


bool compareBySecond(const vector<int>& a, const vector<int>& b) {
    return a[1] < b[1]; // Sort in descending order
}

int n,k;
vector<vector<int>>a;
vector<int>b;
int tot,cur;
int inf = 1e9;
vector<vector<vector<int>>> dp;

// okay greddily just calculate the min number of bottls required to store everything 
// so i get this min no. of bottles k, 
// then we do like a dp[i][j] means min time to fill j bottles if im at ith bottle? 
// so at the ith bottle, if j < k, 

// okay what if we try to fix the final bottles we are going to use then? 

// suppose we already chose some subset of k bottles, then the answer is just total sum of ai - sum of ai in k no?? 

// okay for a set of k bottles that i took, 
// what do i need to maintain? 

// sum of b[1] + b[2] + b[3] + ... b[k] >= tot 
// i also need to know how 

// okay nice, got ac for this, im surprised it didnt tle tho, cos the constraints i think is O(n*n*sum(a)) which can be at wost 10 ^ 8? 
// but i think with smart pruning and finding the earliest index i that we can start our initial dfs from prolly dropts it to like 10 ** 7.5 or smth lol 
// very nice question tho, its basically just a subset sum problem, basically we first find the min k bottles to fill up sum of a, which we can find fast 
// by greedilt sorting b and adding the largest, from there we know the first part of the answer, the 2nd part about min time requires a dp solution 
// because u dont always need to use the k largest bottles, so yeah, this is subset sum, dp state is dp[index of current bottle][how many bottles ive used so far to fill][sum of capacitiy of bj]
// but yeah, the transition is, dp[i][j][sum] = max(dp[i+1][j+1][sum+b[i]] - a[i], dp[i+1][j][sum]) basically just take not take, its just the base cases are slightly more unique 
// and isntead of the usual adding , we subtract the amount of ai from each bottle we choose to take, then at teh end of the bsae case, we ADD total amount of ai intiailly to get 
// the exact amount we have to transfer!

int dfs(int i, int j, int sum) {
    if (j > k){
        return inf;
    }
    if (j == k){
        if (sum < tot){
            return inf;
        }
        return tot;
    }
    if (i == n){
        return inf;
    }
    if (dp[i][j][sum]!=-1){
        return dp[i][j][sum];
    }
    int res = inf;
    res = dfs(i+1,j+1, sum + a[i][1]) - a[i][0];
    res = min(res, dfs(i+1,j,sum));
    dp[i][j][sum] = res;
    return res;
}

int main(){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    // ifstream cin("codeforcesinput.txt"); // comment this if using standard input
    int T;
    // cin >> T;
    T = 1;
    while (T--) {
        cin >> n;
        a.assign(n,vector<int>(2,0));
        b.assign(n,0);
        tot = 0;
        cur = 0;
        k = n;
        for (int i = 0; i <n ;++i){
            cin >> a[i][0];
            tot += a[i][0];
        }
        for (int i = 0; i <n ;++i){
            cin >> a[i][1];
            b[i] = a[i][1];
        }
        // cout << tot << endl;
        sort(b.begin(),b.end());
        for (int i = n-1; i >=0; --i){
            cur += b[i];
            if (cur >= tot){
                k = n-i;
                break;
            }
        }
        int first = 0;
        for (int i = 0; i <n;++i){
            cur = b[i];
            bool ok = 0;
            int cnt = 1;
            for (int j = n-1;j > i; --j){
                cur += b[j];
                cnt++;
                if (cnt > k){
                    break;
                }
                if (cur >= tot){
                    ok = 1;
                    break;
                }
            }
            if (ok){
                first = i;
                break;
            }
        }
        // cout << first << endl;
        sort(a.begin(), a.end(), compareBySecond);
        dp.assign(n+1, vector<vector<int>>(k+1, vector<int>(tot+1,-1)));
        int res = dfs(first,0,0);
        cout << k << " " <<  res << endl;
       
    }
    return 0;
}

