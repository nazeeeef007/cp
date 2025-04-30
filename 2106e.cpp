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
typedef long long ll;
#define pb push_back 
#define mp make_pair
#define FOR(i, n) for(int i = 0; i < (n); i++)
using namespace std;
// reverse(arr.begin() + i, arr.begin() + j + 1); Reverse the subarray from index i to j
// string res2(res.begin(),res.end()); convert array of letters to a string, similar to pythons "".join(res)
// int mid = left + (right - left) / 2; correct way to bin search to avoid overflow
// sort(c.begin(), c.end(), [](const vector<int>& x, const vector<int>& y) {
//     return x[0] > y[0]; // Compare first elements in descending order
// });

// for combinatorics qns with modulo and factorial etc  

// const int MOD = 998244353;
// int MAX_N = 500005;


// vector<long long> fact(MAX_N + 1), invFact(MAX_N + 1);


// long long modPow(long long base, long long exp, long long mod) {
//     long long res = 1;
//     while (exp > 0) {
//         if (exp % 2) res = (res * base) % mod;
//         base = (base * base) % mod;
//         exp /= 2;
//     }
//     return res;
// }


// void precomputeFactorials() {
//     fact[0] = invFact[0] = 1;
//     for (int i = 1; i <= MAX_N; i++) {
//         fact[i] = (fact[i - 1] * i) % MOD;
//     }
//     invFact[MAX_N] = modPow(fact[MAX_N], MOD - 2, MOD);
//     for (int i = MAX_N - 1; i >= 1; i--) {
//         invFact[i] = (invFact[i + 1] * (i + 1)) % MOD;
//     }
//     invFact[0] = 1;
// }



// long long nCr(int n, int r) {
//     if (r > n || r < 0) return 0;
//     return (((fact[n] * invFact[r]) % MOD) * invFact[n - r]) % MOD;
// }

// for dsu rmb to initialise these
vector<int> parentf;
vector<int> rankf;
// vector<int> parentg;
// vector<int> rankg;
// parent.resize(n);
// rank.resize(n, 0);
// for (int i = 0; i < n; ++i) {
//     parent[i] = i; // Each node is its own parent
// }



// int sqrt_n = static_cast<int>(std::sqrt(n));  // Take the square root and cast to in
// Function to compare by largest first value
bool compareByFirst(const vector<int>& a, const vector<int>& b) {
    return a[0] > b[0]; // Sort in descending order
}

// Function to compare by largest second value
bool compareBySecond(const vector<ll>& a, const vector<ll>& b) {
    return a[1] > b[1]; // Sort in descending order
}

bool compareBySecondThenFirst(const vector<int>& a, const vector<int>& b) {
    if (a[1] == b[1]) {
        return a[0] > b[0]; // If second values are equal, sort by first values
    }
    return a[1] > b[1]; // Sort by second values in descending order
}

// how to sort a graph of [u,v,weight] by weight 
// sort(edges.begin(), edges.end(), [](const vector<int>& a, const vector<int>& b){
//         return a[2] < b[2];
//     }
// );
int mod = 1e9 + 7;
// ll MOD = 998244353;
// ll mod = 998244353;
// findp function with path compression
int findp(int x, vector<int>& parent) {
    if (x != parent[x]) {
        parent[x] = findp(parent[x], parent); // Path compression
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

// author : nazeef
// topic: greedy + binary search, a nice 1800 problem, took me quite long to solve tho, cos i had a hard time tryna understand what they wanted from the qn lol also 
// cos its after my last finals paper and im kinde tired but wtv 
// basically the idea is, u need to greedily choose elements larger or smaller than k so that ur binary search fits, which means u dont care about the abs value 
// of each element, u jsut care whether its smaller or bigger than k. and heres another lemma, if k is at some index eg 2 and we start in some range l,r e [1,17]
// there is a uniqeue sequence of left and rights we must take in our binarys earch to reach k's index of 2, so this isnt random, its deterministic. a small 
// greedy idea is, how shld we swap? to kill 2 birds with one stone, anytime we need a bigger number than k, we can swap it with some other index that needed a 
// smaller number than k, so u only need 2 swaps! also its impossible if k doesnt lie in range[l,r], or eg u need 5 smaller, but k <= 5, which means impossible 





void solve(int n, int m, vector<int> & a , vector<vector<int>> & q ) {
    vector<int> pos(n+1,0);
    for (int i = 0; i <n ; ++i) {
        pos[a[i]] = i;
    }
    for (int i = 0; i <m ; ++i) {
        int l = q[i][0]-1;
        int r = q[i][1]-1;
        int k = q[i][2];
        if (pos[k] < l || pos[k] > r) {
            cout << -1 << " ";
            continue;
        }
        int p = pos[k];
        int maxi = k+1;
        int mini = k-1;
        int big = k+1;
        int small = k-1;
        while (l <= r) {
            int mid = l + (r - l) / 2;
            if (a[mid] == k) {
                break;
            }
            if (a[mid] < k) {
                if (mid > p) {
                    maxi++;
                    r = mid -1;
                    big++;
                    
                }
                else {
                    l = mid + 1;
                    small--;
                }

            }
            else if (a[mid] > k) {
                if (mid < p) {
                    mini--;
                    l = mid +1;
                    small--;
                }
                else{
                    r = mid - 1;
                    big++;
                }
            }
        }
        if (big > n+1 || small < 0) {
            cout << -1 << " ";
            continue;
        }   
        int upper = max(maxi - k -1, k - mini - 1);
        int low = min(maxi - k -1, k - mini - 1); 
        cout << low* 2 + (upper - low) * 2 << " ";  

    }
    cout << endl;


    
}





int main() { 
    ios::sync_with_stdio(false); // Unsync C++ streams from C streams
    cin.tie(nullptr); // Untie cin from cout for faster input/output
    // vector<int> p(2e5+1,1);
    // vector<int> pf(2e5+1,1e6);
    // p[0] = p[1] = 0;
    // pf[1] = 1;
    // for (int i = 2; i < (int)(sqrt(2e5+1)); i++){
    //     if (p[i]){
    //         pf[i] = i; 
    //         for (int j = i*i; j<2e5+1;j += i){
    //             p[j] = 0;
    //             pf[j] = min(pf[j],i);
    //         } 
    //     }
    // }
    int q;
    cin >> q;  // Read number of test cases
    // precomputeFactorials();
    
    for (int _ = 0; _ < q; ++_) {
        int n,m,k,x,y;
        // n = 26;
        // ll n,k; 
        // int m;
        // int c ; // Declare n and m
        cin >> n >> m;  // Read n
        // vector<vector<vector<ll>>> b(n);
        // cin >> n ;
        // cin >> m; 
        // vector<char> b(m);
        // string a;
        // a.resize(n);
        // vector<vector<ll>> b(n, vector<ll>(2));
        vector<int> a(n);
        // vector<string> a(n);
        // vector<vector<string>> b(m,vector<string>(n));
        // vector<int> b(k);
        // vector<int> c(m);
        // vector<vector<int>> a(n-1,vector<int>(2));
        vector<vector<int>> b(m,vector<int>(3));
   
        for (int i = 0; i < n; i++) {
            cin >> a[i];
        }
        // for (int i = 0; i < k; i++) {
        //     cin >> b[i];
        // }
        // for (int i = 0; i < m; i++) {
        //     cin >> c[i];
        // }
        // string a;
        
        // // vector<vector<int>> a(m,vector<int>(2));
        // string b;
        // cin >> a >> b;
        // cin >> a[1];
        // string c;
        // cin >> m ;
        // vector<vector<int>> b(m,vector<int>(3));
        // // vector<int> a(n);  // Initialize vector of size n
        // // vector<int> result(n);  // Initialize result vector of size n
        // for (int i = 0; i<m; i++){
        //     for (int j = 0; j < n; ++j) {
        //         cin >> b[i][j] ;
    
        //     }
        // }
        // for (int i = 0; i < m; i++) {
        //     cin >> a[i][0] >> a[i][1] ;
        // }
        for (int i = 0; i < m; i++) {
            cin >> b[i][0] >> b[i][1] >> b[i][2] ;
        }
        // for (int i = 0; i < n; i++) {
        //     cin >> b[i];
        // }
        // cin >> m;
        
        solve(n,m,a,b);

    // Print the result for the current test case in a single line
    // cout << "Case #" << (_ + 1) << ": ";
    // for (int i = 0; i < result.size(); ++i) {
    //     cout << result[i];  // Print each element
    //     if (i < result.size() - 1) cout << " ";  // Print space between elements
    // }
    // cout << endl;  // New line after printing all elements
    }

    return 0;
}


// int main() {
//     ios::sync_with_stdio(false); // Unsync C++ streams from C streams
//     cin.tie(nullptr); // Untie cin from cout for faster input/output
//     ifstream inputFile("codeforcesinput.txt");  // Open file for reading
//     if (!inputFile) {
//         cerr << "Error opening file!" << endl;
//         return 1;
//     }
//     // vector<int> p(2e5+1,1);
//     // vector<int> pf(2e5+1,1e6);
//     // p[0] = p[1] = 0;
//     // pf[1] = 1;
//     // for (int i = 2; i < (int)(sqrt(2e5+1)); i++){
//     //     if (p[i]){
//     //         pf[i] = i;
//     //         for (int j = i*i; j<2e5+1;j += i){
//     //             p[j] = 0;
//     //             pf[j] = min(pf[j],i);
//     //         }
//     //     }
//     // }
//     int q;
//     inputFile >> q;  // Reading number of test cases
//     if (inputFile.fail()) {
//         cerr << "Error reading test case count!" << endl;
//         return 1;
//     }

//     for (int _ = 0; _ < q; ++_) {  // Process each test case
//         // vis.clear();
//         // int m;
//         int n,m,k,x, y;
//         // ll n,k;
//         // int m;
//         // long n,k;
//         inputFile >> n >> m  ;// Read n and k for each test case
//         // vector<vector<int>> a(n, vector<int>());
       
//         // // a.resize(n);
//         // if (inputFile.fail()) {
//         //     cerr << "Error reading n or k!" << endl;
//         //     return 1;
//         // }
//         // vector<vector<ll>> b(n, vector<ll>(2));
//         // string s;
//         vector<int> a(n);
//         // vector<vector<int>> b(n-1, vector<int>(2));
//         // vector<vector<int>> a(n-1, vector<int>(2));  
//         // vector<int> a(n);
//         vector<vector<int>> b(m, vector<int>(3));
//         // string a;
//         // string b;
//         // // string c;
//         // vector<char> b(m);
//         // vector<int> b(k);
//         // vector<int> c(m);
//         // // // vector<vector<string>> c(k, vector<string>(2));
//         // inputFile >> a[0];
//         // inputFile >> a[1];
//         // inputFile >> a ;
//         // inputFile >> n >> k;
//         // vector<vector<int>> b(k, vector<int>(2));
    
//         // Read vector a
//         for (int i = 0; i < n; i++) {
//             inputFile >> a[i];
//         }

//         // inputFile >> m;
//         // vector<vector<int>> b(m, vector<int>(3));
//         // for (int i = 0; i < n; i++) {
//         //     int m;
//         //     inputFile >> m;
//         //     b[i].resize(m);
//         //     for (int j = 0; j < m; ++j) {
//         //         ll cj, dj;
//         //         inputFile >> cj >> dj;
//         //         b[i][j] = {cj, dj};  // Store the pair in the i-th block
//         //     }
//         // }
//         // for (int i = 0; i < m; i++) {
//         //     for (int j = 0; j <n; j++){
//         //         inputFile >> b[i][j] ;
//         //     }
//         // }
        
//         // for (int i = 0; i < n; i++) {
//         //     int d;
//         //     inputFile >> d;
//         //     a[i] = d-1;
//         //     if (inputFile.fail()) {
//         //         cerr << "Error reading a[" << i << "]!" << endl;
//         //         return 1;
//         //     }
//         // }
//         // inputFile >> m;
//         // vector<vector<int>> a(m, vector<int>(2));
//         // inputFile >> x;
//         // cout << a[0]<<endl;
//         // // // Read vector b
//         // for (int i = 0; i < m; i++) {
//         //     // vector<int>c(3);
//         //     inputFile >> a[i][0] >> a[i][1]  ;
            
//         // }
//         for (int i = 0; i < m; i++) {
//             // vector<int>c(3);
            
//             inputFile >> b[i][0] >> b[i][1] >> b[i][2]  ;
            
//         }
//         // for (int i = 0; i < k; i++) {
//         //     inputFile >> b[i];
//         // }
//         // for (int i = 0; i < m; i++) {
//         //     inputFile >> c[i];
//         // }
        
        

//         // Call the solve function with the read data
//         solve(n,m,a,b);
        
        
//     }

//     inputFile.close();  // Close the file when done
//     return 0;
// }

















// Tarjan's for directed graphs, why do we need stack for? to ensure we only update low[i] for BACK EDGES THAT ARE IN OUR CURR RECURSION PATH
// go to gfg page to udnerstand , because if u use a simple visited array, u might wrongly account for CROSS edges, where u have a child that is alr visited
// but he isnt in ur current recursion path/stack! besides finding SCC in directed graph, tarjans can be used for findign bridges in undirected graph and cycle finding
// okay update: this is the full implementation to find all nodes that are part of a cycle, because tarjans  helps to find sccs, but u need a bit more to find the 
// nodes that are part of some cycle!
// vector<vector<int>> graph;
// vector<int> visited, res, stack2, low, disc, cycle, cnt, visited2;
// int n,k,time2;
// stack<int> st;

// void dfs(int i) {
//     low[i] = disc[i] = time2++;
//     st.push(i);
//     stack2[i] = 1;  // Mark as in the stack

//     for (int j : graph[i]) {
//         if (disc[j] == -1) { // Unvisited node
//             dfs(j);
//             low[i] = min(low[i], low[j]);
//         } 
//         else if (stack2[j]) { // Back edge to an ancestor
//             low[i] = min(low[i], disc[j]);
//         }
//     }

//     // If `i` is the root of an SCC
//     if (low[i] == disc[i]) {
//         vector<int> scc;
//         while (!st.empty()) {
//             int v = st.top(); st.pop();
//             stack2[v] = 0; // Mark as out of stack
//             scc.push_back(v);
//             if (v == i) break; // Stop when reaching the root
//         }

//         // If SCC has more than 1 node, mark them as cycle nodes
//         if (scc.size() > 1) {
//             for (int v : scc) {
//                 cycle[v] = 1; // Mark nodes in cycle
//             }
//         }
//     }
// }


// class DSU {
// private:
//     vector<int> parent, rank, weights;
// public:
//     DSU(int n) {
//         parent = vector<int>(n);
//         rank = vector<int>(n, 0);
//         weights = vector<int>(n, 131071);
//         for (int i = 0; i < n; ++i)
//             parent[i] = i;
//     }
//     int findp(int x) {
//         if (x != parent[x])
//             parent[x] = findp(parent[x]);
//         return parent[x];
//     }
//     void union_(int x, int y, int weight) {
//         int xx = findp(x);
//         int yy = findp(y);
//         if (rank[xx] < rank[yy])
//             parent[xx] = yy;
//         else
//             parent[yy] = xx;
//         weights[xx] = weights[yy] = weights[xx] & weights[yy] & weight;
//         if (rank[xx] == rank[yy])
//             rank[xx]++;
//     }
//     int minimumCostOfWalk(int x, int y) {
//         if (x == y) return 0;
//         if (findp(x) != findp(y)) return -1;
//         return weights[findp(x)];
//     }
// };



// class SegmentTree {
// private:
//     vector<int> tree; // Segment tree array
//     int n;                 // Size of the input array

//     // Build the segment tree recursively
//     void buildTree(const vector<int>& arr, int node, int start, int end) {
//         if (start == end) {
//             // Leaf node
//             tree[node] = 0;
//         } else {
//             int mid = (start + end) / 2;
//             int leftChild = 2 * node + 1;
//             int rightChild = 2 * node + 2;

//             buildTree(arr, leftChild, start, mid);        // Build left subtree
//             buildTree(arr, rightChild, mid + 1, end);    // Build right subtree
//             tree[node] = tree[leftChild] + tree[rightChild];
//         }
//     }

//     // Query the segment tree recursively for range [L, R]
//     int queryTree(int node, int start, int end, int L, int R) {
//         if (R < start || L > end) {
//             // Completely outside the query range
//             return 0;
//         }
//         if (L <= start && end <= R) {
//             // Completely inside the query range
//             return tree[node];
//         }

//         // Partially inside and outside
//         int mid = (start + end) / 2;
//         int leftChild = queryTree(2 * node + 1, start, mid, L, R);
//         int rightChild = queryTree(2 * node + 2, mid + 1, end, L, R);
//         return leftChild + rightChild;
//     }

//     // Update the segment tree for a single index
//     void updateTree(int node, int start, int end, int idx, int value) {
//         if (start == end) {
//             // Leaf node
//             tree[node] += value;
//         } else {
//             int mid = (start + end) / 2;
//             int leftChild = 2 * node + 1;
//             int rightChild = 2 * node + 2;

//             if (idx <= mid) {
//                 // Update left child
//                 updateTree(leftChild, start, mid, idx, value);
//             } else {
//                 // Update right child
//                 updateTree(rightChild, mid + 1, end, idx, value);
//             }

//             // Update the current node
//             tree[node] = tree[leftChild] + tree[rightChild];
//         }
//     }

// public:
//     SegmentTree(const vector<int>& arr) {
//         n = arr.size();
//         tree.assign(4 * n, 0);
//         buildTree(arr, 0, 0, n - 1); // Build the tree starting from root
//     }

//     int query(int L, int R) {
//         return queryTree(0, 0, n - 1, L, R);
//     }

//     void update(int idx, int value) {
//         updateTree(0, 0, n - 1, idx, value);
//     }
// };
