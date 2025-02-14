#include <iostream>
#include <vector>
#include <fstream>  // For file input
#include <set>
#include <unordered_map>
#include <cmath> // Include cmath for pow
#include <algorithm> //this is for sorting etc 
#include <queue> //this one is for heap and deque 
#include <bitset>
#include <bits/stdc++.h>
#include <numeric> 
#include <map>
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
// https://codeforces.com/problemset?order=BY_RATING_ASC&tags=graphs%2C1900-
long long mod_exp(int base, int exp, int mod) {
    long long result = 1;
    long long b = base % mod; // Ensure base is within mod
    while (exp > 0) {
        if (exp % 2 == 1) { // If exp is odd
            result = (result * b) % mod;
        }
        b = (b * b) % mod; // Square the base
        exp /= 2; // Divide exp by 2
    }
    return result;
}

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
bool compareBySecond(const vector<int>& a, const vector<int>& b) {
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
// ll mod = 1e9 + 7;
ll mod = 998244353;

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
// topic: bfs , shortest paths and dp with dfs on directed unweighted graph, honestly its div 3 so the 2100 rating is inflated, and this was in 2020
// if this was given in 2025 im quite sure it would be at most 1900 maybe even 1800 only lol, anyways yea u just precompute shortest dist from 
// node 1 to every other node using bfs , then the slightly tricky part is the dp, because from any node i, u can only move to some neighbour j 
// that has vis[j] <= vis[i] at most once, and what this means is when ur recursion meets a node with vis[j] <= vis[i], u can just return vis[j] because 
// u wont be able to use the 2nd operation and hence ur answer will never improve. the interesting case is when vis[j] > vis[i], how does this 
// allow us to dp our answer? suppose that vis[i] = 5 and vis[j] = 7, and we already computed dp[j] = 3, this idea of transitivity applies now, 
// because if i can go from i to j, i can definiteyl go to the best node reached by j!, so if dp[j] is already computed just return min (vis[i], dp[j])
// very neat stuff but not too bad, have seen way more complex 2100 rated graph problems lolol


vector<vector<int>> graph;
vector<int> dp;
vector<int> vis;
int n,m;
int dfs(int i){
    if (dp[i]!=1e9){
        return dp[i];
    }
    int best = vis[i];
    for (auto&j:graph[i]){
        if (vis[i] >= vis[j]){
            best = min(best, vis[j]);
        }
        else{
            best = min(best, dfs(j));
        }
    }
    dp[i] = best;
    return dp[i];

}


void solve(int n_, int m_, vector<vector<int>> & edges) {
    n = n_;
    m = m_;
    vis.assign(n+1, 1e9);
    dp.assign(n+1,1e9);
    dp[1] = 0;
    graph.assign(n+1, vector<int>());
    for (int i = 0; i < m ; ++i){
        int u  = edges[i][0];
        int v = edges[i][1];
        graph[u].pb(v);
    }

    queue<vector<int>> q;
    q.push({1,0});
    vis[1] = 0;
    while (!q.empty()){
        vector<int> c = q.front();
        q.pop();
        int i = c[0];
        int d = c[1];
        for (auto&j:graph[i]){
            if (vis[j] > d + 1 ){
                vis[j] = d +1;
                q.push({j, d+1});
            }
        }
    }

    for (int i = 1; i <= n; ++i){
        dfs(i);
    }
    for (int i = 1; i <=n; ++i){
        cout << dp[i] << " ";
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

    for (int _ = 0; _ < q; ++_) {
        int n,m,k;
        // ll n,k; 
        // int m;
        // int c ; // Declare n and m
        cin >> n >>k  ;  // Read n
        // cin >> n ;
        // cin >> m; 
        // vector<char> b(m);
        // string s;
        // a.resize(n);
        // vector<vector<int>> b(n, vector<int>());
        // vector<int> a(n);
        // vector<int> b(m);
        // vector<vector<ll>> b(n-1,vector<ll>(3));
        vector<vector<int>> b(k,vector<int>(2));
   
        // for (int i = 0; i < n; i++) {
        //     cin >> a[i];
        // }
        // for (int i = 0; i < m; i++) {
        //     cin >> b[i];
        // }
        // string a;
        
        // vector<vector<int>> a(m,vector<int>(2));
        // string b;
        // cin >> a >> b;
        // cin >> a[1];
        // string c;
        // cin >> s ;
        // vector<vector<int>> c(n-1,vector<int>(m));
        // // vector<int> a(n);  // Initialize vector of size n
        // // vector<int> result(n);  // Initialize result vector of size n
        // for (int j = 0; j<n-1;j++){
        //     for (int i = 0; i < m; ++i) {
        //         cin >> c[j][i] ;
    
        //     }
        // }
        for (int i = 0; i < k; i++) {
            cin >> b[i][0] >> b[i][1] ;
        }
        // cin >> m;
        // for (int i = 0; i < m; i++) {
        //     cin >> a[i][0] >> a[i][1] ;
        // }
        solve(n,k,b);

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
//         int n,m,k,x;
//         // ll n,k;
//         // int m;
//         // long n,k;
//         inputFile >> n >> m;// Read n and k for each test case
//         // vector<vector<int>> a(n, vector<int>());
       
//         // // a.resize(n);
//         // if (inputFile.fail()) {
//         //     cerr << "Error reading n or k!" << endl;
//         //     return 1;
//         // }
//         vector<vector<int>> b(m, vector<int>(2));
//         // string s;
//         // vector<int> a(n);
//         // vector<vector<ll>> b(n-1, vector<ll>(3));
//         // vector<vector<ll>> a(n, vector<ll>(m));  
//         // string a;
//         // string b;
//         // // string c;
//         // vector<char> b(m);
//         // vector<int> b(m);
//         // // // vector<vector<string>> c(k, vector<string>(2));
//         // inputFile >> a[0];
//         // inputFile >> a[1];
//         // inputFile >> a;
//         // inputFile >> b;
//         // vector<vector<int>> b(k, vector<int>(2));
    
//         // Read vector a
//         // for (int i = 0; i < n; i++) {
//         //     inputFile >> a[i];
//         // }

//         // inputFile >> r1 >> c1 >> r2 >> c2;
//         for (int i = 0; i < m; i++) {
//             inputFile >> b[i][0] >> b[i][1] ;
//         }
//         // for (int i = 0; i < n; i++) {
//         //     for (int j = 0; j <m; j++){
//         //         inputFile >> a[i][j] ;
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
//         //     inputFile >> a[i][0] >> a[i][1] ;
            
//         // }
//         // for (int j =0; j <n-1 ; j++){
//         //     for (int i = 0; i < m; i++) {
//         //         inputFile >> b[j][i];
//         //     }
//         // }
        
        

//         // Call the solve function with the read data
//         solve(n,m,b);
        
        
//     }

//     inputFile.close();  // Close the file when done
//     return 0;
// }





















// vector<int> a;
// vector<vector<int>> dp;
// int n, k, mid;

// int dfs(int i, int j) {
//     if (i == n) {
//         return 0;
//     }
//     if (dp[i][j] != -1) {
//         return dp[i][j];
//     }

//     if (j < mid) {
//         dp[i][j] = max(
//             dfs(i + 1, j + 1),                  // Include this element
//             (a[i] == i - j + 1) + dfs(i + 1, j) // Skip this element
//         );
//     } else {
//         dp[i][j] = (a[i] == i - j + 1) + dfs(i + 1, j); // Only skip allowed
//     }

//     return dp[i][j];
// }

// void solve(int n_, int k_, vector<int>& a_) {
//     // Assign global variables
//     n = n_;
//     k = k_;
//     a = a_;

//     int left = 0, right = n - k + 1, res = INT_MAX;

//     while (left <= right) {
//         mid = left + (right - left) / 2;

//         // Initialize DP table
//         dp.assign(n + 5, vector<int>(mid + 5, -1));

//         // Check feasibility with current mid
//         int ok = dfs(0, 0);

//         if (ok >= k) {
//             res = mid;    // Store feasible result
//             right = mid - 1; // Search for smaller mid
//         } else {
//             left = mid + 1; // Search for larger mid
//         }
//     }

//     cout << ((res == INT_MAX) ? -1 : res) << endl;
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
