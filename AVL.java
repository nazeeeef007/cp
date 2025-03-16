public class AVL {
    AVL parent, left, right;
    int height, weight;
    int key, value; // Store key and value as pair

    public AVL(int key, int value) {
        this.key = key;
        this.value = value;
        this.height = 1; // Leaf nodes have height 1
        this.weight = 1; // Only itself initially
        this.left = this.right = this.parent = null;
    }


    // Method to print in in-order traversal
    public static void print(AVL node) {
        if (node == null) {
            return; // Base case: If the node is null, just return
        }
        print(node.left); // Print left subtree
        System.out.println("(" + node.key + " , " + node.value + ")"); // Print current node
        print(node.right); // Print right subtree
    }

    // Check the relation between current node and the new (key, value) pair
    public int check(int key, int value) {
        if (this.key < key || (this.key == key && this.value < value)) return -1;
        if (this.key == key && this.value == value) return 0;
        return 1;
    }

    public AVL insert(AVL node, int key, int value) {
        if (node == null) return new AVL(key, value); // New node created

        int res = node.check(key, value);

        if (res == 1) {
            node.left = insert(node.left, key, value);
            node.left.parent = node; // Update parent pointer
        } else if (res == -1) {
            node.right = insert(node.right, key, value);
            node.right.parent = node; // Update parent pointer
        } else {
            return node; // Duplicate keys are ignored
        }

        // Update height and weight
        updateHeightAndWeight(node);

        // **Rebalance and return the corrected subtree**
        return rebalance(node);
    }


    private void updateHeightAndWeight(AVL node) {
        node.height = 1 + Math.max(getHeight(node.left), getHeight(node.right));
        node.weight = 1 + getWeight(node.left) + getWeight(node.right);
    }

    public AVL delete(AVL node, int key, int value) {
        if (node == null) {
            return null; // Base case: Node not found
        }

        int res = node.check(key, value);

        if (res == 1) {
            node.left = delete(node.left, key, value);
            if (node.left != null) node.left.parent = node; // Fix parent pointer
        } else if (res == -1) {
            node.right = delete(node.right, key, value);
            if (node.right != null) node.right.parent = node; // Fix parent pointer
        } else {
            // Node to be deleted found

            // Case 1: No child or one child
            if (node.left == null || node.right == null) {
                AVL temp = (node.left != null) ? node.left : node.right;

                if (temp == null) {
                    // No child case
                    return null;
                } else {
                    // One child case
                    temp.parent = node.parent; // Fix parent pointer
                    return temp; // Replace node with its child
                }
            } else {
                // Case 2: Two children - Find successor
                AVL successor = min(node.right);
                node.key = successor.key;
                node.value = successor.value;

                // **Delete the successor** (which must be in the right subtree)
                node.right = delete(node.right, successor.key, successor.value);
                if (node.right != null) node.right.parent = node;
            }
        }

        // **Update height & weight before rebalancing**
        updateHeightAndWeight(node);

        // **Rebalance and return the corrected subtree**
        AVL balancedNode = rebalance(node);
        if (balancedNode.parent != null && balancedNode.parent.left == node) {
            balancedNode.parent.left = balancedNode;
        } else if (balancedNode.parent != null) {
            balancedNode.parent.right = balancedNode;
        }
        return balancedNode;
    }


    // Helper method to find the minimum value node (in-order successor)
    public AVL min(AVL node) {
        while (node.left != null) {
            node = node.left;
        }
        return node;
    }

    public AVL max(AVL node) {
        while (node.right != null) {
            node = node.right;
        }
        return node;
    }

    // Helper methods for height and weight
    private int getHeight(AVL node) {
        return node == null ? 0 : node.height;
    }

    private int getWeight(AVL node) {
        return node == null ? 0 : node.weight;
    }

    private AVL rebalance(AVL node) {
        int balance = getHeight(node.left) - getHeight(node.right);

        if (balance > 1) { // Left-heavy
            if (getHeight(node.left.left) >= getHeight(node.left.right)) {
                return rotateRight(node); // Left-Left case
            } else {
                node.left = rotateLeft(node.left); // Left-Right case
                return rotateRight(node);
            }
        } else if (balance < -1) { // Right-heavy
            if (getHeight(node.right.right) >= getHeight(node.right.left)) {
                return rotateLeft(node); // Right-Right case
            } else {
                node.right = rotateRight(node.right); // Right-Left case
                return rotateLeft(node);
            }
        }

        return node; // No rebalancing needed
    }

    public AVL rotateLeft(AVL node) {
        AVL newRoot = node.right;
        AVL newRightSubtree = newRoot.left;

        newRoot.left = node;
        node.right = newRightSubtree;

        if (newRightSubtree != null) newRightSubtree.parent = node;

        // **Fix parent pointers**
        newRoot.parent = node.parent;
        node.parent = newRoot;

        // Ensure parent of the rotated subtree points to the new root
        if (newRoot.parent != null) {
            if (newRoot.parent.left == node) {
                newRoot.parent.left = newRoot;
            } else {
                newRoot.parent.right = newRoot;
            }
        }

        // **Update heights and weights**
        updateHeightAndWeight(node);
        updateHeightAndWeight(newRoot);

        return newRoot;
    }

    public AVL rotateRight(AVL node) {
        AVL newRoot = node.left;
        AVL newLeftSubtree = newRoot.right;

        newRoot.right = node;
        node.left = newLeftSubtree;

        if (newLeftSubtree != null) newLeftSubtree.parent = node;

        // **Fix parent pointers**
        newRoot.parent = node.parent;
        node.parent = newRoot;

        // Ensure parent of the rotated subtree points to the new root
        if (newRoot.parent != null) {
            if (newRoot.parent.left == node) {
                newRoot.parent.left = newRoot;
            } else {
                newRoot.parent.right = newRoot;
            }
        }

        // **Update heights and weights**
        updateHeightAndWeight(node);
        updateHeightAndWeight(newRoot);

        return newRoot;
    }

    public AVL rank(AVL node, int target) {
        int rank = getWeight(node.left) + 1;
        if (rank == target) {
            return node;
        }
        if (rank > target) {
            return rank(node.left, target);
        }
        return rank(node.right, target - rank);
    }

    public int getNodeRank(AVL node, int key, int value) {
        // Traverse the tree to find the node with the given key and value
        while (node != null) {
            int cur = node.check(key, value);
            if (cur == 0) {
                break; // Node found
            }
            if (cur < 0) {
                node = node.right; // Move to right subtree if key > current node's key
            } else {
                node = node.left; // Move to left subtree if key < current node's key
            }
        }

        // Now calculate the rank of the node
        int rank = getWeight(node.left) + 1; // Start with 1 for the current node itself

        // Traverse up the tree and add the sizes of left subtrees of right children
        while (node != null && node.parent != null) {
            if (node.parent.right == node) {
                // If current node is the right child, add the size of the left subtree of the parent
                rank += getWeight(node.parent.left) + 1;
            }
            node = node.parent; // Move up to the parent
        }

        return rank;
    }





    public static void main(String[] args) {
        AVL root = new AVL(65, 0);
        root = root.insert(root, 52, 0);
        root = root.insert(root, 70, 0);
        root = root.insert(root, 39, 0);
        root = root.insert(root, 58, 0);
        root = root.insert(root, 68, 0);
        root = root.insert(root, 72, 0);
        root = root.insert(root, 23, 0);
        root = root.insert(root, 23, 2);
        root = root.insert(root, 57, 0);
        root = root.insert(root, 57, -1);
        root = root.insert(root, 60, 0);
        root = root.insert(root, 66, 0);
        root = root.insert(root, 55, 0);
        // Print the inorder traversal of tree
        root.print(root);


    }
}
