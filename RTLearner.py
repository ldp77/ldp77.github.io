import numpy as np

class RTLearner():

    def __init__(self, leaf_size=1, verbose=False):
        self.leaf_size = leaf_size
        self.verbose = verbose

    def add_evidence(self, all_x, all_y):
        
        # RECURSIVELY
        # Determine feature most strongly correlated with Y
        def build_tree(data_x, data_y):

            if data_x.shape[0] == 1:
                return np.array([-1, data_y[0], -1, -1])
            if np.all(data_y == data_y[0]):
                return np.array([-1, data_y[0], -1, -1])

            num_samples, num_features = data_x.shape

            if num_samples <= self.leaf_size:
                return np.array([-1, np.mean(data_y), -1, -1])

            strongest_feature_idx = np.random.randint(0, num_features)

            # Splitval is the median of that feature across all data
            splitval = np.median(data_x[:, strongest_feature_idx])

            # Combine X data with Y data into a singular matrix with all data
            all_data = np.insert(data_x, num_features, data_y, axis=1)

            # Divide Data by Into Left-X, Left-Y, Right-X, Right-Y
            left_data = all_data[all_data[:,strongest_feature_idx] <= splitval]
            right_data = all_data[all_data[:,strongest_feature_idx] > splitval]
            left_x, left_y = left_data[:, 0:-1], left_data[:, -1]
            right_x, right_y = right_data[:, 0:-1], right_data[:, -1]

            # Edge case handle- If Right-X is empty, recursion would never end
            # To avoid this, move one record from left to right
            if right_x.size == 0:
                right_x, right_y = left_x[0], left_y[0:1]
                left_x, left_y = left_x[1:], left_y[1:]

            if self.verbose:
                print("LX: {}, LY: {}, RX: {}, RY: {}".format(left_x, left_y, right_x, right_y))
                print("SFI: {}".format(strongest_feature_idx))
                print("Splitval: {}".format(splitval))

            left_tree = build_tree(left_x, left_y)
            right_tree = build_tree(right_x, right_y)
            root = np.array([strongest_feature_idx, splitval, 1, left_tree.shape[0]//4+1])

            return np.concatenate((root, left_tree, right_tree))

        tree = build_tree(all_x, all_y)
        self.tree = np.array_split(tree, len(tree)/4)

        return

    def query(self, data_x):
        
        def one_query(x_test):
            tree_idx = 0
            while True:
                current_node = self.tree[tree_idx]
                feature_idx, value, left, right = int(current_node[0]), current_node[1], int(current_node[2]), int(current_node[3])

                if feature_idx < 0:
                    return value

                if x_test[feature_idx] <= value:
                    tree_idx += left
                else:
                    tree_idx += right 
            pass

        return np.array([one_query(data_x[i]) for i in range(len(data_x))])

"""
Node: {
    factor, splitval, left, right
}

LeafNode: {
    None, splitval, None, None
}
"""