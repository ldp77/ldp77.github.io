import numpy as np

class BagLearner:

    def __init__(self, learner, kwargs={}, bags=20, boost=False, verbose=False):
        self.learner = learner
        self.learners = [learner() for i in range(bags)]
        self.kwargs = kwargs
        self.bags = bags
        self.boost = boost
        self.verbose = verbose

    def add_evidence(self, data_x, data_y):
        # Combine X and Y data
        all_data = np.insert(data_x, data_x.shape[1], data_y, axis=1)

        # Select bags of data
        def get_bag():
            include_idx = [np.random.randint(0, len(data_y)) for i in range(len(data_y))]
            return all_data[include_idx]
        
        data_bags = [get_bag() for i in range(self.bags)]

        for i in range(self.bags):
            bag_x, bag_y = data_bags[i][:, 0:-1], data_bags[i][:, -1]
            self.learners[i].add_evidence(bag_x, bag_y)

        return

    def query(self, data_x):

        return np.mean([learner.query(data_x) for learner in self.learners], axis=0)
        

        return