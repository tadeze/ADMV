import numpy as np

class BaggedModel(object):
    def __init__(self, model=None, num_model=100, nsample=100):
        self.models = []
        self.num_model = num_model
        self.model_features = np.nan
        self.model = model
        self.nsample = nsample
        if model is None:
            raise ValueError("Basic model object is required.")
    def train(self, X):
        assert isinstance(X, np.ndarray)
        n, d  = X.shape
        self.model_features = np.zeros([self.num_model,d])
        d_subset = np.ceil(d/np.sqrt(d))

        for model in range(self.num_model):
            sample_index = np.random.choice(n, self.nsample, False)
            model_l = self.model()

