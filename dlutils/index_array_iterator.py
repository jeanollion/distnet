from keras_preprocessing.image import Iterator
import numpy as np

class IndexArrayIterator(Iterator):
    def __init__(self, n, batch_size, shuffle, seed):
        super().__init__(n, batch_size, shuffle, seed)
        self.allowed_indexes=np.arange(self.n)

    def set_allowed_indexes(self, indexes):
        self.allowed_indexes=indexes
        self.n=len(indexes)

    def _set_index_array(self):
        if self.shuffle:
            self.index_array = np.random.permutation(self.allowed_indexes)
        else:
            self.index_array = self.allowed_indexes
