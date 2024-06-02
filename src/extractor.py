import numpy as np
from mapper import Mapper
class BigramExtractor():
    def __init__(self, mapper: Mapper):
        self.mapper = mapper

    def extract_features(self, ex, add_to_indexer=False):
        features = np.zeros(len(self.indexer))
        for i in range(len(ex.words) - 1):
            w = ex.words[i] + "||" + ex.words[i + 1]
            feat_idx = self.mapper.add_and_get_index(w) \
                if add_to_indexer else self.mapper.index_of(w)
            if feat_idx != -1:
                features[feat_idx] += 1.0
        return features