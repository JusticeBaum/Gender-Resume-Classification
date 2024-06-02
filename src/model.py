from extractor import BigramExtractor
import numpy as np
class LogisticRegressionGenderClassifier():
    def __init__(self, feat_extractor: BigramExtractor, train_examples, num_iters=50, reg_lambda=0.0, learning_rate=0.1):
        self.feat_extractor = feat_extractor
        self.examples = train_examples
        self.num_iters = num_iters
        self.reg_lambda = reg_lambda
        self.learning_rate = learning_rate

        self.weights = np.zeros((len(feat_extractor.indexer), 1))
        self.bias = 0

        self.train(feat_extractor, train_examples, num_iters, reg_lambda, learning_rate)

    def train(self, feat_extractor: BigramExtractor, train_examples, num_iters=50, reg_lambda=0.0, learning_rate=0.1):
          for epoch in range(num_iters):
            for example in train_examples:
                x = feat_extractor.extract_features(example)
                x = x.reshape(-1, 1)
                y = example.label
                z = np.dot(self.weights.T, x) + self.bias
                a = 1 / (1 + np.exp(-z))
                dw = np.dot(x, (a - y).T)
                db = np.sum(a - y)
                self.weights -= learning_rate * (dw + reg_lambda * self.weights)
                self.bias -= learning_rate * db

    def predict(self, ex, threshold = 0.5):
        x = self.feat_extractor.extract_features(ex)
        z = np.dot(x, self.weights) + self.bias
        predicted_probability = 1 / (1 + np.exp(-z))
        predicted_label = 1 if predicted_probability >= threshold else 0
        return predicted_label, predicted_probability