import sys
from extractor import BigramExtractor
from gender_example import GenderExample
from mapper import Mapper
from model import LogisticRegressionGenderClassifier

def train(ex, feat, reg_lambda):
    return LogisticRegressionGenderClassifier(feat, ex, num_iters=50, reg_lambda=reg_lambda, learning_rate=0.1)

def init_and_train(exs, reg_lambda=0.0):
    map = Mapper()
    
    for ex in exs:
        for i in range(len(ex.words) - 1):
            word = ex.words[i] + "||" + ex.words[i + 1]
            map.add_and_get_index(word)
    
    extractor = BigramExtractor(map)

    train(exs, extractor, reg_lambda = reg_lambda)

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 main.py [path to training data] [path to test data]")
        return
    
    train = sys.argv[1]
    test = sys.arv[2]

    train_gender = GenderExample.read_gender_examples(train)
    test_gender = GenderExample.read_gender_examples(train)
        
        


if __name__ == "__main__":
    main()