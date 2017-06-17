
def sigmoid(z):
       return 1.0 / (1 + np.exp(-z))
       
class LogisticRegression():
    def __init__(self):
        self.weight = np.array([], dtype=float)

    def fit(self, X, y, alpha=0.01, iter_nums=1000):
        X = np.array(X)
        sample_nums, feature_nums = X.shape[0], X.shape[1] + 1
        y = np.array(y).reshape(sample_nums, 1)
        X = np.column_stack((np.ones(sample_nums), X))
        init_weight = np.zeros(feature_nums, dtype=float).reshape(feature_nums, 1)
        for i in range(iter_nums):
            h = sigmoid(np.dot(X, init_weight))
            init_weight += alpha * np.dot(X.T, (y - h))
        self.weight = init_weight
        return self

    def predict(self, X):
        X = np.array(X)
        sample_nums = X.shape[0]
        X = np.column_stack((np.ones(sample_nums), X))
        result = []
        for sample in X:
            if sigmoid(np.dot(sample, self.weight)) >= 0.5:
                result.append(1)
            else:
                result.append(0)
        return result

    def score(self, X, y_true):
        y_predict = self.predict(X)
        return accuracy_score(y_predict, y_true)
