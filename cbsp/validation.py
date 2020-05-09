"""Exhaustive search using crossvalidated weighted least squares regression.

The module searches for the best <num_features> describing the data
among the features in a feature matrix.
It can do so iteratively for models of different numbers of features and
then creates a feature count from the best models. The features are defined
through columns in a feature matrix DataFrame.

    Example usage:
        es = ExhaustiveSearch(num_features=3)
        es.fit_iter(X, y, weights, max_features=10)
"""
from itertools import combinations

from tqdm.auto import tqdm
import numpy as np

from cbsp.utils import crossvalidate

class ExhaustiveSearch:
    """Exhaustive search using crossvalidated weighted least squares regression.
    
    Args:
        num_features (int): Features included in the learning rule, between 1 and 27. Defaults to 3.
        n_splits (int, optional): Splits for cross validation. Defaults to 5.
        alpha (int, optional): Regularization parameter. Defaults to 0.
        use_weights_for_r2 (bool, optional): Evaluate R in the weighted feature space. Defaults to True.
        use_adj_r2 (bool, optional): Adjust R with respect to number of features. Defaults to True.

    Attributes:
        estimators (Dict[int, List[str]]): indexed 'table' with all possible feature combinations - binom(27, num_features).
        rs (Dict[int, float]): goodness of fit for all feature combinations in estimators.
        coefs (Dict[int, List[float]]): coefficients for all features combinations in estimators.
        coefs_std (Dict[int, List[float]]): standard deviation per coefficient in coefs.
        feat_count (Dict[str, int]): Number of occurence of each feature in the best rules. 
                                     Mutated iteratively when using fit_iter.
    """

    def __init__(self, num_features=3, 
                        n_splits=5,
                        alpha=0,
                        use_weights_for_r2=True,
                        use_adj_r2=True):

        self.num_features = num_features
        self.n_splits = n_splits
        self.alpha = alpha
        self.use_weights_for_r2 = use_weights_for_r2
        self.use_adj_r2 = use_adj_r2
        self.estimators = {}
        self.rs = {}
        self.coefs = {}
        self.coefs_std = {}
        self.feat_count = {}

    def fit(self, X, y, weights):
        """Performs a single exhaustive search for learning rules of length 'num_features'.

        Args:
            X (array): feature matrix of shape (#samples, #features).
            y (array): target of shape #samples.
            weights (array): weights of shape #samples.
        """
        self.estimators = {i: c for i, c in enumerate(list(combinations(X.columns, self.num_features)))}
        self.feat_count = {c: 0 for c in X.columns}
        self._fit(X, y, weights)
        self.pprint()

    def _fit(self, X, y, weights):
        """
        Fits all possible features combinations in X to y and updates rs, coefs and coefs_std.
        """
        for idx, comb in tqdm(self.estimators.items()):
            X_temp = X[[c for c in comb]].values
            X_temp = np.ascontiguousarray(X_temp)
            self.rs[idx], coefs, coefs_std = crossvalidate(X_temp,
                                                               y,
                                                               weights,
                                                               self.n_splits,
                                                               alpha=self.alpha,
                                                               use_weights_for_r2=self.use_weights_for_r2,
                                                               use_adj_r2=self.use_adj_r2)
            self.coefs[idx] = coefs
            self.coefs_std[idx] = coefs_std
        self.coefs = {idx: self.coefs[idx] for idx in self.rs}
        self.coefs_std = {idx: self.coefs_std[idx] for idx in self.rs}
        self._sort()

    def fit_iter(self, X, y, weights, min_features=3, max_features=10):
        """Iterative exhaustive search for all combinations of num_features to max_features.

        Args:
            X (pd.DataFrame): feature matrix of shape (#samples, #features).
            y (array): target of shape #samples.
            weights (array): weights of shape #samples.
            min_features (int): minimal number of features considered for the model.
            max_features (int): maximal number of features considered for the model.

        Returns:
            tuple: num_features, best_rs
                with 
                    num_features (array)
                    best_rs (array)

        Note, the attribute rs, estimators, coefs, coefs_std are overwritten in each iteration, while the feature occurence
        of the best fit models are iteratively counted and updated in feat_count.
        """
        self.num_features = min_features
        self.feat_count = {c: 0 for c in X.columns}
        best_rs = []
        while self.num_features <= max_features:
            self.rs = {}  
            self.estimators = {i: c for i, c in enumerate(list(combinations(X.columns, self.num_features)))}
            self._fit(X, y, weights)
            self.pprint((0, 1))  # here we print the best model for the current number of features.
            best_rs.append(next(iter(self.rs.values())))
            self._count_best_features()
            self.num_features += 1
        self.num_features -= 1  # the num_features is 1 higher than what actually has been fitted after the loop
        return np.arange(min_features, max_features + 1), np.array(best_rs)

    def predict(self, X, estimator='best'):
        """Predicts rate-based synaptic plasticity.

        Args:
            X (pd.DataFrame): feature matrix of shape (#samples, #features)
                            from which the features given by estimator will be selected.
            estimator (str or tuple, optional): by default 'best', the best estimator will be used. 
                            A custom estimator can be provided as tuple, e.g. ('u*v', 'u*v*w', 'u*w**2').
        Returns:
            array: rate-based synaptic plasticity as X @ b
        """
        X = X[list(next(iter(self.estimators.values()))) if estimator=='best' else list(estimator)].values
        b = next(iter(self.coefs.values())) if estimator=='best' else self.coefs[self._index(estimator)]
        return X @ b

    def get_results(self, estimator=('u*v', 'u*v*w', 'u*w**2')):
        """Returns results for a certain feature tuple.

        Features in the feature tuple must be in ~cbsp.utils.featstr.
        """
        index = self._index(estimator)
        counts = [self.feat_count[feat] for feat in estimator]
        results = {estimator: {"r": self.rs[index],
                                "coefs": self.coefs[index],
                                "coefs_std": self.coefs[index],
                                "counts": counts}}
        return results

    def pprint(self, start_end = (0, 3)):
        """
        Prints all start to end best models.
        """
        msg = f'{self.num_features} best features:\n'
        start, end = start_end
        results = [(self.estimators[idx], self.rs[idx]) for idx in self.rs.keys()]
        for i, (comb, r) in enumerate(results[start:end], start):
            msg += f'\t{i+1}.: {comb}, {r:.3f}\n'
        print(msg)

    def _index(self, estimator):
        """
        Returns the index for a given estimator.
        """
        return {feats: idx for idx, feats in self.estimators.items()}[estimator] 

    def _sort(self):
        """
        Sorts the features combinations according to their goodness of fit.
        """
        self.rs = dict(sorted(self.rs.items(), key=lambda item: (item[1], item[0]), reverse=True))
        index = [idx for idx in self.rs]
        self.estimators = {idx: self.estimators[idx] for idx in index}
        self.coefs = {idx: self.coefs[idx] for idx in index}
        self.coefs_std = {idx: self.coefs_std[idx] for idx in index}

    def _count_best_features(self):
        """
        Increases the counts of feature occurence in best fit models.
        """
        best = self.estimators[next(iter(self.rs))]
        for feature in self.feat_count:
            if feature in best:
                self.feat_count[feature] += 1
