from sklearn.linear_model import Ridge
from catboost import CatBoostRegressor
from sklearn.neural_network import MLPRegressor

from sklearn.base import BaseEstimator, RegressorMixin


class RidgeWithGBRT(RegressorMixin, BaseEstimator):
    def __init__(self,
                 tree_coef=0.8,
                 tree_n_estimators=100,
                 tree_depth=3,
                 tree_learning_rate=0.01,
                 ridge_alpha=0.1,
                 tree_l2_leaf_reg=3.0,
                 ):
        self.tree_coef = tree_coef
        self.ridge_alpha = ridge_alpha
        self.tree_n_estimators = tree_n_estimators
        self.tree_depth = tree_depth
        self.tree_learning_rate = tree_learning_rate
        self.tree_l2_leaf_reg = tree_l2_leaf_reg

    def fit(self, X, y):
        self.linear_model = Ridge(
            alpha=self.ridge_alpha
        )
        self.tree_model = CatBoostRegressor(
            verbose=False,
            n_estimators=self.tree_n_estimators,
            max_depth=self.tree_depth,
            learning_rate=self.tree_learning_rate,
            l2_leaf_reg=self.tree_l2_leaf_reg,
        )
        self.linear_model.fit(X, y)
        residuals = y - self.linear_model.predict(X)
        self.tree_model.fit(X, residuals)

    def predict(self, X):
        return self.linear_model.predict(X) + self.tree_coef * self.tree_model.predict(X)


class RidgeWithMLP(RegressorMixin, BaseEstimator):
    def __init__(self,
                 mlp_coef=0.8,
                 mlp_hidden_layer_sizes=(100, 100),
                 mlp_max_iter=1000,
                 ridge_alpha=0.1,
                 mlp_solver='adam',
                 mlp_activation='relu',
                 ):
        self.mlp_coef = mlp_coef
        self.mlp_activation = mlp_activation
        self.ridge_alpha = ridge_alpha
        self.mlp_hidden_layer_sizes = mlp_hidden_layer_sizes
        self.mlp_max_iter = mlp_max_iter
        self.mlp_solver = mlp_solver

    def fit(self, X, y):
        self.linear_model = Ridge(
            alpha=self.ridge_alpha
        )
        self.mlp_model = MLPRegressor(
            hidden_layer_sizes=self.mlp_hidden_layer_sizes,
            solver=self.mlp_solver,
            activation=self.mlp_activation,
            max_iter=self.mlp_max_iter,
        )
        self.linear_model.fit(X, y)
        residuals = y - self.linear_model.predict(X)
        self.mlp_model.fit(X, residuals)

    def predict(self, X):
        return self.linear_model.predict(X) + self.mlp_coef * self.mlp_model.predict(X)