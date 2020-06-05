import math
import numpy as np

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

class Model(object):
    """
    This class contains different ml_models and allows testing each one of them
    """

    def __init__(self):
        self.ml_model = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.best_parameters = None

    def set_ml_model(self, ml_model):
        """Set current machine learning model"""
        self.ml_model = ml_model

    def set_train_data(self, x_train, y_train):
        """Set training data"""
        self.x_train = x_train
        self.y_train = y_train

    def set_test_data(self, x_test, y_test):
        """Set test data"""
        self.x_test = x_test
        self.y_test = y_test


    def optimize_hyperparameters(self, n_folds, n_iterations):
        """Optimizes the hyperparameters of a machine learning model by calling its
           corresponding optimization function.
           The optimized hyperparameters are stored in the attribute self.best_parameters.
        Args:
           n_folds: Number of folds used to optimize the hyperparameters during cross-validation.
           n_iterations: Number of iterations used to optimize the hyperparameters.
        """

        model_name = self.ml_model.__class__.__name__

        if model_name == "RandomForestRegressor":
            self.optimize_for_random_forest_regressor(n_folds, n_iterations)

        if model_name == "ElasticNet":
            self.optimize_for_elastic_net(n_folds, n_iterations)

        if model_name == "DecisionTreeRegressor":
            self.optimize_for_decision_tree_regressor(n_folds, n_iterations)

        if model_name == "KNeighborsRegressor":
            self.optimize_for_k_neighbors_regressor(n_folds, n_iterations)

        if model_name == "LinearRegression":
            self.optimize_for_linear_regression()

    def optimize_parameters_with_random_search(self, n_folds, n_iterations, random_grid):
        """Optimizes the hyperparameters of a machine learning model by using random search.
           The optimized hyperparameters are stored in the attribute self.best_parameters.
        Args:
           n_folds: Number of folds used to optimize the hyperparameters during cross-validation.
           n_iterations: Number of iterations used to optimize the hyperparameters.
           random_grid: Grid containing possible hyperparameter values for the machine learning
                        algorithm that is going to be optimized.
        """
        model_optimization = RandomizedSearchCV(estimator=self.ml_model,
                                                param_distributions=random_grid,
                                                n_iter=n_iterations, cv=n_folds, verbose=1,
                                                n_jobs=-1, scoring='neg_root_mean_squared_error')

        # Fit the random search model
        model_optimization.fit(self.x_train, self.y_train)
        self.best_parameters = model_optimization.best_params_

    def optimize_for_random_forest_regressor(self, n_folds, n_iterations):
        """Optimizes the specific hyperparameters of the random forest regressor algorithm.
           The optimized hyperparameters are stored in the attribute self.best_parameters.
           The model in self.ml_model is also updated with the best hyperparameters, but not trained
           yet.
		Args:
           n_folds: Number of folds used to optimize the hyperparameters during cross-validation.
           n_iterations: Number of iterations used to optimize the hyperparameters.
        """

        n_estimators = [int(x) for x in np.linspace(start=1, stop=10, num=5)] # Number of trees
        max_features = ['auto', 'sqrt']
        max_depth = [int(x) for x in np.linspace(start=1, stop=10, num=5)]
        max_depth.append(None)
        min_samples_split = [2, 5, 10]
        min_samples_leaf = [1, 2, 4]
        bootstrap = [True, False]

        random_grid = {'n_estimators': n_estimators,
                       'max_features': max_features,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf,
                       'bootstrap': bootstrap}

        self.optimize_parameters_with_random_search(n_folds, n_iterations, random_grid)

        self.ml_model = RandomForestRegressor(n_estimators=self.best_parameters['n_estimators'],
                                              max_features=self.best_parameters['max_features'],
                                              max_depth=self.best_parameters['max_depth'],
                                              min_samples_split=self.best_parameters['min_samples_split'],
                                              min_samples_leaf=self.best_parameters['min_samples_leaf'],
                                              bootstrap=self.best_parameters['bootstrap'],
                                              n_jobs=-1,
                                              criterion='mse')

    def optimize_for_elastic_net(self, n_folds, n_iterations):
        """Optimizes the specific hyperparameters of the ElasticNet algorithm.
           The optimized hyperparameters are stored in the attribute self.best_parameters.
           The model in self.ml_model is also updated with the best hyperparameters,
           but not trained yet.
        Args:
           n_folds: Number of folds used to optimize the hyperparameters during cross-validation.
           n_iterations: Number of iterations used to optimize the hyperparameters.
        """

        alpha = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
        l1_ratio = [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        max_iter = [1, 5, 10, 100]

        random_grid = {'alpha': alpha,
                       'l1_ratio': l1_ratio,
                       'max_iter': max_iter}

        self.optimize_parameters_with_random_search(n_folds, n_iterations, random_grid)

        self.ml_model = ElasticNet(alpha=self.best_parameters['alpha'],
                                   l1_ratio=self.best_parameters['l1_ratio'],
                                   max_iter=self.best_parameters['max_iter'])

    def optimize_for_decision_tree_regressor(self, n_folds, n_iterations):
        """Optimizes the specific hyperparameters of the decision tree regressor algorithm.
           The optimized hyperparameters are stored in the attribute self.best_parameters.
           The model in self.ml_model is also updated with the best hyperparameters,
           but not trained yet.
        Args:
           n_folds: Number of folds used to optimize the hyperparameters during cross-validation.
           n_iterations: Number of iterations used to optimize the hyperparameters.
        """

        min_samples_split = [2, 5, 10]
        min_samples_leaf = [1, 2, 4]
        max_depth = [int(x) for x in np.linspace(start=10, stop=100, num=10)]
        max_leaf_nodes = [5, 20, 100]

        random_grid = {'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf,
                       'max_depth': max_depth,
                       'max_leaf_nodes': max_leaf_nodes}

        self.optimize_parameters_with_random_search(n_folds, n_iterations, random_grid)

        self.ml_model = DecisionTreeRegressor(min_samples_split=self.best_parameters['min_samples_split'],
                                              min_samples_leaf=self.best_parameters['min_samples_leaf'],
                                              max_depth=self.best_parameters['max_depth'],
                                              max_leaf_nodes=self.best_parameters['max_leaf_nodes'],
                                              criterion='mse')

    def optimize_for_k_neighbors_regressor(self, n_folds, n_iterations):
        """Optimizes the specific hyperparameters of the k-neighbors regressor algorithm.
           The optimized hyperparameters are stored in the attribute self.best_parameters.
           The model in self.ml_model is also updated with the best hyperparameters,
           but not trained yet.
        Args:
           n_folds: Number of folds used to optimize the hyperparameters during cross-validation.
           n_iterations: Number of iterations used to optimize the hyperparameters.
        """

        n_neighbors = [1, 2, 3, 4, 5, 6, 7, 8, 10, 15]
        weights = ['uniform', 'distance']
        metric = ['euclidean']

        random_grid = {'n_neighbors': n_neighbors,
                       'weights': weights,
                       'metric': metric}
        self.optimize_parameters_with_random_search(n_folds, n_iterations, random_grid)

        self.ml_model = KNeighborsRegressor(n_neighbors=self.best_parameters['n_neighbors'],
                                            weights=self.best_parameters['weights'],
                                            metric=self.best_parameters['metric'],
                                            n_jobs=-1)

    def optimize_for_linear_regression(self):
        """Set the model as linear regression.
		   Linear regression has no hyperparameters to optimize.
        """
        self.ml_model = LinearRegression(n_jobs=-1)
        self.best_parameters = []

    def evaluate_model(self):
        """Evaluate the accuracy obtained by the current machine learning model.
           Accuracy is defined as the root mean squared error between reference values and
           the values predicted by the model.
        Returns:
           The accuracy of the current machine learning model.
        """

        self.ml_model.fit(self.x_train, self.y_train)
        y_predicted = self.ml_model.predict(self.x_test).reshape(self.y_test.shape[0], 1)
        accuracy = math.sqrt(mean_squared_error(y_predicted, self.y_test)) #The root mean squared error
        return accuracy
