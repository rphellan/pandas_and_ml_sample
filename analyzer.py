from __future__ import print_function #To avoid pylint complaining about parenthesis in print
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
import ml_models

def test_sales_data_by_product_shop_month(sales_data, sales_prod_shop_month):
    """
    Verify that sales_product_shop_month was correctly calculated, with the right information.
    The sum of all sales of a product should be the same for both sales_data and
    sales_product_shop_month.
    Args:
        sales_data: Dataframe containing sales data.
        sales_data: Dataframe containing sales data grouped by product, shop, and month.
    """

    list_of_items_to_test = [0, 10, 20]

    for item_id in list_of_items_to_test:
        sales_ref = sales_data.loc[sales_data["item_id"] ==
                                   item_id].groupby('item_id')['total_amount'].sum().to_numpy()[0]
        sales_result = sales_prod_shop_month.loc[sales_prod_shop_month["item_id"] ==
                                                 item_id].groupby('item_id')['total_amount'].sum().to_numpy()[0]
        assert sales_ref == sales_result

def do_data_sanity_check(sales_data):
    """
    Report if there are any missing values or duplicates in the data.
    Args:
        sales_data: Dataframe containing sales data.
    """
    print("-----------Data sanity check-----------")
    print("Missing values count")
    print(sales_data.isnull().sum())
    print('Number of duplicates:', sales_data.duplicated().sum())

def print_data_information(sales_data):
    """
    Report some general information about the data, including the 10 first samples,
    meta information of the dataframe, and a stastitical description of the contained values.
    Args:
        sales_data: Dataframe containing sales data.
    """
    print("----------First top records----------")
    print(sales_data.head(10))
    print("-----------Meta information-----------")
    print(sales_data.info())
    print("-----------Descriptive statistics-----------")
    print(sales_data.describe())

def prepare_data_to_analyze(sales_data):
    """
    Separate the sales data into training and testing sets.
    A new column, total_amount, is added to predict the estimated sales of a product for an
    specific month.
    Args:
        sales_data: Dataframe containing sales data.
    Returns:
        Training and testing datasets.
    """
    sales_data["total_amount"] = sales_data["item_price"] * sales_data["item_cnt_day"]

    grouping_criteria = ['item_id', 'date_block_num', 'shop_id']
    sales_data_by_product_shop_month = sales_data.groupby(grouping_criteria).sum().reset_index()

    test_sales_data_by_product_shop_month(sales_data, sales_data_by_product_shop_month)

    sales_data_sample = sales_data_by_product_shop_month.sample(frac=0.1)
    sales_data_as_np_array = sales_data_sample.values
    x_data = np.concatenate((sales_data_as_np_array[:, 0].reshape(sales_data_as_np_array.shape[0], 1),
                             sales_data_as_np_array[:, 1].reshape(sales_data_as_np_array.shape[0], 1),
                             sales_data_as_np_array[:, 2].reshape(sales_data_as_np_array.shape[0], 1)),
                            axis=1)
    y_data = sales_data_as_np_array[:, 5].reshape(sales_data_as_np_array.shape[0], 1)

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2)

    return x_train, x_test, y_train, y_test

def evaluate_ml_models(x_train, x_test, y_train, y_test):
    """
    Report the accuracy of the selected machine learning models to predict
    the sales per month of a product in a store.
    Args:
        x_train: Numpy array containing a list of products, stores, and months of sale,
                 to train the selected machine learning algorithms.
        x_test: Numpy array containing the total sales for the data listed in x_train.
        y_train: Numpy array containing a list of products, stores, and months of sale,
                 to test and measure the accuracy of the selected machine learning algorithms.
        y_test: Numpy array containing the total sales for the data listed in y_train.
    """
    a_model = ml_models.Model()
    a_model.set_train_data(x_train, y_train.ravel())
    a_model.set_test_data(x_test, y_test.ravel())

    iterations = 20
    folds = 5

    list_of_models = [LinearRegression(), ElasticNet(), RandomForestRegressor(),
                      DecisionTreeRegressor(), KNeighborsRegressor()]
    models_info = []
    for model in list_of_models:
        a_model.set_ml_model(model)
        a_model.optimize_hyperparameters(folds, iterations)

        model_name = model.__class__.__name__
        accuracy = a_model.evaluate_model()
        models_info.append([model_name, accuracy])

    print("-----------Accuracy results for each model-----------")
    for model_info in models_info:
        print ("Model:", model_info[0], "Accuracy:", model_info[1])

def main():
    """
    This program compares the accuracy of different machine learning algorithms to
    predict total sales for every product and store in the next month.
    The algorithms being compared are linear regression, elastic net, random forest regressor,
    decision tree regressor, and k neighbors regressor
    """

    sales_data = pd.read_csv('./input/sales_train.csv')
    print_data_information(sales_data)
    do_data_sanity_check(sales_data)
    x_train, x_test, y_train, y_test = prepare_data_to_analyze(sales_data)
    evaluate_ml_models(x_train, x_test, y_train, y_test)

if __name__ == "__main__":
    main()
