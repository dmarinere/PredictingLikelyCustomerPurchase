import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

from model import Modelling
from pred_utilities import DealwithSample
from preprocessing import Preprocessor

def main():
    data = pd.read_csv('/home/mahveotm/Desktop/Project/PredictingLikelyCustomerPurchase/data/bank-additional-full.csv', sep=';')
    process = Preprocessor(data)
    data, columns = process._divide_data(data, "y")
    categorical, numerical = process._classify_data(columns, data)
    transformed, y  = process._preprocess_data(categorical, numerical)
    #dealing with our imbalanced data by oversampling the data set
    X, y = DealwithSample(transformed, y, method="RandomOverSampler")
    model = Modelling()
    x_train, x_test, y_train, y_test = model._splitdata(X, y)
    model.Prediction(x_train, x_test, y_train, y_test,
                    method="multilayerperceptron")



if __name__ == "__main__":
    main()

