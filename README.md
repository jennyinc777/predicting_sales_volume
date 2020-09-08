# Kaggle Competition:  Predicting Sales Volume

## Problem Statement

> In this competition you will work with a challenging time-series dataset consisting of daily sales data, kindly provided by one of the largest Russian software firms - 1C Company. 
We are asking you to predict total sales for every product and store in the next month. By solving this competition you will be able to apply and enhance your data science skills.

[source](https://www.kaggle.com/c/competitive-data-science-predict-future-sales/overview)

## Executive Summary

The Predicting Sales Volume Kaggle Competition consists of using a 3 million row data set to predict monthly sales volume by shop and by item id.  There are many challenges that arise when we work with a dataset this big.

First, I thought that breaking down the dataset into logical chunks, perhaps by `shop_id`, can be a great idea.  Working with a small set of data gives me a general look into how to work with the data without having to worry about computational speed.  For example, when I was working with just shop \#42, I built a seasonal decomposition graph relatively quickly.



I was also able to preprocess my data such as appending all items to all shops by month and pivot my dataset for a neural network.  Then knowing that I can do it for one shop, I then built out my code in a Python script and iterated through all the shops.

Second, normal iterative processes will take much more time so finding Python libraries that are fast and efficient will be our friend here.  A Python library called `itertools` helped a lot which was an idea given by this [notebook](https://www.kaggle.com/gordotron85/future-sales-xgboost-top-3#Preprocessing).

Next, I implemented a Long Short Term Memory (LSTM) model with an `EarlyStopping(patience = 5)` regularization technique.  I tweaked the learning rate along with the epochs since I noticed a bit of oscillation in my neural net.  In the end, I decided on a learning rate of 0.0001 and 15 epochs.  Even though my LSTM was iterating through all shops, I was careful to watch over each epoch.  For the most part, shops were not too overfit and had a validation loss of close to 1 but there were a handful that had some abnormal validation loss scores.  This could be due to outliers in that particular shop.

Some hard lessons that I learned in this project is that the preprocessing step could be the bottleneck for most complex datasets when you are working through a project.  Dedicating ample time to this step could save a lot of heartache in your process.  The data given has no way to enter a neural network without being cleaned and preprocessed dramatically.

## Conclusion

- RMSE score:  1.59829
- Preprocessing step is the most important step for time-series neural nets. 
- Complex datasets → spend a lot of time in preprocessing
- It was interesting to see validation loss scores increase dramatically for specific stores -- this may have something to do with outliers in that particular shop.
- Vast majority of items in test file should be 0. Can our model predict this? Understanding test file is important as well

## Next Steps

- Since RMSE is heavily affected by outliers, it would be best to remove any outliers.
- Remove stationarity
- Attempt a multilayer perceptron (MLP) model which was mentioned in
an [article](https://machinelearningmastery.com/suitability-long-short-term-memory-networks-time-series-forecasting/) that compared LSTM’s with MLP’s.
- Using other gradient descent boosting techniques (XGBoost).
- There is an `item_category` column that could be useful to use in a model (making it a multi-variate model).

## Download data

### Through Kaggle's API:

> `kaggle competitions download -c competitive-data-science-predict-future-sales`

### Through competition's [website](https://www.kaggle.com/c/competitive-data-science-predict-future-sales/data)

## Data Dictionary

| Files | Description |
|-|-|
| `1_cleaning.ipynb` | groups data by shop and item id, creates 42 csv files by shop |
| `2_eda.ipynb` | creates graphs |
| `3_preprocessing.py` | creates a folder called `clean_shops` and outputs csv files of preprocessed shop data |
| `4_modeling.ipynb` | uses `modeling.py` to train and predict |
| `5_submit.ipynb` | transforms `test.csv`, concatenates all prediction shop files and saves submission |
| `modeling.py` | train/test/split, LSTM model, and test file transformations |
| `presentation.pdf` | final presentation |