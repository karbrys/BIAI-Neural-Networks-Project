import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

train_data = pd.read_csv("data/train.csv")  # loading the data
test_data = pd.read_csv("data/test.csv")  # ('../data/test.csv')

# Train data has column: "Survived", test data doesn't

train_data.head()
print("Total number of rows in training data ", train_data.shape[0])
print("Total number of rows in test data ", test_data.shape[0])

# Statistics of survival between genders
((train_data.groupby(['Sex', 'Survived']).Survived.count() * 100) / train_data.groupby('Sex').Survived.count())
# The result is that 74% of women 18% of men survived

# Handling missing values:

# Getting rid of 'Cabin' column as it is not of significance to survival
train_data.drop('Cabin', axis=1, inplace=True)
test_data.drop('Cabin', axis=1, inplace=True)

# Age column has several missing values, so they will be filled with mean
train_data.Age.fillna(train_data.Age.mean(), inplace=True)
test_data.Age.fillna(test_data.Age.mean(), inplace=True)

# Age column mean
print(train_data.Age.mean())
print(test_data.Age.mean())

# Changing string values of gender to categorical ones
train_data.Sex = train_data.Sex.map({'female': 1, 'male': 0})
test_data.Sex = test_data.Sex.map({'female': 1, 'male': 0})

# Rounding the values in Age column
train_data.Age = train_data.Age.apply(np.ceil)
test_data.Age = test_data.Age.apply(np.ceil)

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(test_data)
