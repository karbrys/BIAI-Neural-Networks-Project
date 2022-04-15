import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

train_data = pd.read_csv(
    "C:/Users/User/Desktop/infa materialy/SEM 6/BIAI/BIAI-Neural-Networks-Project/data/train.csv")  # loading the data
test_data = pd.read_csv(
    "C:/Users/User/Desktop/infa materialy/SEM 6/BIAI/BIAI-Neural-Networks-Project/data/test.csv")  # ('../data/test.csv')

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
for data in train_data:
    data.Age.fillna(data.Age.mean(), inplace=True)
for data in test_data:
    data.Age.fillna(data.Age.mean(), inplace=True)


# Changing string values of gender to categorical ones
def change_gender(x):
    if x == 'male':
        return 0
    elif x == 'female':
        return 1


train_data.Sex = train_data.Sex.apply(change_gender)
test_data.Sex = test_data.Sex.apply(change_gender)
