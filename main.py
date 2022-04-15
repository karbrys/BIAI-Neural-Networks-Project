import os
import pandas as pd
import src.data_clean.clean as dc

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Train data has column: "Survived", test data doesn't
train_data = pd.read_csv("data/train.csv")  # loading the data
test_data = pd.read_csv("data/test.csv")  # ('../data/test.csv')

train_data.head()
print("Total number of rows in training data: ", train_data.shape[0])
print("Total number of rows in test data: ", test_data.shape[0])

# Statistics of survival between genders
((train_data.groupby(['Sex', 'Survived']).Survived.count() * 100) / train_data.groupby('Sex').Survived.count())
# The result is that 74% of women 18% of men survived

# Cleaning and handling missing values in data sets:
train_data, test_data = dc.clean_data_sets(train_data, test_data)

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(train_data)
