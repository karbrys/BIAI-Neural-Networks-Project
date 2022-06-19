import os
import pandas as pd
import tensorflow as tf
from keras.integration_test.preprocessing_test_utils import BATCH_SIZE
from tensorflow import keras
from tensorflow.python.keras import layers
import src.data_clean.clean as dc
import src.model.model as md
import matplotlib.pyplot as plt
import stats.plots as sth
import seaborn as sns
from sklearn import metrics

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Train data has column: "Survived", test data doesn't
train_data = pd.read_csv("data/train.csv")  # loading the data
test_data = pd.read_csv("data/test.csv")  # ('../data/test.csv')
validate_data = pd.read_csv("data/validate.csv")  # ('../data/validate.csv')

train_data.head()
print("Total number of rows in training data: ", train_data.shape[0])
print("Total number of rows in test data: ", test_data.shape[0])
print("Total number of rows in validate data: ", validate_data.shape[0])

# Statistics of survival between genders
((train_data.groupby(['Sex', 'Survived']).Survived.count() * 100) / train_data.groupby('Sex').Survived.count())
# The result is that 74% of women 18% of men survived

# Cleaning and handling missing values in data sets:
train_data_clean, test_data_clean, validate_data_clean = dc.clean_data_sets(train_data, test_data, validate_data)

# Histograms
# sth.create_survival_age_histogram(train_data_clean)
# sth.create_survival_pclass_histogram(train_data_clean)
# sth.create_correlation(train_data_clean)

# Take out target column - in our case survived column
target = train_data_clean.pop('Survived')
validate_target = validate_data_clean.pop('Survived')

# Converting pandas data sets to tensorflow data sets
train_data_clean_tf = tf.convert_to_tensor(train_data_clean)
test_data_clean_tf = tf.convert_to_tensor(test_data_clean)

# Calling normalizer to set the layer's mean and standard-deviation
normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(train_data_clean)

# Creating model
model = md.create_model_11(normalizer)

# Fitting model
history = model.fit(train_data_clean, target, batch_size=32, verbose=2,
                    epochs=200)  # combinations also with:batch_size=64 epochs=100

# Plotting the Loss and Accuracy over 250 epochs
# pd.DataFrame(history.history).plot(title="Loss and Accuracy")
# plt.savefig('plot11a.png')
# plt.show()

# Model summary
model.summary()

# Evaluation
print(model.evaluate(validate_data_clean, validate_target))

# Prediction for test data
predict = model.predict(test_data_clean)
predict = (predict > 0.5).astype(int).ravel()
print(predict)

# Submission to csv
# result = pd.DataFrame({"Pclass": test_data_clean.Pclass, "Sex": test_data_clean.Sex, "Age": test_data_clean.Age, "Survived": predict})
# result.to_csv("final_result_m11a.csv", index=False)


# Plotting the confusion matrix in heatmap
# matrix = metrics.confusion_matrix(target, predict)
# sns.heatmap(matrix, annot = True,fmt = 'g')
# plt.savefig('confusion.png')
# plt.show()

# Reading predictions
# df = pd.read_csv("training_results/final_result_m1.csv")
# count = (df['Survived'] == 1).sum()
# print('One: ', count)
# count = (df['Survived'] == 0).sum()
# print('Zero: ', count)
