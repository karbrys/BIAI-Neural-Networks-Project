import numpy as np


def clean_data_sets(train_data_set, test_data_set, validate_data_set):
    # Getting rid of columns that are not significant to survival
    train_data_set.drop(columns=['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],
                        inplace=True)
    test_data_set.drop(columns=['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],
                       inplace=True)
    validate_data_set.drop(columns=['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],
                       inplace=True)

    # Age column has several missing values, so they will be filled with mean
    train_data_set.Age.fillna(train_data_set.Age.mean(), inplace=True)
    test_data_set.Age.fillna(test_data_set.Age.mean(), inplace=True)
    validate_data_set.fillna(test_data_set.Age.mean(), inplace=True)

    # Age column mean
    print("Age column mean in training data: ", train_data_set.Age.mean())
    print("Age column mean in test data: ", test_data_set.Age.mean())
    print("Age column mean in validate data: ", validate_data_set.Age.mean())

    # Changing string values of gender to categorical ones
    train_data_set.Sex = train_data_set.Sex.map({'female': 1, 'male': 0})
    test_data_set.Sex = test_data_set.Sex.map({'female': 1, 'male': 0})
    validate_data_set.Sex = validate_data_set.Sex.map({'female': 1, 'male': 0})

    # Rounding values in Age column
    train_data_set.Age = train_data_set.Age.apply(np.ceil)
    test_data_set.Age = test_data_set.Age.apply(np.ceil)
    validate_data_set.Age = validate_data_set.Age.apply(np.ceil)
    return train_data_set, test_data_set, validate_data_set
