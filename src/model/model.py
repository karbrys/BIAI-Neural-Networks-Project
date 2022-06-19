import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import layers
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential, load_model


def create_model_1(normalizer):
    # Creating model layers
    model = tf.keras.Sequential([
        normalizer,
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(6, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    # Compiling model
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model


# GOOD
def create_model_2(normalizer):
    # Creating model layers
    model2 = tf.keras.Sequential([
        normalizer,
        tf.keras.layers.Dense(20, activation='sigmoid'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(12, activation='relu'),
        tf.keras.layers.Dense(8, activation='sigmoid'),
        tf.keras.layers.Dense(1)
    ])

    # Compiling model
    model2.compile(optimizer='adam',
                   loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                   metrics=['accuracy'])
    return model2


def create_model_3(normalizer):
    # Creating model layers
    model3 = tf.keras.Sequential([
        normalizer,
        tf.keras.layers.Dense(10, activation='sigmoid'),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    # Compiling model
    model3.compile(optimizer='adam',
                   loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                   metrics=['accuracy'])
    return model3


def create_model_4(normalizer):
    # Creating model layers
    model4 = tf.keras.Sequential([
        normalizer,
        tf.keras.layers.Dense(12, activation='relu'),
        tf.keras.layers.Dense(8, activation='sigmoid'),
        tf.keras.layers.Dense(4, activation='sigmoid'),
        tf.keras.layers.Dense(1)
    ])

    # Compiling model
    model4.compile(optimizer='adam',
                   loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                   metrics=['accuracy'])
    return model4


def create_model_5(normalizer):
    # Creating model layers
    model5 = tf.keras.Sequential([
        normalizer,
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(1,  activation='relu')
    ])

    # Compiling model
    model5.compile(optimizer='adam',
                   loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                   metrics=['accuracy'])
    return model5


def create_model_6(normalizer):
    # Creating model layers
    model6 = tf.keras.Sequential([
        normalizer,
        tf.keras.layers.Dense(30, activation='relu'),
        tf.keras.layers.Dense(20, activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    # Compiling model
    model6.compile(optimizer='adam',
                   loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                   metrics=['accuracy'])
    return model6


def create_model_7(normalizer):
    # Creating model layers
    model7 = tf.keras.Sequential([
        normalizer,
        tf.keras.layers.Dense(10, activation='tanh'),
        tf.keras.layers.Dense(10, activation='tanh'),
        tf.keras.layers.Dense(1)
    ])

    # Compiling model
    model7.compile(optimizer='adam',
                   loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                   metrics=['accuracy'])
    return model7


def create_model_8(normalizer):
    # Creating model layers
    model8 = tf.keras.Sequential([
        normalizer,
        tf.keras.layers.Dense(15, activation='softplus'),
        tf.keras.layers.Dense(10, activation='softplus'),
        tf.keras.layers.Dense(5, activation='softplus'),
        tf.keras.layers.Dense(1)
    ])

    # Compiling model
    model8.compile(optimizer='adam',
                    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                    metrics=['accuracy'])
    return model8


def create_model_9(normalizer):
    # Creating model layers
    model9 = tf.keras.Sequential([
        normalizer,
        tf.keras.layers.Dense(12, activation='relu'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(4, activation='relu'),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    # Compiling model
    model9.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])
    return model9
