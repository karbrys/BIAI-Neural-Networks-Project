import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import layers
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential, load_model


def create_model(normalizer):
    # Creating model layers
    model = tf.keras.Sequential([
            normalizer,
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dense(1)
        ])

    # Compiling model
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model

def create_model2(normalizer):
    # Creating model layers
    model2 = tf.keras.Sequential([
            normalizer,
            tf.keras.layers.Dense(10, activation='sigmoid'),
            tf.keras.layers.Dense(10, activation='sigmoid'),
            tf.keras.layers.Dense(1)
        ])

    # Compiling model
    model2.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model2

def create_model3(normalizer):
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

def create_model4(normalizer):
    # Creating model layers
    model4 = tf.keras.Sequential([
            normalizer,
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dense(10, activation='sigmoid'),
            tf.keras.layers.Dense(1)
        ])

    # Compiling model
    model4.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model4

def create_model5(normalizer):
    # Creating model layers
    model5 = tf.keras.Sequential([
            normalizer,
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dense(1)
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
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dense(1)
        ])

    # Compiling model
    model6.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model6