import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image_dataset_from_directory

from tensorflow.keras.models import load_model


def get_dataset(folder_path):
    dataset = image_dataset_from_directory(
        folder_path,
        batch_size=16,
        image_size=(28, 28))

    return dataset


def make_model():
    inputs = keras.Input(shape=(28, 28, 3))
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Flatten()(x)
    x = layers.Dense(900, activation="relu")(x)
    x = layers.Dense(900, activation="relu")(x)
    x = layers.Dense(900, activation="relu")(x)
    # x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(6, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="Adam", loss="sparse_categorical_crossentropy")
    return model


def train():
    # model.summary()
    # Compile the model

    # Train the model for 1 epoch from Numpy data
    train_dataset = get_dataset('/home/fedor/Desktop/network/archive/six-shapes-dataset-v1/six-shapes/train')
    val_dataset = get_dataset('/home/fedor/Desktop/network/archive/six-shapes-dataset-v1/six-shapes/val')
    print("Fit on NumPy data")
    history = model.fit(train_dataset, batch_size=32, epochs=10, validation_data=val_dataset)
    train_2 = get_dataset('/home/fedor/Desktop/network/archive/six-shapes-dataset-v2/six-shapes/train')
    val_2 = get_dataset('/home/fedor/Desktop/network/archive/six-shapes-dataset-v2/six-shapes/val')
    history_2 = model.fit(train_2, batch_size=32, epochs=10, validation_data=val_2)


def test():
    test_dataset = get_dataset('/home/fedor/Desktop/network/archive/six-shapes-dataset-v1/six-shapes/test')
    # test_2 = get_dataset('/home/fedor/Desktop/network/archive/six-shapes-dataset-v2/six-shapes/test')
    history = model.evaluate(test_dataset, batch_size=1500)


def predict(file):
    dataset = image_dataset_from_directory(
        folder_path=file,
        batch_size=16,
        image_size=(28, 28))

    return dataset


if __name__ == '__main__':
    model = load_model('my_model.h5')
    test()
    # model.save('my_model.h5')
