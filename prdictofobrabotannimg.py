from tensorflow.keras.models import load_model
from learning import get_dataset
import cv2
import tensorflow as tf


def get_geometric_shape(predictions):
    y = 0
    res_i = 0
    for i in range(6):
        x = predictions[0][i]
        if x > y:
            y = x
            res_i = i
    return class_names[res_i]


model = load_model('my_model.h5')
class_names = ('circle', 'kite', 'parallelogram', 'square', 'trapezoid', 'triangle')


# test_dataset = get_dataset('/home/fedor/Desktop/network/archive/six-shapes-dataset-v1/six-shapes/test')
# loss = model.evaluate(test_dataset)  # returns loss and metrics
# print(loss)
def predict():
    image = cv2.imread('img1.png')

    predictions = model.predict(image)

    geometric_shape = get_geometric_shape(predictions)
    return geometric_shape


if __name__ == '__main__':
    print(predict())
