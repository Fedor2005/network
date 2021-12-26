from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image_dataset_from_directory


def get_dataset(folder_path):
    dataset = image_dataset_from_directory(
        folder_path,
        batch_size=32,
        image_size=(28, 28))

    return dataset


def get_geometric_shape(predictions,
                        class_names=('circle', 'kite',
                                     'parallelogram', 'square',
                                     'trapezoid', 'triangle')):
    y = 0
    res_i = 0
    for i in range(6):
        x = predictions[0][i]
        if x > y:
            y = x
            res_i = i

    if y <= 0.8:
        return '''Not recognized'''
    else:
        return class_names[res_i]


def predictx(file, md):
    predictions = md.predict(file)
    print(predictions)
    geometric_shape = get_geometric_shape(predictions)
    return geometric_shape


def main():
    model = load_model('my_model.h5')
    x = get_dataset('archive/test')
    return predictx(x, model)

if __name__ == '__main__':
    main()