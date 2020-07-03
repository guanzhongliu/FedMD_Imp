import numpy as np
from sklearn import svm
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from load_data import load_MNIST_data, load_EMNIST_data, generate_partial_data
import _pickle as pickle


def getNum(dataset):
    num = dataset.size / 28 / 28
    return int(num)


if __name__ == '__main__':
    x, y, test_x, test_y = load_MNIST_data(standarized=True, verbose=True)
    X_train_EMNIST, y_train_EMNIST, X_test_EMNIST, y_test_EMNIST \
        = load_EMNIST_data("./dataset/emnist-letters.mat",
                           standarized=True, verbose=True)
    X_tmp, y_tmp = generate_partial_data(X=X_test_EMNIST, y=y_test_EMNIST,
                                         class_in_use=[10, 11, 12, 13, 14, 15], verbose=True)

    x = np.reshape(x, (getNum(x), -1))
    test_x = np.reshape(test_x, (getNum(test_x), -1))
    model = svm.LinearSVC(verbose=True, max_iter=3000)
    model.fit(x, y)

    z = model.predict(test_x)

    print('准确率:', np.sum(z == test_y) / z.size)

    # with open('./model.pkl', 'wb') as file:
    #     pickle.dump(model, file)
