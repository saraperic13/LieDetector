import pandas
from keras import optimizers
from keras.layers import Dense, Activation
from keras.models import Sequential
from keras.models import model_from_json
from sklearn import model_selection
import numpy as np


def load_data(file_name, training=False):
    data = pandas.read_csv(file_name)
    target = data['answer']
    data = data.drop('answer', axis=1)

    if training:
        model = model_selection.StratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=1)
        gen = model.split(data, target)

        train_x, train_y, test_x, test_y = [], [], [], []
        for train_index, test_index in gen:
            train_x = data.loc[train_index]
            train_y = target.loc[train_index]
            test_x = data.loc[test_index]
            test_y = target.loc[test_index]

        return train_x, train_y, test_x, test_y

    return data, target


def fit(x_train, y_train):
    model = Sequential()
    model.add(Dense(100, input_dim=4, init="uniform",
                    activation="relu"))
    model.add(Dense(1))
    model.add(Activation("sigmoid"))

    model.compile(optimizer=optimizers.SGD(lr=0.1, momentum=0.9),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=500)
    return model


def evaluate(x_test, y_test, model, file_name="model"):
    score = model.evaluate(x_test, y_test)
    model_json = model.to_json()
    with open(file_name + ".json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights(file_name + ".h5")
    print("[INFO] Saved model to disk")

    return score


def load_model(file_name="model"):
    json_file = open(file_name+".json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(file_name + ".h5")
    print("[INFO] Loaded model from disk")
    return loaded_model


def predict(data, model=None):
    if model is None:
        model = load_model()
    prediction = model.predict(np.array(data))
    return prediction

def main(training=False):
    if training:
        train_x, train_y, test_x, test_y = load_data("files/datasetExtracted.csv", training=True)
        model = fit(train_x, train_y)
        score = evaluate(test_x, test_y, model)
        print(score)
    else:
        model = load_model()
        print(predict(np.array([[0.6333, 0.6333, 2, 1]]), model))


main(training=True)