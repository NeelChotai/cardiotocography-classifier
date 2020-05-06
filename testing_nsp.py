import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import to_categorical
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from rbf.rbf_keras.rbflayer import RBFLayer, InitCentersRandom

neurons = 0

features = ["LB", "AC", "FM", "UC", "ASTV", "MSTV", "ALTV", "MLTV", "DL"]
data = pd.read_excel("CTG.xls", sheet_name="Raw Data")
data = data.dropna()
input_data = data.loc[:, features].values
target = data.loc[:, ["NSP"]].values

scalar = StandardScaler()
input_data = scalar.fit_transform(input_data)
target = scalar.fit_transform(target)

encoder = LabelEncoder()
encoded_target = encoder.fit_transform(target)
encoded_target = to_categorical(encoded_target)

def model():
    model = Sequential()
    model.add(Dense(neurons, input_dim=len(features), activation="relu"))
    model.add(Dense(neurons))
    model.add(Dense(neurons))
    model.add(Dense(3, activation="softmax")) # 3 output nodes, one for each class
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model

def model_rbf():
    model = Sequential()
    model.add(RBFLayer(neurons, initializer=InitCentersRandom(input_data), betas=2.0, input_shape=(len(features),)))
    model.add(Dense(3)) # 3 output nodes, one for each class
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model

with open("testing_nsp.txt", "a") as testing:
    testing.write("MLP (3-layers):\n")
    
    for i in [1, 3, 5, 10, 15, 20, 25, 30]:
        neurons = i
        estimator = KerasClassifier(build_fn=model, epochs=200, batch_size=5)
        x_train, x_test, y_train, y_test = train_test_split(input_data, encoded_target, test_size=0.3)

        estimator.fit(x_train, y_train)
        y_pred = estimator.predict(x_test)
        y_pred = to_categorical(y_pred)

        testing.write("Neurons: {}\n".format(neurons))
        testing.write("NSP accuracy: %.2f%%\n" % metrics.accuracy_score(y_test, y_pred))
        testing.write("NSP misclassification rate: %.2f%%\n" % (1 - metrics.accuracy_score(y_test, y_pred)))
        testing.write("NSP mean squared error: %.2f%%\n" % metrics.mean_squared_error(y_test, y_pred))
        testing.write("NSP recall: %.2f%%\n" % metrics.recall_score(y_test, y_pred, average="macro"))
        testing.write("NSP precision: %.2f%%\n" % metrics.precision_score(y_test, y_pred, average="macro"))
        testing.write("\n")