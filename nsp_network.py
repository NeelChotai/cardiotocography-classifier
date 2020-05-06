import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.losses import categorical_crossentropy
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import to_categorical
from sklearn import metrics
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

features = ["LB", "AC", "FM", "UC", "ASTV", "MSTV", "ALTV", "MLTV", "DL"]
data = pd.read_excel("CTG.xls", sheet_name="Raw Data")
data.dropna(inplace=True)
data.drop_duplicates(keep="first", inplace=True)
input_data = data.loc[:, features].values
target = data.loc[:, ["NSP"]].values

scalar = StandardScaler()
input_data = scalar.fit_transform(input_data)
target = scalar.fit_transform(target)

encoder = LabelEncoder().fit_transform(target)
encoded_target = to_categorical(encoder)

def model():
    model = Sequential()
    model.add(Dense(15, input_dim=len(features), activation="relu")) # 15 nodes in hidden layer, 4 input nodes
    model.add(Dense(15)) # second input layer, 15 nodes
    model.add(Dense(3, activation="softmax")) # 3 output nodes, one for each class
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model

accuracy, std_dev, misclassification, mse, recall, precision = [], [], [], [], [], []

for x in range(10):
    estimator = KerasClassifier(build_fn=model, epochs=200, batch_size=32)
    kfold = KFold(n_splits=10) # 10 folds, not shuffled
    results = cross_val_score(estimator, input_data, encoded_target, cv=kfold, n_jobs=-1)
    x_train, x_test, y_train, y_test = train_test_split(input_data, encoded_target, test_size=0.3)
    estimator.fit(x_train, y_train)
    y_pred = estimator.predict(x_test)
    y_pred = to_categorical(y_pred)
    accuracy.append(results.mean()*100)
    std_dev.append(results.std()*100)
    misclassification.append((1 - results.mean())*100)
    mse.append(metrics.mean_squared_error(y_test, y_pred))
    recall.append(metrics.recall_score(y_test, y_pred, average="macro")*100)
    precision.append(metrics.precision_score(y_test, y_pred, average="macro")*100)


with open("output_nsp.txt", "a") as output:
    output.write("NSP accuracy: %.2f%% (%.2f%%)\n" % (np.mean(accuracy), np.mean(std_dev)))
    output.write("NSP misclassification rate: %.2f%% (%.2f%%)\n" % (np.mean(misclassification), np.std(misclassification)))
    output.write("NSP mean squared error: %.2f (%.2f)\n" % (np.mean(mse), np.std(mse)))
    output.write("NSP recall: %.2f%% (%.2f%%)\n" % (np.mean(recall), np.std(recall)))
    output.write("NSP precision: %.2f%% (%.2f%%)\n" % (np.mean(precision), np.std(precision)))