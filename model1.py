import sys

import model1

sys.path.append("..")
import numpy as np
import pandas as pd
import pylab as pb
import GPy
import sklearn.metrics as sm
from sklearn.model_selection import train_test_split
from scipy.io import loadmat
import pickle

weather = pd.read_csv("weather.Ulsan.6hr.1980-2017.csv")
weather = weather[["time", "pressure", "wind_speed", "precipitation", "humidity", "temperature"]]
weather.drop(weather.index[0:54000], 0, inplace=True)
weather["time"] = list(range(len(weather)))
data = np.array(weather)

X = data[:, :-1]
X = X - X.min()
X = 2 * (X / X.max()) - 1
y = data[:, -1].reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
#Z =1000
#kern = GPy.kern.Exponential(5, lengthscale=1) * GPy.kern.RBF(5, lengthscale=1)
#model1 = GPy.models.SparseGPRegression(X_train, y_train, kern, num_inducing=Z)
#model1.optimize()
# save the model
#file1 = open('model1', 'wb')
#pickle.dump(model1, file1)
#file1.close()
# open a model
file1 = open('model1', 'rb')
mod1 = pickle.load(file1)
file1.close()
print(mod1)

mean, variance = mod1.predict(X_test)
print("Score: ", round(sm.r2_score(mean, y_test), 2))
pb.plot(y_test, color="dodgerblue", lw=5, alpha=1)
pb.plot(mean, color="red", lw=1, alpha=0.5, label="Predicted")
pb.plot(y_test, "x", color='black', label="Observation", alpha=1)
pb.xlabel("Time Sequence")
pb.ylabel("Temperature")
pb.legend()
pb.title("Sparse Gaussian Process Regression")
pb.show()






