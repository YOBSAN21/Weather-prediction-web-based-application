import sys

sys.path.append("..")
import numpy as np
import pandas as pd
import pylab as pb
import urllib.request
import GPy
import sklearn.metrics as sm
from sklearn.model_selection import train_test_split
from scipy.io import loadmat
import pickle

weather = pd.read_csv("weather.Ulsan.6hr.1980-2017.csv")
weather = weather[["time", "pressure", "wind_speed", "temperature", "precipitation","humidity"]]

weather.drop(weather.index[0:54000], 0, inplace=True)
weather["time"] = list(range(len(weather)))
data = np.array(weather)

X = data[:, :-1]
X = X - X.min()
X = 2 * (X / X.max()) - 1
y = data[:, -1].reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
#Z = 50
#kern = GPy.kern.Exponential(5, lengthscale=1) * GPy.kern.RBF(5, lengthscale=1)
#model2 = GPy.models.GPRegression(X_train, y_train, kern)
#model2.optimize()

#file2 = open('model2', 'wb')
#pickle.dump(model2, file2)
#file2.close()
# open a model
file2 = open('model2', 'rb')
mod2 = pickle.load(file2)
file2.close()
print(mod2)

mean, variance = mod2.predict(X_test)
print("Score: ", round(sm.r2_score(mean, y_test), 2))
pb.plot(y_test, color="dodgerblue", lw=5, alpha=1)
pb.plot(mean, color="red", lw=1, alpha=0.5, label="Predicted")
pb.plot(y_test, "x", color='black', label="Observation", alpha=1)
pb.xlabel("Time Sequence")
pb.ylabel("Temperature")
pb.legend()
pb.title("Gaussian Process Regression")
pb.show()
