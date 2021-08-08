import sys

sys.path.append("..")
import numpy as np
import pandas as pd
import GPy
import pylab as pb
import sklearn.metrics as sm
from sklearn.model_selection import train_test_split
import pickle

weather = pd.read_csv("weather.Ulsan.6hr.1980-2017.csv")
weather = weather[["time", "wind_speed", "humidity", 'temperature', "pressure"]]

weather.drop(weather.index[0:54000], 0, inplace=True)
weather["time"] = list(range(len(weather)))
data = np.array(weather)

X = data[:, :-1]
X = X - X.min()
X = 2 * (X / X.max()) - 1
y = data[:, -1].reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
#Z = 500
#kern = GPy.kern.Exponential(4, lengthscale=1)*GPy.kern.RBF(4, lengthscale=1)
#model3 = GPy.models.GPRegression(X_train, y_train, kern)
#model3.optimize()

# save the model
#file3 = open('model3', 'wb')
#pickle.dump(model3, file3)
#file3.close()
# open a mode3
file3 = open('model3', 'rb')
mod3 = pickle.load(file3)
file3.close()

mean, variance = mod3.predict(X_test)
print("R2 score for pressure: ", round(sm.r2_score(mean, y_test), 2))
pb.plot(y_test, color="dodgerblue", lw=5, alpha=1)
pb.plot(mean, color="red", lw=1, alpha=0.5, label="Predicted")
pb.plot(y_test, "x", color='black', label="Observation", alpha=1)
pb.show()
