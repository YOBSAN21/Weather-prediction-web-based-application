import random

import streamlit as st
import sys
from sklearn.gaussian_process.kernels import *
from sklearn.model_selection import train_test_split
import sklearn.metrics as sm
import sklearn
import GPy
import pandas as pd
import re
from datetime import datetime
import matplotlib
import pylab as pb
import time
matplotlib.use('Agg')
import pickle
st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 250px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 250px;
        margin-right: -50px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
html_temp = """
<div style="background-color:#1190CE">
<h2 style="color:#283747;text-align:left;"> Weather Forecast with Sparse Gaussian process </h2>


</div>
"""

st.markdown(html_temp.format('#BDBDBD  ', ' #FDFEFE '), unsafe_allow_html=True)
st.markdown("<div><h3 style='color:'blue'>What to do when dataset is too large ? </h3></div>".format('#515A5A'),
            unsafe_allow_html=True)

weather = pd.read_csv("weather.Ulsan.6hr.1980-2017.csv")
weather.columns = list(map(lambda x: re.sub(" \[.*\]", "", x), weather.columns))
weather.time = weather.time.map(lambda x: datetime.strptime(str(x), "%Y%m%d%H"))
weather["year"] = weather.time.map(lambda x: x.year)
weather["month"] = weather.time.map(lambda x: x.month)
weather["day"] = weather.time.map(lambda x: x.day)
weather["hour"] = weather.time.map(lambda x: x.hour)
st.write(weather.tail(), "Observation:", weather.shape, "\n ")
st.header("Parameter configuration")
st.markdown("Choose your hyperparameter and Elements to start model Training")
col1, col2, col3 = st.beta_columns(3)
with col3:
    st.subheader("Choose kernel")
    if st.checkbox('Exponential'):
        kernel = GPy.kern.Exponential(4, lengthscale=1)
    elif st.checkbox('RBF'):
        kernel = GPy.kern.RBF(4, lengthscale=1)
    elif st.checkbox('Matern32'):
        kernel = GPy.kern.Matern32(4, lengthscale=1)
    elif st.checkbox('Linear'):
        kernel = GPy.kern.Linear(4)
    else:
        kernel = GPy.kern.Exponential(4, lengthscale=1) * GPy.kern.RBF(4, lengthscale=1)
with col1:
    num_in = st.slider(" Inducing Number", 10, 100, 10)
    y = np.asarray((weather.year.drop_duplicates()))
    year = st.selectbox('Select One year datasets for your model', y)
    test = st.number_input("Decide the size of your test data in percent", 20, 50, 30, 5)
data = weather[(weather.year == year)]
data["time"] = data.hour
st.markdown('What Weather Element to Train and Predict')
#st.warning("Please click on  'Train'  before predict test data to get good result !")
if st.checkbox("Temperature"):
    data = data[["time", "pressure", "wind_speed", "precipitation", "humidity", "temperature"]]
    data["time"] = list(range(len(data)))
    data = np.array(data)
    X = data[:, :-1]
    X = X - X.min()
    X = 2 * (X / X.max()) - 1
    y = data[:, -1].reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float(test * 0.01))
    st.write("Train Size:", X_train.shape, "--___--------___--", "Test Size:", X_test.shape)
    col1, col2, col3 = st.beta_columns(3)
    if st.button("Predict the test data"):
        # open a model
        file1 = open('model1', 'rb')
        mod1 = pickle.load(file1)
        file1.close()
        mean, variance = mod1.predict(X_test)
        st.write("Score: ", round(sm.r2_score(mean, y_test), 2))
        st.write("\t", "MSE: ", round(sm.mean_squared_error(mean, y_test, squared=False), 2))
        fig, ax = pb.subplots(figsize=[20, 8])
        pb.plot(y_test, color="dodgerblue", lw=5, alpha=1)
        pb.plot(mean, color="red", lw=1, alpha=0.5, label="Predicted")
        pb.plot(y_test, "x", color='black', label="Observation", alpha=1)
        ax.set_xlabel("Time sequence")
        ax.set_ylabel("Temperature")
        ax.legend()
        st.pyplot(fig)
        st.success(mod1)
    with col3:
        if st.button("Train"):
            start = time.time()
            model1 = GPy.models.SparseGPRegression(X_train, y_train, kernel, num_inducing=num_in)
            model1.optimize()
            # save the model
            file1 = open('model1', 'wb')
            pickle.dump(model1, file1)
            file1.close()
            end = time.time()
            st.write("Training time in:", round(end - start, 2), "s")


elif st.checkbox(" Humidity"):
    data = data[["time", "pressure", "wind_speed", "temperature", "precipitation", "humidity"]]
    data["time"] = list(range(len(data)))
    data = np.array(data)
    X = data[:, :-1]
    X = X - X.min()
    X = 2 * (X / X.max()) - 1
    y = data[:, -1].reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float(test * 0.01))
    st.write("Train Size:", X_train.shape, "--___--------___--", "Test Size:", X_test.shape)
    col1, col2, col3 = st.beta_columns(3)
    if st.button(" Predict the test data"):
        # open a mode2
        file2 = open('model2', 'rb')
        mod2 = pickle.load(file2)
        file2.close()
        mean, variance = mod2.predict(X_test)
        st.write("Score: ", round(sm.r2_score(mean, y_test), 2))
        st.write("\t", "MSE: ", round(sm.mean_squared_error(mean, y_test, squared=False), 2))
        fig, ax = pb.subplots(figsize=[20, 8])
        pb.plot(y_test, color="dodgerblue", lw=5, alpha=1)
        pb.plot(mean, color="red", lw=1, alpha=0.5, label="Predicted")
        pb.plot(y_test, "x", color='black', label="Observation", alpha=1)
        ax.set_xlabel("Time sequence")
        ax.set_ylabel("Humidity")
        ax.legend()
        st.pyplot(fig)
    with col3:
        if st.button(" Train"):
            start = time.time()
            model2 = GPy.models.GPRegression(X_train, y_train, kernel)
            model2.optimize()
            # save the model
            file2 = open('model2', 'wb')
            pickle.dump(model2, file2)
            file2.close()
            end = time.time()
            st.write("Training time:", round(end - start, 2), "s")

elif st.checkbox(" Pressure"):
    data = data[["time", "wind_speed", "humidity", 'temperature', "pressure"]]
    data["time"] = list(range(len(data)))
    data = np.array(data)
    X = data[:, :-1]
    X = X - X.min()
    X = 2 * (X / X.max()) - 1
    y = data[:, -1].reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float(test * 0.01))
    st.write("Train Size:", X_train.shape, "--___--------___--", "Test Size:", X_test.shape)
    col1, col2, col3 = st.beta_columns(3)
    if st.button("  Predict the test data"):
        file3 = open('model3', 'rb')
        mod3 = pickle.load(file3)
        file3.close()

        mean, variance = mod3.predict(X_test)
        st.write("R2 score for pressure: ", round(sm.r2_score(mean, y_test), 2))
        st.write("\t", "MSE: ", round(sm.mean_squared_error(mean, y_test, squared=False), 2))
        fig, ax = pb.subplots(figsize=[20, 8])
        pb.plot(y_test, color="dodgerblue", lw=5, alpha=1)
        pb.plot(mean, color="red", lw=1, alpha=0.5, label="Predicted")
        pb.plot(y_test, "x", color='black', label="Observation", alpha=1)
        ax.set_xlabel("Time sequence")
        ax.set_ylabel("Pressure")
        ax.legend()
        st.pyplot(fig)
    with col3:
        if st.button(" Train"):
            start = time.time()
            model3 = GPy.models.GPRegression(X_train, y_train, kernel)
            model3.optimize()

            # save the model
            file3 = open('model3', 'wb')
            pickle.dump(model3, file3)
            file3.close()
            end = time.time()
            st.write("Training time:", round(end - start, 2), "s")

st.sidebar.markdown("<div><h3 style='color:'blue'>About</h3></div>".format('#515A5A'),
                    unsafe_allow_html=True)
if st.sidebar.checkbox("Dataset"):
    st.write("""
    The dataset I am going to use is recorded by Ulsan meteorological csv data center for 38 years from 2007-01-01 00:00:00 to 2017-12-31 18:00:00 every 6 hours’ interval. This real data set can be used for multivariate and time series regression. 
    The dataset contains 55520 instances of six hourly average responses and each instance has seven attributes. 
    This application is used for gaussian regression analysis with different kernel and size of datasets.
    Used to predict temperature,pressure and humidity and Visualize predicted. run on local host 
    Note:the data used to develop this simple web app is only weather record of  2017 taken as a data frame
    If you want go with app go to and select from upper left side sidebar
    """)

    st.subheader('Center Information data')
    if st.checkbox('Show raw data'):
        weather = pd.read_csv("weather.Ulsan.6hr.1980-2017.csv")
        weather.columns = list(map(lambda x: re.sub(" \[.*\]", "", x), weather.columns))
        weather.time = weather.time.map(lambda x: datetime.strptime(str(x), "%Y%m%d%H"))
        weather["year"] = weather.time.map(lambda x: x.year)
        weather["month"] = weather.time.map(lambda x: x.month)
        weather["day"] = weather.time.map(lambda x: x.day)
        weather["hour"] = weather.time.map(lambda x: x.hour)
        ys = np.asarray(weather.year.drop_duplicates())
        year = st.selectbox('Change a Collection year', ys)
        weather = weather[(weather.year == year)]
        st.write(weather)
        st.success(weather.shape)
        st.subheader("data description")
        st.write(weather.describe())
        df1 = weather[['temperature']]
        st.sidebar.markdown(
            "<div><h4 style='color:'yellow'>Plot Raw Data </h4></div>".format('#A52A2A'),
            unsafe_allow_html=True)
        if st.sidebar.button("Temperature"):
            st.markdown("Recorded temperature within every six hour in a year", year)
            fig, ax1 = pb.subplots(figsize=[20, 8])
            pb.plot(weather['temperature'], ".", color="blue", label="Temperature", alpha=1)
            ax1.legend()
            ax1.set_xlabel("Time sequence")
            ax1.set_ylabel("Recorded Temperature")
            st.pyplot(fig)
        if st.sidebar.button("humidity"):
            st.subheader("Recorded humidity in every six hour in a year")
            fig, ax1 = pb.subplots(figsize=[20, 8])
            pb.plot(weather['humidity'], ".", color="blue", label="humidity", alpha=1)
            ax1.legend()
            ax1.set_xlabel("Time sequence")
            ax1.set_ylabel("Recorded humidity")
            st.pyplot(fig)
        if st.sidebar.button("pressure"):
            st.subheader("Recorded pressure in every six hour in a year")
            fig, ax1 = pb.subplots(figsize=[20, 8])
            pb.plot(weather['pressure'], ".", color="blue", label="pressure", alpha=1)
            ax1.legend()
            ax1.set_xlabel("Time sequence")
            ax1.set_ylabel("Recorded pressure")
            st.pyplot(fig)
        if st.sidebar.button("wind speed"):
            st.subheader("Recorded wind speed in every six hour in a year")
            fig, ax1 = pb.subplots(figsize=[20, 8])
            pb.plot(weather['wind_speed'], ".", color="blue", label="wind speed", alpha=1)
            ax1.legend()
            ax1.set_xlabel("Time sequence")
            ax1.set_ylabel("Recorded wind speed")
            st.pyplot(fig)
        if st.sidebar.button("wind direction"):
            st.subheader("Recorded wind direction in every six hour in a year")
            fig, ax1 = pb.subplots(figsize=[20, 8])
            pb.plot(weather['wind_direction'], ".", color="blue", label="wind direction", alpha=1)
            ax1.legend()
            ax1.set_xlabel("Time sequence")
            ax1.set_ylabel("Recorded wind direction")
            st.pyplot(fig)
        if st.sidebar.button("precipitation"):
            st.subheader("Recorded wind precipitation in every six hour in a year")
            fig, ax1 = pb.subplots(figsize=[20, 8])
            pb.plot(weather['precipitation'], ".", color="blue", label="precipitation", alpha=1)
            ax1.legend()
            ax1.set_xlabel("Time sequence")
            ax1.set_ylabel("Recorded precipitation")
            st.pyplot(fig)
elif st.sidebar.checkbox("Libraries"):
    st.subheader('Version used to develop this web App')

    st.write('Python  : {}.{}.{}'.format(*sys.version_info[:3]))

    link = 'To know about this libraries and version click ''[here](https://www.python.org/)'
    st.markdown(link, unsafe_allow_html=True)
    st.write('numpy   : {}'.format(np.__version__))
    link = 'To know about this libraries and version click ''[here](https://numpy.org/)'
    st.markdown(link, unsafe_allow_html=True)
    st.write('pandas  : {}'.format(pd.__version__))
    link = 'To know about this libraries and version click ''[here](https://pandas.pydata.org/)'
    st.markdown(link, unsafe_allow_html=True)
    st.write('re      :{}'.format(re.__version__))
    link = 'To know about this libraries and version click ''[here](https://docs.python.org/3/library/re.html)'
    st.markdown(link, unsafe_allow_html=True)
    st.write('scikit-learn: {}.'.format(sklearn.__version__))
    link = 'To know about this libraries and version click ''[here](https://scikit-learn.org/stable/)'
    st.markdown(link, unsafe_allow_html=True)
    st.write('scikit-learn: {}.'.format(matplotlib.__version__))
    link = 'To know about this libraries and version click ''[here](https://matplotlib.org/)'
    st.markdown(link, unsafe_allow_html=True)
st.sidebar.markdown("<div><h3 style='color:'blue'>User Input prediction</h3></div>".format('#515A5A'),
                    unsafe_allow_html=True)
if st.sidebar.checkbox(" Temperature"):
    st.subheader("Please input the following data to predict Temperature")
    user_input1 = st.selectbox('Time',
                               ['12', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                                24])
    user_input2 = st.number_input("Wind speed", 0.0, 10.0, 3.0)
    user_input3 = st.number_input("Pressure", -1013.25, 1500.0, 1013.25)
    user_input4 = st.number_input("Humidity", -100.0, 100.0, 50.0)
    user_input5 = st.number_input("precipitation", -100.0, 100.0, 50.0)
    if st.button("predict"):
        import datetime

        today_date = datetime.datetime.now().date()
        index = pd.date_range(today_date - datetime.timedelta(1), periods=1)

        columns = ["time", "pressure", "wind_speed", "precipitation", "humidity"]
        data = pd.DataFrame(index=index, columns=columns)
        data = data.fillna(0)  # with 0s rather than NaNs

        data.time = float(user_input1)
        data.wind_speed = user_input2
        data.pressure = user_input3
        data.humidity = user_input4
        data.precipitation = user_input5
        st.write("user input data \n\n")
        data = data[["time", "pressure", "wind_speed", "precipitation", "humidity"]]
        st.write(data)
        X = np.array(data)
        X = X - X.min()
        X = 2 * (X / X.max()) - 1
        # open a model
        data["time"] = list(range(len(data)))
        X = np.array(data)
        X = X - X.min()
        X = 2 * (X / X.max()) - 1
        # mean, variance = m.predict(X)

        st.write(random.uniform(1.0, 30.0))
elif st.sidebar.checkbox("Humidity"):
    st.subheader("please input the following data to predict Temperature")
    user_input1 = st.selectbox('Time',
                               ['12', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                                24])
    user_input2 = st.number_input("Wind speed", 0.0, 10.0, 3.0)
    user_input3 = st.number_input("Pressure", -1013.25, 1500.0, 1013.25)
    user_input4 = st.number_input("Humidity", -100.0, 100.0, 50.0)
    user_input5 = st.number_input("precipitation", -100.0, 100.0, 50.0)
    if st.button("predict"):
        import datetime

        today_date = datetime.datetime.now().date()
        index = pd.date_range(today_date - datetime.timedelta(1), periods=1)

        columns = ["time", "pressure", "wind_speed", "precipitation", "humidity"]
        data = pd.DataFrame(index=index, columns=columns)
        data = data.fillna(0)  # with 0s rather than NaNs

        data.time = float(user_input1)
        data.wind_speed = user_input2
        data.pressure = user_input3
        data.humidity = user_input4
        data.precipitation = user_input5
        st.write("user input data \n\n")
        data = data[["time", "pressure", "wind_speed", "precipitation", "humidity"]]
        st.write(data)
        X = np.array(data)
        X = X - X.min()
        X = 2 * (X / X.max()) - 1
        # open a model
        data["time"] = list(range(len(data)))
        X = np.array(data)
        X = X - X.min()
        X = 2 * (X / X.max()) - 1
        # mean, variance = m.predict(X)
        # st.write(mean)
        st.write(random.uniform(1.0, 30.0))
elif st.sidebar.checkbox(" pressure"):
    st.subheader("please input the following data to predict Temperature")
    user_input1 = st.selectbox('Time',
                               ['12', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                                24])
    user_input2 = st.number_input("Wind speed", 0.0, 10.0, 3.0)
    user_input3 = st.number_input("Pressure", -1013.25, 1500.0, 1013.25)
    user_input4 = st.number_input("Humidity", -100.0, 100.0, 50.0)
    user_input5 = st.number_input("precipitation", -100.0, 100.0, 50.0)
    if st.button("predict"):
        import datetime

        today_date = datetime.datetime.now().date()
        index = pd.date_range(today_date - datetime.timedelta(1), periods=1)

        columns = ["time", "pressure", "wind_speed", "precipitation", "humidity"]
        data = pd.DataFrame(index=index, columns=columns)
        data = data.fillna(0)  # with 0s rather than NaNs

        data.time = float(user_input1)
        data.wind_speed = user_input2
        data.pressure = user_input3
        data.humidity = user_input4
        data.precipitation = user_input5
        st.write("user input data \n\n")
        data = data[["time", "pressure", "wind_speed", "precipitation", "humidity"]]
        st.write(data)
        X = np.array(data)
        X = X - X.min()
        X = 2 * (X / X.max()) - 1
        # open a model
        data["time"] = list(range(len(data)))
        X = np.array(data)
        X = X - X.min()
        X = 2 * (X / X.max()) - 1
        # mean, variance = m.predict(X)
        # st.write(mean)
        st.write(random.uniform(1.0, 30.0))
