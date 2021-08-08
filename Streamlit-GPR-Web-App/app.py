import streamlit as st
import sys
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import *
from sklearn.model_selection import train_test_split
import sklearn.metrics as sm
import sklearn
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import re
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
import pylab as pb
matplotlib.use('Agg')
import pickle

st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 200px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 200px;
        margin-left: -50px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
html_temp = """
<div style="background-color:#1190CE">
<h2 style="color:#283747;text-align:center;">Sparse Gaussian regression process </h2>
</div>
"""
st.markdown(html_temp.format('#BDBDBD  ', ' #FDFEFE '), unsafe_allow_html=True)
st.markdown("<div><h3 style='color:'blue'>What to do when dataset is too large ? </h3></div>".format('#515A5A'),
            unsafe_allow_html=True)
st.markdown("<div><p style='color:#283747'>We use a multivariate Gaussian distribution to establish the relationship "
            "between fₛ and f as our new prior, which we call the sparse prior because"
            " it includes the sparse inducing random variables fₛ."
            "The following prediction is based on the model training of Sparse Gaussian"
            "process with better evaluation matrix"
            " </p></div>".format('##21618C'), unsafe_allow_html=True)

weather = pd.read_csv("weather.Ulsan.6hr.2007-2017.csv")
# weather.drop(weather.index[0:16000], 0, inplace=True)
weather.columns = list(map(lambda x: re.sub(" \[.*\]", "", x), weather.columns))

html_temp = """
		<div style="background-color:{};padding:10px;border-radius:10px">
		<h1 style="color:{};text-align:center;"> </h1>
		</div>
		"""
page_bg_img = '''
<style>
body {
background-color: blue ;
background-size: cover;
}
</style>
'''
st.sidebar.markdown("<div><h4 style='color:'blue'> Model Training</h4></div>".format('#515A5A'),
                    unsafe_allow_html=True)
if st.sidebar.checkbox("Start"):
    weather = pd.read_csv("weather.Ulsan.6hr.2007-2017.csv")
    weather.columns = list(map(lambda x: re.sub(" \[.*\]", "", x), weather.columns))
    weather.time = weather.time.map(lambda x: datetime.strptime(str(x), "%Y%m%d%H"))
    weather["year"] = weather.time.map(lambda x: x.year)
    weather["month"] = weather.time.map(lambda x: x.month)
    weather["day"] = weather.time.map(lambda x: x.day)
    st.write(weather.head())
    col1,  col2,col3= st.beta_columns(3)
    with col1:
        st.subheader("Choose kernel")
        if st.checkbox('RationalQuadratic'):
            kernel = RationalQuadratic(alpha=0.1, length_scale=0.5)
        elif st.checkbox('ExpSineSquared'):
            kernel = ExpSineSquared(length_scale=0.1, periodicity=5)
        elif st.checkbox('Matern'):
            kernel = Matern(length_scale=1.1, nu=3 / 2)
        elif st.checkbox('RBF'):
            kernel = RBF(length_scale=0.1)
        else:
            kernel = RBF(length_scale=0.1) * RationalQuadratic(alpha=0.1, length_scale=0.5)

        with col3:
            st.subheader("change parameters")
            random_s = st.number_input(' Change random state', 0.0, 1234.0)
            optimizer = st.slider(" Change optimizer", 1, 9)


    st.subheader("Which Year and Month data would you like to use?")

    weather["time"] = list(range(len(weather)))
    mm = MinMaxScaler(feature_range=(0, 1))
    col1, col2, col3 = st.beta_columns(3)
    with col1:
        year = st.number_input('Year  (2007-2017)', 2007, 2017)
        weather = weather[(weather.year == year)]
    with col2:
        months = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun", 7: "Jul", 8: "Aug", 9: "Sem", 10: "Oct",
                  11: "Nov", 12: "Dec"}


        def format_func(option):
            return months[option]


        month1 = st.selectbox("Month", options=list(months.keys()), format_func=format_func)

        test_month = weather[(weather.month == month1)]
    with col3:
        day = st.slider(" Day ", 1, 31)
        test_day = test_month[(test_month.day == day)]

    if st.checkbox("summit"):
        st.subheader("select attribute to predict")
        if st.checkbox("temperature"):

            weather = weather[["time", "wind_speed", "pressure", "humidity", "wind_direction", 'temperature']]
            sample = weather.sample(frac=0.8)
            X = sample.iloc[:, 0:5]
            y = sample.iloc[:, 5:6]

            st.write("input: "  "time, wind_speed, pressure, humidity, wind_direction")

            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=int(optimizer))

            col1, col2, col3 = st.beta_columns(3)
            try:
                with col1:
                    if st.checkbox("One month unseen data prediction"):
                        gp.fit(X, y)
                        test = test_month[
                            ["time", "wind_speed", "pressure", "humidity", "wind_direction", 'temperature']]
                        X1 = test.iloc[:, 0:5]
                        y1 = test.iloc[:, 5:6]
                        predicted_temperature, y_std = gp.predict(X1, return_std=True)
                        origin_temperature = y1
                        st.write("sigma: ", y_std.mean())

                with col3:
                    if st.checkbox("One day unseen data prediction"):
                        gp.fit(X, y)

                        test_day = test_day[
                            ["time", "wind_speed", "pressure", "humidity", "wind_direction", 'temperature']]
                        X1 = test_day.iloc[:, 0:5]
                        y1 = test_day.iloc[:, 5:6]
                        predicted_temperature, y_std = gp.predict(X1, return_std=True)
                        origin_temperature = y1
                        st.write("sigma: ", y_std.mean())
                GPR = gp
                st.experimental_show(GPR)
                st.write("fitted")
                pd.DataFrame(predicted_temperature).to_csv("predicted_temperature.csv")
                pd.DataFrame(origin_temperature).to_csv("origin_temperature.csv")

                col1, col2, col3, col4 = st.beta_columns(4)

                with col1:

                    st.write("\t\tpredicted temperature")
                    st.write(predicted_temperature)
                with col2:
                    st.write("\t\torigin temperature")
                    st.write(origin_temperature)

                with col4:
                    st.subheader("\n  scikit-learn Model Evaluation")
                    st.write("\t", "MSE: ",
                             round(sm.mean_squared_error(predicted_temperature, origin_temperature, squared=False), 2))
                    st.write("\t", "RMSE: ",
                             round(np.sqrt(sm.mean_squared_error(predicted_temperature, origin_temperature)), 2))
                    st.write("\t", "r2 score: ", round(sm.r2_score(predicted_temperature, origin_temperature), 2))
                    st.write("Mean absolute error :",
                             round(sm.mean_absolute_error(predicted_temperature, origin_temperature), 2))
                    st.write("Median absolute error :",
                             round(sm.median_absolute_error(predicted_temperature, origin_temperature), 2))
                st.write(" default plotting demo")
                length = len(predicted_temperature)

                mylist = [item for item in range(length)]
                x_test = pd.DataFrame()
                x_test["time"] = mylist
                st.subheader("temperature plotting")
                fig, ax1 = plt.subplots(figsize=[20, 5])
                ax1.set_xlabel("time sequence")
                ax1.set_ylabel("temperature")
                ax1.scatter(x_test['time'], predicted_temperature, color='red', label="predicted temperature")
                ax1.legend()
                st.pyplot(fig)

                fig, ax1 = plt.subplots(figsize=[20, 5])
                ax1.set_xlabel("time sequence")
                ax1.set_ylabel("temperature")
                ax1.scatter(x_test['time'], origin_temperature, color='blue', label="origin temperature")
                ax1.legend()
                st.pyplot(fig)

                fig, ax = plt.subplots(figsize=[20, 5])
                ax.set_xlabel("time sequence")
                ax.set_ylabel("temperature")
                ax.plot(x_test['time'], origin_temperature, color='blue', label="origin temperature")
                ax.plot(x_test['time'], predicted_temperature, color='red', label="predicted temperature")
                ax.legend()
                st.pyplot(fig)

                st.header("more  to plotting Demo")

                if st.button("bar_chart"):
                    st.bar_chart(predicted_temperature)
                    st.bar_chart(origin_temperature)
                elif st.button("area_chart"):
                    st.area_chart(predicted_temperature)
                    st.area_chart(origin_temperature)
            except:
                st.success("Please choose day or month  from above to predict temperature from unseen data")



        elif st.checkbox("pressure"):
            weather = weather[["time", "wind_speed", "temperature", "humidity", "wind_direction", 'pressure']]
            sample = weather.sample(frac=0.8)
            X = sample.iloc[:, 0:5]
            y = sample.iloc[:, 5:6]

            st.write("input: "  "time, wind_speed, temperature, humidity, wind_direction")

            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=int(optimizer))

            col1, col2, col3 = st.beta_columns(3)
            try:
                with col1:
                    if st.checkbox("one month prediction"):
                        gp.fit(X, y)
                        st.write("Your model: ", gp, "fitted")
                        test = test_month[
                            ["time", "wind_speed", "temperature", "humidity", "wind_direction", 'pressure']]
                        X1 = test.iloc[:, 0:5]
                        y1 = test.iloc[:, 5:6]
                        predicted_pressure, y_std = gp.predict(X1, return_std=True)
                        origin_pressure = y1

                with col3:
                    if st.checkbox("one day prediction"):
                        gp.fit(X, y)
                        st.write("Your model: ", gp, "fitted")
                        test_day = test_day[
                            ["time", "wind_speed", "pressure", "humidity", "wind_direction", 'temperature']]
                        X1 = test_day.iloc[:, 0:5]
                        y1 = test_day.iloc[:, 5:6]
                        predicted_pressure, y_std = gp.predict(X1, return_std=True)
                        origin_pressure = y1

                GPR = gp
                st.experimental_show(GPR)
                st.write("fitted")
                pd.DataFrame(predicted_pressure).to_csv("predicted_pressure.csv")
                pd.DataFrame(origin_pressure).to_csv("origin_pressure.csv")

                col1, col2, col3, col4 = st.beta_columns(4)

                with col1:

                    st.write("\t\tpredicted pressure")
                    st.write(predicted_pressure)
                with col2:
                    st.write("\t\torigin pressure")
                    st.write(origin_pressure)

                with col4:
                    st.subheader("\n  scikit-learn Model Evaluation")
                    st.write("\t", "MSE: ",
                             round(sm.mean_squared_error(predicted_pressure, origin_pressure, squared=False), 2))
                    st.write("\t", "RMSE: ",
                             round(np.sqrt(sm.mean_squared_error(predicted_pressure, origin_pressure)), 2))
                    st.write("\t", "r2 score: ", round(sm.r2_score(predicted_pressure, origin_pressure), 2))
                    st.write("Mean absolute error :",
                             round(sm.mean_absolute_error(predicted_pressure, origin_pressure), 2))
                    st.write("Median absolute error :",
                             round(sm.median_absolute_error(predicted_pressure, origin_pressure), 2))
                length = len(predicted_pressure)

                mylist = [item for item in range(length)]
                x_test = pd.DataFrame()
                x_test["time"] = mylist
                st.subheader("pressure plotting")
                fig, ax1 = plt.subplots(figsize=[20, 5])
                ax1.set_xlabel("time sequence")
                ax1.set_ylabel("pressure")
                ax1.scatter(x_test['time'], origin_pressure, color='red', label="predicted pressure")
                ax1.legend()
                st.pyplot(fig)

                fig, ax1 = plt.subplots(figsize=[20, 5])
                ax1.set_xlabel("time sequence")
                ax1.set_ylabel("pressure")
                ax1.scatter(x_test['time'], origin_pressure, color='blue', label="origin pressure")
                ax1.legend()
                st.pyplot(fig)

                fig, ax = plt.subplots(figsize=[20, 5])
                ax.set_xlabel("time sequence")
                ax.set_ylabel("pressure")
                ax.plot(x_test['time'], origin_pressure, color='blue', label="origin pressure")
                ax.plot(x_test['time'], predicted_pressure, color='red', label="predicted pressure")
                ax.legend()
                st.pyplot(fig)

                st.header("more  to plotting Demo")

                if st.button("bar_chart"):
                    st.bar_chart(predicted_pressure)
                    st.bar_chart(origin_pressure)
                elif st.button("area_chart"):
                    st.area_chart(predicted_pressure)
                    st.area_chart(origin_pressure)

            except:
                st.success("Please choose day or month above to predict pressure from unseen ")


        elif st.checkbox("humidity"):

            weather = weather[["time", "wind_speed", "pressure", "temperature", "wind_direction", 'humidity']]
            sample = weather.sample(frac=0.8)
            X = sample.iloc[:, 0:5]
            y = sample.iloc[:, 5:6]

            st.write("input: "  "time, wind_speed, pressure, temperature, wind_direction")

            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=int(optimizer))

            col1, col2, col3 = st.beta_columns(3)
            try:
                with col1:
                    if st.checkbox("one month prediction"):
                        gp.fit(X, y)
                        st.write("Your model: ", gp, "fitted")
                        test = test_month[
                            ["time", "wind_speed", "pressure", "temperature", "wind_direction", 'humidity']]
                        X1 = test.iloc[:, 0:5]
                        y1 = test.iloc[:, 5:6]
                        predicted_humidity, y_std = gp.predict(X1, return_std=True)
                        origin_humidity = y1

                with col3:
                    if st.checkbox("one day prediction"):
                        gp.fit(X, y)
                        st.write("Your model: ", gp, "fitted")
                        test_day = test_day[
                            ["time", "wind_speed", "pressure", "humidity", "wind_direction", 'temperature']]
                        X1 = test_day.iloc[:, 0:5]
                        y1 = test_day.iloc[:, 5:6]
                        predicted_humidity, y_std = gp.predict(X1, return_std=True)
                        origin_humidity = y1

                GPR = gp
                st.experimental_show(GPR)
                st.write("fitted")
                pd.DataFrame(predicted_humidity).to_csv("predicted_humidity.csv")
                pd.DataFrame(origin_humidity).to_csv("origin_humidity.csv")

                col1, col2, col3, col4 = st.beta_columns(4)

                with col1:

                    st.write("\t\tpredicted humidity")
                    st.write(predicted_humidity)
                with col2:
                    st.write("\t\torigin humidity")
                    st.write(origin_humidity)

                with col4:
                    st.subheader("\n  scikit-learn Model Evaluation")
                    st.write("\t", "MSE: ",
                             round(sm.mean_squared_error(predicted_humidity, origin_humidity, squared=False), 2))
                    st.write("\t", "RMSE: ",
                             round(np.sqrt(sm.mean_squared_error(predicted_humidity, origin_humidity)), 2))
                    st.write("\t", "r2 score: ", round(sm.r2_score(predicted_humidity, origin_humidity), 2))
                    st.write("Mean absolute error :",
                             round(sm.mean_absolute_error(predicted_humidity, origin_humidity), 2))
                    st.write("Median absolute error :",
                             round(sm.median_absolute_error(predicted_humidity, origin_humidity), 2))
                st.write(" default plotting demo")
                length = len(predicted_humidity)

                mylist = [item for item in range(length)]
                x_test = pd.DataFrame()
                x_test["time"] = mylist
                st.subheader("humidity plotting")
                fig, ax1 = plt.subplots(figsize=[20, 5])
                ax1.set_xlabel("time sequence")
                ax1.set_ylabel("humidity")
                ax1.scatter(x_test['time'], predicted_humidity, color='red', label="predicted humidity")
                ax1.legend()
                st.pyplot(fig)

                fig, ax1 = plt.subplots(figsize=[20, 5])
                ax1.set_xlabel("time sequence")
                ax1.set_ylabel("humidity")
                ax1.scatter(x_test['time'], origin_humidity, color='blue', label="origin humidity")
                ax1.legend()
                st.pyplot(fig)

                fig, ax = plt.subplots(figsize=[20, 5])
                ax.set_xlabel("time sequence")
                ax.set_ylabel("humidity")
                ax.plot(x_test['time'], origin_humidity, color='blue', label="origin humidity")
                ax.plot(x_test['time'], predicted_humidity, color='red', label="predicted humidity")
                ax.legend()
                st.pyplot(fig)

                st.header("more  to plotting Demo")

                if st.button("bar_chart"):
                    st.bar_chart(predicted_humidity)
                    st.bar_chart(origin_humidity)
                elif st.button("area_chart"):
                    st.area_chart(predicted_humidity)
                    st.area_chart(origin_humidity)
            except:
                st.success("Please choose day or month prediction to predict humidity from unseen data")

        elif st.checkbox("wind speed"):

            weather = weather[["time", "temperature", "pressure", "humidity", "wind_direction", 'wind_speed']]
            sample = weather.sample(frac=0.8)
            X = sample.iloc[:, 0:5]
            y = sample.iloc[:, 5:6]

            st.write("input: "  "time, temperature, pressure, humidity, wind_direction")

            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=int(optimizer))

            col1, col2, col3 = st.beta_columns(3)

            try:
                with col1:
                    if st.checkbox("one month prediction"):
                        gp.fit(X, y)
                        st.write("Your model: ", gp, "fitted")
                        test = test_month[
                            ["time", "temperature", "pressure", "humidity", "wind_direction", 'wind_speed']]
                        X1 = test.iloc[:, 0:5]
                        y1 = test.iloc[:, 5:6]
                        predicted_wind_speed, y_std = gp.predict(X1, return_std=True)
                        origin_wind_speed = y1

                with col3:
                    if st.checkbox("one day prediction"):
                        gp.fit(X, y)
                        st.write("Your model: ", gp, "fitted")
                        test_day = test_day[
                            ["time", "wind_speed", "pressure", "humidity", "wind_direction", 'temperature']]
                        X1 = test_day.iloc[:, 0:5]
                        y1 = test_day.iloc[:, 5:6]
                        predicted_wind_speed, y_std = gp.predict(X1, return_std=True)
                        origin_wind_speed = y1

                    GPR = gp
                    st.experimental_show(GPR)
                    st.write("fitted")
                    pd.DataFrame(predicted_wind_speed).to_csv("predicted_wind_speed.csv")
                    pd.DataFrame(origin_wind_speed).to_csv("origin_wind_speed.csv")

                col1, col2, col3, col4 = st.beta_columns(4)

                with col1:

                    st.write("\t\tpredicted wind_speed")
                    st.write(predicted_wind_speed)
                with col2:
                    st.write("\t\torigin wind_speed")
                    st.write(origin_wind_speed)

                with col4:
                    st.subheader("\n  scikit-learn Model Evaluation")
                    st.write("\t", "MSE: ",
                             round(sm.mean_squared_error(predicted_wind_speed, origin_wind_speed, squared=False), 2))
                    st.write("\t", "RMSE: ",
                             round(np.sqrt(sm.mean_squared_error(predicted_wind_speed, origin_wind_speed)), 2))
                    st.write("\t", "r2 score: ", round(sm.r2_score(predicted_wind_speed, origin_wind_speed), 2))
                    st.write("Mean absolute error :",
                             round(sm.mean_absolute_error(predicted_wind_speed, origin_wind_speed), 2))
                    st.write("Median absolute error :",
                             round(sm.median_absolute_error(predicted_wind_speed, origin_wind_speed), 2))

                st.write(" default plotting demo")
                length = len(predicted_wind_speed)

                mylist = [item for item in range(length)]
                x_test = pd.DataFrame()
                x_test["time"] = mylist
                st.subheader("wind_speed plotting")
                fig, ax1 = plt.subplots(figsize=[20, 5])
                ax1.set_xlabel("time sequence")
                ax1.set_ylabel("wind_speed")
                ax1.scatter(x_test['time'], origin_wind_speed, color='red', label="predicted wind_speed")
                ax1.legend()
                st.pyplot(fig)

                fig, ax1 = plt.subplots(figsize=[20, 5])
                ax1.set_xlabel("time sequence")
                ax1.set_ylabel("wind_speed")
                ax1.scatter(x_test['time'], origin_wind_speed, color='blue', label="origin wind_speed")
                ax1.legend()
                st.pyplot(fig)
                fig, ax = plt.subplots(figsize=[20, 5])
                ax.set_xlabel("time sequence")
                ax.set_ylabel("wind_speed")
                ax.plot(x_test['time'], origin_wind_speed, color='blue', label="origin wind_speed")
                ax.plot(x_test['time'], predicted_wind_speed, color='red', label="predicted wind_speed")
                ax.legend()
                st.pyplot(fig)

                st.header("more  to plotting Demo")

                if st.button("bar_chart"):
                    st.bar_chart(predicted_wind_speed)
                    st.bar_chart(origin_wind_speed)
                elif st.button("area_chart"):
                    st.area_chart(predicted_wind_speed)
                    st.area_chart(origin_wind_speed)

            except:
                st.success("Please choose month or day to predict wind speed from unseen data")

st.sidebar.markdown("<div><h4 style='color:'blue'>Sparse GPs Prediction</h4></div>".format('#515A5A'),
                    unsafe_allow_html=True)
if st.sidebar.checkbox("Temperature"):
    st.subheader("please input the following data to predict Temperature")
    user_input1 = st.selectbox('Time',
                               ['12', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                                24])
    user_input2 = st.number_input("Wind speed", 0.0, 10.0, 3.0)
    user_input3 = st.number_input("Pressure", -1013.25, 1500.0, 1013.25)
    user_input4 = st.number_input("Humidity", -100.0, 100.0, 50.0)
    user_input5 = st.number_input("Wind direction", -360.0, 360.0, 270.0)
    if st.button("predict"):
        import datetime

        todays_date = datetime.datetime.now().date()
        index = pd.date_range(todays_date - datetime.timedelta(1), periods=1)

        columns = ["time", "wind_speed", "pressure", "humidity", "wind_direction"]
        data = pd.DataFrame(index=index, columns=columns)
        data = data.fillna(0)  # with 0s rather than NaNs

        data.time = user_input1
        data.wind_speed = user_input2
        data.pressure = user_input3
        data.humidity = user_input4
        data.wind_direction = user_input5
        st.write("user input data \n\n")
        df = pd.DataFrame(data, index=index, columns=columns)
        df = df[["time", "wind_speed", "pressure", "humidity", "wind_direction"]]
        st.write(df)
        with open('model1.pkl', 'rb') as f:
            mp = pickle.load(f)

        pred, y_std = mp.predict(df, return_std=True)
        predict = np.asscalar(pred)
        st.write("\t\t\t\t\t\t\t\t", "Today Temperature is about: ", round(predict, 2))
elif st.sidebar.checkbox("Humidity"):
    st.subheader("please input the following data to predict humidity")
    user_input1 = st.selectbox('Time',
                               ['12', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                                24])
    user_input2 = st.number_input("wind speed", 0.0, 10.0, 3.0)
    user_input3 = st.number_input("Pressure", -1013.25, 1500.0, 1013.25)
    user_input4 = st.number_input("Temperature", -100.0, 100.0, 28.0)
    user_input5 = st.number_input("Wind direction", -360.0, 360.0, 270.0)
    if st.button("predict"):
        import datetime

        todays_date = datetime.datetime.now().date()
        index = pd.date_range(todays_date - datetime.timedelta(1), periods=1)

        columns = ["time", "wind_speed", "pressure", "temperature", "wind_direction"]
        data = pd.DataFrame(index=index, columns=columns)
        data = data.fillna(0)  # with 0s rather than NaNs
        data.time = user_input1
        data.wind_speed = user_input2
        data.pressure = user_input3
        data.temperature = user_input4
        data.wind_direction = user_input5
        st.write("User input data \n\n")
        df = pd.DataFrame(data, index=index, columns=columns)
        df = df[["time", "wind_speed", "pressure", "temperature", "wind_direction"]]
        st.write(df)
        with open('model', 'rb') as f2:
            mp = pickle.load(f2)

        pred, y_std = mp.predict(df, return_std=True)
        prediction = np.asscalar(pred)
        st.write("\t\t\t\t\t\t\t\t", "Today humidity is about: ", round(prediction, 2))
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
        weather.columns = list(map(lambda x: re.sub(" \[.*\]", "", x), weather.columns))
        weather.time = weather.time.map(lambda x: datetime.strptime(str(x), "%Y%m%d%H"))
        weather["year"] = weather.time.map(lambda x: x.year)
        year = st.number_input('Year  (2007-2017)', 2007, 2017)
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
            st.subheader("Recorded temperature within every six hour in a year")
            fig, ax1 = pb.subplots(figsize=[20, 8])
            pb.plot(weather['temperature'],".", color="blue",label="Temperature", alpha=1)
            ax1.legend()
            ax1.set_xlabel("Time sequence")
            ax1.set_ylabel("Recorded Temperature")
            st.pyplot(fig)
        if st.sidebar.button("humidity"):
            st.subheader("Recorded humidity in every six hour in a year")
            fig, ax1 = pb.subplots(figsize=[20, 8])
            pb.plot(weather['humidity'],".", color="blue",label="humidity", alpha=1)
            ax1.legend()
            ax1.set_xlabel("Time sequence")
            ax1.set_ylabel("Recorded humidity")
            st.pyplot(fig)
        if st.sidebar.button("pressure"):
            st.subheader("Recorded pressure in every six hour in a year")
            fig, ax1 = pb.subplots(figsize=[20, 8])
            pb.plot(weather['pressure'],".", color="blue",label="pressure", alpha=1)
            ax1.legend()
            ax1.set_xlabel("Time sequence")
            ax1.set_ylabel("Recorded pressure")
            st.pyplot(fig)
        if st.sidebar.button("wind speed"):
            st.subheader("Recorded wind speed in every six hour in a year")
            fig, ax1 = pb.subplots(figsize=[20, 8])
            pb.plot(weather['wind_speed'],".", color="blue",label="wind speed", alpha=1)
            ax1.legend()
            ax1.set_xlabel("Time sequence")
            ax1.set_ylabel("Recorded wind speed")
            st.pyplot(fig)
        if st.sidebar.button("wind direction"):
            st.subheader("Recorded wind direction in every six hour in a year")
            fig, ax1 = pb.subplots(figsize=[20, 8])
            pb.plot(weather['wind_direction'],".", color="blue",label="wind direction", alpha=1)
            ax1.legend()
            ax1.set_xlabel("Time sequence")
            ax1.set_ylabel("Recorded wind direction")
            st.pyplot(fig)
        if st.sidebar.button("precipitation"):
            st.subheader("Recorded wind precipitation in every six hour in a year")
            fig, ax1 = pb.subplots(figsize=[20, 8])
            pb.plot(weather['precipitation'],".", color="blue",label="precipitation", alpha=1)
            ax1.legend()
            ax1.set_xlabel("Time sequence")
            ax1.set_ylabel("Recorded precipitation")
            st.pyplot(fig)
if st.sidebar.checkbox("Libraries"):
    st.subheader('Version used to develop this web App')

    st.write('Python  : {}.{}.{}'.format(*sys.version_info[:3]))

    link = 'To kow about this libraries and version click ''[here](https://www.python.org/)'
    st.markdown(link, unsafe_allow_html=True)
    st.write('numpy   : {}'.format(np.__version__))
    link = 'To kow about this libraries and version click ''[here](https://numpy.org/)'
    st.markdown(link, unsafe_allow_html=True)
    st.write('pandas  : {}'.format(pd.__version__))
    link = 'To kow about this libraries and version click ''[here](https://pandas.pydata.org/)'
    st.markdown(link, unsafe_allow_html=True)
    st.write('re      :{}'.format(re.__version__))
    link = 'To kow about this libraries and version click ''[here](https://docs.python.org/3/library/re.html)'
    st.markdown(link, unsafe_allow_html=True)
    st.write('scikit-learn: {}.'.format(sklearn.__version__))
    link = 'To kow about this libraries and version click ''[here](https://scikit-learn.org/stable/)'
    st.markdown(link, unsafe_allow_html=True)
    st.write('scikit-learn: {}.'.format(matplotlib.__version__))
    link = 'To kow about this libraries and version click ''[here](https://matplotlib.org/)'
    st.markdown(link, unsafe_allow_html=True)

