import pandas as pd
import progressbar as pb

from statsmodels.tsa.arima.model import ARIMA
from matplotlib import pyplot as plt
from math import sqrt
from sklearn.metrics import mean_squared_error

#----------------------------[ IPERPARAMETERS ]--------------------------------

#AR - Autoregression = emphasizes the dependent relationship between an 
#     observation and its preceding or ‘lagged’ observations. The lag order, 
#     representing the number of lag observations incorporated in the model.

AR = 20


# I - Integrated = achieve a stationary time series, one that doesn’t exhibit 
#     trend or seasonality, differencing is applied. 
#     It typically involves subtracting an observation from its preceding 
#     observation. Degree of differencing, denoting the number of times raw 
#     observations undergo differencing.

I = 4


#MA - Moving Average = component zeroes in on the relationship between an 
#     observation and the residual error from a moving average model based on 
#     lagged observations. Order of moving average, indicating the size of the 
#     moving average window.

MA = 0


#The (p,d,q) order of the model for the autoregressive, differences, and
#moving average components. d is always an integer, while p and q may
#either be integers or lists of integers.


# SIZE = train set dimension (percentage) [0:SIZE] for train and [SIZE:len(x)]
#        for test

SIZE = 0.70

#------------------------------------------------------------------------------

def create_model(data) -> ARIMA:

    model = ARIMA(data, order=(AR, I, MA))
    model_fit = model.fit()

    return model_fit



def ARIMA_Analisys(data: pd.DataFrame) -> None:

    model_fit = create_model(data)
    residuals = pd.DataFrame(model_fit.resid)

    print(model_fit.summary())
    print(residuals.describe())


def ARIMA_Predictions(data: pd.DataFrame):

    values = data.values
    size = int(len(values)*SIZE)
    train = values[0:size]
    test = values[size:len(values)]
    history = [x for x in train]
    predictions = list()
    bar = pb.ProgressBar(maxval= len(test),
                         widgets=[pb.Bar('=', '[', ']'), ' ', pb.Percentage()])
    bar.start()

    for x in range(len(test)):

        model_fit = create_model(history)
        output = model_fit.forecast()
        pred = output[0]
        predictions.append(pred)
        obs = test[x]
        history.append(obs)
        bar.update(x)

        #print('predicted=%f, expected=%f' % (pred, obs))

    bar.finish()
    sqm = sqrt(mean_squared_error(test, predictions))

    return predictions, sqm
 
