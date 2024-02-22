import yfinance as yf
import pandas as pd

from matplotlib import pyplot as plt
from arima import SIZE
from arima import ARIMA_Analisys
from arima import ARIMA_Predictions
from tens import TENSORFLOW_Prediction


#----------------------------[ IPERPARAMETERS ]--------------------------------

#PERIOD = stocks data downloaded YTD es: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,max.

PERIOD = "5Y"

#------------------------------------------------------------------------------

def get_data() -> yf.Ticker:

    return yf.Ticker(input("Quale titolo vuoi scaricare?\n"))



def main():
    
    data = get_data()
    history = data.history(PERIOD)

    while (history.empty):

        print("Nessun dato trovato")
        data = get_data()
        history = data.history(PERIOD)

    info = data.info

    print("Dati scaricati di:\n")
    print(f"Nome:\t{info['longName']}\n")
    print("Inizio analisi...")

    history = history['Close']
    history.index = pd.to_datetime(history.index).date

    ARIMA_Analisys(history)
    pred1, sqm1 = ARIMA_Predictions(history)
    pred2, sqm2 = TENSORFLOW_Prediction(history)

    print(f"Scarto quadratico medio\n\nARIMA:\t\t\t{sqm1}\nTENSORFLOW:\t\t{sqm2}")

    plt.title(info['longName'])
    plt.xticks(rotation = 45)
    plt.plot(history[int(len(history.values)*SIZE):].index, pred1, color = "red")
    plt.plot(history[int(len(history.values)*SIZE):].index, pred2, color = "orange")
    plt.plot(history[int(len(history.values)*SIZE):].index, history.values[int(len(history.values)*SIZE):], color = "blue")
    plt.legend(['ARIMA', 'Tensorflow', 'Stock Data'])
    plt.show()



main()

