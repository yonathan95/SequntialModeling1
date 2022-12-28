from exchange_dataset import get_exchange_dataloader
from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy as np
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)

# numExamples is max 32
def arima_model(numExamples):
    # split into train and test sets
    train_set = get_exchange_dataloader() 
    total_mse = 0
    for i, (x, y) in enumerate(train_set):
        x = np.resize(x.numpy(),(x.shape[0],x.shape[1]))
        for j in range(numExamples):
            predictions = []
            history = x[j][:10].tolist()
            tempTest = x[j][11:].tolist()
            try:   
                # walk-forward validation
                for t in range(len(tempTest)):
                    model = ARIMA(history, order=(4,1,1))
                    model_fit = model.fit()
                    output = model_fit.forecast()
                    yhat = output[0]
                    predictions.append(yhat)
                    obs = tempTest[t]
                    history.append(obs)
            except Exception as e :
                continue
            total_mse += mean_squared_error(history[10:], predictions)
        break
    print('Test MSE: %.3f' % total_mse)
    with open('/home/yonathan95/SequntialModeling1/Assignment2/arima_exchange_final_loss.txt', 'w') as fp:
        fp.write("%s\n" % total_mse)
            
def main():
    arima_model(10)
  
  
# Using the special variable 
# __name__
if __name__=="__main__":
    main()