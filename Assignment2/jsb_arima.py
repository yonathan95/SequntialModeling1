from scipy.io import loadmat
import numpy as np
import os
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from pandas import datetime
from pandas import read_csv
from pandas import DataFrame
from statsmodels.tsa.arima.model import ARIMA
from matplotlib import pyplot
from math import sqrt
from sklearn.metrics import mean_squared_error
from jsb_datasets import load_jsb
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)

NUMBERS_OF_NOTES = 88

def CreateNoteTablePerNote(arr,i):
    tableNote = {}
    for index in range(88):
        tableNote[index] = []
    for j in range(len(arr[i])):
        for k in range(len(arr[i][j])):
            tableNote[k].append(arr[i][j][k])
    return tableNote
                

def CreateNoteTablePerNote1(arr):
    tableNote = {}
    for i in range(88):
        tableNote[i] = []
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            for k in range(len(arr[i][j])):
                tableNote[k].append(arr[i][j][k])
    return tableNote
                

#predict numExamples exampels
def ARIMA_Model(numExamples):
    # split into train and test sets
    train, _, _ = load_jsb()
    total_mse = 0
    for j in range(numExamples):
        tableNotesTrain = CreateNoteTablePerNote(train,j)
        predictions = []
        history = [] 
        for note in range(NUMBERS_OF_NOTES):
            if max(tableNotesTrain[note]) == 0:
                continue
            hisotryPerNode = tableNotesTrain[note][:10]
            tempTest = tableNotesTrain[note][11:]
            try:   
                # walk-forward validation
                for t in range(len(tempTest)):
                    model = ARIMA(hisotryPerNode, order=(4,1,1))
                    model_fit = model.fit()
                    output = model_fit.forecast()
                    yhat = threshold(output[0])
                    predictions.append(yhat)
                    obs = tempTest[t]
                    hisotryPerNode.append(obs)
                    history.append(obs)
            except :
                continue
        total_mse += mean_squared_error(history, predictions)
    print('Test MSE: %.3f' % total_mse)
    with open('/home/yonathan95/SequntialModeling1/Assignment2/arima_jsb_final_loss.txt', 'w') as fp:
        fp.write("%s\n" % total_mse)

        

def threshold(x):
    if x >= 0.5 :
        return 1
    return 0    

def parser(x):
 return datetime.strptime('190'+x, '%Y-%m')
 

def main():
    ARIMA_Model(10)
  
  
# Using the special variable 
# __name__
if __name__=="__main__":
    main()



