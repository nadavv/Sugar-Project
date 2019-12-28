# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 20:14:35 2018

@author: opher
"""
from matplotlib import pyplot as plt
import numpy as np


# =============================================================================
# show predictions and test series
# =============================================================================
def vis_1_prediction (y_test_unnorm,preds_unnorm,biased_preds_unnorm,
                 NsamplesPerDay_y,
                 Ndays_train,NHours_train,
                 NMinutes_test,current_time,
                 Nepoches,NcellsLSTM,activeFunc,LearningRate,
                 Error_L1,Error_L1_bias,Error_L2,Error_L2_bias):
    
    date_prediction = '{:%d-%m-%Y}'.format(current_time)
    time_prediction = '{:%H:%M}'.format(current_time)
    
    # Results graph   
    plt.figure()
    plt.plot(np.arange(NsamplesPerDay_y),preds_unnorm[0,0,:],marker='o',label='Probabilistic Prediction')
    plt.plot(np.arange(NsamplesPerDay_y),y_test_unnorm[0,0,:],marker='o',label='Real Values')
    plt.plot(np.arange(NsamplesPerDay_y),biased_preds_unnorm[0,0,:],marker='o',label='Biased Probabilistic Prediction')
    plt.title('Probabilistic Prediction vs. Real Values' 
              '\n' + date_prediction + ' ' + time_prediction +
              '\n training days: ' + str(Ndays_train) + 
              ' , hours before curret time: ' + str(NHours_train) +
              ' , minutes for prediction: ' + str(NMinutes_test) + 
              '\n epoches: ' + str(Nepoches) + ' , LSTM cells: ' + str(NcellsLSTM) + 
              ' , Activation: ' + activeFunc + ', Learn Rate: ' + str(LearningRate) )
    plt.ylabel('Suger level')
    plt.xlabel('time')
    plt.legend()
    plt.grid()
        
    
    
    # Error graph   
    plt.figure()
    
    plt.subplot(121)
    plt.plot(np.arange(NsamplesPerDay_y),Error_L1,marker='o',label='Error_L1')
    plt.plot(np.arange(NsamplesPerDay_y),Error_L1_bias,marker='o',label='Error_L1 Bias')
    plt.title('Error L1')
    plt.ylabel('Error')
    plt.xlabel('time')
    plt.legend()
    plt.grid()
    
    plt.subplot(122)
    plt.plot(np.arange(NsamplesPerDay_y),Error_L2,marker='o',label='Error_L2')
    plt.plot(np.arange(NsamplesPerDay_y),Error_L2_bias,marker='o',label='Error_L2 Bias')
    plt.title('Error L2')
    plt.ylabel('Error')
    plt.xlabel('time')
    plt.legend()
    plt.grid()
    
    plt.suptitle('Errors' 
                 '\n' + date_prediction + ' ' + time_prediction +
                 '\n training days: ' + str(Ndays_train) + 
                 ' , hours before curret time: ' + str(NHours_train) +
                 ' , minutes for prediction: ' + str(NMinutes_test) + 
                 '\n epoches: ' + str(Nepoches) + ' , LSTM cells: ' + str(NcellsLSTM) + 
                 ' , Activation: ' + activeFunc + ', Learn Rate: ' + str(LearningRate) )
    
    
    
    
    
    
def vis_multi_error_prediction ():