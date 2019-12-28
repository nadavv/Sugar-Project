# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 15:53:08 2018

@author: opher
"""
import os
import network_updated as network
import data_manipulation as dm
import visualization_results as vis
import datetime

# =============================================================================
# parameters for comparisons:
#     number of days for trainings
#     time for train in each day
#     time for predition
#     
#     lstm amount of cells
#     type of optimizer
#     loss function
#     
#     which value for results evaluation:
#         max difference
#         mean difference
# =============================================================================


#def main():
    
working_directory = os.path.join('..','Libre')
time_without_data = 3

# dataframe from preprocessing
df = dm.data_preprocessing(working_directory,time_without_data)

# current  date and time: (year,month,day,hour,minute)
current_time = datetime.datetime(2017,4,17,14,0)

# training parameters:
NHours_train = 2        # hours
NMinutes_test = 30      # minutes
Ndays_train = 30        # days

# network paramters:
Nepoches = 100
NcellsLSTM = 90
activeFunc = 'sigmoid'
LearningRate = 0.005

df_max,df_min,df_normalized = network.data_normalization(df)

NsamplesPerDay_x,NsamplesPerDay_y,x_train,y_train,x_test,y_test = network.data_arragment(
                    df_normalized,
                    Ndays_train,NHours_train,NMinutes_test,
                    current_time)

preds,biased_preds = network.LSTM_build_train (
                    NsamplesPerDay_x,NsamplesPerDay_y,
                    x_train,x_test,y_train,
                    Nepoches,NcellsLSTM,activeFunc,LearningRate)

y_test_unnorm,preds_unnorm,biased_preds_unnorm = network.data_unnormalization(df_max,df_min,y_test,preds,biased_preds)

Error_L1,Error_L1_bias,Error_L2,Error_L2_bias = network.error_calculation(y_test_unnorm,preds_unnorm,biased_preds_unnorm)

vis.vis_1_prediction(y_test_unnorm,preds_unnorm,biased_preds_unnorm,
                 NsamplesPerDay_y,
                 Ndays_train,NHours_train,
                 NMinutes_test,current_time,
                 Nepoches,NcellsLSTM,activeFunc,LearningRate,
                 Error_L1,Error_L1_bias,Error_L2,Error_L2_bias)


#main()



















