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
import numpy as np

#%%

working_directory = os.path.join('..','Libre')
time_without_data = 3
# dataframe from preprocessing
df = dm.data_preprocessing(working_directory,time_without_data)
# data normalization
df_max,df_min,df_normalized = network.data_normalization(df)
# datetime format: datetime.datetime(YEAR,MONTH,DAY,HOUR,MINUTE)
current_time = datetime.datetime(2017,8,31,15,30)


#%%
# =============================================================================
# Prediction with the trainand network parameters below
# =============================================================================

# training parameters:
NHours_train = 2        # hours
NMinutes_test = 30      # minutes
Ndays_train = 30        # days
# network paramters:
Nepoches = 100
NcellsLSTM = 90
activeFunc = 'tanh'
LearningRate = 0.005

# data arragement before inserting the to the network
NsamplesPerDay_x,NsamplesPerDay_y,x_train,y_train,x_test,y_test = network.data_arragment(
                    df_normalized,
                    Ndays_train,NHours_train,NMinutes_test,
                    current_time)

# LSTM network build ,trainings and predictions
preds,biased_preds = network.LSTM_build_train (
                    NsamplesPerDay_x,NsamplesPerDay_y,
                    x_train,x_test,y_train,
                    Nepoches,NcellsLSTM,activeFunc,LearningRate)

# Prediction data unnormalization
y_test_unnorm,preds_unnorm,biased_preds_unnorm = network.data_unnormalization(df_max,df_min,y_test,preds,biased_preds)

# Error calculation
ABS_error_bias, MAE_bias, MAE_bias_STD = network.error_calculation(
            y_test_unnorm,preds_unnorm,biased_preds_unnorm)

# Display Results 
vis.vis_1_prediction(y_test_unnorm,preds_unnorm,biased_preds_unnorm,
                 NsamplesPerDay_y,
                 Ndays_train,NHours_train,
                 NMinutes_test,current_time,
                 Nepoches,NcellsLSTM,activeFunc,LearningRate,
                 ABS_error_bias,MAE_bias,MAE_bias_STD)


#%%
# =============================================================================
# Number of train days
# =============================================================================
# training parameters:
NHours_train = 2        # hours
NMinutes_test = 30      # minutes
# network paramters:
Nepoches = 90
NcellsLSTM = 90
activeFunc = 'tanh'
LearningRate = 0.005

Ndays_train = np.array([1 ,5, 10, 20, 30, 50, 70, 85, 100, 120, 140, 160, 180, 200,
                        230, 260, 300])       # days
#ErrorL2_vec = []
#ErrorL2_max_vec = []
MAE_vec = []
MAE_STD = []

for days in Ndays_train:

    Ndays = int(days)
    NsamplesPerDay_x,NsamplesPerDay_y,x_train,y_train,x_test,y_test = network.data_arragment(
                        df_normalized,
                        Ndays,NHours_train,NMinutes_test,
                        current_time)
    
    preds,biased_preds = network.LSTM_build_train (
                        NsamplesPerDay_x,NsamplesPerDay_y,
                        x_train,x_test,y_train,
                        Nepoches,NcellsLSTM,activeFunc,LearningRate)
    
    y_test_unnorm,preds_unnorm,biased_preds_unnorm = network.data_unnormalization(df_max,df_min,y_test,preds,biased_preds)
    
    ABS_error_bias, MAE_bias, MAE_bias_STD = network.error_calculation(
                y_test_unnorm,preds_unnorm,biased_preds_unnorm)

    MAE_vec.append(MAE_bias)
    MAE_STD.append(MAE_bias_STD)
#    ErrorL2_max_vec.append(Error_L2_bias_max)
    

MAE_vec = np.array(MAE_vec)
MAE_STD = np.array(MAE_STD)

vis.vis_multi_error_prediction_days (
             Ndays_train,NHours_train,
             NMinutes_test,current_time,
             Nepoches,NcellsLSTM,activeFunc,LearningRate,
             MAE_vec,MAE_STD)

#%%
# =============================================================================
# Number of train Hours
# =============================================================================
# training parameters:
NMinutes_test = 30      # minutes
Ndays_train = 120        # days
# network paramters:
Nepoches = 90
NcellsLSTM = 90
activeFunc = 'tanh'
LearningRate = 0.005

NHours_train = np.array([0.1 ,0.5, 0.7, 1, 1.3, 1.7, 2, 2.5, 3])       # hours
#ErrorL2_vec = []
#ErrorL2_max_vec = []
MAE_vec = []
MAE_STD = []

for hours in NHours_train:

    Nhours = float(hours)
    NsamplesPerDay_x,NsamplesPerDay_y,x_train,y_train,x_test,y_test = network.data_arragment(
                        df_normalized,
                        Ndays_train,Nhours,NMinutes_test,
                        current_time)
    
    preds,biased_preds = network.LSTM_build_train (
                        NsamplesPerDay_x,NsamplesPerDay_y,
                        x_train,x_test,y_train,
                        Nepoches,NcellsLSTM,activeFunc,LearningRate)
    
    y_test_unnorm,preds_unnorm,biased_preds_unnorm = network.data_unnormalization(df_max,df_min,y_test,preds,biased_preds)
    
    ABS_error_bias, MAE_bias, MAE_bias_STD = network.error_calculation(
                y_test_unnorm,preds_unnorm,biased_preds_unnorm)

    MAE_vec.append(MAE_bias)
    MAE_STD.append(MAE_bias_STD)
    
  
MAE_vec = np.array(MAE_vec)
MAE_STD = np.array(MAE_STD)

vis.vis_multi_error_train_hours (
             Ndays_train,NHours_train,
             NMinutes_test,current_time,
             Nepoches,NcellsLSTM,activeFunc,LearningRate,
             MAE_vec,MAE_STD)

#%%
# =============================================================================
# Prediction Time
# =============================================================================
# training parameters:
NHours_train = 1        # hours
Ndays_train = 120        # days
# network paramters:
Nepoches = 100
NcellsLSTM = 90
activeFunc = 'tanh'
LearningRate = 0.005

NMinutes_test = np.array([5 ,10, 20, 30, 40, 50, 60, 70, 80, 90 , 100, 120])       # minutes
#ErrorL2_vec = []
#ErrorL2_max_vec = []
MAE_vec = []
MAE_STD = []

for minutes in NMinutes_test:

    Nminutes = int(minutes)
    NsamplesPerDay_x,NsamplesPerDay_y,x_train,y_train,x_test,y_test = network.data_arragment(
                        df_normalized,
                        Ndays_train,NHours_train,Nminutes,
                        current_time)
    
    preds,biased_preds = network.LSTM_build_train (
                        NsamplesPerDay_x,NsamplesPerDay_y,
                        x_train,x_test,y_train,
                        Nepoches,NcellsLSTM,activeFunc,LearningRate)
    
    y_test_unnorm,preds_unnorm,biased_preds_unnorm = network.data_unnormalization(df_max,df_min,y_test,preds,biased_preds)
    
    ABS_error_bias, MAE_bias, MAE_bias_STD = network.error_calculation(
                y_test_unnorm,preds_unnorm,biased_preds_unnorm)

    MAE_vec.append(MAE_bias)
    MAE_STD.append(MAE_bias_STD)
    

MAE_vec = np.array(MAE_vec)
MAE_STD = np.array(MAE_STD)

vis.vis_multi_error_pred_time (
             Ndays_train,NHours_train,
             NMinutes_test,current_time,
             Nepoches,NcellsLSTM,activeFunc,LearningRate,
             MAE_vec,MAE_STD)


#%%
# =============================================================================
# Number of LSTM Cells
# =============================================================================
# training parameters:
NHours_train = 1        # hours
NMinutes_test = 30      # minutes
Ndays_train = 120        # days
# network paramters:
Nepoches = 100
activeFunc = 'tanh'
LearningRate = 0.005

NcellsLSTM = np.array([5 ,10, 20, 40, 60, 90, 110, 120, 140, 170, 200])       # cells
#ErrorL2_vec = []
#ErrorL2_max_vec = []
MAE_vec = []
MAE_STD = []

for cells in NcellsLSTM:

    Ncells = int(cells)
    NsamplesPerDay_x,NsamplesPerDay_y,x_train,y_train,x_test,y_test = network.data_arragment(
                        df_normalized,
                        Ndays_train,NHours_train,NMinutes_test,
                        current_time)
    
    preds,biased_preds = network.LSTM_build_train (
                        NsamplesPerDay_x,NsamplesPerDay_y,
                        x_train,x_test,y_train,
                        Nepoches,Ncells,activeFunc,LearningRate)
    
    y_test_unnorm,preds_unnorm,biased_preds_unnorm = network.data_unnormalization(df_max,df_min,y_test,preds,biased_preds)
    
    ABS_error_bias, MAE_bias, MAE_bias_STD = network.error_calculation(
                y_test_unnorm,preds_unnorm,biased_preds_unnorm)

    MAE_vec.append(MAE_bias)
    MAE_STD.append(MAE_bias_STD)
    

MAE_vec = np.array(MAE_vec)
MAE_STD = np.array(MAE_STD)

vis.vis_multi_error_LSTM_cells (
             Ndays_train,NHours_train,
             NMinutes_test,current_time,
             Nepoches,NcellsLSTM,activeFunc,LearningRate,
             MAE_vec,MAE_STD)


#%%
# =============================================================================
# Number of epoches
# =============================================================================

# training parameters:
NHours_train = 1        # hours
NMinutes_test = 30      # minutes
Ndays_train = 120        # days
# network paramters:
NcellsLSTM = 90
activeFunc = 'tanh'
LearningRate = 0.005

#Nepoches = np.array([5, 10, 20, 40, 90, 120, 150, 200, 300, 500, 700, 1000, 1500, 2000, 3000])
Nepoches = np.array([1, 50, 100, 200, 400, 700, 1000, 1500, 2000, 3000, 4000, 5000, 7000])       # epoches
#ErrorL2_vec = []
#ErrorL2_max_vec = []
MAE_vec = []
MAE_STD = []

for epoch in Nepoches:

    Nepoch = int(epoch)
    NsamplesPerDay_x,NsamplesPerDay_y,x_train,y_train,x_test,y_test = network.data_arragment(
                        df_normalized,
                        Ndays_train,NHours_train,NMinutes_test,
                        current_time)
    
    preds,biased_preds = network.LSTM_build_train (
                        NsamplesPerDay_x,NsamplesPerDay_y,
                        x_train,x_test,y_train,
                        Nepoch,NcellsLSTM,activeFunc,LearningRate)
    
    y_test_unnorm,preds_unnorm,biased_preds_unnorm = network.data_unnormalization(df_max,df_min,y_test,preds,biased_preds)
    
    ABS_error_bias, MAE_bias, MAE_bias_STD = network.error_calculation(
                y_test_unnorm,preds_unnorm,biased_preds_unnorm)

    MAE_vec.append(MAE_bias)
    MAE_STD.append(MAE_bias_STD)
    

MAE_vec = np.array(MAE_vec)
MAE_STD = np.array(MAE_STD)

vis.vis_multi_error_Nepoches (
             Ndays_train,NHours_train,
             NMinutes_test,current_time,
             Nepoches,NcellsLSTM,activeFunc,LearningRate,
             MAE_vec,MAE_STD)

#%%
# =============================================================================
# Learning Rate
# =============================================================================
# training parameters:
NHours_train = 1        # hours
NMinutes_test = 30      # minutes
Ndays_train = 120        # days
# network paramters:
Nepoches = 200
NcellsLSTM = 90
activeFunc = 'tanh'

LearningRate = np.array([0.001, 0.002, 0.005, 0.01, 0.02, 0.04, 0.07, 0.1, 0.2, 0.4, 0.6, 0.9])       # rate
#ErrorL2_vec = []
#ErrorL2_max_vec = []
MAE_vec = []
MAE_STD = []

for rate in LearningRate:

    Nrate = int(rate)
    NsamplesPerDay_x,NsamplesPerDay_y,x_train,y_train,x_test,y_test = network.data_arragment(
                        df_normalized,
                        Ndays_train,NHours_train,NMinutes_test,
                        current_time)
    
    preds,biased_preds = network.LSTM_build_train (
                        NsamplesPerDay_x,NsamplesPerDay_y,
                        x_train,x_test,y_train,
                        Nepoches,NcellsLSTM,activeFunc,Nrate)
    
    y_test_unnorm,preds_unnorm,biased_preds_unnorm = network.data_unnormalization(df_max,df_min,y_test,preds,biased_preds)
    
    ABS_error_bias, MAE_bias, MAE_bias_STD = network.error_calculation(
                y_test_unnorm,preds_unnorm,biased_preds_unnorm)

    MAE_vec.append(MAE_bias)
    MAE_STD.append(MAE_bias_STD)
    

MAE_vec = np.array(MAE_vec)
MAE_STD = np.array(MAE_STD)

vis.vis_multi_error_LearningRate (
             Ndays_train,NHours_train,
             NMinutes_test,current_time,
             Nepoches,NcellsLSTM,activeFunc,LearningRate,
             MAE_vec,MAE_STD)



#%%
# =============================================================================
# Activation Function - check which functions?????
# =============================================================================
# training parameters:
NHours_train = 1        # hours
NMinutes_test = 30      # minutes
Ndays_train = 120        # days
# network paramters:
Nepoches = 200
NcellsLSTM = 90
LearningRate = 0.005

activeFunc = ['sigmoid', 'tanh','relu']       # function
#ErrorL2_vec = []
#ErrorL2_max_vec = []
MAE_vec = []
MAE_STD = []

for func in activeFunc:

    NsamplesPerDay_x,NsamplesPerDay_y,x_train,y_train,x_test,y_test = network.data_arragment(
                        df_normalized,
                        Ndays_train,NHours_train,NMinutes_test,
                        current_time)
    
    preds,biased_preds = network.LSTM_build_train (
                        NsamplesPerDay_x,NsamplesPerDay_y,
                        x_train,x_test,y_train,
                        Nepoches,NcellsLSTM,func,LearningRate)
    
    y_test_unnorm,preds_unnorm,biased_preds_unnorm = network.data_unnormalization(df_max,df_min,y_test,preds,biased_preds)
    
    ABS_error_bias, MAE_bias, MAE_bias_STD = network.error_calculation(
                y_test_unnorm,preds_unnorm,biased_preds_unnorm)

    MAE_vec.append(MAE_bias)
    MAE_STD.append(MAE_bias_STD)
    
MAE_vec = np.array(MAE_vec)
MAE_STD = np.array(MAE_STD)

vis.vis_multi_error_Active_Func (
             Ndays_train,NHours_train,
             NMinutes_test,current_time,
             Nepoches,NcellsLSTM,activeFunc,LearningRate,
             MAE_vec,MAE_STD)

#%%




