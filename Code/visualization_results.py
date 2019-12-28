# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 20:14:35 2018

@author: opher
"""
import os
from matplotlib import pyplot as plt
import numpy as np

results_directory = os.path.join('..','Results')

# =============================================================================
# show predictions and test series
# =============================================================================
def vis_1_prediction (y_test_unnorm,preds_unnorm,biased_preds_unnorm,
                 NsamplesPerDay_y,
                 Ndays_train,NHours_train,
                 NMinutes_test,current_time,
                 Nepoches,NcellsLSTM,activeFunc,LearningRate,
                 ABS_error_bias,MAE_bias,MAE_bias_STD):
    
    date_prediction = '{:%d-%m-%Y}'.format(current_time)
    time_prediction = '{:%H-%M}'.format(current_time)
    
    # Results graph   
    plt.figure(figsize=(10,7))
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
    plt.ylabel('Suger Level (mg/deciliter)')
    plt.xlabel('time (minutes)')
    plt.legend()
    plt.grid()
    
    
    file_name = os.path.join(results_directory,
                             'Probabilistic Prediction vs. Real Values ' 
                             + date_prediction + ' ' + time_prediction + '.png')
    ii = 0
    while os.path.isfile(file_name):
        ii+=1
        file_name = os.path.join(results_directory,
                             'Probabilistic Prediction vs. Real Values ' + 
                             + date_prediction + ' ' + time_prediction + ' #' + str(ii) + '.png')
    
    plt.savefig(file_name, bbox_inches='tight')
    
    # Error graph   
    plt.figure(figsize=(10,7))
    
#    plt.subplot(121)
#    plt.plot(np.arange(NsamplesPerDay_y),L1error,marker='o',label='Absulot Error')
    plt.plot(np.arange(NsamplesPerDay_y),ABS_error_bias,marker='o',label='Absolut Error Bias')
    plt.title('Absolut Error' 
                 '\n' + date_prediction + ' ' + time_prediction +
                 '\n training days: ' + str(Ndays_train) + 
                 ' , hours before curret time: ' + str(NHours_train) +
                 ' , minutes for prediction: ' + str(NMinutes_test) + 
                 '\n epoches: ' + str(Nepoches) + ' , LSTM cells: ' + str(NcellsLSTM) + 
                 ' , Activation: ' + activeFunc + ', Learn Rate: ' + str(LearningRate) +
                 '\n MAE = ' + '{0:.2f}'.format(MAE_bias) + ' , STD = ' + '{0:.2f}'.format(MAE_bias_STD) )
    plt.ylabel('Absolut Error')
    plt.xlabel('time(minutes)')
    plt.legend()
    plt.grid()
    
#    plt.subplot(122)
#    plt.plot(np.arange(NsamplesPerDay_y),L2error,marker='o',label='Square Error')
#    plt.plot(np.arange(NsamplesPerDay_y),L2error_bias,marker='o',label='Square Error Bias')
#    plt.title('Error L2')
#    plt.ylabel('Error')
#    plt.xlabel('time (minutes)')
#    plt.legend()
#    plt.grid()
    

    file_name = os.path.join(results_directory,
                             'Errors - Probabilistic Prediction vs. Real Values '
                             + date_prediction + ' ' + time_prediction + '.png')
    ii = 0
    while os.path.isfile(file_name):
        ii+=1
        file_name = os.path.join(results_directory,
                             'Errors - Probabilistic Prediction vs. Real Values ' + 
                             date_prediction + ' ' + time_prediction + ' #' + str(ii) + '.png')
  
    plt.savefig(file_name, bbox_inches='tight')
    
    
def vis_multi_error_prediction_days (
                 Ndays_train,NHours_train,
                 NMinutes_test,current_time,
                 Nepoches,NcellsLSTM,activeFunc,LearningRate,
                 MAE_vec,MAE_STD):
    
    date_prediction = '{:%d-%m-%Y}'.format(current_time)
    time_prediction = '{:%H-%M}'.format(current_time)
    labels = np.array([str(x) for x in Ndays_train])
#    labels = np.array2string(Ndays_train)
    Ndays_train_list=np.ndarray.tolist(Ndays_train)
  
    plt.figure(figsize=(18,7))
    plt.subplot(211)
    plt.plot(Ndays_train,MAE_vec,marker='o',label='MAE')
#    plt.title('MAE vs. Number of train days')
    plt.ylabel('MAE')
    plt.xlabel('Number of training days')
    plt.legend()
    plt.grid()
    plt.xticks(Ndays_train_list,labels)
    
    plt.subplot(212)
    plt.plot(Ndays_train,MAE_STD,marker='o',label='MAE STD')
#    plt.title('MAE STD vs. Number of train days')
    plt.ylabel('STD')
    plt.xlabel('Number of training days')
    plt.legend()
    plt.grid()
    plt.xticks(Ndays_train_list,labels)
    
    plt.suptitle('MAE and STD vs. Number of train days' 
              '\n' + date_prediction + ' ' + time_prediction +
              '\n hours before curret time: ' + str(NHours_train) +
              ' , minutes for prediction: ' + str(NMinutes_test) + 
              '\n epoches: ' + str(Nepoches) + ' , LSTM cells: ' + str(NcellsLSTM) + 
              ' , Activation: ' + activeFunc + ', Learn Rate: ' + str(LearningRate))

    file_name = os.path.join(results_directory,
                             'Error vs Number of train days ' 
                             + date_prediction + ' ' + time_prediction +  '.png')

    ii = 0
    while os.path.isfile(file_name):
        ii+=1
        file_name = os.path.join(results_directory,
                             'Error vs Number of train days ' + 
                             date_prediction + ' ' + time_prediction + ' #' + str(ii) + '.png')
                             
    plt.savefig(file_name, bbox_inches='tight')
    
    
def vis_multi_error_train_hours (
                 Ndays_train,NHours_train,
                 NMinutes_test,current_time,
                 Nepoches,NcellsLSTM,activeFunc,LearningRate,
                 MAE_vec,MAE_STD):
    
    date_prediction = '{:%d-%m-%Y}'.format(current_time)
    time_prediction = '{:%H-%M}'.format(current_time)
    labels = np.array([str(x) for x in NHours_train])
#    labels = np.array2string(Ndays_train)
    NHours_train_list=np.ndarray.tolist(NHours_train)
    
    plt.figure(figsize=(18,7))
    plt.subplot(211)
    plt.plot(NHours_train,MAE_vec,marker='o',label='MAE')
#    plt.title('MAE vs. Number of train days')
    plt.ylabel('MAE')
    plt.xlabel('Number of train hours')
    plt.legend()
    plt.grid()
    plt.xticks(NHours_train_list,labels)
    
    plt.subplot(212)
    plt.plot(NHours_train,MAE_STD,marker='o',label='MAE STD')
#    plt.title('MAE STD vs. Number of train days')
    plt.ylabel('STD')
    plt.xlabel('Number of train hours')
    plt.legend()
    plt.grid()
    plt.xticks(NHours_train_list,labels)
    
    plt.suptitle('MAE and STD vs. Number of train hours' 
              '\n' + date_prediction + ' ' + time_prediction +
              '\n Number train days: ' + str(Ndays_train) +
              ' , minutes for prediction: ' + str(NMinutes_test) + 
              '\n epoches: ' + str(Nepoches) + ' , LSTM cells: ' + str(NcellsLSTM) + 
              ' , Activation: ' + activeFunc + ', Learn Rate: ' + str(LearningRate) )
      
    file_name = os.path.join(results_directory,
                             'Error vs Number of train hours '
                             + date_prediction + ' ' + time_prediction +'.png')

    ii = 0
    while os.path.isfile(file_name):
        ii+=1
        file_name = os.path.join(results_directory,
                             'Error vs Number of train hours ' + 
                             date_prediction + ' ' + time_prediction + ' #' + str(ii) + '.png')
                             
    plt.savefig(file_name, bbox_inches='tight')

    
def vis_multi_error_pred_time (
                 Ndays_train,NHours_train,
                 NMinutes_test,current_time,
                 Nepoches,NcellsLSTM,activeFunc,LearningRate,
                 MAE_vec,MAE_STD):
    
    date_prediction = '{:%d-%m-%Y}'.format(current_time)
    time_prediction = '{:%H-%M}'.format(current_time)
    labels = np.array([str(x) for x in NMinutes_test])
#    labels = np.array2string(Ndays_train)
    NMinutes_test_list=np.ndarray.tolist(NMinutes_test)
    
    plt.figure(figsize=(18,7))
    plt.subplot(211)
    plt.plot(NMinutes_test,MAE_vec,marker='o',label='MAE')
#    plt.title('MAE vs. Number of train days')
    plt.ylabel('MAE')
    plt.xlabel('Number of Prediction Minutes')
    plt.legend()
    plt.grid()
    plt.xticks(NMinutes_test_list,labels)
    
    plt.subplot(212)
    plt.plot(NMinutes_test,MAE_STD,marker='o',label='MAE STD')
#    plt.title('MAE STD vs. Number of train days')
    plt.ylabel('STD')
    plt.xlabel('Number of Prediction Minutes')
    plt.legend()
    plt.grid()
    plt.xticks(NMinutes_test_list,labels)
    
    plt.suptitle('MAE and STD vs. Number of Prediction Minutes' 
              '\n' + date_prediction + ' ' + time_prediction +
              '\n Number train days: ' + str(Ndays_train) +
              ' , hours before curret time: ' + str(NHours_train) + 
              '\n epoches: ' + str(Nepoches) + ' , LSTM cells: ' + str(NcellsLSTM) + 
              ' , Activation: ' + activeFunc + ', Learn Rate: ' + str(LearningRate) )
      
    file_name = os.path.join(results_directory,
                             'Error vs Number of Prediction Minutes '
                             + date_prediction + ' ' + time_prediction +'.png')
    ii = 0
    while os.path.isfile(file_name):
        ii+=1
        file_name = os.path.join(results_directory,
                             'Error vs Number of Prediction Minutes ' + 
                             date_prediction + ' ' + time_prediction + ' #' + str(ii) + '.png')
        
    plt.savefig(file_name, bbox_inches='tight')    
    
def vis_multi_error_LSTM_cells (
                 Ndays_train,NHours_train,
                 NMinutes_test,current_time,
                 Nepoches,NcellsLSTM,activeFunc,LearningRate,
                 MAE_vec,MAE_STD):
    
    date_prediction = '{:%d-%m-%Y}'.format(current_time)
    time_prediction = '{:%H-%M}'.format(current_time)
    labels = np.array([str(x) for x in NcellsLSTM])
#    labels = np.array2string(Ndays_train)
    NcellsLSTM_list=np.ndarray.tolist(NcellsLSTM)
    
    plt.figure(figsize=(18,7))
    plt.subplot(211)
    plt.plot(NcellsLSTM,MAE_vec,marker='o',label='MAE')
#    plt.title('MAE vs. Number of train days')
    plt.ylabel('MAE')
    plt.xlabel('Number of LSTM Cells')
    plt.legend()
    plt.grid()
    plt.xticks(NcellsLSTM_list,labels)
    
    plt.subplot(212)
    plt.plot(NcellsLSTM,MAE_STD,marker='o',label='MAE STD')
#    plt.title('MAE STD vs. Number of train days')
    plt.ylabel('STD')
    plt.xlabel('Number of LSTM Cells')
    plt.legend()
    plt.grid()
    plt.xticks(NcellsLSTM_list,labels)
    
    plt.suptitle('MAE and STD vs. Number of LSTM Cells' 
              '\n' + date_prediction + ' ' + time_prediction +
              '\n Number train days: ' + str(Ndays_train) +
              ' , hours before curret time: ' + str(NHours_train) + 
              ' , minutes for prediction: ' + str(NMinutes_test) +
              '\n epoches: ' + str(Nepoches) +  
              ' , Activation: ' + activeFunc + ', Learn Rate: ' + str(LearningRate) )
         
    file_name = os.path.join(results_directory,
                             'Error vs Number of LSTM Cells '
                             + date_prediction + ' ' + time_prediction + '.png')
    ii = 0
    while os.path.isfile(file_name):
        ii+=1
        file_name = os.path.join(results_directory,
                             'Error vs Number of LSTM Cells ' + 
                             date_prediction + ' ' + time_prediction + ' #' + str(ii) + '.png')
            
    plt.savefig(file_name, bbox_inches='tight')        
    
def vis_multi_error_LearningRate (
                 Ndays_train,NHours_train,
                 NMinutes_test,current_time,
                 Nepoches,NcellsLSTM,activeFunc,LearningRate,
                 MAE_vec,MAE_STD):
    
    date_prediction = '{:%d-%m-%Y}'.format(current_time)
    time_prediction = '{:%H-%M}'.format(current_time)
    labels = np.array([str(x) for x in LearningRate])
#    labels = np.array2string(Ndays_train)
    LearningRate_list=np.ndarray.tolist(LearningRate)
    
    plt.figure(figsize=(24,7))
    plt.subplot(211)
    plt.plot(LearningRate,MAE_vec,marker='o',label='MAE')
#    plt.title('MAE vs. Number of train days')
    plt.ylabel('MAE')
    plt.xlabel('Learning Rate')
    plt.legend()
    plt.grid()
    plt.xticks(LearningRate_list,labels)
    
    plt.subplot(212)
    plt.plot(LearningRate,MAE_STD,marker='o',label='MAE STD')
#    plt.title('MAE STD vs. Number of train days')
    plt.ylabel('STD')
    plt.xlabel('Learning Rate')
    plt.legend()
    plt.grid()
    plt.xticks(LearningRate_list,labels)
    
    plt.suptitle('MAE and STD vs. Learning Rate' 
              '\n' + date_prediction + ' ' + time_prediction +
              '\n Number train days: ' + str(Ndays_train) +
              ' , hours before curret time: ' + str(NHours_train) + 
              ' , minutes for prediction: ' + str(NMinutes_test) +
              '\n epoches: ' + str(Nepoches) + ' , LSTM cells: ' + str(NcellsLSTM) +  
              ' , Activation: ' + activeFunc)
      
    file_name = os.path.join(results_directory,
                             'Error vs Learning Rate ' 
                             + date_prediction + ' ' + time_prediction + '.png')
    ii = 0
    while os.path.isfile(file_name):
        ii+=1
        file_name = os.path.join(results_directory,
                             'Error vs Learning Rate ' + 
                             date_prediction + ' ' + time_prediction + ' #' + str(ii) + '.png')    
            
    plt.savefig(file_name, bbox_inches='tight')  

    
def vis_multi_error_Nepoches (
                 Ndays_train,NHours_train,
                 NMinutes_test,current_time,
                 Nepoches,NcellsLSTM,activeFunc,LearningRate,
                 MAE_vec,MAE_STD):
    
    date_prediction = '{:%d-%m-%Y}'.format(current_time)
    time_prediction = '{:%H-%M}'.format(current_time)
    labels = np.array([str(x) for x in Nepoches])
#    labels = np.array2string(Ndays_train)
    Nepoches_list=np.ndarray.tolist(Nepoches)
    
    plt.figure(figsize=(22,7))
    plt.subplot(211)
    plt.plot(Nepoches,MAE_vec,marker='o',label='MAE')
#    plt.title('MAE vs. Number of train days')
    plt.ylabel('MAE')
    plt.xlabel('Train Epoches')
    plt.legend()
    plt.grid()
    plt.xticks(Nepoches_list,labels)
    
    plt.subplot(212)
    plt.plot(Nepoches,MAE_STD,marker='o',label='MAE STD')
#    plt.title('MAE STD vs. Number of train days')
    plt.ylabel('STD')
    plt.xlabel('Train Epoches')
    plt.legend()
    plt.grid()
    plt.xticks(Nepoches_list,labels)
    
    plt.suptitle('MAE and STD vs. Train Epoches' 
              '\n' + date_prediction + ' ' + time_prediction +
              '\n Number train days: ' + str(Ndays_train) +
              ' , hours before curret time: ' + str(NHours_train) + 
              ' , minutes for prediction: ' + str(NMinutes_test) +
              '\n LSTM cells: ' + str(NcellsLSTM) +  
              ' , Activation: ' + activeFunc + ', Learn Rate: ' + str(LearningRate))
      
    file_name = os.path.join(results_directory,
                             'Error vs Number of Train Epoches ' 
                             + date_prediction + ' ' + time_prediction + '.png')
    ii = 0
    while os.path.isfile(file_name):
        ii+=1
        file_name = os.path.join(results_directory,
                             'Error vs Number of Train Epoches ' + 
                             date_prediction + ' ' + time_prediction + ' #' + str(ii) + '.png')
        
    plt.savefig(file_name, bbox_inches='tight') 
    
    
def vis_multi_error_Active_Func (
                 Ndays_train,NHours_train,
                 NMinutes_test,current_time,
                 Nepoches,NcellsLSTM,activeFunc,LearningRate,
                 MAE_vec,MAE_STD):
    
    date_prediction = '{:%d-%m-%Y}'.format(current_time)
    time_prediction = '{:%H-%M}'.format(current_time)
    labels = activeFunc
#    labels = np.array([str(x) for x in LearningRate])
#    labels = np.array2string(Ndays_train)
#    LearningRate_list=np.ndarray.tolist(LearningRate)
    plot_vec = range(len(activeFunc))
    
    plt.figure(figsize=(18,7))
    plt.subplot(211)
    plt.plot(plot_vec,MAE_vec,marker='o',label='MAE')
#    plt.title('MAE vs. Number of train days')
    plt.ylabel('MAE')
    plt.xlabel('Activation Function')
    plt.legend()
    plt.grid()
    plt.xticks(plot_vec,labels)
    
    plt.subplot(212)
    plt.plot(plot_vec,MAE_STD,marker='o',label='MAE STD')
#    plt.title('MAE STD vs. Number of train days')
    plt.ylabel('STD')
    plt.xlabel('Activation Function')
    plt.legend()
    plt.grid()
    plt.xticks(plot_vec,labels)
    
    plt.suptitle('MAE and STD vs. Activation Function' 
              '\n' + date_prediction + ' ' + time_prediction +
              '\n Number train days: ' + str(Ndays_train) +
              ' , hours before curret time: ' + str(NHours_train) + 
              ' , minutes for prediction: ' + str(NMinutes_test) +
              '\n epoches: ' + str(Nepoches) + ' , LSTM cells: ' + str(NcellsLSTM) + 
              ', Learn Rate: ' + str(LearningRate))
            
    file_name = os.path.join(results_directory,
                             'Error vs Activation Function '
                             + date_prediction + ' ' + time_prediction + '.png')
    ii = 0
    while os.path.isfile(file_name):
        ii+=1
        file_name = os.path.join(results_directory,
                             'Error vs Activation Function ' + 
                             + date_prediction + ' ' + time_prediction + ' #' + str(ii) + '.png')
        
    plt.savefig(file_name, bbox_inches='tight')     
    
    
    
    
    
    
    
    
    
    
    
    