# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 11:53:46 2018

@author: sopherb
"""
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 11:39:36 2017

@author: nadavv


# -*- coding: utf-8 -*-
"""
#Spyder Editor

import numpy as np

from sklearn import preprocessing
import datetime

from keras.layers import LSTM, Dense, Input, TimeDistributed #, Masking
from keras.models import Model
from keras import optimizers
#from keras.optimizers import RMSprop
from keras.preprocessing import sequence


def data_normalization(df):
    #    df = dm.df_withoutBD
    df_max = df['Glucose (mg/dL)'].max()
    df_min = df['Glucose (mg/dL)'].min()
    
    # Create x, where x the 'scores' column's values as floats
    x = df[['Glucose (mg/dL)']].values.astype(float)
    
    # Create a minimum and maximum processor object
    min_max_scaler = preprocessing.MinMaxScaler()
    
    # Create an object to transform the data to fit minmax processor
    x_scaled = min_max_scaler.fit_transform(x)
    
    # Run the normalizer on the dataframe
    df_normalized = df.copy()
    df_normalized['Glucose (mg/dL)'] = x_scaled
    
    return(df_max,df_min,df_normalized)

# =============================================================================
# Define date and time for processing.
# =============================================================================
#NHours_train = 2
#NMinutes_test = 30
#Ndays_train = 30
#current_time = datetime.datetime(2017,6,20,15,0)        
#class datetime.datetime(year, month, day[, hour[, minute[, second[, microsecond[, tzinfo]]]]])



def data_arragment (df_normalized,Ndays_train,NHours_train,
                    NMinutes_test,current_time):

    NsamplesPerDay_x = NHours_train*60
    NsamplesPerDay_y = NMinutes_test
    
#    start_hour = datetime.datetime(2017,6,20,15)
#    end_hour = start_hour + datetime.timedelta(hours=NHours_train,minutes=-1)
    
    start_hour = current_time + datetime.timedelta(minutes=-(NsamplesPerDay_x-1))
    end_hour = current_time
    
    test_start = end_hour + datetime.timedelta(minutes=1)
    test_end = test_start + datetime.timedelta(minutes=NMinutes_test-1)
    
#    start_date = start_hour.date()
#    end_date = start_date + datetime.timedelta(days=Ndays_train)
    
    start_date = current_time.date() + datetime.timedelta(days=-Ndays_train)
    end_date = current_time.date() + datetime.timedelta(days=-1)
    Ndays_test = 1 # Don't change it!
    test_date = current_time.date()
    
    # =============================================================================
    # =============================================================================
    
    df_relevant_time_x = df_normalized.between_time('{:%H:%M}'.format(start_hour),'{:%H:%M}'.format(end_hour))
    df_relevant_time_y = df_normalized.between_time('{:%H:%M}'.format(test_start),'{:%H:%M}'.format(test_end))
    
    df_relevant_time_x = df_relevant_time_x['Glucose (mg/dL)']
    df_relevant_time_y = df_relevant_time_y['Glucose (mg/dL)']
    
    x_train_ = df_relevant_time_x.loc[start_date:test_date]
    y_train_ = df_relevant_time_y.loc[start_date:test_date]
    
    x_test_ =  df_relevant_time_x.loc[test_date:test_date+datetime.timedelta(days=1)]
    y_test_ =  df_relevant_time_y.loc[test_date:test_date+datetime.timedelta(days=1)]
    
    it_train = np.arange(Ndays_train)
    it_test = np.arange(Ndays_test)
    
    # =============================================================================
    # Amout of samples per day (between the relevant hours)
    # =============================================================================
    
    x_train_ = [x_train_[i*NsamplesPerDay_x:i*NsamplesPerDay_x+NsamplesPerDay_x] for i in it_train]
    y_train_ = [y_train_[i*NsamplesPerDay_y:i*NsamplesPerDay_y+NsamplesPerDay_y] for i in it_train]
    
    x_test_ = [x_test_[i*NsamplesPerDay_x:i*NsamplesPerDay_x+NsamplesPerDay_x] for i in it_test]
    y_test_ = [y_test_[i*NsamplesPerDay_y:i*NsamplesPerDay_y+NsamplesPerDay_y] for i in it_test]
    
    maxlen_x = NsamplesPerDay_x
    maxlen_y = NsamplesPerDay_y
    
    x_train = sequence.pad_sequences(x_train_, dtype = 'float', maxlen=maxlen_x, padding = 'post', truncating = 'post')
    y_train = sequence.pad_sequences(y_train_, dtype = 'float', maxlen=maxlen_y, padding = 'post', truncating = 'post')
    
    x_test = sequence.pad_sequences(x_test_, dtype = 'float', maxlen=maxlen_x, padding = 'post', truncating = 'post')
    y_test = sequence.pad_sequences(y_test_, dtype = 'float', maxlen=maxlen_y, padding = 'post', truncating = 'post')
    
    # =============================================================================
    # Shape for the data: ( #of days, #of variables == 1, #of samples)
    # =============================================================================
    
    x_train = np.reshape(x_train, ( Ndays_train, 1,NsamplesPerDay_x))
    y_train = np.reshape(y_train, ( Ndays_train, 1, NsamplesPerDay_y))
    
    x_test = np.reshape(x_test, ( Ndays_test, 1, NsamplesPerDay_x))
    y_test = np.reshape(y_test, ( Ndays_test, 1, NsamplesPerDay_y))
    
    print ("x_train shape: %s | y_train shape: %s " % ((str(x_train.shape)),str(y_train.shape)))
    print ("x_test shape: %s | y_test shape: %s " % ((str(x_test.shape)),str(y_test.shape)))
    
    return (NsamplesPerDay_x,NsamplesPerDay_y,x_train,y_train,x_test,y_test)

# =============================================================================
# LSTM Network creation
# =============================================================================

def LSTM_build_train (NsamplesPerDay_x,NsamplesPerDay_y,x_train,x_test,y_train,Nepoches,NcellsLSTM,activeFunc,LearningRate):
    x = Input(shape = (None,NsamplesPerDay_x),name='input')
    #mask = Masking(0, name='input_masked')(x)
    
    #lstm_kwargs = {'dropout_W' : 0.25, 'dropout_U':0.1, 'return_sequences': True}
    lstm1 = LSTM(NcellsLSTM, activation='sigmoid',name='lstm1',dropout = 0.25 ,return_sequences=True)(x)
     
    output = TimeDistributed(Dense(NsamplesPerDay_y, activation = activeFunc),  name='output')(lstm1) 
    model = Model(input=x,outputs=output)
     
    optimizer = optimizers.RMSprop(lr=LearningRate)
#    optimizer = optimizers.Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
#    optimizer = optimizers.optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
#    optimizer = optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)

    model.compile(optimizer=optimizer, loss='mean_squared_error',metrics=['accuracy'])
     
    model.summary()
    
    # LSTM Network training
    model.fit(x_train, y_train, batch_size=None, epochs=Nepoches,verbose=1)

    # LSTM Network prediction
    preds = model.predict(x_test)
    preds.shape
    
    first_pred_bias = x_test[0,0,-1] - preds[0,0,0]
    biased_preds = preds + first_pred_bias
    return(preds,biased_preds)

# =============================================================================
# unnormalized the prediction series and test series for evaluation
# =============================================================================
def data_unnormalization(df_max,df_min,y_test,preds,biased_preds):
    preds_unnorm = preds * (df_max - df_min) + df_min
    y_test_unnorm = y_test * (df_max - df_min) + df_min
    biased_preds_unnorm = biased_preds * (df_max - df_min) + df_min
    
    return(y_test_unnorm,preds_unnorm,biased_preds_unnorm)


# =============================================================================
# Error calculation per value for 1 series
# =============================================================================
def error_1_calculation (y_test_unnorm,preds_unnorm,biased_preds_unnorm):
    
    Error_L1 = abs(y_test_unnorm[0,0,:]-preds_unnorm[0,0,:])
    Error_L1_bias = abs(y_test_unnorm[0,0,:]-biased_preds_unnorm[0,0,:])
    
    Error_L2 = (y_test_unnorm[0,0,:]-preds_unnorm[0,0,:])**2
    Error_L2_bias = (y_test_unnorm[0,0,:]-biased_preds_unnorm[0,0,:])**2
    
    
    return Error_L1,Error_L1_bias,Error_L2,Error_L2_bias

# =============================================================================
# Error calculation per series
# =============================================================================
def error_multi_calculation (y_test_unnorm,preds_unnorm,biased_preds_unnorm):
    
    L1error = abs(y_test_unnorm[0,0,:]-preds_unnorm[0,0,:])
    L1error_bias = abs(abs(y_test_unnorm[0,0,:]-biased_preds_unnorm[0,0,:]))
    
    Error_L1 = sum(L1error)
    Error_L1_bias = sum(L1error_bias)
    Error_L1_bias_min = min(L1error_bias)
    Error_L1_bias_max = max(L1error_bias)
    
    L2error = (y_test_unnorm[0,0,:]-preds_unnorm[0,0,:])**2
    L2error_bias = (y_test_unnorm[0,0,:]-biased_preds_unnorm[0,0,:])**2

    Error_L2 = sum(L2error)
    Error_L2_bias = sum(L2error_bias)
    Error_L2_bias_min=min(Error_L2_bias)
    Error_L2_bias_max=max(Error_L2_bias)
    
    return Error_L1_bias_max, Error_L2_bias_max

# =============================================================================
# =============================================================================








