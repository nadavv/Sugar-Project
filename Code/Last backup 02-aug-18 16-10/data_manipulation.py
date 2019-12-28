# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 11:39:36 2017

@author: nadavv


# -*- coding: utf-8 -*-
"""
#Spyder Editor
import os
import pandas as pd
import numpy as np
from datetime import timedelta


# working_directory = 'D:\Project2\SugarProj\Libre'
# working_directory = 'C:\\Users\\Nadav\\Google Drive\\Include\\Studies\\Project\\Project2\\SugarProj\\Libre'
# working_directory = os.path.join('..','Libre')

def data_preprocessing (working_directory, time_without_data):
    firstFile = os.listdir(working_directory)[0]
    file_path = os.path.join(working_directory,firstFile)
    df = pd.read_csv(file_path, sep='\t', header=0,skiprows =1)    
    
    for file_name in os.listdir(working_directory):
        if file_name.endswith(".txt") and not file_name.startswith(firstFile):
    #            file_path = "%s\%s" % (working_directory, file_name)
                file_path = os.path.join(working_directory,file_name)
                df = df.append(pd.read_csv(file_path, sep='\t', header=0,skiprows =1))    
        else:
            continue
        
    df.drop(df.columns[5:-1], axis=1, inplace=True)
    df.drop(df.columns[0], axis=1, inplace=True)
    df.drop(df.columns[1], axis=1, inplace=True)
    df['Glucose (mg/dL)'] = df['Historic Glucose (mg/dL)'].combine_first(df['Scan Glucose (mg/dL)'])
    df.drop(df.columns[1], axis=1, inplace=True)
    df.drop(df.columns[1], axis=1, inplace=True)
    df['Time'] =pd.to_datetime(df.Time)
    df = df.sort_values(by='Time',ascending=True)
    df = df.drop_duplicates('Time')
    df['Time Difference'] = df['Time'] - df['Time'].shift(1)
    df = df[['Time', 'Time Difference','Glucose (mg/dL)']]
    df['Real Sample'] = 1
    df['Glucose (mg/dL)'] = pd.to_numeric(df['Glucose (mg/dL)'], errors='raise',downcast ='signed')
    df = df[np.isfinite(df['Glucose (mg/dL)'])] #deleting entries which are not glucose values
    
#    threshold = 3
    threshold = time_without_data
    badDataDayMask = (df['Time Difference'] > timedelta(hours=threshold))
    badDates = df.loc[badDataDayMask]['Time'].dt.date
    df['Time just date'] = df['Time'].dt.date
    df['Bad Date'] = df['Time just date'].isin(badDates)
    df.drop('Time just date', axis=1,inplace=True)
    df.drop('Time Difference', axis=1,inplace=True)
    
    df = df.set_index('Time')
    df = df.reindex(pd.date_range(start=df.index.min(),end=df.index.max(),freq='1min')) 
    df.interpolate(method='linear')
    df['Glucose (mg/dL)'].interpolate(inplace=True)
    df['Glucose (mg/dL)'] = df['Glucose (mg/dL)'].astype(np.uint16) # uint16 resolution is fine
    df['Real Sample']= df['Real Sample'].fillna(False)
    
    df['Time'] = df.index
    df['Date'] = df['Time'].dt.date
    df['Bad Date'] = df['Date'].isin(badDates)
    df.drop('Time',axis=1,inplace=True)
    
    df_withoutBD = df
    df_withoutBD = df_withoutBD.drop(df_withoutBD.index[df_withoutBD['Bad Date'] == True])
    
    return df_withoutBD










