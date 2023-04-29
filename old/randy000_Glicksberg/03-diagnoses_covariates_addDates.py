import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pickle
import statsmodels.api as sm
import seaborn as sns

icd9 = pd.read_parquet('tidy_data/icd9.parquet')
#convert ICD code column to floats, turning the ones starting with letters to NaNs
icd9['ONTOLOGY_ID'] = pd.to_numeric(icd9['ONTOLOGY_ID'], errors='coerce')

icd10 = pd.read_parquet('tidy_data/icd10.parquet')

#import demographics
person = pd.read_csv('data/ehr-csv/person/person-detail.csv')
#Remove patients born in or before 1900
person = person[person.YOB>1900].reset_index(drop=True)

#dict with row indices of all unique MRNs
person_mrn = pickle.load(open('tidy_data/person_mrn_groups.p', 'rb'))
person_mtx = np.matrix(person)

hiv_icd9 = ['42', '79.53', 'V08']
hiv_icd10 = ['B20', 'B97.35', 'Z21']
sickle_icd9 = ['282.41', '282.42', '282.5', '282.6', '282.60', '282.62', '282.63', '282.64', '282.68', '282.69']
sickle_icd10 = ['D57', 'D57.0', 'D57.00', 'D57.01', 'D57.02', 'D57.03', 'D57.09', 'D57.1', 'D57.2', 'D57.20',
               'D57.21', 'D57.211', 'D57.212', 'D57.213', 'D57.218', 'D57.219', 'D57.3']
hepc_icd9 = ['70.41', '70.44', '70.51', '70.54', '70.7', '70.70', '70.71']
hepc_icd10 = ['B18.2', 'B17.11', 'B17.10', 'B19.20', 'B19.21']
#TODO: combine 03-diagnoses_covariates and 04_ncd_sud_diagnoses into one script 
# ncd_icd9 = []
# ncd_icd10 = []
# sud_icd9 = []
# sud_icd10 = []
 #all unique ICD codes in icd9.parquet between 303 <= x < 304 and 305 <= x < 305.1
aud_icd9 = ['303.02,' '303.03', '303.9', '303.01', '303.0', '303.91', '303.92', '303.93', '305.0', '305.02', '305.03', '305.01']
aud_icd10 = ['F10.10','F10.120','F10.121','F10.129','F10.14', 'F10.180', 'F10.19',
                                              'F10.20', 'F10.21', 'F10.220', 'F10.221', 'F10.229', 'F10.230',
                                              'F10.231', 'F10.232', 'F10.239', 'F10.24', 'F10.27', 'F10.29', 
                                              'F10.920','F10.929', 'F10.94', 'F10.950', 'F10.951', 'F10.959',
                                              'F10.96', 'F10.980', 'F10.982', 'F10.99']

#Get dates of diagnoses

for codes_icd9, codes_icd10, name in zip([#hiv_icd9, sickle_icd9, hepc_icd9,
                                         aud_icd9], 
                                         [#hiv_icd10, sickle_icd10, hepc_icd10,
                                         aud_icd10],
                                         [#'hiv', 'sickle', 'hepc',
                                          'aud']):
    df1 = icd9[icd9.CODE.isin(codes_icd9)]
    df2 = icd10[icd10.CODE.isin(codes_icd10)]
    df = pd.concat([df1, df2])
    print(df1.shape, df2.shape, df.shape)
    dx_date = []

    for i,mrn in enumerate(df.MRN):
        if i%1_000_000==0:
            print(i)

        #Get patient birthday
        person_row = person_mtx[[person_mrn[mrn]]][0][0] #person.iloc[person_mrn[mrn]]
        long_month_name = person_row[0,3]
        datetime_object = datetime.strptime(long_month_name, "%B")
        month_number = datetime_object.month
        s = f'{person_row[0,4]}/{month_number}/1'
        birth_date = datetime.strptime(s, "%Y/%m/%d")

        #Calculate date of dx
        modified_date = birth_date + timedelta(days=int(df.AGE_IN_DAYS.values[i]))
        dx_date.append(modified_date)

    

    df['date'] = dx_date

    df.to_csv(f'tidy_data/{name}_icd.csv', index=None)
