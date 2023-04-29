import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pickle
import statsmodels.api as sm
import seaborn as sns

person = pd.read_csv('data/ehr-csv/person/person-detail.csv')
person = person[person.YOB>1900].reset_index(drop=True)
person_mrn = pickle.load(open('tidy_data/person_mrn_groups.p', 'rb'))
person_mtx = np.matrix(person)

for filename in [#'data/ehr-csv/encounter/person-encounter.csv',
		 #'tidy_data/medication.parquet', 
		 #'tidy_data/icd9.parquet',
                 #'tidy_data/icd10.csv',
		#'data/ehr-csv/lab/person-lab.csv', 
		'data/ehr-csv/vitals/person-vitals.csv']:
    if ".csv" in filename:
        if 'person-lab' in filename:
            df = pd.read_csv(filename, names=['MRN', 'CODE', 'AGE_IN_DAYS', 'TIMESTAMP', 'LAB', 'TYPE'])
        elif 'person-vitals' in filename:
            df = pd.read_csv(filename, names=['MRN', 'CODE', 'AGE_IN_DAYS', 'TIMESTAMP', 'VITAL', 'UNIT'])
            df = df.drop(columns=['VITAL'])
        else:
            df = pd.read_csv(filename)
    elif ".parquet" in filename:
        df = pd.read_parquet(filename)
        
    
    with open("log.txt", "a") as myfile:
        myfile.write(f"imported {filename}")
        
    if "MRN" in df.columns:
        mrncol = "MRN"
    elif "MEDICAL_RECORD_NUMBER" in df.columns:
        mrncol = "MEDICAL_RECORD_NUMBER"
        
    if "BEGIN_DATE_AGE_IN_DAYS" in df.columns:
        datecol = "BEGIN_DATE_AGE_IN_DAYS"
    elif "AGE_IN_DAYS" in df.columns:
        datecol = "AGE_IN_DAYS"
        
    df = df[df[mrncol].isin(list(person_mrn.keys()))]
    
    dates = []

    #Get dates 
    for i,mrn in enumerate(df[mrncol]):
        if i%1_000_000==0:
            with open("log.txt", "a") as myfile:
                myfile.write(str(i))

        #Get patient birthday
        person_row = person_mtx[[person_mrn[mrn]]][0][0] #person.iloc[person_mrn[mrn]]
        long_month_name = person_row[0,3]
        datetime_object = datetime.strptime(long_month_name, "%B")
        month_number = datetime_object.month
        s = f'{person_row[0,4]}/{month_number}/1'
        birth_date = datetime.strptime(s, "%Y/%m/%d")

        #Calculate date of opioid prescription
        modified_date = birth_date + timedelta(days=int(df[datecol].values[i]))
        dates.append(modified_date.date())

    df['date'] = dates
    slashidx = filename.rfind('/')
    dotidx = filename.rfind('.')
    newname = filename[slashidx+1:dotidx]
    df.to_parquet(f'tidy_data/{newname}.parquet', index=False)
    del df
                 
                 
                 
         