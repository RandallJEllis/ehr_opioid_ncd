import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import date2num
import datetime

'''Preprocess the raw data in the following steps:
1. Remove records before before 2006 based on an email from Jacob Weiser (jacob.weiser@mssm.edu), Thursday October 6th, 2022, 11:00am: 

>We do have a little over 1,000 records in the database from the 1990’s, I assume we got some of these from legacy systems, I also saw a few records dated about 1900 which are most likely just error records..
In terms of from when the data is well populated, 2006-2007 is when we started using epic here at Sinai -at least some departments- so disregarding data prior to these years makes sense as they will be sparsely populated.

2. Convert date columns to datetime format
3. Save files as parquet files
'''

#MEDICATIONS
med = pd.read_csv('../raw_data/1794_Medications.txt', sep='|', encoding='cp1252', low_memory=False)
print(med.shape)
med.MEDICATION_START_DATE = pd.to_datetime(med.MEDICATION_START_DATE)
med.MEDICATION_END_DATE = pd.to_datetime(med.MEDICATION_END_DATE)
med = med[(med.MEDICATION_START_DATE>='2006-1-1') & (med.MEDICATION_START_DATE<='2040-1-1') &
          (med.MEDICATION_END_DATE>='2006-1-1') & (med.MEDICATION_END_DATE<='2040-1-1') ]
print(med.shape)
med.PHARMACEUTICAL_CLASS = med.PHARMACEUTICAL_CLASS.str.upper()
med.MEDICATION_NAME = med.MEDICATION_NAME.str.upper()
med.MEDICATION_GENERIC_NAME = med.MEDICATION_GENERIC_NAME.str.upper()
med.to_parquet('../tidy_data/Medications.parquet')


#DIAGNOSES
diagnoses = pd.read_csv('../raw_data/1794_Diagnosis.txt', sep='|', encoding='cp1252', low_memory=False)
print(diagnoses.shape)
diagnoses.DIAGNOSIS_DATE = pd.to_datetime(diagnoses.DIAGNOSIS_DATE)
diagnoses = diagnoses[(diagnoses.DIAGNOSIS_DATE>='2006-1-1') & (diagnoses.DIAGNOSIS_DATE<='2040-1-1')]
print(diagnoses.shape)
diagnoses.ENCOUNTER_DATE = pd.to_datetime(diagnoses.ENCOUNTER_DATE)
diagnoses = diagnoses[(diagnoses.ENCOUNTER_DATE>='2006-1-1') & (diagnoses.ENCOUNTER_DATE<='2040-1-1')]
print(diagnoses.shape)
diagnoses.to_parquet('../tidy_data/Diagnosis.parquet')


#PATIENTS
patients = pd.read_csv('../raw_data/1794_Patient.txt', sep='|', encoding='cp1252', low_memory=False)
print(patients.shape)
patients['YOB'] = 2022-patients['AGE']
patients = patients[patients.YOB>=1900]
print(patients.shape)
patients = patients.drop(columns=['PATIENT_NAME'])
patients = patients.loc[:,['MRN', 'AGE', 'RACE', 'SEX', 'YOB']]

new_patients = pd.read_csv('../raw_data/1794_Control2.txt', sep='|', encoding='cp1252', low_memory=False)
new_patients["ENCOUNTER_DATE"] = pd.to_datetime(new_patients["ENCOUNTER_DATE"], errors='coerce')
new_patients["DOB"] = pd.to_datetime(new_patients["DOB"], errors='coerce')
new_patients['YOB'] = list(new_patients.DOB.dt.year)
new_patients = new_patients[(new_patients["ENCOUNTER_DATE"].dt.year<2023) & 
                            (new_patients["ENCOUNTER_DATE"].dt.year>2005)]
new_patients = new_patients.loc[:,['MRN', 'AGE', 'RACE', 'SEX', 'YOB']]
new_patients = new_patients.drop_duplicates()

all_pts = pd.concat([patients,new_patients]).reset_index(drop=True)
all_pts.to_parquet('../tidy_data/Patient.parquet')