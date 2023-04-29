import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import date2num
import datetime
from collections import Counter

'''Preprocess the raw data in the following steps:
1. Remove records before before 2006 based on an email from Jacob Weiser (jacob.weiser@mssm.edu), Thursday October 6th, 2022, 11:00am: 

>We do have a little over 1,000 records in the database from the 1990’s, I assume we got some of these from legacy systems, I also saw a few records dated about 1900 which are most likely just error records..
In terms of from when the data is well populated, 2006-2007 is when we started using epic here at Sinai -at least some departments- so disregarding data prior to these years makes sense as they will be sparsely populated.

2. Convert date columns to datetime format
3. Save files as parquet files
'''

#MEDICATIONS
med = pd.read_csv('../raw_data/Medications.txt', sep='|', encoding='cp1252', low_memory=False)
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
diagnoses = pd.read_csv('../raw_data/Diagnosis.txt', sep='|', encoding='cp1252', low_memory=False)
print(diagnoses.shape)
#Remove 'Medical History' entries as these are not new diagnoses (email on 11/23/2022 with Priyal/Praveen)
diagnoses = diagnoses[diagnoses.DIAGNOSIS_TYPE!='Medical History']
print(diagnoses.shape)
#Convert DIAGNOSIS_DATE to datetime data type
diagnoses.DIAGNOSIS_DATE = pd.to_datetime(diagnoses.DIAGNOSIS_DATE)
diagnoses = diagnoses[(diagnoses.DIAGNOSIS_DATE>='2006-1-1') & (diagnoses.DIAGNOSIS_DATE<='2040-1-1')]
print(diagnoses.shape)
#Convert ENCOUNTER_DATE to datetime data type
diagnoses.ENCOUNTER_DATE = pd.to_datetime(diagnoses.ENCOUNTER_DATE)
diagnoses = diagnoses[(diagnoses.ENCOUNTER_DATE>='2006-1-1') & (diagnoses.ENCOUNTER_DATE<='2040-1-1')]
print(diagnoses.shape)
diagnoses.to_parquet('../tidy_data/Diagnosis.parquet')


#PATIENTS
patients = pd.read_csv('../raw_data/Patient.txt', sep='|', encoding='cp1252', low_memory=False)
print(patients.shape)
#Remove names
patients = patients.drop(columns=['PATIENT_NAME'])
#Subset relevant columns
patients = patients.loc[:,['MRN', 'AGE', 'DOB', 'RACE', 'SEX']]
#Convert DOB to datetime data type
patients["DOB"] = pd.to_datetime(patients["DOB"], errors='coerce')
#Remove patients born before January 1st, 1900
patients = patients[patients.DOB > '1900-01-01']
#Remove patients with NAN age
patients = patients[~patients.AGE.isnull()]

'''
Remove patients with less than 5 encounters. We will later check if patients have 5+ encounters during each 
particular enrollment period, but removing patients with <5 encounters now will make things run faster.
'''
enc = pd.read_csv('../raw_data/Encounters.txt', sep='|', encoding='cp1252', low_memory=False)
occurrences_mrn = Counter(enc.MRN).most_common()

#Remove patients with <5 encounters
mrns_to_remove = []
for entry in occurrences_mrn:
    if entry[1]<5:
        mrns_to_remove.append(entry[0])

#467,756 patients removed
patients = patients[~patients.MRN.isin(mrns_to_remove)]
patients.to_parquet('../tidy_data/Patient.parquet')
enc = enc[~enc.MRN.isin(mrns_to_remove)]
enc['ENCOUNTER_DATE'] = pd.to_datetime(enc['ENCOUNTER_DATE'], errors='coerce')
enc.to_parquet('../tidy_data/Encounters.parquet')