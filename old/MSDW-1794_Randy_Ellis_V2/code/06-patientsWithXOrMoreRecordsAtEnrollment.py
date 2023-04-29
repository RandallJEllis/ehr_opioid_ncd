import pandas as pd
import os
import numpy as np
from collections import Counter
import pickle
from datetime import datetime
import sys

'''
The purpose of this script is to identify the MRNs with num_recs or more records during the enrollment period to avoid including patients who are not steady patients. 
'''

begin_enroll_years = sys.argv[1] #list
n = len(begin_enroll_years)
begin_enroll_years = begin_enroll_years[1:n-1]
begin_enroll_years = begin_enroll_years.split(',')

end_enroll_years = sys.argv[2] #list
n = len(end_enroll_years)
end_enroll_years = end_enroll_years[1:n-1]
end_enroll_years = end_enroll_years.split(',')

#by_encounters = sys.argv[3]

# if by_encounters = 1:
#     df = pd.read_parquet(f'tidy_data/person-encounter.parquet')
#     for begin_enroll, end_enroll in zip(begin_enroll_years, end_enroll_years):
#         df_sub = df[(df['date'] >= pd.to_datetime(f'{begin_enroll}-01-01')) & 
#                         (df['date'] < pd.to_datetime(f'{end_enroll}-01-01'))]
        
#         if "MRN" in df.columns:
#             mrncol = "MRN"
#         elif "MEDICAL_RECORD_NUMBER" in df.columns:
#             mrncol = "MEDICAL_RECORD_NUMBER"
            
#         df_sub.loc[df_sub.ENCOUNTER_TYPE.isin(['Ambulatory Surgery', 'Outpatient In A Bed',
#                                  'Preadmit Ambulatory Surgery/Testing', 'Preadmit Testing'])] = 'Outpatient'
#         df_sub.loc[df_sub.ENCOUNTER_TYPE=='Preadmit Inpatient'] = 'Inpatient'
#         df_sub = df_sub[df_sub.ENCOUNTER_TYPE.isin(['Inpatient','Outpatient','Emergency'])]
        
#         mrn_idx = df_sub.groupby([mrncol]).groups

#         counter = Counter(df_sub.mrncol)
#         finalmrns = [mrn for mrn, occurrences in counter.items() if occurrences >= num_recs]
#         pickle.dump(finalmrns, open(f'tidy_data/mrns_{num_recs}OrMoreEncounters_{begin_enroll}_{end_enroll}.p', 'wb'))
        
# else:
for begin_enroll, end_enroll in zip(begin_enroll_years, end_enroll_years):
    mrns_enroll = []
    for file in ['person-vitals.parquet', 'person-encounter.parquet', 'icd10.parquet', 'person-lab.parquet',
                'medication.parquet', 'icd9.parquet']:

        df = pd.read_parquet(f'tidy_data/{file}')

        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        with open("log.txt", "a") as myfile:
            myfile.write(f"Imported {file}, Current Time = {current_time}")    

        if "MRN" in df.columns:
            mrncol = "MRN"
        elif "MEDICAL_RECORD_NUMBER" in df.columns:
            mrncol = "MEDICAL_RECORD_NUMBER"

        df_sub = df[(df['date'] >= pd.to_datetime(f'{begin_enroll}-01-01')) & 
                    (df['date'] < pd.to_datetime(f'{end_enroll}-01-01'))]

        mrns_enroll.extend(df_sub[mrncol])

        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        with open("log.txt", "a") as myfile:
            myfile.write(f"{file} - Added {begin_enroll}-{end_enroll} MRNs, Current Time = {current_time}")

    cc = Counter(mrns_enroll)
    #finalmrns = [mrn for mrn, occurrences in counter.items() if occurrences >= num_recs]
    d = {}
    for key, value in cc.items():
        d[key] = value
    df_recs = pd.DataFrame({'mrn':d.keys(), 'records':d.values()})
    df_recs.to_parquet(f'tidy_data/mrns_numRecords_{begin_enroll}_{end_enroll}.parquet')
    #pickle.dump(finalmrns, open(f'tidy_data/mrns_{num_recs}OrMoreRecords_{begin_enroll}_{end_enroll}.p', 'wb'))

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    with open("log.txt", "a") as myfile:
        myfile.write(f"Saved {begin_enroll}-{end_enroll} MRNs, Current Time = {current_time}") 

    
