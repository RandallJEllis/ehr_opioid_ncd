import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pickle
import statsmodels.api as sm
import seaborn as sns

person = pd.read_csv('data/ehr-csv/person/person-detail.csv')
#Remove patients born in or before 1900
person = person[person.YOB>1900].reset_index(drop=True)

person.reset_index(drop=True, inplace=True)
print('loaded data')
Subset all patients with one or two+ rows, so that we don't have to iterate through patients with one row
mrn_rowCounts = person.MEDICAL_RECORD_NUMBER.value_counts() 
mrn_oneRow = mrn_rowCounts[mrn_rowCounts<2].index
mrn_twoOrMoreRows = mrn_rowCounts[mrn_rowCounts>1].index

demographic_twoOrMore = person[person['MEDICAL_RECORD_NUMBER'].isin(mrn_twoOrMoreRows)]
demographic_twoOrMore.reset_index(drop=True, inplace=True)
print('about to make idx')
idx = demographic_twoOrMore.groupby('MEDICAL_RECORD_NUMBER').groups
print('made idx')
mrn_values = list(idx.values())
demographic_twoOrMore = np.matrix(demographic_twoOrMore)

demo_individuals = []
print('about to start loop')
count=0
for i in range(len(mrn_values)):
    if count%50000==0:
        print(count)

    mrn_demo = demographic_twoOrMore[list(mrn_values[i]), :]

    mrn_demo_clean = mrn_demo[np.where((mrn_demo[:,2]!='Unknown') & (mrn_demo[:,1]!='Unknown'))[0],:]

    if mrn_demo_clean.shape[0] > 0:
        demo_individuals.append(mrn_demo_clean[0, :])
    else:
        demo_individuals.append(mrn_demo[0, :])

    count+=1

demo_individuals_arr = np.concatenate(demo_individuals)
demo_twoOrMore_df = pd.DataFrame(demo_individuals_arr, columns = person.columns)
print(demo_twoOrMore_df.shape)

mrn_oneRow = list(mrn_oneRow)
person_oneRow = person[person.MEDICAL_RECORD_NUMBER.isin(mrn_oneRow)]
person_clean = pd.concat([person_oneRow, demo_twoOrMore_df])
person_clean['GENDER'] = person_clean['GENDER'].replace(['MSDW_UNKNOWN', 'NOT AVAILABLE', 'Indeterminant',
                                                         'Unknown'], 'other')
person_clean['RACE'] = person_clean['RACE'].replace(['Unk', 'MSDW_UNKNOWN', None, 'NOT AVAILABLE',
                                                     'Other', 'Unknown'], 'other')
person_clean['DECEASED'] = person_clean['DECEASED'].replace([None], 'NOT AVAILABLE')
person_clean.to_parquet('tidy_data/person_cleaned.parquet', index=False)
