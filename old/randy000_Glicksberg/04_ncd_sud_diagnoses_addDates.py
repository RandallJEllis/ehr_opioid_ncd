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

# #Subset diagnoses related to neurocognitive disease
# senile_vasc_dementia = icd9[(icd9.ONTOLOGY_ID>=290) & ((icd9.ONTOLOGY_ID<=290.43))]
# senile_vasc_dementia_icd10 = icd10[icd10.ONTOLOGY_ID.isin(['F01', 'F01.5', 'F01.50','F01.51', 'F02', 'F02.8', 
#                                                           'F02.80', 'F02.81', 'F03', 'F03.9', 'F03.90', 'F03.91'])]

# intell_disab = icd9[(icd9.ONTOLOGY_ID>=317) & ((icd9.ONTOLOGY_ID<=319))]
# intell_disab_icd10 = icd10[icd10.ONTOLOGY_ID.isin(['F70', 'F71', 'F72', 'F73'])]

# alz_mci = icd9[(icd9.ONTOLOGY_ID>=331) & ((icd9.ONTOLOGY_ID<332))]
# alz_mci_icd10 = icd10[icd10.ONTOLOGY_ID.isin(['G30.0', 'G30.1', 'G30.8', 'G30.9', 'G31.01', 'G31.09', 'G31.1', 
#                                               'G31.2', 'G31.82', 'G31.83', 'G31.84', 'G31.85', 'G31.89', 'G31.9',])]

# print(senile_vasc_dementia.shape, senile_vasc_dementia_icd10.shape, intell_disab.shape, intell_disab_icd10.shape,
#      alz_mci.shape, alz_mci_icd10.shape)
# ncd_icd = pd.concat([senile_vasc_dementia, senile_vasc_dementia_icd10, intell_disab, intell_disab_icd10,
#                      alz_mci, alz_mci_icd10]).reset_index(drop=True)

# #add dates to ncd_icd

# ncd_dates = []

# for i,mrn in enumerate(ncd_icd.MRN):
#     if i%1_000_000==0:
#         print(i)
#     person_row = person_mtx[[person_mrn[mrn]]][0][0] #person.iloc[person_mrn[mrn]]
#     long_month_name = person_row[0,3]
#     datetime_object = datetime.strptime(long_month_name, "%B")
#     month_number = datetime_object.month
#     s = f'{person_row[0,4]}/{month_number}/1'
#     birth_date = datetime.strptime(s, "%Y/%m/%d")

#     modified_date = birth_date + timedelta(days=int(ncd_icd.AGE_IN_DAYS.values[i]))
#     ncd_dates.append(modified_date)

# ncd_icd['date'] = ncd_dates

# ncd_icd.to_csv('ncd_icd.csv')


#SUD
#substance dependence or abuse including alcohol
sud_icd = icd9[(icd9.ONTOLOGY_ID>=303) & ((icd9.ONTOLOGY_ID<306))].reset_index(drop=True)
sud_icd_icd10 = icd10[icd10.ONTOLOGY_ID.isin(['F10.10',
 'F10.120',
 'F10.121',
 'F10.129',
 'F10.14',
 'F10.180',
 'F10.19',
 'F10.20',
 'F10.21',
 'F10.220',
 'F10.221',
 'F10.229',
 'F10.230',
 'F10.231',
 'F10.232',
 'F10.239',
 'F10.24',
 'F10.27',
 'F10.29',
 'F10.920',
 'F10.929',
 'F10.94',
 'F10.950',
 'F10.951',
 'F10.959',
 'F10.96',
 'F10.980',
 'F10.982',
 'F10.99',
 'F11.10',
 'F11.120',
 'F11.129',
 'F11.20',
 'F11.21',
 'F11.220',
 'F11.221',
 'F11.222',
 'F11.229',
 'F11.23',
 'F11.24',
 'F11.29',
 'F11.90',
 'F11.921',
 'F11.94',
 'F11.988',
 'F11.99',
 'F12.10',
 'F12.120',
 'F12.122',
 'F12.129',
 'F12.151',
 'F12.180',
 'F12.19',
 'F12.20',
 'F12.21',
 'F12.220',
 'F12.221',
 'F12.229',
 'F12.251',
 'F12.259',
 'F12.90',
 'F12.920',
 'F12.921',
 'F12.922',
 'F12.929',
 'F12.950',
 'F12.951',
 'F12.959',
 'F12.980',
 'F12.988',
 'F12.99',
 'F13.10',
 'F13.20',
 'F13.21',
 'F13.229',
 'F13.231',
 'F13.239',
 'F13.90',
 'F13.920',
 'F13.929',
 'F13.94',
 'F13.951',
 'F14.10',
 'F14.120',
 'F14.122',
 'F14.129',
 'F14.14',
 'F14.151',
 'F14.180',
 'F14.20',
 'F14.21',
 'F14.220',
 'F14.221',
 'F14.222',
 'F14.229',
 'F14.23',
 'F14.24',
 'F14.250',
 'F14.90',
 'F14.920',
 'F14.921',
 'F14.929',
 'F14.94',
 'F14.951',
 'F14.980',
 'F14.988',
 'F15.10',
 'F15.120',
 'F15.188',
 'F15.20',
 'F15.21',
 'F15.23',
 'F15.90',
 'F15.929',
 'F15.950',
 'F15.980',
 'F15.99',
 'F16.10',
 'F16.188',
 'F16.20',
 'F16.283',
 'F17.200',
 'F17.201',
 'F17.203',
 'F17.209',
 'F17.210',
 'F17.211',
 'F17.213',
 'F17.218',
 'F17.219',
 'F17.290',
 'F18.10',
 'F18.20',
 'F19.10',
 'F19.120',
 'F19.121',
 'F19.122',
 'F19.129',
 'F19.20',
 'F19.21',
 'F19.230',
 'F19.231',
 'F19.239',
 'F19.90',
 'F19.921',
 'F19.929',
 'F19.930',
 'F19.931',
 'F19.939',
 'F19.94',
 'F19.950',
 'F19.951',
 'F19.959',
 'F19.96',
 'F19.97',
 'F19.980',
 'F19.982',
 'F19.988',
 'F19.99'])]

print(sud_icd.shape, sud_icd_icd10.shape)
sud_icd = pd.concat([sud_icd, sud_icd_icd10])
print(sud_icd.shape)

#add dates to sud_icd

sud_dates = []

for i,mrn in enumerate(sud_icd.MRN):
    if i%1_000_000==0:
        print(i)
    person_row = person_mtx[[person_mrn[mrn]]][0][0] #person.iloc[person_mrn[mrn]]
    long_month_name = person_row[0,3]
    datetime_object = datetime.strptime(long_month_name, "%B")
    month_number = datetime_object.month
    s = f'{person_row[0,4]}/{month_number}/1'
    birth_date = datetime.strptime(s, "%Y/%m/%d")

    modified_date = birth_date + timedelta(days=int(sud_icd.AGE_IN_DAYS.values[i]))
    sud_dates.append(modified_date)

sud_icd['date'] = sud_dates

sud_icd.to_csv('tidy_data/sud_icd.csv')