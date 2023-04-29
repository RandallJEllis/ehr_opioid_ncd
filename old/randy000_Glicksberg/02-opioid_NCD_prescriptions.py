import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pickle
import statsmodels.api as sm
import seaborn as sns

med = pd.read_parquet('tidy_data/medication.parquet')


'''Subset medications for opioids and neurocognitive drugs
ONTOLOGY_ID shows RxNorm codes
N02A	Opioids
A06AH	Peripheral opioid receptor antagonists
N01AH	Opioid anesthetics
N02AG	Opioids in combination with antispasmodics
N02AJ	Opioids in combination with non-opioid analgesics
N02AX	Other opioids
N07BC	Drugs used in opioid dependence
N02AJ09	codeine and other non-opioid analgesics
N02AJ03	dihydrocodeine and other non-opioid analgesics
N02AJ15	tramadol and other non-opioid analgesics

Find codes for drugs at RxNorm: https://mor.nlm.nih.gov/RxClass/search?query

Did not use opioid anesthetics 
Did not use peripheral opioid agonists

#Example from the link: "acetaminophen / codeine" has an RXCUI code of 817579
'''
opioid_codes = ['817579', '352362', '135095', '1819', '1841', '710303', '3290', '22713', '23088', '4337', '3423',
               '484259', '6754', '6761', '7052', '7238', '1545902', '1806700', '7676', '7804', '7894', '7814','8001',
               '8119', '8354', '8785', '787390', '10597', '10689', '3304', '236913', '28863', '6813','7894', '480',
               '17933', '8143', '73032', '56795', '5489', '1841', '6378', '7804','2670',
               '3290','23405','3304','237005','7533',]
opi_prescrip = med[med.ONTOLOGY_ID.isin(opioid_codes)].reset_index(drop=True)

#N06D (anti-dementia drugs): https://mor.nlm.nih.gov/RxClass/search?query=ANTI-DEMENTIA%20DRUGS%7CATC1-4&searchBy=class&sourceIds=N06D&drugSources=atc1-4%7Catc%2Cepc%7Cfdaspl%2Cmeshpa%7Cmesh%2Cdisease%7Cmedrt%2Cchem%7Cfdaspl%2Cmoa%7Cfdaspl%2Cpe%7Cfdaspl%2Cpk%7Cmedrt%2Ctc%7Cfmtsme%2Cva%7Cva%2Cdispos%7Csnomedct%2Cstruct%7Csnomedct%2Cschedule%7Crxnorm
ncd_codes = ['135447', '1430990', '4637', '6719', '183379', '10318']
ncd_prescrip = med[med.ONTOLOGY_ID.isin(ncd_codes)].reset_index(drop=True)

#import demographics
person = pd.read_csv('data/ehr-csv/person/person-detail.csv')
#Remove patients born in or before 1900
person = person[person.YOB>1900].reset_index(drop=True)

#use groupby([column]).groups to get the row indices of all unique values of [column], turn into dict
# person_mrn = person.groupby('MEDICAL_RECORD_NUMBER').groups
# pickle.dump(person_mrn, open('person_mrn_groups.p', 'wb'))
person_mrn = pickle.load(open('tidy_data/person_mrn_groups.p', 'rb'))

person_mtx = np.matrix(person)

#Get dates of opioid and NCD prescriptions

opi_presc_date = []
ncd_presc_date = []

#Get dates of opioid prescriptions
for i,mrn in enumerate(opi_prescrip.MRN):
    if i%1_000_000==0:
        print(i)
        
    #Get patient birthday
    person_row = person_mtx[[person_mrn[mrn]]][0][0]
    long_month_name = person_row[0,3]
    datetime_object = datetime.strptime(long_month_name, "%B")
    month_number = datetime_object.month
    s = f'{person_row[0,4]}/{month_number}/1'
    birth_date = datetime.strptime(s, "%Y/%m/%d")
    
    #Calculate date of opioid prescription
    modified_date = birth_date + timedelta(days=int(opi_prescrip.AGE_IN_DAYS.values[i]))
    opi_presc_date.append(modified_date)

#Get dates of NCD prescriptions
for i,mrn in enumerate(ncd_prescrip.MRN):
    if i%1_000_000==0:
        print(i)
    person_row = person_mtx[[person_mrn[mrn]]][0][0]
    long_month_name = person_row[0,3]
    datetime_object = datetime.strptime(long_month_name, "%B")
    month_number = datetime_object.month
    s = f'{person_row[0,4]}/{month_number}/1'
    birth_date = datetime.strptime(s, "%Y/%m/%d")
    
    #Calculate date of NCD prescription
    modified_date = birth_date + timedelta(days=int(ncd_prescrip.AGE_IN_DAYS.values[i]))
    ncd_presc_date.append(modified_date)

opi_prescrip['date'] = opi_presc_date
ncd_prescrip['date'] = ncd_presc_date

opi_prescrip.to_parquet('tidy_data/opi_prescrip.parquet')
ncd_prescrip.to_parquet('tidy_data/ncd_prescrip.parquet')
