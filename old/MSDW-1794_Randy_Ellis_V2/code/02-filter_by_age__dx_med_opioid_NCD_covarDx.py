import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import date2num
import datetime

'''
1. Filter patient list, meds, and diagnoses by age 45, since this is our minimum age for including patients showing NCD
2. Export specific files for all opioid prescriptions (for enrollment), NCD diagnoses/prescriptions (for exclusion and follow-up),
   SUD diagnoses (covariate), AUD diagnoses (covariate), HIV/HepC/Sickle-cell diagnoses (covars)
'''

#filter meds, diagnoses, and persons by age 45 or older
med = pd.read_parquet('../tidy_data/Medications.parquet')
med = med[med.AGE_AT_ENCOUNTER>=45]
med.to_parquet('../tidy_data/Medications_over45.parquet')

dx = pd.read_parquet('../tidy_data/Diagnosis.parquet')
dx = dx[dx.AGE_AT_ENCOUNTER>=45]
dx.to_parquet('../tidy_data/Diagnosis_over45.parquet')

# person = pd.read_parquet('../tidy_data/Patient.parquet')
# person = person[person.AGE>=45]
# person.to_parquet('../tidy_data/Patient_over45.parquet')

#NCD diagnoses
svd9 = dx[dx.CODE.str.startswith('290')]
svd10 = dx[dx.CODE.str.startswith(('F01', 'F01.5', 'F01.50','F01.51', 'F02', 'F02.8',
                                     'F02.80', 'F02.81', 'F03', 'F03.9', 'F03.90', 'F03.91'))]
intell_disab = dx[dx.CODE.str.startswith(('317','318','319'))]
intell_disab_icd10 = dx[dx.CODE.str.startswith(('F70', 'F71', 'F72', 'F73'))]

alz_mci = dx[dx.CODE.str.startswith('331')]
alz_mci_icd10 = dx[dx.CODE.str.startswith(('G30.0', 'G30.1', 'G30.8', 'G30.9', 'G31.01', 'G31.09', 'G31.1',
                                              'G31.2', 'G31.82', 'G31.83', 'G31.84', 'G31.85', 'G31.89', 'G31.9'))]

ncd_dx = pd.concat([svd9,svd10,intell_disab,intell_disab_icd10,alz_mci,alz_mci_icd10]).reset_index(drop=True)
ncd_dx.to_parquet('../tidy_data/ncd_diagnoses.parquet')

#SUD diagnoses
sud_icd9 = dx[dx.CODE.str.startswith(('303','304','305'))]
sud_icd10 = dx[dx.CODE.str.startswith(('F10','F11','F12','F13','F14','F15','F16','F17','F18','F19'))]

sud_icd = pd.concat([sud_icd9, sud_icd10]).reset_index(drop=True)
sud_icd.to_parquet('../tidy_data/sud_diagnoses.parquet')

#AUD diagnoses
aud_icd9 = dx[dx.CODE.str.startswith(('303.02,' '303.03', '303.9', '303.01', '303.0', '303.91', '303.92', '303.93', '305.0',
                                      '305.02', '305.03', '305.01'))]
aud_icd10 = dx[dx.CODE.str.startswith(('F10'))]
aud_icd = pd.concat([aud_icd9, aud_icd10]).reset_index(drop=True)
aud_icd.to_parquet('../tidy_data/aud_diagnoses.parquet')

#HIV diagnoses
hiv_icd9 = dx[dx.CODE.str.startswith(('42', '79.53', 'V08'))]
hiv_icd10 = dx[dx.CODE.str.startswith(('B20', 'B97.35', 'Z21'))]
hiv_icd = pd.concat([hiv_icd9, hiv_icd10]).reset_index(drop=True)
hiv_icd.to_parquet('../tidy_data/hiv_diagnoses.parquet')

#Sickle-cell diagnoses
sickle_icd9 = dx[dx.CODE.str.startswith(('282.41', '282.42', '282.5', '282.6', '282.60', '282.62', '282.63', '282.64',
                                         '282.68', '282.69'))]
sickle_icd10 = dx[dx.CODE.str.startswith(('D57', 'D57.0', 'D57.00', 'D57.01', 'D57.02', 'D57.03', 'D57.09', 'D57.1', 'D57.2', 
                                          'D57.20', 'D57.21', 'D57.211', 'D57.212', 'D57.213', 'D57.218', 'D57.219', 'D57.3'))]
sickle_icd = pd.concat([sickle_icd9, sickle_icd10]).reset_index(drop=True)
sickle_icd.to_parquet('../tidy_data/sickle_diagnoses.parquet')

#HepC diagnoses
hepc_icd9 = dx[dx.CODE.str.startswith(('70.41', '70.44', '70.51', '70.54', '70.7', '70.70', '70.71'))]
hepc_icd10 = dx[dx.CODE.str.startswith(('B18.2', 'B17.11', 'B17.10', 'B19.20', 'B19.21'))]
hepc_icd = pd.concat([hepc_icd9, hepc_icd10]).reset_index(drop=True)
hepc_icd.to_parquet('../tidy_data/hepc_diagnoses.parquet')

#MEDICATIONS
drug_list = pd.read_excel('../psychiatricDrugsforMSDW.xlsx')
drug_list.columns = ['WHO_code','drug','drug_class']
drug_list['drug'] = drug_list['drug'].str.upper()
drug_list['drug'] = drug_list['drug'].apply(lambda x: str(x).replace(u'\xa0', u''))
drug_list['drug_class'] = drug_list['drug_class'].str.upper()

#opioid medications
opioid_med = med[med.PHARMACEUTICAL_CLASS.str.contains('OPIOID')]
opioid_med.to_parquet('../tidy_data/opioid_med.parquet')

#ncd medications
antidementia = list(set(drug_list[drug_list.drug_class=='ANTI-DEMENTIA DRUGS'].drug))
antidementia_string = "|".join(antidementia)
antidementia = med[med.MEDICATION_NAME.str.contains(antidementia_string)] 

ncd_med = med[med.PHARMACEUTICAL_CLASS.str.contains('ALZHEIMER|CHOLINESTERASE')]
myasthenia_gravis = list(set(ncd_med.MEDICATION_GENERIC_NAME) ^ set(antidementia.MEDICATION_GENERIC_NAME))
ncd_med = ncd_med[~ncd_med.MEDICATION_GENERIC_NAME.isin(myasthenia_gravis)].reset_index(drop=True)
ncd_med.to_parquet('../tidy_data/ncd_med.parquet')
