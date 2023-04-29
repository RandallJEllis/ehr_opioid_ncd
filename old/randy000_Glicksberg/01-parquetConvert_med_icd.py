#Used preprocess_medication_icd9.R to create medication.csv and ICD9.csv
#Goals: 
#Change datatypes of ICD codes
#Filter out patients with very old ages
#Convert these to parquet

import pandas as pd

icd9 = pd.read_csv('ICD9.csv')
icd9["CODE"] = icd9["CODE"].astype(str)
icd9["ONTOLOGY_ID"] = icd9["ONTOLOGY_ID"].astype(str)
icd9 = icd9[icd9.AGE_IN_DAYS<44000]
icd9.to_parquet('tidy_data/icd9.parquet')

icd10 = pd.read_csv('icd10.csv')
icd10["CODE"] = icd10["CODE"].astype(str)
icd10["ONTOLOGY_ID"] = icd10["ONTOLOGY_ID"].astype(str)
icd10 = icd10[icd10.AGE_IN_DAYS<44000]
icd10.to_parquet('tidy_data/icd10.parquet')

med = pd.read_csv('medication.csv')
med = med[med.AGE_IN_DAYS<44000]
med.to_parquet('tidy_data/medication.parquet')
