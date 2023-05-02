# Convert to parquet for faster loading as we work with the data
import pandas as pd
from datetime import datetime
import numpy as np

### patients
patient = pd.read_csv('../raw_data/data_participant.tsv', sep='\t')
# Referring to DNANexus, renaming columns to make sense
# 34 = year of birth; 52 = month of birth; 31 = sex (0 is female; 1 is male); 42040 = GP clinical event records; 41259 = Records in HES inpatient main dataset
patient = patient.rename(columns={'34-0.0':'yob', '52-0.0':'mob', '31-0.0':'sex', '42040-0.0':'gp_records', '41259-0.0':'inpatient_records'})
patient['sex'] = patient['sex'].replace({0: 'Female', 1: 'Male'})
patient.to_parquet('../tidy_data/patient.parquet')

### prescriptions
med = pd.read_csv('../raw_data/data_gp_scripts.tsv', sep='\t') 

# remove prescriptions before 1990 (~34k out of 58M total)
med.issue_date = pd.to_datetime(med.issue_date)
med = med[med.issue_date.dt.year>=1990]

# create subsets based on whether drug name, read2 or BNF are NaN
# only remove a prescription if all three are NaN
missing_drug = med[med.drug_name.isna()]
remove = missing_drug[(missing_drug.bnf_code.isna()) & (missing_drug.read_2.isna())]
med = med.drop(labels=remove.index) # 5 prescriptions removed
med = med.reset_index(drop=True)

med.drug_name = med.drug_name.str.upper()

# create a column for patient age at time of prescription
med['AGE_AT_ENCOUNTER'] = np.nan
unique_eids = list(set(med.eid))

# patient groups
pt_gp = patient.groupby('eid').groups
# med patient groups
med_pt_gp = med.groupby('eid').groups

# iterate over unique patients and calculate age at encounter by subtracting date of birth from date of prescription
for i,eid in enumerate(unique_eids):
    birthday = datetime(patient.iloc[pt_gp[eid]].yob, patient.iloc[pt_gp[eid]].mob, 1)
    med.AGE_AT_ENCOUNTER[med_pt_gp[eid]] = (med.issue_date[med_pt_gp[eid]] - birthday).dt.days
#convert to years 
med['AGE_AT_ENCOUNTER'] = med['AGE_AT_ENCOUNTER'] / 365.25
med.to_parquet('../tidy_data/med.parquet')

# Read2-to-BNF table
read_v2_drugs_bnf = pd.read_excel('../all_lkps_maps_v3.xlsx', sheet_name='read_v2_drugs_bnf')
read_v2_drugs_bnf.to_parquet('../tidy_data/read_v2_drugs_bnf.parquet')

# BNF lookup
bnf_lkp = pd.read_excel('../all_lkps_maps_v3.xlsx', sheet_name='bnf_lkp')
bnf_lkp.BNF_Presentation_Code = bnf_lkp.BNF_Presentation_Code.astype(str)
bnf_lkp.to_parquet('../tidy_data/bnf_lkp.parquet')

# Read2 lookup
read_v2_drugs_lkp = pd.read_excel('../all_lkps_maps_v3.xlsx', sheet_name='read_v2_drugs_lkp')
read_v2_drugs_lkp.to_parquet('../tidy_data/read_v2_drugs_lkp.parquet')

### read2 - ICD9 and ICD10
read2_icd9 = pd.read_excel('../all_lkps_maps_v3.xlsx', sheet_name='read_v2_icd9')
read2_icd9.to_parquet('../tidy_data/readv2_icd9.parquet')
read2_icd10 = pd.read_excel('../all_lkps_maps_v3.xlsx', sheet_name='read_v2_icd10')
read2_icd10.to_parquet('../tidy_data/readv2_icd10.parquet')

### read3 - ICD9 and ICD10
read3_icd9 = pd.read_excel('../all_lkps_maps_v3.xlsx', sheet_name='read_ctv3_icd9')
read3_icd9.to_parquet('../tidy_data/readv3_icd9.parquet')
read3_icd10 = pd.read_excel('../all_lkps_maps_v3.xlsx', sheet_name='read_ctv3_icd10')
read3_icd10.to_parquet('../tidy_data/readv3_icd10.parquet')

### read2 and read3 lookups
read2_lkp = pd.read_excel('../all_lkps_maps_v3.xlsx', sheet_name='read_v2_lkp')
read2_lkp.to_parquet('../tidy_data/read2_lkp.parquet')
read3_lkp = pd.read_excel('../all_lkps_maps_v3.xlsx', sheet_name='read_ctv3_lkp')
read3_lkp.to_parquet('../tidy_data/read3_lkp.parquet')

### diagnoses
# Re-exported data_gp_clinical on May 1st, 2023 to include Read CTV3 codes (86M rows that had NaN in the read_2 column)
gp_clin = pd.read_csv('../raw_data/data_gp_clinical_may12023.tsv', sep='\t')
# There are ZERO rows with both read_2 and read_3 codes and ZERO rows with neither code
gp_clin = gp_clin.drop(columns=['eid(gp_clinical - eid)'])
gp_clin = gp_clin.rename(columns={'eid(participant - eid)':'eid'})
# remove diagnoses before 1990 (~1.5M of ~123M total)
gp_clin.event_dt = pd.to_datetime(gp_clin.event_dt)
gp_clin = gp_clin[gp_clin.event_dt.dt.year>=1990]
gp_clin = gp_clin.reset_index(drop=True)
gp_clin.to_parquet('../tidy_data/data_gp_clinical.parquet')

### hospital admissions and diagnoses
hesin = pd.read_csv('../raw_data/data_hesin.tsv', sep='\t')
# remove diagnoses before 1990 (~57k of ~3.9M total)
hesin.epistart = pd.to_datetime(hesin.epistart)
hesin = hesin[hesin.epistart.dt.year>=1990]
hesin = hesin.reset_index(drop=True)
hesin.to_parquet('../tidy_data/data_hesin.parquet')

hesin_diag = pd.read_csv('../raw_data/data_hesin_diag.tsv', sep='\t')
# create column as tuple of patient ID and instance index
hesin['eid_ins_index'] = list(zip(hesin.eid, hesin.ins_index))
hesin_diag['eid_ins_index'] = list(zip(hesin_diag.eid, hesin_diag.ins_index))
# set overlap of tuples
overlap_eid_ins_index = set(hesin.eid_ins_index).intersection(set(hesin_diag.eid_ins_index))
# remove diagnoses with no corresponding tuple in episode dataframe (cannot know dates of diagnoses otherwise)
# removes ~80k diagnoses out of ~15.2M total
hesin_diag_overlap = hesin_diag[hesin_diag.eid_ins_index.isin(overlap_eid_ins_index)]
hesin_diag_overlap = hesin_diag_overlap.merge(hesin.loc[:,['eid_ins_index','epistart','epiend','epidur']], how='left', on='eid_ins_index')
hesin_diag_overlap = hesin_diag_overlap.reset_index(drop=True)
hesin_diag_overlap = hesin_diag_overlap.drop(columns=['eid_ins_index'])
hesin_diag_overlap.to_parquet('../tidy_data/data_hesin_diag.parquet')

### icd9 and icd10
icd9_lkp = pd.read_excel('../all_lkps_maps_v3.xlsx', sheet_name='icd9_lkp')
icd9_lkp.rename(columns={'DESCRIPTION_ICD9':'DESCRIPTION'}, inplace=True)
icd9_lkp.to_parquet('../tidy_data/icd9_lkp.parquet')

icd10_lkp = pd.read_excel('../all_lkps_maps_v3.xlsx', sheet_name='icd10_lkp')
icd10_lkp = icd10_lkp.iloc[:,:5]
# add rows for icd10 codes ending in X where the X is removed
for i,code in enumerate(icd10_lkp.ALT_CODE):
    if code and code[-1]=='X':
        icd10_lkp.loc[icd10_lkp.index.max() + 1] = icd10_lkp.loc[i]  # Insert a new row below the first row with the same values
        icd10_lkp.loc[icd10_lkp.index.max(), 'ALT_CODE'] = code[:-1]
icd10_lkp.to_parquet('../tidy_data/icd10_lkp.parquet')

# Create encounters file based on hospitalization diagnoses, GP clinical events, and prescriptions
enc = pd.concat([hesin_diag_overlap[['eid','epistart']].rename(columns={'epistart':'event_dt'}),
                    gp_clin[['eid','event_dt']],
                    med[['eid','issue_date']].rename(columns={'issue_date':'event_dt'})
                    ])
enc.to_parquet('../tidy_data/encounters.parquet')    