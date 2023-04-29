import pandas as pd

### Outpatient diagnoses
dx = pd.read_parquet('../tidy_data/data_gp_clinical.parquet')
# Read2-to-ICD9 mapping table
read2_icd9 = pd.read_parquet('../tidy_data/readv2_icd9.parquet')
# Read2-to-ICD10 mapping table
read2_icd10 = pd.read_parquet('../tidy_data/readv2_icd10.parquet')
# ICD9 lookup table
icd9_lkp = pd.read_parquet('../tidy_data/icd9_lkp.parquet')
# ICD10 lookup table
icd10_lkp = pd.read_parquet('../tidy_data/icd10_lkp.parquet')

print(dx.shape) # (122389889, 3)

# Merge to Read2_ICD9 mapping table
icd9 = dx.merge(read2_icd9.loc[:,['read_code','icd9_code']], how='left', left_on='read_2', right_on='read_code')
print(icd9.shape) # (122389889, 3)
icd9 = icd9[~icd9.read_code.isna()]
print(icd9.shape) # (1652584, 3)

# Merge to Read2_ICD10 mapping table
icd10 = dx.merge(read2_icd10.loc[:,['read_code','icd10_code']], how='left', left_on='read_2', right_on='read_code')
print(icd10.shape) # (122389889, 3)
icd10 = icd10[~icd10.read_code.isna()]
print(icd10.shape) # (1634256, 3)

# Merge with ICD lookup tables to add code descriptions
# CAUTION: The ICD codes are not unique across ICD9 and ICD10. Codes beginning with E and V are common to both.
# To avoid duplicates, we will reset the index and merge on the ICD9 code. Then we will set the index back to the original. 
# This will allow us later on to merge the two dataframes together without duplicates.
icd9 = icd9.reset_index().merge(icd9_lkp, how='left', left_on='icd9_code', right_on='ICD9').set_index('index')
icd9.drop(columns=['read_code','ICD9'], inplace=True)
icd9['icd_type'] = 9

# 186,824 rows map to either icd9/10 lookups but not the other
icd10 = icd10.reset_index().merge(icd10_lkp[['ALT_CODE','DESCRIPTION']], how='left', left_on='icd10_code',
                                  right_on='ALT_CODE').set_index('index')
icd10.drop(columns=['read_code','ALT_CODE'], inplace=True)
icd10['icd_type'] = 10

# 186,824 rows map to either icd9/10 lookups but not the other
print(f'{len(set(icd9.index).symmetric_difference(set(icd10.index)))} rows map to either icd9/10 lookups but not the other')

# 102,576 rows map to icd9 lookups but not icd10
print(f'{len(set(icd9.index).difference(set(icd10.index)))} rows map to icd9 lookups but not icd10')

# 84,248 rows map to icd10 lookups but not icd9
print(f'{len(set(icd10.index).difference(set(icd9.index)))} rows map to icd10 lookups but not icd9')

# Combine icd9 with the subset of icd10 that is unique
icd9.rename(columns={'icd9_code': 'icd_code', 'DESCRIPTION_ICD9': 'DESCRIPTION'}, inplace=True)
icd10.rename(columns={'icd10_code': 'icd_code'}, inplace=True)
icd = pd.concat([icd9, icd10.loc[set(icd10.index).difference(set(icd9.index))]])

icd.to_parquet('../tidy_data/icd_data_gp_clinical.parquet')

### Inpatient diagnoses
def merge_descriptions(icd_type, df):
    if icd_type == 'icd9':
        lkp_column, icd_lkp = 'ICD9', icd9_lkp
        df_column = 'diag_icd9'
    elif icd_type == 'icd10':
        lkp_column, icd_lkp  = 'ALT_CODE', icd10_lkp[['ALT_CODE', 'DESCRIPTION']]
        df_column = 'diag_icd10'

    df = df.reset_index().merge(icd_lkp, how='left', left_on=df_column, right_on=lkp_column).set_index('index')
    df.drop(columns=[lkp_column], inplace=True)
    return df

# Read in the inpatient data
hesin_diag = pd.read_parquet('../tidy_data/data_hesin_diag.parquet')

# Merge with ICD lookup tables to add code descriptions
hesin_diag_icd9 = merge_descriptions('icd9', hesin_diag[(~hesin_diag.diag_icd9.isna())])
hesin_diag_icd10 = merge_descriptions('icd10', hesin_diag[(~hesin_diag.diag_icd10.isna())])

hesin_diag_icd9.rename(columns={'diag_icd9': 'icd_code'}, inplace=True)
hesin_diag_icd9.drop(columns=['diag_icd10','diag_icd9_nb'], inplace=True)
hesin_diag_icd9['icd_type'] = 9
hesin_diag_icd10.rename(columns={'diag_icd10': 'icd_code'}, inplace=True)
hesin_diag_icd10.drop(columns=['diag_icd9','diag_icd9_nb'], inplace=True)
hesin_diag_icd10['icd_type'] = 10

# Combine the two dataframes
hesin_diag = pd.concat([hesin_diag_icd9, hesin_diag_icd10])

# Save the data
hesin_diag.to_parquet('../tidy_data/icd_data_hesin_diag.parquet')

# Merge the two dataframes together
icd.event_dt = pd.to_datetime(icd.event_dt)
hesin_diag.rename(columns={'epiend':'event_dt'}, inplace=True)
hesin_diag.event_dt = pd.to_datetime(hesin_diag.event_dt)

# Merge the two dataframes together
# 16886307 rows
all_dx = pd.concat([icd[['eid','event_dt','icd_code','DESCRIPTION','icd_type']],
            hesin_diag[['eid','event_dt','icd_code','DESCRIPTION','icd_type']]])
# 15824799 rows, removing 1061508 duplicates
all_dx = all_dx.drop_duplicates().reset_index(drop=True, inplace=True) 

# Save the data
all_dx.to_parquet('../tidy_data/icd_data_all_dx.parquet')