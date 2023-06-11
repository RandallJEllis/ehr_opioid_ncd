import pandas as pd

# import diagnoses data
icd = pd.read_parquet('../tidy_data/icd_data_all_dx.parquet')

# Create a dictionary of icd codes and their indices
icd9_gps = icd[icd.icd_type==9].groupby('icd_code').groups
icd10_gps = icd[icd.icd_type==10].groupby('icd_code').groups

# Function to get all diagnoses that start with a given code
def get_dx(icd_type, codes):
    if icd_type == 'icd9':
        icd_gps = icd9_gps
    elif icd_type == 'icd10':
        icd_gps = icd10_gps

    row_idx = [list(icd_gps[key]) for code in codes for key in icd_gps if key.startswith(code)]
    row_idx = [elem for sublist in row_idx for elem in sublist]
    return icd.iloc[row_idx]

# Define diagnoses of interest
ncd_codes = {
    'icd9': ['290', '317', '318', '319', '331'],
    'icd10': ['F01', 'F02', 'F03', 'F70', 'F71', 'F72', 'F73', 'G30', 'G31']
}

sud_codes = {
    'icd9': ['303', '304', '305'],
    'icd10': ['F10', 'F11', 'F12', 'F13', 'F14', 'F15', 'F16', 'F17', 'F18', 'F19']
}

oud_codes = {
    'icd9': ['3040', '3047', '3055'],
    'icd10': ['F11']
}

aud_codes = {
    'icd9': ['303', '3050'],
    'icd10': ['F10']
}

tobacco_codes = {
    'icd9': ['3051'],
    'icd10': ['F17']
}

hiv_codes = {
    'icd9': ['042', '0795', 'V08'],
    'icd10': ['B20', 'Z21'] # codes weren't granular enough to have B9735
}

sickle_codes = {
    'icd9': ['2824', '2825', '2826'], # Had to do it like this because codes weren't granular similarly to MSHS
    'icd10': ['D57']
}

hepc_codes = {
    'icd9': ['0704', '0705', '0707'],
    'icd10': ['B17', 'B18', 'B19']
}

# Export specific files for all diagnoses of interest
ncd_icd = pd.concat([get_dx(system, codes) for system, codes in ncd_codes.items()]).reset_index(drop=True)
ncd_icd.to_parquet('../tidy_data/ncd_diagnoses.parquet')

sud_icd = pd.concat([get_dx(system, codes) for system, codes in sud_codes.items()]).reset_index(drop=True)
sud_icd.to_parquet('../tidy_data/sud_diagnoses.parquet')

oud_icd = pd.concat([get_dx(system, codes) for system, codes in oud_codes.items()]).reset_index(drop=True)
oud_icd.to_parquet('../tidy_data/oud_diagnoses.parquet')

aud_icd = pd.concat([get_dx(system, codes) for system, codes in aud_codes.items()]).reset_index(drop=True)
aud_icd.to_parquet('../tidy_data/aud_diagnoses.parquet')

tobacco_icd = pd.concat([get_dx(system, codes) for system, codes in tobacco_codes.items()]).reset_index(drop=True)
tobacco_icd.to_parquet('../tidy_data/tobacco_diagnoses.parquet')

hiv_icd = pd.concat([get_dx(system, codes) for system, codes in hiv_codes.items()]).reset_index(drop=True)
hiv_icd.to_parquet('../tidy_data/hiv_diagnoses.parquet')

sickle_icd = pd.concat([get_dx(system, codes) for system, codes in sickle_codes.items()]).reset_index(drop=True)
sickle_icd.to_parquet('../tidy_data/sickle_diagnoses.parquet')

hepc_icd = pd.concat([get_dx(system, codes) for system, codes in hepc_codes.items()]).reset_index(drop=True)
hepc_icd.to_parquet('../tidy_data/hepc_diagnoses.parquet')

