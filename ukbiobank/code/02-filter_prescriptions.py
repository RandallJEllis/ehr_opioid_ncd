import pandas as pd

med = pd.read_parquet('../tidy_data/med.parquet')

### Opioid drugs
# ReadV2 for opioid prescriptions begins with dj; BNF begins with 040702 (opioid analgesics) or 041003 (opioid dependence)
opioid_df = med[(med.read_2.str.startswith('dj', na=False)) | 
                (med.bnf_code.str.startswith('040702', na=False)) | 
                (med.bnf_code.str.startswith('041003', na=False)) | 
                (med.bnf_code.str.startswith('04.07.02', na=False)) |
                (med.bnf_code.str.startswith('04.10.03', na=False))
                ]
opioid_df.to_parquet('../tidy_data/opioid_med.parquet')

### Anti-dementia drugs
# ReadV2 for antidementia prescriptions begins with dy (CENTRAL ACETYLCHOLINESTERASE INHIBITOR) or
# dB (OTHER ANTIDEMENTIA DRUGS); BNF begins with 0411      
antidementia_df = med[(med.read_2.str.startswith('dy', na=False)) | 
                      (med.read_2.str.startswith('dB', na=False)) |
                      (med.bnf_code.str.startswith('0411', na=False)) | 
                      (med.bnf_code.str.startswith('04.11', na=False)) 
           ]
antidementia_df.to_parquet('../tidy_data/ncd_med.parquet')
