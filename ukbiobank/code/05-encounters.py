import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from datetime import datetime, timedelta

# import inpatient diagnoses data
# every diagnosis only has the start date of the episode, not the specific day of the diagnosis
hesin_diag = pd.read_parquet('../tidy_data/data_hesin_diag.parquet')

# import clinical events
gp_clin = pd.read_parquet('../tidy_data/data_gp_clinical.parquet')

# concatenate
enc = pd.concat([hesin_diag[['eid','epistart']].rename(columns={'epistart':'event_dt'}),
                    gp_clin[['eid','event_dt']]
                    ])
enc = enc.drop_duplicates()
enc = enc.reset_index(drop=True)
enc.to_parquet('../tidy_data/encounters.parquet')      

# Count total patients and number of patients with 5+ encounters
eid_cnt = Counter(enc.eid)
arr = np.array([x for x in eid_cnt.values()])
# 468397 total, 350483 with 5+ encounters
print(len(eid_cnt), sum(arr>5))