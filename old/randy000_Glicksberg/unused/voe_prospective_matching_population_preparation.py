import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pickle
import statsmodels.api as sm
import seaborn as sns


now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)

person = pd.read_parquet('person_cleaned.parquet')
opi_prescrip = pd.read_parquet('opi_prescrip.parquet')
ncd_prescrip = pd.read_parquet('ncd_prescrip.parquet')
ncd_icd = pd.read_csv('ncd_icd.csv')
sud_icd = pd.read_csv('sud_icd.csv')
hiv_icd = pd.read_csv('hiv_icd.csv')
sickle_icd = pd.read_csv('sickle_icd.csv')
hepc_icd = pd.read_csv('hepc_icd.csv')


c=1
control_N=[]
opioid_N = []
coefs = []
ps = []
low_interval = []
high_interval = []
num_control_ncd = []
num_opioid_ncd = []
followup_interval_col = []
beg_year_col = []
end_year_col = []
num_op_col = []
hx_sud_col = []
ncd_thresh_col = []
hx_sickle_col = []
hx_hepc_col = []
hx_hiv_col = []
con_opi_prescrip_col = []

for followup_interval in [5, 10]:
    for beg_year,end_year in zip([2002,2005],[2005,2008]):
        for num_op in [5, 10]:
            for hx_sud in [0, 1]:
                for ncd_thresh in [45,55]:
                    
                    #for con_opi_prescrip in [0,1]:
                    
                    print('Building populations.')
                    now = datetime.now()
                    current_time = now.strftime("%H:%M:%S")
                    print("Current Time =", current_time)
                    
                    print(c)

                    #con_opi_prescrip_col.append(con_opi_prescrip)

                    #Subset prescriptions for enrollment period 
                    begin = f'{beg_year}-01-01'
                    end = f'{end_year}-01-01'
                    followup = f'{end_year+followup_interval}-01-01'
                    ncd_age_threshold = 365.25*ncd_thresh
                    num_opi_prescrip = num_op

                    enrollment = opi_prescrip[(opi_prescrip['date']>=begin) & (opi_prescrip['date']<end)]
                    mrn_enrollment = enrollment.groupby('MRN').groups

                    mrn_enrollment_keys = list(mrn_enrollment.keys())

                    #identify patients with sufficient opioid prescriptions during the enrollment period
                    opioid_cohort_mrns = []
                    for mrn in mrn_enrollment_keys:
                        if len(mrn_enrollment[mrn]) >= num_opi_prescrip:
                            opioid_cohort_mrns.append(mrn)

                    ncd_icd_followup = ncd_icd[(ncd_icd['date']>=end) & (ncd_icd['date']<followup)]
                    ncd_prescrip_followup = ncd_prescrip[(ncd_prescrip['date']>=end) & (ncd_prescrip['date']<followup)]
                    ncd_followup = set(ncd_icd_followup.MRN).union(ncd_prescrip_followup.MRN)

                    #remove pts with any ncd dx or meds before end of enrollment or before age threshold 
                    ncd_icd_exclude = ncd_icd[(ncd_icd.date<end) | (ncd_icd.AGE_IN_DAYS < ncd_age_threshold)]
                    ncd_prescrip_exclude = ncd_prescrip[(ncd_prescrip.date<end) | (ncd_prescrip.AGE_IN_DAYS < ncd_age_threshold)]
                    all_exclude = list(set(ncd_icd_exclude.MRN).union(set(ncd_prescrip_exclude.MRN)))

                    #if con_opi_prescrip==0:
                    control_cohort = person[(~person.MEDICAL_RECORD_NUMBER.isin(all_exclude)) & 
                       (~person.MEDICAL_RECORD_NUMBER.isin(opioid_cohort_mrns)) & 
                       (~person.MEDICAL_RECORD_NUMBER.isin(set(opi_prescrip.MRN))) & 
                       (~person.MEDICAL_RECORD_NUMBER.isin(set(sud_icd.MRN))) & 
                       (person.YOB<end_year-ncd_thresh)]
                       
#                                     else:
#                                         control_cohort = person[(~person.MEDICAL_RECORD_NUMBER.isin(all_exclude)) & 
#                                            (~person.MEDICAL_RECORD_NUMBER.isin(opioid_cohort_mrns)) & 
#                                            #(person.MEDICAL_RECORD_NUMBER.isin(set(opi_prescrip.MRN))) & 
#                                            (~person.MEDICAL_RECORD_NUMBER.isin(set(sud_icd.MRN))) & 
#                                            (person.YOB<1988) & 
#                                            (person.YOB>1940)]

                    print(control_cohort.shape)

                    if hx_sud==1:
                        opioid_cohort = person[(~person.MEDICAL_RECORD_NUMBER.isin(all_exclude)) & 
                                               (person.MEDICAL_RECORD_NUMBER.isin(opioid_cohort_mrns)) & 
                                               (person.YOB<end_year-ncd_thresh)]
                    else:
                        opioid_cohort = person[(~person.MEDICAL_RECORD_NUMBER.isin(all_exclude)) & 
                                               (person.MEDICAL_RECORD_NUMBER.isin(opioid_cohort_mrns)) &                                                                (~person.MEDICAL_RECORD_NUMBER.isin(set(sud_icd.MRN))) & 
                                               (person.YOB<end_year-ncd_thresh)]
                                               

                    print(opioid_cohort.shape)

                    

                    class_labels = [0]*control_cohort.shape[0] + [1]*opioid_cohort.shape[0]
                    pop = pd.concat([control_cohort, opioid_cohort])
                    pop['label'] = class_labels

                    pop['ncd'] = pop.MEDICAL_RECORD_NUMBER.isin(ncd_followup)
                    pop['ncd'] = pop['ncd'].astype(int)
        
                    sickle_sub = sickle_icd[sickle_icd.date<followup]
                    pop['sickle'] = pop.MEDICAL_RECORD_NUMBER.isin(sickle_sub.MRN)
                    pop['sickle'] = pop['sickle'].astype(int)
                    
                    hepc_sub = hepc_icd[hepc_icd.date<followup]
                    pop['hepc'] = pop.MEDICAL_RECORD_NUMBER.isin(hepc_icd.MRN)
                    pop['hepc'] = pop['hepc'].astype(int)
                    
                    hiv_sub = hiv_icd[hiv_icd.date<followup]
                    pop['hiv'] = pop.MEDICAL_RECORD_NUMBER.isin(hiv_icd.MRN)
                    pop['hiv'] = pop['hiv'].astype(int)
                    
                    pop.to_csv(f'matching_populations/population_{c}.csv')
                    
                    

                    c+=1

print(f'{c} specifications finished.')                     
             
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)
                                                   
                                                   
                    
