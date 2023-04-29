import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pickle
import statsmodels.api as sm
import seaborn as sns

'''TODO
Make separate VOE script for AUD as predictor for NCD outcome
include patients with records across inpatient, emergency, and outpatient (Glicksberg email, though this would drastically reduce the N)
'''

new_line = '\n' #for use with f-strings to make the log easier to read
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)
path = 'tidy_data/'
person = pd.read_parquet(path+'person_cleaned.parquet')
opi_prescrip = pd.read_parquet(path+'opi_prescrip.parquet')
ncd_prescrip = pd.read_parquet(path+'ncd_prescrip.parquet')
ncd_icd = pd.read_csv(path+'ncd_icd.csv')
sud_icd = pd.read_csv(path+'sud_icd.csv')
#sud_icd = sud_icd[~sud_icd.ONTOLOGY_ID.isin([304.0,
#  304.01,
#  304.02,
#  304.03,304.7,
#  304.71,
#  304.72,
#  304.73,305.5,
#  305.51,
#  305.52,
#  305.53,'F11.10',
#  'F11.120',
#  'F11.129',
#  'F11.20',
#  'F11.21',
#  'F11.220',
#  'F11.221',
#  'F11.222',
#  'F11.229',
#  'F11.23',
#  'F11.24',
#  'F11.29',
#  'F11.90',
#  'F11.921',
#  'F11.94',
#  'F11.988',
#  'F11.99'])]
aud_icd = pd.read_csv(path+'aud_icd.csv')
hiv_icd = pd.read_csv(path+'hiv_icd.csv')
sickle_icd = pd.read_csv(path+'sickle_icd.csv')
hepc_icd = pd.read_csv(path+'hepc_icd.csv')
numrecsthreshold = 2 #number of records each included patient must have

for beg_year,end_year in zip([2003, 2006],[2006, 2009]):
    
    c=1
    control_N = []
    opioid_N = []
    control_mean_age = []
    opioid_mean_age = []
    control_sd_age = []
    opioid_sd_age = []
    control_perc_male = []
    opioid_perc_male = []
    control_perc_female = []
    opioid_perc_female = []
    num_opRx = []
    coefs = []
    ps = []
    low_interval = [] #confidence interval of OR
    high_interval = [] #confidence interval of OR
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
    hx_aud_col = []
    hx_sud_covar_col = []
    num_recs_covar_col = []
    con_opi_prescrip_col = []

    numrecs_enroll = pd.read_parquet(f'{path}mrns_numRecords_{beg_year}_{end_year}.parquet')#pickle.load(open(path+'mrns_numRecords_{beg_year}_{end_year}.p', 'rb'))
    for followup_interval in [5, 10]:
        for num_op in [5, 10, 15]:
            for hx_sud in [0, 1]:
                for ncd_thresh in [45,55]:
                    for num_recs_covar in [0, 1]:
                        
                        #the data stop at 2016, so skip 10-year follow-up for the second enrollment period
                        if followup_interval==10 and beg_year==2006:
                            continue
                        
                        #for con_opi_prescrip in [0,1]:
                        now = datetime.now()
                        current_time = now.strftime("%H:%M:%S")
                        with open("log.txt", "a") as myfile:
                            myfile.write(f'Building populations. {current_time}{new_line}')
                        
                        with open("log.txt", "a") as myfile:
                            myfile.write(f'{str(c)}{new_line}')

                        #con_opi_prescrip_col.append(con_opi_prescrip)

                        #Subset prescriptions for enrollment period 
                        begin = f'{beg_year}-01-01'
                        end = f'{end_year}-01-01'
                        followup = f'{end_year+followup_interval}-01-01'
                        ncd_age_threshold = 365.25*ncd_thresh

                        enrollment = opi_prescrip[(opi_prescrip['date']>=begin) & (opi_prescrip['date']<end)]
                        mrn_enrollment = enrollment.groupby('MRN').groups

                        mrn_enrollment_keys = list(mrn_enrollment.keys())

                        #identify patients with sufficient opioid prescriptions during the enrollment period
                        opioid_cohort_mrns = []
                        for mrn in mrn_enrollment_keys:
                            if len(mrn_enrollment[mrn]) >= num_op:
                                opioid_cohort_mrns.append(mrn)

                        #tabulate number of opioid prescriptions per patient during enrollment
                        num_oprx = [len(v) for k,v in mrn_enrollment.items()]
                        mrn_oprx = [k for k,v in mrn_enrollment.items()]
                        oprx_df = pd.DataFrame({'mrn': mrn_oprx, 'num_op': num_oprx})

                        #identify patients with NCD on followup (i.e., after enrollment, before followup)
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
                        (person.MEDICAL_RECORD_NUMBER.isin(numrecs_enroll.mrn)) &  
                        (person.YOB<end_year-ncd_thresh)]
                        
    #                                     else:
    #                                         control_cohort = person[(~person.MEDICAL_RECORD_NUMBER.isin(all_exclude)) & 
    #                                            (~person.MEDICAL_RECORD_NUMBER.isin(opioid_cohort_mrns)) & 
    #                                            #(person.MEDICAL_RECORD_NUMBER.isin(set(opi_prescrip.MRN))) & 
    #                                            (~person.MEDICAL_RECORD_NUMBER.isin(set(sud_icd.MRN))) & 
    #                                            (person.YOB<1988) & 
    #                                            (person.YOB>1940)]

                        with open("log.txt", "a") as myfile:
                            myfile.write(f'Control cohort: {control_cohort.shape[0]}{new_line}')

                        # if hx_sud==1:
                        opioid_cohort = person[(~person.MEDICAL_RECORD_NUMBER.isin(all_exclude)) & 
                                            (person.MEDICAL_RECORD_NUMBER.isin(opioid_cohort_mrns)) & 
                                            (person.MEDICAL_RECORD_NUMBER.isin(numrecs_enroll.mrn))	&
                                            (person.YOB<end_year-ncd_thresh)]
                        # else:
                        #     opioid_cohort = person[(~person.MEDICAL_RECORD_NUMBER.isin(all_exclude)) & 
                        #                         (person.MEDICAL_RECORD_NUMBER.isin(opioid_cohort_mrns)) & 
                        #                         (~person.MEDICAL_RECORD_NUMBER.isin(set(sud_icd.MRN))) & 
                        #                         (person.MEDICAL_RECORD_NUMBER.isin(numrecs_enroll.mrn))	&
                        #                         (person.YOB<end_year-ncd_thresh)]
                                                

                        with open("log.txt", "a") as myfile:
                            myfile.write(f'Opioid cohort: {opioid_cohort.shape[0]}{new_line}')
                        
                        control_cohort.loc[:,'age'] = end_year - control_cohort['YOB']
                        opioid_cohort.loc[:,'age'] = end_year - opioid_cohort['YOB']
                        scalar_con_mean_age = np.mean(control_cohort['age'])
                        scalar_opi_mean_age = np.mean(opioid_cohort['age'])
                        scalar_con_sd_age = np.std(control_cohort['age'])
                        scalar_opi_sd_age = np.std(opioid_cohort['age'])
                        scalar_con_perc_male = control_cohort[control_cohort.GENDER=='Male'].shape[0]/control_cohort.shape[0]
                        scalar_opi_perc_male = opioid_cohort[opioid_cohort.GENDER=='Male'].shape[0]/opioid_cohort.shape[0]
                        scalar_con_perc_female = control_cohort[control_cohort.GENDER=='Female'].shape[0]/control_cohort.shape[0]
                        scalar_opi_perc_female = opioid_cohort[opioid_cohort.GENDER=='Female'].shape[0]/opioid_cohort.shape[0]

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

                        aud_sub = aud_icd[aud_icd.date<followup]
                        pop['aud'] = pop.MEDICAL_RECORD_NUMBER.isin(aud_icd.MRN)
                        pop['aud'] = pop['aud'].astype(int)

                        sud_sub = sud_icd[sud_icd.date<followup]
                        pop['sud'] = pop.MEDICAL_RECORD_NUMBER.isin(sud_icd.MRN)
                        pop['sud'] = pop['sud'].astype(int)

                        pop = pd.merge(pop, numrecs_enroll, left_on='MEDICAL_RECORD_NUMBER', right_on='mrn')
                        pop = pop[pop.records>numrecsthreshold]

                        pop = pd.merge(pop, oprx_df, how='left', on='mrn')
                        pop.num_op.fillna(0, inplace=True)

                        with open("log.txt", "a") as myfile:
                            myfile.write(f"{sum(pop[pop.label==0]['ncd'])} NCD pts in the control group{new_line}{sum(pop[pop.label==1]['ncd'])} NCD pts in the opioid group.{new_line}")
            
                        for hx_sickle in [0,1]:
                            for hx_hepc in [0,1]:
                                for hx_hiv in [0,1]:
                                    for hx_aud in [0,1]:
                                        for hx_sud_covar in [0,1]:
                                            for num_recs_covar in [0,1]:
                                                for oprx in [0,1]:

                                                    if hx_sud==0 and hx_sud_covar==1:
                                                        continue

                                    
                                                    with open("log.txt", "a") as myfile:
                                                        myfile.write(f'{str(c)}{new_line}')

                                                    now = datetime.now()
                                                    current_time = now.strftime("%H:%M:%S")
                                                    with open("log.txt", "a") as myfile:
                                                        myfile.write(f"Current Time = {current_time}{new_line}")


                                                    #update results CSV
                                                    if c>1:
                                                        df = pd.DataFrame({
                                                            'control_N': control_N,
                                                            'opioid_N': opioid_N,
                                                            'control_AgeMean': control_mean_age,
                                                            'control_AgeSD': control_sd_age,
                                                            'opioid_AgeMean': opioid_mean_age,
                                                            'opioid_AgeSD': opioid_sd_age,
                                                            'control_male%': control_perc_male,
                                                            'control_female%': control_perc_female,
                                                            'opioid_male%': opioid_perc_male,
                                                            'opioid_female%': opioid_perc_female,
                                                            'OR': coefs,
                                                            '.025': low_interval,
                                                            '.975': high_interval,
                                                            'p': ps,
                                                            'num_control_ncd': num_control_ncd,
                                                            'num_opioid_ncd': num_opioid_ncd,
                                                            'followup_time': followup_interval_col,
                                                            'start_enroll': beg_year_col,
                                                            'end_enroll': end_year_col,
                                                            'opioid_rx_enroll': num_op_col,
                                                            'hx_sud_opioid_enroll': hx_sud_col,
                                                            'ncd_age_threshold': ncd_thresh_col,
                                                            'hx_sickle': hx_sickle_col,
                                                            'hx_hepc': hx_hepc_col,
                                                            'hx_hiv': hx_hiv_col,
                                                            'hx_aud': hx_aud_col,
                                                            'hx_sud_covar': hx_sud_covar_col,
                                                            'num_recs_covar': num_recs_covar_col,
                                                            'num_opRx': num_opRx

                                                            #'control_opi_prescrip': con_opi_prescrip_col
                                                            })            
                                                        df.to_csv(f'voe_outputs/OUD/OPIOIDRX_voe_{beg_year}_{end_year}_{numrecsthreshold}OrMoreRecs.csv', index=None)

                                                    formula = "ncd ~ C(label) + YOB + C(GENDER)"
                                                    if hx_sickle: 
                                                        formula = f'{formula} + C(sickle)'
                                                    if hx_hepc:
                                                        formula = f'{formula} + C(hepc)'
                                                    if hx_hiv:
                                                        formula = f'{formula} + C(hiv)'
                                                    if hx_aud:
                                                        formula = f'{formula} + C(aud)'
                                                    if hx_sud_covar:
                                                        formula = f'{formula} + C(sud)'
                                                    if num_recs_covar:
                                                        formula = f'{formula} + records'
                                                    #include number of opioid prescriptions as a variable
                                                    if oprx:
                                                        formula = f'{formula} + num_op'

                                                    res = sm.formula.glm(formula,  family=sm.families.Binomial(), data=pop).fit() 
                                                    results_summary = res.summary()

                                                    # Note that tables is a list. The table at index 1 is the "core" table. Additionally, read_html puts dfs in a list, so we want index 0
                                                    results_as_html = results_summary.tables[1].as_html()
                                                    fifi = pd.read_html(results_as_html, header=0, index_col=0)[0]
                                                    fifi.to_csv(f'voe_outputs/OUD/OPIOIDRX_voe_{beg_year}_{end_year}_{numrecsthreshold}OrMoreRecs_{c}.csv')


                                                    a=res.summary2().tables[1]
                                                    coefs.append(a.loc['C(label)[T.1]', 'Coef.'])
                                                    ps.append(a.loc['C(label)[T.1]', 'P>|z|'])
                                                    low_interval.append(a.loc['C(label)[T.1]', '[0.025'])
                                                    high_interval.append(a.loc['C(label)[T.1]', '0.975]'])

                                                    num_control_ncd.append(sum(pop[pop.label==0]['ncd']))
                                                    num_opioid_ncd.append(sum(pop[pop.label==1]['ncd']))
                                                    control_N.append(control_cohort.shape[0])
                                                    opioid_N.append(opioid_cohort.shape[0])
                                                    followup_interval_col.append(followup_interval)
                                                    beg_year_col.append(beg_year)
                                                    end_year_col.append(end_year)
                                                    num_op_col.append(num_op)
                                                    hx_sud_col.append(hx_sud)
                                                    ncd_thresh_col.append(ncd_thresh)
                                                    hx_sickle_col.append(hx_sickle)
                                                    hx_hepc_col.append(hx_hepc)
                                                    hx_hiv_col.append(hx_hiv)
                                                    hx_aud_col.append(hx_aud)
                                                    hx_sud_covar_col.append(hx_sud_covar)
                                                    num_recs_covar_col.append(num_recs_covar)
                                                    control_cohort.loc[:,'age'] = end_year - control_cohort['YOB']
                                                    opioid_cohort.loc[:,'age'] = end_year - opioid_cohort['YOB']
                                                    control_mean_age.append(scalar_con_mean_age)
                                                    opioid_mean_age.append(scalar_opi_mean_age)
                                                    control_sd_age.append(scalar_con_sd_age)
                                                    opioid_sd_age.append(scalar_opi_sd_age)
                                                    control_perc_male.append(scalar_con_perc_male)
                                                    opioid_perc_male.append(scalar_opi_perc_male)
                                                    control_perc_female.append(scalar_con_perc_female)
                                                    opioid_perc_female.append(scalar_opi_perc_female)
                                                    num_opRx.append(oprx)

                                                    c+=1

    with open("log.txt", "a") as myfile:
        myfile.write(f'{c} specifications finished.{new_line}')                     
    df = pd.DataFrame(
        {'control_N': control_N,
        'opioid_N': opioid_N,
        'control_AgeMean': control_mean_age,
        'control_AgeSD': control_sd_age,
        'opioid_AgeMean': opioid_mean_age,
        'opioid_AgeSD': opioid_sd_age,
        'control_male%': control_perc_male,
        'control_female%': control_perc_female,
        'opioid_male%': opioid_perc_male,
        'opioid_female%': opioid_perc_female,
        'OR': coefs,
        '.025': low_interval,
        '.975': high_interval,
        'p': ps,
        'num_control_ncd': num_control_ncd,
        'num_opioid_ncd': num_opioid_ncd,
        'followup_time': followup_interval_col,
        'start_enroll': beg_year_col,
        'end_enroll': end_year_col,
        'opioid_rx_enroll': num_op_col,
        'hx_sud_opioid_enroll': hx_sud_col,
        'ncd_age_threshold': ncd_thresh_col,
        'hx_sickle': hx_sickle_col,
        'hx_hepc': hx_hepc_col,
        'hx_hiv': hx_hiv_col,
        'hx_aud': hx_aud_col,
        'hx_sud_covar': hx_sud_covar_col,
        'num_recs_covar': num_recs_covar_col,
        'num_opRx': num_opRx
        #'control_opi_prescrip': con_opi_prescrip_col
        })            
    df.to_csv(f'voe_outputs/OUD/OPIOIDRX_voe_{beg_year}_{end_year}_{numrecsthreshold}OrMoreRecs.csv', index=None)

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    with open("log.txt", "a") as myfile:
        myfile.write(f"Current Time = {current_time}{new_line}")                                                   
                                                   
                    
