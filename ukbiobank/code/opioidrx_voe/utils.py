import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pickle
import statsmodels.api as sm
import seaborn as sns
from collections import Counter

'''
Helper functions to simplify the VoE scripts
'''

def import_data():
    path = '../../tidy_data/'
    person = pd.read_parquet(path+'patient.parquet')
    encounters = pd.read_parquet(path+'encounters.parquet')
    
    opi_prescrip = pd.read_parquet(path+'opioid_med.parquet')
    #only keep prescriptions with a certain duration
    # opi_prescrip['med_duration'] = (opi_prescrip.MEDICATION_END_DATE - opi_prescrip.MEDICATION_START_DATE).dt.days
    # opi_prescrip = opi_prescrip[opi_prescrip.med_duration>=10]

    ncd_prescrip = pd.read_parquet(path+'ncd_med.parquet')
    ncd_icd = pd.read_parquet(path+'ncd_diagnoses.parquet')

    # sud_icd = pd.read_parquet(path+'sud_diagnoses.parquet')
    # WE USED OUD FOR THIS!
    sud_icd = pd.read_parquet(path+'oud_diagnoses.parquet')
    aud_icd = pd.read_parquet(path+'aud_diagnoses.parquet')
    tobacco_icd = pd.read_parquet(path+'tobacco_diagnoses.parquet')

    hiv_icd = pd.read_parquet(path+'hiv_diagnoses.parquet')
    sickle_icd = pd.read_parquet(path+'sickle_diagnoses.parquet')
    # depression_icd = pd.read_parquet(path+'depression_diagnoses.parquet')
    # anxiety_icd = pd.read_parquet(path+'anxiety_diagnoses.parquet')
    return person,encounters,opi_prescrip,ncd_prescrip,ncd_icd,sud_icd,aud_icd,tobacco_icd,hiv_icd,sickle_icd

def initialize_empty_lists():
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
    stderrs = []
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
    hx_hiv_col = []
    hx_aud_col = []
    hx_tobacco_col = []
    hx_sud_covar_col = []
    hx_depression_col = []
    hx_anxiety_col = []
    hx_mat_col = []
    num_recs_covar_col = []
    con_opi_prescrip_col = []
    return c,control_N,opioid_N,control_mean_age,opioid_mean_age,control_sd_age,opioid_sd_age,control_perc_male,opioid_perc_male,control_perc_female,\
        opioid_perc_female,coefs,stderrs,ps,low_interval,high_interval,num_control_ncd,num_opioid_ncd,followup_interval_col,beg_year_col,end_year_col,num_op_col,\
            ncd_thresh_col,hx_sickle_col,hx_hiv_col,hx_aud_col,hx_tobacco_col,hx_sud_covar_col,hx_depression_col,hx_anxiety_col,hx_mat_col

def opioid_prescriptions(opi_prescrip, beg_year, end_year, followup_interval):
    #set dates for beginning and end of enrollment, and followup date
    begin = f'{beg_year}-01-01'
    end = f'{end_year}-01-01'
    followup = f'{end_year+followup_interval}-01-01'

    #subset opioid prescriptions for enrollment
    opioids_timewindow = opi_prescrip[(opi_prescrip['issue_date']>=begin) & 
                                            (opi_prescrip['issue_date']<end)]
    mrn_opioid_counts = opioids_timewindow.groupby('eid').groups
    return begin,end,followup,mrn_opioid_counts#,mrn_opioids

def opioid_enrollment(num_op, mrn_opioid_counts):#, mrn_opioids):
    #subset eids with a qualifying number of opioid prescriptions for enrollment
    opioid_cohort_mrns = [key for (key,value) in mrn_opioid_counts.items() if len(value) >= num_op]
    return opioid_cohort_mrns

def mrn_greaterthan3_opioids(opi_prescrip, followup):
    #identify eids with more than 3 opioid prescriptions to remove from the control group as the control patients can have up to 3 prescriptions
    mrn_remove = []
    opioid_rx_before_followup = opi_prescrip[opi_prescrip['issue_date']<followup]
    opioid_rx_before_followup_groups = opioid_rx_before_followup.groupby('eid').groups
    mrn_remove = [key for (key,value) in opioid_rx_before_followup_groups.items() if len(value) > 3]
    return mrn_remove

def ncd_patients(ncd_prescrip, ncd_icd, end, followup):
    #identify eids who develop NCD between the end of enrollment and followup based on 3+ ICD codes and prescriptions
    ncd_icd_followup = ncd_icd[(ncd_icd['event_dt']>=end) & 
                                (ncd_icd['event_dt']<=followup)].reset_index(drop=True)
    ncd_icd_groups = ncd_icd_followup.groupby('eid').groups
    ncd_icd_fu_mrns = [key for (key,value) in ncd_icd_groups.items() if len(value) >= 3]
    
    #identify patients with 1+ NCD prescriptions upon followup
    ncd_prescrip_followup = ncd_prescrip[(ncd_prescrip['issue_date']>=end) & 
                                        (ncd_prescrip['issue_date']<=followup)].reset_index(drop=True)
    ncd_prescrip_groups = ncd_prescrip_followup.groupby('eid').groups
    ncd_prescrip_fu_mrns = set(ncd_prescrip_followup.eid)
    ncd_followup = list(set(ncd_icd_fu_mrns).union(ncd_prescrip_fu_mrns))

    #identify age of NCD onset by finding earliest date of dx or prescriptions for patients that meet NCD criteria
    age_onset = []
    for mrn in ncd_followup:
        if mrn in ncd_icd_fu_mrns and mrn not in ncd_prescrip_fu_mrns:
            mrn_dx = ncd_icd_followup.iloc[ncd_icd_groups[mrn]]
            earliest_dx_date = min(mrn_dx.event_dt)
            age_onset.append(mrn_dx[mrn_dx.event_dt==earliest_dx_date].AGE_AT_ENCOUNTER.values[0])
            continue

        if mrn in ncd_prescrip_fu_mrns and mrn not in ncd_icd_fu_mrns:
            mrn_rx = ncd_prescrip_followup.iloc[ncd_prescrip_groups[mrn]]
            earliest_rx_date = min(mrn_rx.issue_date)
            age_onset.append(mrn_rx[mrn_rx.issue_date==earliest_rx_date].AGE_AT_ENCOUNTER.values[0])
            continue
    
        if mrn in ncd_prescrip_fu_mrns and mrn in ncd_icd_fu_mrns:
            mrn_dx = ncd_icd_followup.iloc[ncd_icd_groups[mrn]]
            earliest_dx_date = min(mrn_dx.event_dt)
            mrn_rx = ncd_prescrip_followup.iloc[ncd_prescrip_groups[mrn]]
            earliest_rx_date = min(mrn_rx.issue_date)

            if earliest_dx_date < earliest_rx_date:
                age_onset.append(mrn_dx[mrn_dx.event_dt==earliest_dx_date].AGE_AT_ENCOUNTER.values[0])
                continue
            elif earliest_dx_date > earliest_rx_date:
                age_onset.append(mrn_rx[mrn_rx.issue_date==earliest_rx_date].AGE_AT_ENCOUNTER.values[0])
                continue
            elif earliest_dx_date == earliest_rx_date:
                age_onset.append(mrn_rx[mrn_rx.issue_date==earliest_rx_date].AGE_AT_ENCOUNTER.values[0])
                continue
    
    #export dataframe of eids and age of NCD onset
    ncd_followup_df = pd.DataFrame({'eid':ncd_followup, 'age_onset':age_onset})
    return ncd_followup, ncd_followup_df

def extract_mrns_with_3ormore_icd_codes(df, followup):
    #extract eids with 3+ rows in a dataframe with ICD codes
    df_sub = df[df.event_dt<followup]
    df_icd_groups = df_sub.groupby('eid').groups
    df_icd_fu_mrns = [key for (key,value) in df_icd_groups.items() if len(value) >= 3]
    return df_icd_fu_mrns

def controldxs_filter_patients_3ormore_icd_codes(sud_icd, aud_icd, tobacco_icd, hiv_icd, sickle_icd, followup):
    #sickle cell
    sickle_icd_fu_mrns = extract_mrns_with_3ormore_icd_codes(sickle_icd, followup)
                
    #HIV
    hiv_icd_fu_mrns = extract_mrns_with_3ormore_icd_codes(hiv_icd, followup)

    #AUD
    aud_icd_fu_mrns = extract_mrns_with_3ormore_icd_codes(aud_icd, followup)

    #tobacco
    tobacco_icd_fu_mrns = extract_mrns_with_3ormore_icd_codes(tobacco_icd, followup)

    #SUD
    sud_icd_fu_mrns = extract_mrns_with_3ormore_icd_codes(sud_icd, followup)

    return sickle_icd_fu_mrns, hiv_icd_fu_mrns, aud_icd_fu_mrns, tobacco_icd_fu_mrns, sud_icd_fu_mrns#

def exclude_patients_ncd_before_or_during_enrollment(ncd_prescrip, ncd_icd, ncd_thresh, end):
    ncd_icd_exclude = ncd_icd[(ncd_icd.event_dt<end) | 
                                            (ncd_icd.AGE_AT_ENCOUNTER<ncd_thresh)]
    ncd_prescrip_exclude = ncd_prescrip[(ncd_prescrip.issue_date<end) | 
                                                    (ncd_prescrip.AGE_AT_ENCOUNTER<ncd_thresh)]
    all_exclude = list(set(ncd_icd_exclude.eid).union(set(ncd_prescrip_exclude.eid)))
    return all_exclude

def remove_patients_lessthan5_encounters(encounters, begin, end, control_cohort, opioid_cohort):
    enc_period = encounters[(encounters.event_dt>=begin) & (encounters.event_dt<end)]
    mrn_enc_count = Counter(enc_period.eid).most_common()
    #each entry of mrn_enc_count is a length-2 tuple of (eid, number of encounters)
    mrn_enc_keep = [entry[0] for entry in mrn_enc_count if entry[1] >=5]
                #subet patients with adequate encounters
    control_cohort = control_cohort[control_cohort.eid.isin(mrn_enc_keep)]
    opioid_cohort = opioid_cohort[opioid_cohort.eid.isin(mrn_enc_keep)]
    return control_cohort,opioid_cohort

def mean_sd_age_percent_sex(end_year, control_cohort, opioid_cohort):
    control_cohort.loc[:,'age'] = end_year - control_cohort['yob']
    opioid_cohort.loc[:,'age'] = end_year - opioid_cohort['yob']
    scalar_con_mean_age = np.mean(control_cohort['age'])
    scalar_opi_mean_age = np.mean(opioid_cohort['age'])
    scalar_con_sd_age = np.std(control_cohort['age'])
    scalar_opi_sd_age = np.std(opioid_cohort['age'])
    scalar_con_perc_male = control_cohort[control_cohort.sex=='Male'].shape[0]/control_cohort.shape[0]
    scalar_opi_perc_male = opioid_cohort[opioid_cohort.sex=='Male'].shape[0]/opioid_cohort.shape[0]
    scalar_con_perc_female = control_cohort[control_cohort.sex=='Female'].shape[0]/control_cohort.shape[0]
    scalar_opi_perc_female = opioid_cohort[opioid_cohort.sex=='Female'].shape[0]/opioid_cohort.shape[0]
    return scalar_con_mean_age,scalar_opi_mean_age,scalar_con_sd_age,scalar_opi_sd_age,scalar_con_perc_male,scalar_opi_perc_male,scalar_con_perc_female,scalar_opi_perc_female

def build_population(sickle_icd_fu_mrns, hiv_icd_fu_mrns, aud_icd_fu_mrns, tobacco_icd_fu_mrns, \
     sud_icd_fu_mrns, #depression_icd_fu_mrns, anxiety_icd_fu_mrns, \
     ncd_followup, control_cohort, opioid_cohort):
    class_labels = [0]*control_cohort.shape[0] + [1]*opioid_cohort.shape[0]
    pop = pd.concat([control_cohort, opioid_cohort])
    pop['label'] = class_labels

    pop['ncd'] = pop.eid.isin(ncd_followup)
    pop['ncd'] = pop['ncd'].astype(int)
    
    pop['sickle'] = pop.eid.isin(sickle_icd_fu_mrns)
    pop['sickle'] = pop['sickle'].astype(int)
    
    pop['hiv'] = pop.eid.isin(hiv_icd_fu_mrns)
    pop['hiv'] = pop['hiv'].astype(int)
    
    pop['aud'] = pop.eid.isin(aud_icd_fu_mrns)
    pop['aud'] = pop['aud'].astype(int)

    pop['tobacco'] = pop.eid.isin(tobacco_icd_fu_mrns)
    pop['tobacco'] = pop['tobacco'].astype(int)
    
    pop['sud'] = pop.eid.isin(sud_icd_fu_mrns)
    pop['sud'] = pop['sud'].astype(int)
    
    # pop['depression'] = pop.eid.isin(depression_icd_fu_mrns)
    # pop['depression'] = pop['depression'].astype(int)

    # pop['anxiety'] = pop.eid.isin(anxiety_icd_fu_mrns)
    # pop['anxiety'] = pop['anxiety'].astype(int)
    
    return pop

def MAT(opi_prescrip, followup, pop):
    # mat_col = []
    opi_prescrip = opi_prescrip[~opi_prescrip.drug_name.isna()]
    mat_df = opi_prescrip[opi_prescrip.drug_name.str.contains('METHADONE|BUPRENORPHINE|LOFEXIDINE|NALTREXONE')]
    mat_df = mat_df[mat_df['issue_date']<followup]
    # mat_df_groups = mat_df.groupby('eid').groups
    mat_mrns = list(set(mat_df.eid))
    
    # mat_mrns = [key for (key,value) in mat_df_groups.items() if len(value) >= 3]
    # mat_df_before_followup_mrns = list(mat_df_groups.keys())

    # mat_mrns = []
    # for mrn in mat_df_before_followup_mrns:
    #     if len(mat_df_groups[mrn]) >= 3:
    #         mat_mrns.append(mrn)

    pop['MAT'] = pop.eid.isin(mat_mrns)
    pop['MAT'] = pop['MAT'].astype(int)
    # for mrn in pop.eid:
    #     if mrn not in mat_mrns:
    #         mat_col.append(0)
    #     else:
    #         mat_col.append(1)

    # pop['MAT'] = mat_col

    return pop

def opioid_rx_counts(opi_prescrip, followup, pop):
    #obtain number of opioid prescriptions before followup for each patient
    opioid_rx_before_followup = opi_prescrip[opi_prescrip['issue_date']<followup]
    opioid_rx_before_followup_groups = opioid_rx_before_followup.groupby('eid').groups

    opi_mrns = [k for (k,v) in opioid_rx_before_followup_groups.items()]
    num_rx = [len(v) for (k,v) in opioid_rx_before_followup_groups.items()]
    opi_df = pd.DataFrame({'eid':opi_mrns, 'opioid_count': num_rx})
    pop = pop.merge(opi_df, how='left')
    pop.opioid_count.fillna(0, inplace=True)
    return pop
    
def update_results_csv(beg_year, end_year, c, control_N, opioid_N, control_mean_age, opioid_mean_age, control_sd_age, opioid_sd_age, control_perc_male,\
     opioid_perc_male, control_perc_female, opioid_perc_female, coefs, stderrs, ps, low_interval, high_interval, num_control_ncd, num_opioid_ncd, \
        followup_interval_col, beg_year_col, end_year_col, num_op_col, ncd_thresh_col, hx_sickle_col, hx_hiv_col, hx_aud_col, hx_tobacco_col,\
            hx_sud_covar_col, hx_mat_col,output_path):
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
                        'coef': coefs,
                        'stderr': stderrs,
                        '.025': low_interval,
                        '.975': high_interval,
                        'p': ps,
                        'num_control_ncd': num_control_ncd,
                        'num_opioid_ncd': num_opioid_ncd,
                        'followup_time': followup_interval_col,
                        'start_enroll': beg_year_col,
                        'end_enroll': end_year_col,
                        'opioid_rx_enroll': num_op_col,
                        'ncd_age_threshold': ncd_thresh_col,
                        'hx_sickle': hx_sickle_col,
                        'hx_hiv': hx_hiv_col,
                        'hx_aud': hx_aud_col,
                        'hx_tobacco': hx_tobacco_col,
                        'hx_sud_covar': hx_sud_covar_col,
                        'hx_MAT': hx_mat_col
                        })  
        PATH = f'{output_path}/analyses/period_summaries/'
        if not os.path.exists(PATH):
            os.makedirs(PATH)           
        df.to_csv(f'{PATH}voe_{beg_year}_{end_year}.csv', index=None)

def statistical_model(hx_sickle, hx_hiv, hx_aud, hx_tobacco, hx_sud_covar, hx_mat, opioid_predictor='binary_exposure',
                        outcome='ncd'):
    if opioid_predictor=='binary_exposure':
        if outcome=='ncd':
            formula = f"{outcome} ~ C(label) + yob + C(sex)"
        elif outcome=='age_onset':
            formula = f"{outcome} ~ C(label) + C(sex)"
    elif opioid_predictor=='prescription_count':
        if outcome=='ncd':
            formula = f"{outcome} ~ opioid_count + yob + C(sex)"
        elif outcome=='age_onset':
            formula = f"{outcome} ~ opioid_count + C(sex)"

    if hx_sickle: 
        formula = f'{formula} + C(sickle)'
    if hx_hiv:
        formula = f'{formula} + C(hiv)'
    if hx_aud:
        formula = f'{formula} + C(aud)'
    if hx_tobacco:
        formula = f'{formula} + C(tobacco)'
    if hx_sud_covar:
        formula = f'{formula} + C(label)*C(sud)'
    if hx_mat:
        formula = f'{formula} + C(MAT)'
    return formula

def save_coefficient_data(beg_year, end_year, c, pop, formula,output_path, outcome='ncd'):
    if outcome=='ncd':
        res = sm.formula.glm(formula, family=sm.families.Binomial(), data=pop).fit() 
    elif outcome=='age_onset':
        res = sm.formula.glm(formula, data=pop).fit() 
    results_summary = res.summary()

    # Note that tables is a list. The table at index 1 is the "core" table. Additionally, read_html puts dfs in a list, so we want index 0
    results_as_html = results_summary.tables[1].as_html()
    fifi = pd.read_html(results_as_html, header=0, index_col=0)[0]
    PATH = f'{output_path}/analyses/single_expts/'
    if not os.path.exists(PATH):
        os.makedirs(PATH)  
    fifi.to_csv(f'{PATH}/voe_{beg_year}_{end_year}_{c}.csv')
    return res

def append_data_to_lists(beg_year, end_year, control_N, opioid_N, control_mean_age, opioid_mean_age, control_sd_age, opioid_sd_age, control_perc_male,\
     opioid_perc_male, control_perc_female, opioid_perc_female, coefs, stderrs, ps, low_interval, high_interval, num_control_ncd, num_opioid_ncd,\
         followup_interval_col, beg_year_col, end_year_col, num_op_col, ncd_thresh_col, hx_sickle_col, hx_hiv_col, hx_aud_col,\
             hx_tobacco_col, hx_sud_covar_col, hx_mat_col, followup_interval, num_op, ncd_thresh, control_cohort, opioid_cohort,\
                 scalar_con_mean_age, scalar_opi_mean_age, scalar_con_sd_age, scalar_opi_sd_age, scalar_con_perc_male, scalar_opi_perc_male,\
                     scalar_con_perc_female, scalar_opi_perc_female, pop, hx_sickle, hx_hiv, hx_aud, hx_tobacco, hx_sud_covar,\
                        hx_mat, res, opioid_predictor='binary_exposure'):
    
    if opioid_predictor=='binary_exposure':
        table_var = 'C(label)[T.1]'
    elif opioid_predictor=='prescription_count':
        table_var = 'opioid_count'
    a=res.summary2().tables[1]
    coefs.append(a.loc[f'{table_var}', 'Coef.'])
    stderrs.append(a.loc[f'{table_var}', 'Std.Err.'])
    ps.append(a.loc[f'{table_var}', 'P>|z|'])
    low_interval.append(a.loc[f'{table_var}', '[0.025'])
    high_interval.append(a.loc[f'{table_var}', '0.975]'])

    num_control_ncd.append(sum(pop[pop.label==0]['ncd']))
    num_opioid_ncd.append(sum(pop[pop.label==1]['ncd']))
    control_N.append(pop[pop.label==0].shape[0])
    opioid_N.append(pop[pop.label==1].shape[0])
    followup_interval_col.append(followup_interval)
    beg_year_col.append(beg_year)
    end_year_col.append(end_year)
    num_op_col.append(num_op)
    ncd_thresh_col.append(ncd_thresh)
    hx_sickle_col.append(hx_sickle)
    hx_hiv_col.append(hx_hiv)
    hx_aud_col.append(hx_aud)
    hx_tobacco_col.append(hx_tobacco)
    hx_sud_covar_col.append(hx_sud_covar)
    hx_mat_col.append(hx_mat)
    control_mean_age.append(scalar_con_mean_age)
    opioid_mean_age.append(scalar_opi_mean_age)
    control_sd_age.append(scalar_con_sd_age)
    opioid_sd_age.append(scalar_opi_sd_age)
    control_perc_male.append(scalar_con_perc_male)
    opioid_perc_male.append(scalar_opi_perc_male)
    control_perc_female.append(scalar_con_perc_female)
    opioid_perc_female.append(scalar_opi_perc_female)

def export_final_data_enrollment_period(new_line, beg_year, end_year, c, control_N, opioid_N, control_mean_age, opioid_mean_age, control_sd_age,\
     opioid_sd_age, control_perc_male, opioid_perc_male, control_perc_female, opioid_perc_female, coefs, stderrs, ps, low_interval, high_interval,\
         num_control_ncd, num_opioid_ncd, followup_interval_col, beg_year_col, end_year_col, num_op_col, ncd_thresh_col, hx_sickle_col,\
              hx_hiv_col, hx_aud_col, hx_tobacco_col, hx_sud_covar_col, hx_mat_col,output_path):
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
        'coef': coefs,
        'stderr': stderrs,
        '.025': low_interval,
        '.975': high_interval,
        'p': ps,
        'num_control_ncd': num_control_ncd,
        'num_opioid_ncd': num_opioid_ncd,
        'followup_time': followup_interval_col,
        'start_enroll': beg_year_col,
        'end_enroll': end_year_col,
        'opioid_rx_enroll': num_op_col,
        'ncd_age_threshold': ncd_thresh_col,
        'hx_sickle': hx_sickle_col,
        'hx_hiv': hx_hiv_col,
        'hx_aud': hx_aud_col,
        'hx_tobacco': hx_tobacco_col,
        'hx_sud_covar': hx_sud_covar_col,
        'hx_MAT': hx_mat_col
        })   
    PATH = f'{output_path}/analyses/period_summaries/'
    if not os.path.exists(PATH):
        os.makedirs(PATH)         
    df.to_csv(f'{PATH}voe_{beg_year}_{end_year}.csv', index=None)