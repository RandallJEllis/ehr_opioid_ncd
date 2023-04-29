import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pickle
import statsmodels.api as sm
import seaborn as sns
from collections import Counter
from utils_AUD import *
import sys

#choose outcome: binary outcome ('ncd') of NCD or age of onset ('age_onset')
outcome = sys.argv[1] #'ncd' or 'age_onset'

print(f'Outcome variable is {outcome}')

if outcome != 'age_onset':
    output_path = f'../../voe_outputs/aud/controlsNoAUDDX/binary_outcome/controlVarOUD'
else:
    output_path = f'../../voe_outputs/aud/controlsNoAUDDX/age_onset_ncd/controlVarOUD'

new_line = '\n' #for use with f-strings to make the log easier to read
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)

person, encounters, opi_prescrip, ncd_prescrip, ncd_icd, sud_icd, aud_icd, tobacco_icd, hiv_icd, sickle_icd, hepc_icd,depression_icd,anxiety_icd = import_data()

for beg_year,end_year in zip([2006,2007,2008,2009,2010,2011,2012,2013,2014],
                             [2009,2010,2011,2012,2013,2014,2015,2016,2017]):
    
    c, control_N, aud_N, control_mean_age, aud_mean_age, control_sd_age, aud_sd_age, control_perc_male, aud_perc_male, control_perc_female,\
         aud_perc_female, coefs, stderrs, ps, low_interval, high_interval, num_control_ncd, num_aud_ncd, followup_interval_col, beg_year_col, end_year_col,\
             num_op_col, ncd_thresh_col, hx_sickle_col, hx_hepc_col, hx_hiv_col, hx_aud_col, hx_tobacco_col,hx_sud_covar_col, hx_depression_col,\
                 hx_anxiety_col,hx_mat_col,control_opioid_rx_count_col = initialize_empty_lists()

    for followup_interval in [5, 10]:
        #the data stop at 2022, so skip 10-year follow-up when the end_year is later than 2012
        if followup_interval==10 and end_year>2012:
            continue

        #set dates for beginning and end of enrollment, and followup date
        begin = f'{beg_year}-01-01'
        end = f'{end_year}-01-01'
        followup = f'{end_year+followup_interval}-01-01'

        #Subset prescriptions for enrollment period 
        # mrn_opioid_counts = opioid_prescriptions(opi_prescrip, beg_year, end_year, followup_interval)

        #identify patients with NCD on followup (i.e., after enrollment, before followup)
        #3 ICD codes or 1 prescription
        ncd_followup,ncd_followup_df = ncd_patients(ncd_prescrip, ncd_icd, end, followup)

        #subset diseases being controlled for by time, and patients by 3+ ICD codes
        sickle_icd_fu_mrns, hepc_icd_fu_mrns, hiv_icd_fu_mrns, aud_icd_fu_mrns, tobacco_icd_fu_mrns, sud_icd_fu_mrns,depression_icd_fu_mrns, \
            anxiety_icd_fu_mrns = controldxs_filter_patients_3ormore_icd_codes(sud_icd, aud_icd, tobacco_icd, hiv_icd, sickle_icd, hepc_icd,\
                 depression_icd,anxiety_icd,followup)
                
        # for num_op in [5, 10, 15]:
            
        #identify exposed patients with sufficient opioid prescriptions during the enrollment period
        # opioid_cohort_mrns = opioid_enrollment(num_op, mrn_opioid_counts)
        
        #identify all patients with >3 opioid prescriptions before followup to remove from the controls
        # mrns_greaterthan3_opioids_remove_from_control_cohort = mrn_greaterthan3_opioids(opi_prescrip, followup)

        for ncd_thresh in [45,55,65]:

            #remove pts with any ncd dx or meds before end of enrollment or before age threshold 
            all_exclude = exclude_patients_ncd_before_or_during_enrollment(ncd_prescrip, ncd_icd, ncd_thresh, end)

            #build control cohort. remove patients with NCD before threshold age, patients with SUD or AUD diagnoses, patients who are too young to be enrolled
            control_cohort = person[(~person.MRN.isin(all_exclude)) & 
                            #(~person.MRN.isin(mrns_greaterthan3_opioids_remove_from_control_cohort)) &   
                            #(~person.MRN.isin(mrn_opioids)) &    
                            # (~person.MRN.isin(set(sud_icd.MRN))) &
                            (~person.MRN.isin(set(aud_icd[aud_icd.DIAGNOSIS_DATE<followup].MRN))) &
                            (person.YOB<(end_year-ncd_thresh))]

            #build AUD cohort
            aud_cohort = person[(~person.MRN.isin(all_exclude)) & 
                            (person.MRN.isin(aud_icd_fu_mrns)) & 
                            (person.YOB<(end_year-ncd_thresh))]

            with open("log.txt", "a") as myfile:
                myfile.write(f'AUD cohort: {aud_cohort.shape[0]}{new_line}')

            #remove patients with <5 encounters during enrollment
            control_cohort, aud_cohort = remove_patients_lessthan5_encounters(encounters, begin, end, control_cohort, aud_cohort)

            #calculate mean/SD of age and percentage of each sex for controls and AUD patients 
            scalar_con_mean_age, scalar_aud_mean_age, scalar_con_sd_age, scalar_aud_sd_age, scalar_con_perc_male, scalar_aud_perc_male, \
                scalar_con_perc_female, scalar_aud_perc_female = mean_sd_age_percent_sex(end_year, control_cohort, aud_cohort)
            
            #label patients by cohort, NCD outcome, and filter patients with 3+ ICD codes for controlled diagnoses
            pop = build_population(sickle_icd_fu_mrns, hepc_icd_fu_mrns, hiv_icd_fu_mrns, aud_icd_fu_mrns, tobacco_icd_fu_mrns, sud_icd_fu_mrns, \
                depression_icd_fu_mrns,anxiety_icd_fu_mrns,
                ncd_followup, control_cohort, aud_cohort)
            
            # number of opioid prescriptions for each patient
            pop = opioid_rx_counts(opi_prescrip, followup, pop)

            #mark patients with medication-assisted therapy (3+ prescriptions)
            # pop = MAT(opi_prescrip, followup, pop)

            #only retain patients with NCD, add column for age_onset
            if outcome == 'age_onset':
                pop = pop[pop.ncd==1]
                pop = pop.merge(ncd_followup_df.loc[:,['MRN','age_onset']])
            
            #save population
            pop.to_csv(f'{output_path}/populations/voe_{beg_year}_{end_year}_{followup_interval}yearfollowup_{ncd_thresh}NCDageExclusion_{c}.csv')   

            with open("log.txt", "a") as myfile:
                myfile.write(f"{sum(pop[pop.label==0]['ncd'])} NCD pts in the control group{new_line}{sum(pop[pop.label==1]['ncd'])} NCD pts in the AUD group.{new_line}")

            # for hx_sickle in [0,1]:
            #     for hx_hepc in [0,1]:
            #         for hx_hiv in [0,1]:
                        # for hx_aud in [0,1]:
            for hx_tobacco in [0,1]:
                for hx_sud_covar in [0,1]:
                # for hx_depression in [0,1]:
                #     for hx_anxiety in [0,1]:
                #         for control_opioid_rx_count in [0,1]:
                            # for hx_mat in [0,1]:

                            # if hx_aud==1 and hx_sud_covar==1:
                            #     continue

                    with open("log.txt", "a") as myfile:
                        myfile.write(f'{str(c)}{new_line}')

                    now = datetime.now()
                    current_time = now.strftime("%H:%M:%S")
                    with open("log.txt", "a") as myfile:
                        myfile.write(f"Current Time = {current_time}{new_line}")

                    #update results CSV
                    update_results_csv(beg_year, end_year, c, control_N, aud_N, control_mean_age, aud_mean_age, 
                        control_sd_age, aud_sd_age, control_perc_male, aud_perc_male, control_perc_female, 
                            aud_perc_female, coefs, stderrs, ps, low_interval, high_interval, num_control_ncd, num_aud_ncd,
                                    followup_interval_col, beg_year_col, end_year_col,  
                                    #num_op_col, 
                                    ncd_thresh_col, hx_sickle_col,
                                        hx_hepc_col, hx_hiv_col, hx_tobacco_col,
                                        #hx_aud_col, 
                                        hx_sud_covar_col, 
                                            #hx_mat_col, 
                                            hx_depression_col,hx_anxiety_col,control_opioid_rx_count_col,output_path)
                    
                    #set up statistical model
                    formula = statistical_model(hx_tobacco, hx_sud_covar, outcome=outcome) 
                    #hx_sickle, hx_hepc, hx_hiv, #hx_depression,hx_anxiety, control_opioid_rx_count,#hx_mat,\
                            

                    #save coefficients for each individual model/analysis
                    res = save_coefficient_data(beg_year, end_year, c, pop, formula, output_path, outcome=outcome)

                    #append coefficients to larger lists for whole enrollment periods
                    append_data_to_lists(beg_year, end_year, control_N, aud_N, control_mean_age, aud_mean_age, \
                        control_sd_age, aud_sd_age, control_perc_male, aud_perc_male, control_perc_female,\
                                aud_perc_female, coefs, stderrs, ps, low_interval, high_interval, num_control_ncd, num_aud_ncd, \
                                followup_interval_col, beg_year_col, end_year_col, #num_op_col, 
                                ncd_thresh_col, hx_sickle_col, \
                                    hx_hepc_col, hx_hiv_col, hx_tobacco_col, #hx_aud_col, 
                                    hx_sud_covar_col, hx_depression_col, hx_anxiety_col,control_opioid_rx_count_col, \
                                        #hx_mat_col, 
                                        followup_interval, ncd_thresh, control_cohort, aud_cohort, \
                                            scalar_con_mean_age, scalar_aud_mean_age, scalar_con_sd_age, scalar_aud_sd_age,\
                                                    scalar_con_perc_male, scalar_aud_perc_male, scalar_con_perc_female,\
                                                        scalar_aud_perc_female, pop, hx_tobacco,
                                                        #hx_sickle, hx_hepc, hx_hiv, \
                                                            hx_sud_covar, #hx_depression, hx_anxiety,control_opioid_rx_count,#hx_mat, 
                                                            res)

                    c+=1

    export_final_data_enrollment_period(new_line, beg_year, end_year, c, control_N, aud_N, control_mean_age, aud_mean_age, control_sd_age,\
         aud_sd_age, control_perc_male, aud_perc_male, control_perc_female, aud_perc_female, coefs, stderrs, ps, low_interval, high_interval, \
            num_control_ncd, num_aud_ncd, followup_interval_col, beg_year_col, end_year_col, #num_op_col, 
            ncd_thresh_col, #hx_sickle_col, hx_hepc_col, hx_hiv_col, 
            hx_tobacco_col, #hx_aud_col, 
                hx_sud_covar_col, #hx_depression_col, hx_anxiety_col,control_opioid_rx_count_col,#hx_mat_col,
                output_path)

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    with open("log.txt", "a") as myfile:
        myfile.write(f"Current Time = {current_time}{new_line}")