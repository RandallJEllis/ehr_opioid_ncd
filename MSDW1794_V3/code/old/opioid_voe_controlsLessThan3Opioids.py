import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pickle
import statsmodels.api as sm
import seaborn as sns
from collections import Counter
from utils import *

'''TODO
Make separate VOE script for AUD as predictor for NCD outcome
include patients with records across inpatient, emergency, and outpatient (Glicksberg email, though this would drastically reduce the N)
'''

new_line = '\n' #for use with f-strings to make the log easier to read
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)

person, encounters, opi_prescrip, ncd_prescrip, ncd_icd, sud_icd, aud_icd, hiv_icd, sickle_icd, hepc_icd, depression_icd, anxiety_icd = import_data()
#numrecsthreshold

for beg_year,end_year in zip([2006,2007,2008,2009,2010,2011,2012,2013,2014],
                             [2009,2010,2011,2012,2013,2014,2015,2016,2017]):
    
    c, control_N, opioid_N, control_mean_age, opioid_mean_age, control_sd_age, opioid_sd_age, control_perc_male, opioid_perc_male, control_perc_female, opioid_perc_female, coefs, ps, low_interval, high_interval, num_control_ncd, num_opioid_ncd, followup_interval_col, beg_year_col, end_year_col, num_op_col, ncd_thresh_col, hx_sickle_col, hx_hepc_col, hx_hiv_col, hx_aud_col, hx_sud_covar_col, hx_depression_col, hx_anxiety_col = initialize_empty_lists()

    for followup_interval in [5, 10]:
        #the data stop at 2022, so skip 10-year follow-up when the end_year is later than 2012
        if followup_interval==10 and end_year>2012:
            continue

        #Subset prescriptions for enrollment period 
        begin, end, followup, mrn_enrollment, mrn_enrollment_keys = opioid_prescriptions(opi_prescrip, beg_year, end_year, followup_interval)

        #identify patients with NCD on followup (i.e., after enrollment, before followup)
        #3 ICD codes or 1 prescription
        ncd_followup = ncd_patients(ncd_prescrip, ncd_icd, end, followup)

        #subset diseases being controlled for by time, and patients by 3+ ICD codes
        sickle_icd_fu_mrns, hepc_icd_fu_mrns, hiv_icd_fu_mrns, aud_icd_fu_mrns, sud_icd_fu_mrns, depression_icd_fu_mrns, anxiety_icd_fu_mrns = controls_filter_patients_3ormore_icd_codes(sud_icd, aud_icd, hiv_icd, sickle_icd, hepc_icd, depression_icd, anxiety_icd, followup)
                
        for num_op in [5, 10, 15]:
            
            #identify exposed patients with sufficient opioid prescriptions during the enrollment period
            opioid_cohort_mrns = opioid_enrollment(num_op, mrn_enrollment, mrn_enrollment_keys)
            
            #identify control patients with <3 opioid prescriptions
            #EDIT 11/26 2:52PM - TESTING CONDITION OF CONTROLS HAVING NO OPIOIDS DURING ENROLLMENT BUT CAN HAVE OPIOIDS OTHERWISE
            # control_cohort_remove_morethan2opioid_mrns = remove_controls_morethan2_opioids(mrn_enrollment, mrn_enrollment_keys)

            for ncd_thresh in [45,55]:

                #remove pts with any ncd dx or meds before end of enrollment or before age threshold 
                all_exclude = exclude_patients_ncd_before_or_during_enrollment(ncd_prescrip, ncd_icd, ncd_thresh, end)

                control_cohort = person[(~person.MRN.isin(all_exclude)) & 
                (~person.MRN.isin(mrn_enrollment_keys)) &    
#                 (~person.MRN.isin(set(opi_prescrip.MRN))) & 
                (~person.MRN.isin(set(sud_icd.MRN))) &
                (person.YOB<(end_year-ncd_thresh))]

                opioid_cohort = person[(~person.MRN.isin(all_exclude)) & 
                                    (person.MRN.isin(opioid_cohort_mrns)) & 
                                    (person.YOB<(end_year-ncd_thresh))]

                with open("log.txt", "a") as myfile:
                    myfile.write(f'Opioid cohort: {opioid_cohort.shape[0]}{new_line}')

                #remove patients with <5 encounters during enrollment
                control_cohort, opioid_cohort = remove_patients_lessthan5_encounters(encounters, begin, end, control_cohort, opioid_cohort)

                #calculate mean/SD of age and percentage of each sex for controls and opioid-exposed patients 
                scalar_con_mean_age, scalar_opi_mean_age, scalar_con_sd_age, scalar_opi_sd_age, scalar_con_perc_male, scalar_opi_perc_male, scalar_con_perc_female, scalar_opi_perc_female = mean_sd_age_percent_sex(end_year, control_cohort, opioid_cohort)

                pop = build_population(sickle_icd_fu_mrns, hepc_icd_fu_mrns, hiv_icd_fu_mrns, aud_icd_fu_mrns, sud_icd_fu_mrns, depression_icd_fu_mrns, anxiety_icd_fu_mrns, followup, ncd_followup, control_cohort, opioid_cohort)
                                                                  
                with open("log.txt", "a") as myfile:
                    myfile.write(f"{sum(pop[pop.label==0]['ncd'])} NCD pts in the control group{new_line}{sum(pop[pop.label==1]['ncd'])} NCD pts in the opioid group.{new_line}")

                for hx_sickle in [0,1]:
                    for hx_hepc in [0,1]:
                        for hx_hiv in [0,1]:
                            for hx_aud in [0,1]:
                                for hx_sud_covar in [0,1]:
                                    for hx_depression in [0,1]:
                                        for hx_anxiety in [0,1]:

                                            if hx_aud==1 and hx_sud_covar==1:
                                                continue

                                            with open("log.txt", "a") as myfile:
                                                myfile.write(f'{str(c)}{new_line}')

                                            now = datetime.now()
                                            current_time = now.strftime("%H:%M:%S")
                                            with open("log.txt", "a") as myfile:
                                                myfile.write(f"Current Time = {current_time}{new_line}")

                                            #update results CSV
                                            update_results_csv(beg_year, end_year, c, control_N, opioid_N, control_mean_age, opioid_mean_age, control_sd_age, opioid_sd_age, control_perc_male, opioid_perc_male, control_perc_female, opioid_perc_female, coefs, ps, low_interval, high_interval, num_control_ncd, num_opioid_ncd, followup_interval_col, beg_year_col, end_year_col, num_op_col, ncd_thresh_col, hx_sickle_col, hx_hepc_col, hx_hiv_col, hx_aud_col, hx_sud_covar_col, hx_depression_col, hx_anxiety_col)
                                            
                                            #set up statistical model
                                            formula = statistical_model(hx_sickle, hx_hepc, hx_hiv, hx_aud, hx_sud_covar, hx_depression, hx_anxiety)

                                            #save coefficients for each individual model/analysis
                                            res = save_coefficient_data(beg_year, end_year, c, pop, formula)

                                            #append coefficients to larger lists for whole enrollment periods
                                            append_data_to_lists(beg_year, end_year, control_N, opioid_N, control_mean_age, opioid_mean_age, control_sd_age, opioid_sd_age, control_perc_male, opioid_perc_male, control_perc_female, opioid_perc_female, coefs, ps, low_interval, high_interval, num_control_ncd, num_opioid_ncd, followup_interval_col, beg_year_col, end_year_col, num_op_col, ncd_thresh_col, hx_sickle_col, hx_hepc_col, hx_hiv_col, hx_aud_col, hx_sud_covar_col, hx_depression_col, hx_anxiety_col, followup_interval, num_op, ncd_thresh, control_cohort, opioid_cohort, scalar_con_mean_age, scalar_opi_mean_age, scalar_con_sd_age, scalar_opi_sd_age, scalar_con_perc_male, scalar_opi_perc_male, scalar_con_perc_female, scalar_opi_perc_female, pop, hx_sickle, hx_hepc, hx_hiv, hx_aud, hx_sud_covar, hx_depression, hx_anxiety, res)

                                            c+=1

    export_final_data_enrollment_period(new_line, beg_year, end_year, c, control_N, opioid_N, control_mean_age, opioid_mean_age, control_sd_age, opioid_sd_age, control_perc_male, opioid_perc_male, control_perc_female, opioid_perc_female, coefs, ps, low_interval, high_interval, num_control_ncd, num_opioid_ncd, followup_interval_col, beg_year_col, end_year_col, num_op_col, ncd_thresh_col, hx_sickle_col, hx_hepc_col, hx_hiv_col, hx_aud_col, hx_sud_covar_col, hx_depression_col, hx_anxiety_col)

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    with open("log.txt", "a") as myfile:
        myfile.write(f"Current Time = {current_time}{new_line}")