import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import date2num
import datetime
from collections import Counter
import sys

'''
1. Match entries with BNF but no CV2 to "bnf_lkp" sheet, and entries with CV2 but no BNF to "read_v2_drugs_lkp".
2. For med entries with CV2 but no BNF, take the first word of drug name in ALL CAPS, and search it in the 
"Presentation" column in the "bnf_lkp" sheet and try to merge drug class on this
'''

# load prescriptions and lookup table for merging CV2 to BNF
med = pd.read_parquet('../tidy_data/data_gp_scripts.parquet')
read_v2_drugs_bnf = pd.read_parquet('../tidy_data/read_v2_drugs_bnf.parquet')
bnf_lkp = pd.read_parquet('../tidy_data/bnf_lkp.parquet')
read_v2_drugs_lkp = pd.read_parquet('../tidy_data/read_v2_drugs_lkp.parquet')

# create subsets based on whether CV2 or BNF are NaN
# nan_both_bnf_cv2 = med[(med.bnf_code.isna()) & (med.read_2.isna())]
val_cv2_nan_bnf = med[(med.bnf_code.isna()) & ~(med.read_2.isna())]
val_bnf_nan_cv2 = med[~(med.bnf_code.isna()) & (med.read_2.isna())]
val_both = med[~(med.bnf_code.isna()) & ~(med.read_2.isna())]

del med

# # NOT USING: ReadV2 codes in read_v2_drugs_bnf have a space at the end! Remove this
# new_entries = []
# for i,val in enumerate(read_v2_drugs_bnf.read_code):
#     new_entries.append(str(val)[:-1])
# read_v2_drugs_bnf.read_code = new_entries
    
def cv2_with_nan_bnf():
    ### CV2 with NaN BNF Section - merge CV2 codes with CV2 drug names; use drug names to merge BNF drug classes
    # merge CV2/NoBNF with CV2 drug lookup table

    # Some ReadV2 codes in med have '00' at the end! Remove this from the subset we will be merging
    new_entries = []
    for i,val in enumerate(val_cv2_nan_bnf.read_2):
        if str(val)[-2:] == '00':
            new_entries.append(str(val)[:-2])
        else:
            new_entries.append(val)
    val_cv2_nan_bnf.read_2 = new_entries

    merge_val_cv2_nan_bnf = val_cv2_nan_bnf.merge(read_v2_drugs_lkp, how='inner', left_on='read_2', right_on='read_code')
    # merge_val_cv2_nan_bnf = val_cv2_nan_bnf.merge(read_v2_drugs_bnf, how='left', left_on='read_2', right_on='read_code')

    # take first word of CV2 drug name and make it uppercase
    merge_val_cv2_nan_bnf['first_word'] = merge_val_cv2_nan_bnf.term_description.str.split().str.get(0).str.upper()
    # create .groups dict of uppercase first words
    prescrip_term_indices = merge_val_cv2_nan_bnf.groupby('first_word').groups

    # convert bnf_lkp Presentation column (drug + dose) to uppercase and create .groups dict
    bnf_lkp.BNF_Presentation = bnf_lkp.BNF_Presentation.str.upper()
    bnf_term_indices = bnf_lkp.groupby('BNF_Presentation').groups

    del val_cv2_nan_bnf

    # create BNF_Paragraph column, pre-fill with NaNs
    merge_val_cv2_nan_bnf['BNF_Paragraph'] = np.nan

    # list of terms to iterate over and fill BNF Paragraphs for
    prescrip_term = list(prescrip_term_indices.keys())

    for i,term in enumerate(prescrip_term):
        if i%1000==0:
            with open('log.txt', 'w') as f:
                f.write(f'{i} terms done out of {len(prescrip_term)}')
                f.write('\n')
        
        # if term has * in front, remove
        if term[0] == "*":
            search = term[1:]
        else:
            search = term
        
        # find BNF Presentations with search term
        bnf_search_res = [val for key, val in bnf_term_indices.items() if search in key]
        
        # combine indices into one list and subset
        flat_list = [item for sublist in bnf_search_res for item in sublist]
        drug_subset = bnf_lkp.iloc[flat_list]
        
        # check if there 0, 1, or >1 BNF_Paragraph values
        # Note: only 8.55% end up with 'no results', good fill rate!
        if len(set(drug_subset.BNF_Paragraph))>1:
            # print(term)
            # cv2_bnf_paragraph.append(drug_subset.BNF_Paragraph.unique())
            merge_val_cv2_nan_bnf.loc[prescrip_term_indices[term], 'BNF_Paragraph'] = str(drug_subset.BNF_Paragraph.unique())
        elif len(set(drug_subset.BNF_Paragraph))==0:
            merge_val_cv2_nan_bnf.loc[prescrip_term_indices[term], 'BNF_Paragraph'] = 'no results'
        else:
            # cv2_bnf_paragraph.append(drug_subset.BNF_Paragraph.values[0])
            merge_val_cv2_nan_bnf.loc[prescrip_term_indices[term], 'BNF_Paragraph'] = drug_subset.BNF_Paragraph.values[0]

    merge_val_cv2_nan_bnf.to_parquet('../tidy_data/merge_val_cv2_nan_bnf.parquet')

def bnf_fill_paragraph():
    ### BNF Section - merge BNF codes with BNF drug classes
    bnf_fill = pd.concat([val_both, val_bnf_nan_cv2])

    # Create pandas groups for the codes in the data 
    bnf_fill_code_indices = bnf_fill.groupby('bnf_code').groups
    # list of terms to iterate over and fill BNF Paragraphs for
    bnf_fill_code_keys = list(bnf_fill_code_indices.keys())

    # Create pandas groups for the codes in the lookup 
    bnf_presentation_code_indices = bnf_lkp.groupby('BNF_Presentation_Code').groups

    # create BNF_Paragraph column, pre-fill with NaNs
    bnf_fill['BNF_Paragraph'] = np.nan

    # Many BNF codes in the data have periods and 2 characters at the end, and this formatting 
    # differs with the lookup file. To remedy this, remove the periods and last two characters 
    # of all BNF codes in the data that have periods, and update the pandas groups
    for i,term in enumerate(bnf_fill_code_keys):
        if '.' in term:
            changeto = term.replace('.', '')
            changeto = changeto[:-2]
            bnf_fill.iloc[bnf_fill_code_indices[term], 'bnf_code'] = changeto

    for i,term in enumerate(bnf_fill_code_keys):
        if i%1000==0:
            with open('log.txt', 'w') as f:
                f.write(f'{i} codes done out of {len(bnf_fill_code_keys)}')
                f.write('\n')
        
        if '.' in term:
            term = term.replace('.', '')
            search = term[:-2]
        else:
            search = term
        # remove last two charactere of code because they prevent the search from working
        # search = term[:-2]

        # # if term has * in front, remove
        # if term[0] == "*":
        #     search = term[1:]
        # else:
        #     search = term
        
        # find BNF Presentations with search term
        bnf_search_res = [val for key, val in bnf_presentation_code_indices.items() if search in key]
        
        # combine indices into one list and subset
        flat_list = [item for sublist in bnf_search_res for item in sublist]
        drug_subset = bnf_lkp.iloc[flat_list]
        
        # check if there 0, 1, or >1 BNF_Paragraph values
        if len(set(drug_subset.BNF_Paragraph))>1:
            # print(term)
            # cv2_bnf_paragraph.append(drug_subset.BNF_Paragraph.unique())
            bnf_fill.loc[bnf_fill_code_indices[term], 'BNF_Paragraph'] = str(drug_subset.BNF_Paragraph.unique())
        elif len(set(drug_subset.BNF_Paragraph))==0:
            bnf_fill.loc[bnf_fill_code_indices[term], 'BNF_Paragraph'] = 'no results'
        else:
            # cv2_bnf_paragraph.append(drug_subset.BNF_Paragraph.values[0])
            bnf_fill.loc[bnf_fill_code_indices[term], 'BNF_Paragraph'] = drug_subset.BNF_Paragraph.values[0]

    bnf_fill.to_parquet('../tidy_data/bnf_fill_paragraph.parquet')

bnf_fill_paragraph()

# cols = ['eid','bnf_code','drug_name','quantity','issue_date','read_2','BNF_Paragraph']
# sub_merge_val_cv2_nan_bnf = merge_val_cv2_nan_bnf.loc[:,cols]
# sub_bnf_fill = bnf_fill.loc[:,cols]
# final_med = pd.concat([sub_merge_val_cv2_nan_bnf, sub_bnf_fill])