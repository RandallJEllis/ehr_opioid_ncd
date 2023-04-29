import pandas as pd

med = pd.read_parquet('../tidy_data/med.parquet')
# drug_indices = med.groupby('drug_name').groups

# bnf_lkp = pd.read_parquet('../tidy_data/bnf_lkp.parquet')
# for col in bnf_lkp.columns:
#     bnf_lkp[col] = bnf_lkp[col].str.upper()
# bnf_paragraph_indices = bnf_lkp.groupby('BNF_Paragraph').groups

# # from: '../../old/MSDW-1794_Randy_Ellis_V2/code/Untitled.ipynb
# antidementia_rx = 'GALANTAMINE|DONEPEZIL|TACRINE|MEMANTINE|IPIDACRINE|GINKGO FOLIUM|ADUCANUMAB|RIVASTIGMINE|NAMENDA|EXELON|ARICEPT|NEIROMIDIN|AMIRIDINE|ADUHELM|COGNEX|RAZADYNE'.split('|')
# addict_rx = 'NALTREXONE|VARENICLINE|LEVACETYLMETHADOL|CALCIUM CARBIMIDE|LEVOMETHADONE|DISULFIRAM|ACAMPROSATE|BUPRENORPHINE, COMBINATIONS|NICOTINE|BUPRENORPHINE|DIAMORPHINE|LOFEXIDINE|CYTISINICLINE|NALMEFENE|METHADONE|SUBOXONE|NARCAN|ZUBSOLV|BELBUCA|SUBLOCADE|METHADOSE|SUBUTEX'.split('|')
# opioid_rx = 'HYDROMORPHONE|LEVORPHANOL|DEXTROPROPOXYPHENE|MEPTAZINOL|PROPOXYPHENE|HYDROCODONE|METHADONE|NALTREXONE|LEVACETYLMETHADOL|BEZITRAMIDE|HEROIN|OLICERIDINE|NALBUPHINE|MEPERIDINE|TAPENTADOL|CODEINE|PAPAVERETUM|NALOXONE|MORPHINE|TRAMADOL|FENTANYL|PENTAZOCINE|PIRITRAMIDE|PHENAZOCINE|KETOBEMIDONE|DEXTROMORAMIDE|BUTORPHANOL|NICOMORPHINE|OXYMORPHONE|LEVOMETHADONE|OXYCODONE|DIHYDROCODEINE|OPIUM|PETHIDINE|BUPRENORPHINE|TILIDINE|DIAMORPHINE|DEZOCINE|OXYCONTIN|VICODIN|REMIFENTANIL|PERCOCET|ENDOCET|SUFENTANIL|ULTRAM|DILAUDID|MS CONTIN|HYDROMET|DURAGESIC|ULTRACET|NUCYNTA|XTAMPZA|NORCO|ROXANOL|DARVOCET|VICOPROFEN|DEMEROL|ROXICODONE|BUTRANS|KADIAN|ROXICET|ALFENTANIL|OPANA|SUBSYS|MSIR|LORTAB|TUSSIONEX|ACTIQ|HYDROCOD|AVINZA|MORPHABOND'.split('|')

# def unique_bnf_drugs(search):
#     bnf_search_res = [val for key, val in bnf_paragraph_indices.items() if search in key]

#     # combine indices into one list and subset
#     flat_list = [item for sublist in bnf_search_res for item in sublist]
#     drug_subset = bnf_lkp.iloc[flat_list]

#     return drug_subset

# def subset_prescriptions(drugs):
#     search_res=[]
#     for rx in drugs:
#         search_res.extend([val for key, val in drug_indices.items() if rx in key])
#     flat_list = [item for sublist in search_res for item in sublist]
#     rx_df = med.iloc[flat_list]
#     return rx_df

# ### Opioids
# opioids = unique_bnf_drugs('OPIOID')
# bnf_opi = opioids[(~opioids.BNF_Paragraph.str.contains('NON-OPIOID')) &
#            (~opioids.BNF_Paragraph.str.contains('ANTAG'))]
# bnf_opi_drugs = list(set(bnf_opi.BNF_Product))

# # combine MSDW list of drugs with BNF drugs in appropriate class
# opioid_rx.extend(bnf_opi_drugs)

# # final subset of opioid prescriptions
# opioid_df = subset_prescriptions(opioid_rx)

# ReadV2 for opioid prescriptions begins with dj; BNF begins with 040702 or 041003
opioid_df = med[(med.read_2.str.startswith('dj', na=False)) | 
                (med.bnf_code.str.startswith('040702', na=False)) | 
                (med.bnf_code.str.startswith('041003', na=False)) | 
                (med.bnf_code.str.startswith('04.07.02', na=False)) |
                (med.bnf_code.str.startswith('04.10.03', na=False))
                ]
opioid_df.to_parquet('../tidy_data/opioid_med.parquet')

### Anti-dementia drugs
# bnf_antidementia_df = unique_bnf_drugs('DEMENTIA')
# bnf_antidementia_drugs = list(set(bnf_antidementia_df.BNF_Product))

# # combine MSDW list of drugs with BNF drugs in appropriate class
# antidementia_rx.extend(bnf_antidementia_drugs)

# antidementia_rx_df = subset_prescriptions(antidementia_rx)

#TODO: Find BNF codes for antidementia drugs
# ReadV2 for antidementia prescriptions begins with dy and dB; BNF begins with -----        
antidementia_df = med[(med.read_2.str.startswith('dy|dB', na=False)) | (med.bnf_code.str.startswith('040702', na=False)) | 
            (med.bnf_code.str.startswith('041003', na=False)) | (med.bnf_code.str.startswith('04.07.02', na=False)) |
            (med.bnf_code.str.startswith('04.10.03', na=False))]
antidementia_df.to_parquet('../tidy_data/ncd_med.parquet')
