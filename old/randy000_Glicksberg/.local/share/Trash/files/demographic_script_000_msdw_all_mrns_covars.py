import pandas as pd
import pickle

demographic = pd.read_csv('projects/person_detail_cleaned.csv', index_col=0)
demographic.drop(columns=['RELIGION'], inplace=True)
demographic.reset_index(drop=True, inplace=True)
print('loaded data')
#Subset all patients with one or two+ rows, so that we don't have to iterate through patients with one row
mrn_rowCounts = demographic.MEDICAL_RECORD_NUMBER.value_counts() 
mrn_oneRow = mrn_rowCounts[mrn_rowCounts<2].index
mrn_twoOrMoreRows = mrn_rowCounts[mrn_rowCounts>1].index

demographic_twoOrMore = demographic[demographic['MEDICAL_RECORD_NUMBER'].isin(mrn_twoOrMoreRows)]
demographic_twoOrMore.reset_index(drop=True, inplace=True)
print('about to make idx')
idx = demographic_twoOrMore['MEDICAL_RECORD_NUMBER'].reset_index().groupby('MEDICAL_RECORD_NUMBER')['index'].apply(tuple).to_dict()
print('made idx')
mrn_values = list(idx.values())

demo_individuals = []
print('about to start loop')
count=0
for i in range(len(mrn_values)):
    if count%50000==0:
        print(count)


    mrn_demo = demographic_twoOrMore.iloc[list(mrn_values[i]), :]

    mrn_demo_clean = mrn_demo[(mrn_demo['RACE']!='Unknown') & (mrn_demo['GENDER']!='Unknown')]

    if mrn_demo_clean.shape[0] > 0:
        demo_individuals.append(mrn_demo_clean.iloc[0, :])
    else:
        demo_individuals.append(mrn_demo.iloc[0, :])

    count+=1

demo_twoOrMore_df = pd.DataFrame(demo_individuals)
demo_twoOrMore_df.to_csv('demo_twoOrMore.csv')
