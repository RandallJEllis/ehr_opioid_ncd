import pandas as pd
import os
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pickle
import statsmodels.api as sm
import seaborn as sns
from collections import Counter
import scipy.stats as st
from scipy.stats import gaussian_kde

plt.style.use('default')
plt.rcParams["font.weight"] = "bold"
plt.rcParams["font.size"] = 14
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = 'bold'


def importdata(ehr, predictor, outcome):
    """Import data from the EHR and return a pandas dataframe.

    Parameters
    ----------
    ehr : str
        'sinai' or 'ukb'
    predictor : str
        'binary' or 'prescription'
    outcome : str
        'binary' or 'age_onset'

    Returns
    -------
    df : pandas.DataFrame
        Dataframe with the predictor and outcome variables.
    """
    if ehr == 'sinai':
        file_ehr = 'MSDW1794_V3'
    elif ehr == 'ukb':
        file_ehr = 'ukbiobank'
    else:
        raise ValueError('Invalid EHR: %s' % ehr)

    if predictor == 'binary':
        file_pred = 'binary_exposure'
    elif predictor == 'prescription':
        file_pred = 'prescription_count'
    elif predictor == 'aud':
        file_pred = 'aud'
    else:
        raise ValueError('Invalid predictor: %s' % predictor)

    if outcome == 'binary':
        file_out = 'binary_outcome'
    elif outcome == 'age_onset':
        file_out = 'age_onset_ncd'
    else:
        raise ValueError('Invalid outcome: %s' % outcome)

    if file_pred == 'aud':
        if ehr=='sinai': path = f"{file_ehr}/voe_outputs/aud/controlsNoAUDDX/{file_out}/controlVarOUD/analyses/period_summaries/"
        elif ehr=='ukb': path = f'{file_ehr}/voe_outputs/aud/controlsNoAUDDX/{file_out}/controlVarSUD/analyses/period_summaries/'

    else:
        path = f"{file_ehr}/voe_outputs/opioids/controlsLessThan3Opioids/{file_pred}/{file_out}/controlVarOUD/analyses/period_summaries/"
        
    datasets = []

    for enrollment_year in range(1989,2020):
        if os.path.exists(path + f'voe_{enrollment_year}_{enrollment_year+3}.csv'):
            ds = pd.read_csv(path + f'voe_{enrollment_year}_{enrollment_year+3}.csv')
            datasets.append(ds)
    allexpts = pd.concat(datasets)
    print(allexpts.shape)

    #check for duplicate experiments
    allexpts = allexpts.drop_duplicates()
    print(allexpts.shape)

    #remove years with low sample sizes
    print(allexpts.shape)
    if ehr == 'sinai':
        allexpts = allexpts[(allexpts.start_enroll>=2008) & 
            (allexpts['hx_MAT']==0)
            ]
    elif ehr == 'ukb':
        allexpts = allexpts[(allexpts.start_enroll>=2004) & (allexpts.start_enroll<=2010) &
            (allexpts['hx_MAT']==0) #& (allexpts['hx_aud']==0) 
            ]

    print(allexpts.shape)

    # create total N column
    allexpts['total_N'] = allexpts['control_N'] + allexpts['opioid_N']

    #add corrected p-values, OR confidence intervals, percentage of each group with NCD
    if 'age_onset_ncd' not in path:
        allexpts['coef'] = np.exp(allexpts['coef'])
        allexpts['.025'] = np.exp(allexpts['.025'])
        allexpts['.975'] = np.exp(allexpts['.975'])

    # correct for multiple comparisons
    allexpts['bonferroni'] = sm.stats.multipletests(allexpts['p'], alpha=0.05, method='bonferroni')[1]
    allexpts['bh_p'] = sm.stats.multipletests(allexpts['p'], alpha=0.05, method='fdr_bh')[1]

    # percentage of NCD for each group
    allexpts['opi_percent_ncd'] = 100 * (allexpts['num_opioid_ncd'] / allexpts['opioid_N'])
    allexpts['con_percent_ncd'] = 100 * (allexpts['num_control_ncd'] / allexpts['control_N'])

    # percentage of sex for total sample
    allexpts['total_female%'] = ((allexpts['control_female%'] * allexpts['control_N']) + (allexpts['opioid_female%'] * allexpts['opioid_N'])) / allexpts['total_N']

    # mean age for total sample
    allexpts['total_mean_age'] = ((allexpts['control_AgeMean'] * allexpts['control_N']) + (allexpts['opioid_AgeMean'] * allexpts['opioid_N'])) / allexpts['total_N']

    df = allexpts.copy()
    return df, path, file_ehr

def get_summary(predictor, outcome):
    sinai = importdata('sinai', predictor, outcome)
    ukb = importdata('ukb', predictor, outcome)
    df = pd.concat([sinai, ukb])
    print(predictor, outcome, '\n',
    'N', min(df.total_N), max(df.total_N), '\n',
    'Female %', min(df['total_female%']), max(df['total_female%']), '\n',
    'Mean age', min(df['total_mean_age']), max(df['total_mean_age']), '\n',
    'Coef', f'Min: {min(df.coef)},', f'Max: {max(df.coef)},', f'Median: {np.median(df.coef)}', '\n',
    '% significant', sum(df.bh_p<0.05)/len(df), f'Median p: {np.median(df.bh_p)}','\n',
    )
    return df
    
def summ_stats(coefs, pvals):
    '''Summary statistics:
    including the 1st, 50th (median), and 99th percentile of effect size and p-values
    the “relative odds ratio” (Rcoef) as the ratio of the 99th and 1st percentile odds ratio
    the “relative p-value” (RP) as the difference between the 99th and 1st percentile of -log10(adjusted p-value). '''

    #calculate relative effect size
    coefs = sorted(abs(coefs))
    rcoef = round(coefs[int(len(coefs)*.99)] / coefs[int(len(coefs)*.01)],2)
    
    #calculate relative p-value    
    ps = sorted(pvals)
    rp = round(ps[int(len(ps)*.99)] - ps[int(len(ps)*.01)],2)

    return rcoef, rp

def marker_scale(df):
    #min-max scale marker size based on total N of experiments for plotting and legend
    markermin = 5
    markermax = 100
    df['markerscale'] = (markermax-markermin) * ((df['total_N'] - min(df['total_N'])) / \
                                                   (max(df['total_N']) - min(df['total_N']))) + markermin
    df.markerscale = pd.to_numeric(df.markerscale)
    return df,markermin,markermax

def add_legends(df, ax, p1, legend1_label, markermin, markermax):
    #Legend colors/groups
    legend1 = ax.legend(loc=(1.02, 0.5), title=legend1_label, prop={'size': 11})
    for lh in legend1.legendHandles: 
        lh.set_alpha(1)
    ax.add_artist(legend1)

    #Legend showing sample sizes
    kw = dict(prop="sizes", num=5,  fmt="{x:.0f}",
              func=lambda s: s/(markermax-markermin) * (\
                                                      (max(df['total_N']) - min(df['total_N']))\
                                                    ) - markermin + min(df['total_N']))
    legend2 = ax.legend(*p1.legend_elements(**kw),#handles, labels,
                               loc=(1.02, 0), title="Total N", prop={'size': 9})
    ax.add_artist(legend2)
    
    
def axis_lines_labels(coefs, pvals, pcol, xlab):
    
    #vertical dotted line showing median coef
    range_pvals = max(pvals) - min(pvals)
    plt.vlines(np.median(coefs), ymin=\
                     np.median(pvals)-(0.13*range_pvals), #np.median(-np.log10(r[p_col]))-7,#
               ymax=\
                     np.median(pvals)+(0.13*range_pvals), #np.median(-np.log10(r[p_col]))+7,#
                     linestyles='dashed', color='k')
    
    #horizontal dotted line showing median p-value  
    range_coefs = (max(coefs) - min(coefs))
    plt.hlines(np.median(pvals), xmin=\
                     np.median(coefs)-(0.13*range_coefs), #np.median(r.coef)-2,#
               xmax=\
                     np.median(coefs)+(0.13*range_coefs), #np.median(r.coef)+2,#
                     linestyles='dashed', color='k')  
    
    #axis labels
    plt.ylabel('-log$_{10}$ adj. p-value', fontsize=14)
    plt.xlabel(xlab, fontsize=14)
    
    #mins for axis limits and axis lines
    coef_range = max(coefs) - min(coefs)
    xlim_xmin = min(coefs)-0.1*coef_range
    xlim_xmax = max(coefs)+0.1*coef_range
    hlines_xmin = min(coefs)-0.05*coef_range
    hlines_xmax = max(coefs)+0.05*coef_range
        
    #axis lines
    #if plotting odds ratios (binary outcome)
    if 'age_onset_ncd' not in path:
        if max(coefs) < 1:
            xlim_xmax = 1 + (0.1*coef_range)
            plt.hlines(y=-np.log10(0.05),xmin=hlines_xmin, xmax=1 + (0.07*coef_range), color='black') # type: ignore
        elif min(coefs) > 1:
            xlim_xmin = 1 - (0.1*coef_range)
            plt.hlines(y=-np.log10(0.05),xmin=1 - (0.07*coef_range), xmax=hlines_xmax, color='black')
        else:
            plt.hlines(y=-np.log10(0.05),xmin=hlines_xmin, xmax=hlines_xmax, color='black')
        plt.vlines(1, ymin=0, ymax=max(pvals)+0.5, color='black')

    #if plotting betas (age onset outcome)
    elif 'age_onset_ncd' in path:
        if max(coefs) < 0:
            xlim_xmax = 0 + (0.1*coef_range)
            plt.hlines(y=-np.log10(0.05),xmin=hlines_xmin, xmax=0 + (0.07*coef_range), color='black')
        elif min(coefs) > 0:
            xlim_xmin = 0 - (0.1*coef_range)
            plt.hlines(y=-np.log10(0.05),xmin=0 - (0.07*coef_range), xmax=hlines_xmax, color='black')
        else:
            plt.hlines(y=-np.log10(0.05),xmin=hlines_xmin, xmax=hlines_xmax, color='black')
        plt.vlines(0, ymin=0, ymax=max(pvals)+0.5, color='black')
    
    
    #axis limits
    plt.xlim((xlim_xmin, xlim_xmax))
    plt.ylim((-0.1,max(pvals)+0.05*range_pvals))

def add_title():
    #title
    res,rp = summ_stats(df.coef, -np.log10(df.bh_p))
    plt.title(f"RES: {res}\nRP: {rp}", size=12)
    
def formatting(df, pcol, ax, p1, legend1title, markermin, markermax, xlab):
    #axis lines for significance and division of increased/decreased risk
    axis_lines_labels(df.coef, -np.log10(df[pcol]), pcol, xlab)
    add_legends(df, ax, p1, legend1title, markermin, markermax)
    add_title()

def plot_by_controls(df,pcol,legend1title, xlab, alpha):
    df,markermin,markermax = marker_scale(df)
    
    if sum(df.hx_sud_covar)>0:
        nocon = df[(df.hx_aud==0) & (df.hx_sud_covar==0) & (df.hx_tobacco==0)]

        aud = df[(df.hx_aud==1) & (df.hx_sud_covar==0) & (df.hx_tobacco==0)]
        sud = df[(df.hx_aud==0) & (df.hx_sud_covar==1) & (df.hx_tobacco==0)]
        tobacco = df[(df.hx_aud==0) & (df.hx_sud_covar==0) & (df.hx_tobacco==1)]
        
        aud_sud = df[(df.hx_aud==1) & (df.hx_sud_covar==1) & (df.hx_tobacco==0)]
        aud_tobacco = df[(df.hx_aud==1) & (df.hx_sud_covar==0) & (df.hx_tobacco==1)]
        sud_tobacco = df[(df.hx_aud==0) & (df.hx_sud_covar==1) & (df.hx_tobacco==1)]
        aud_sud_tobacco = df[(df.hx_aud==1) & (df.hx_sud_covar==1) & (df.hx_tobacco==1)]

    #     print(df.shape)
    #     print(nocon.shape[0]+aud.shape[0]+sud.shape[0]+tobacco.shape[0]+aud_sud.shape[0]+aud_tobacco.shape[0]+\
    #           sud_tobacco.shape[0]+aud_sud_tobacco.shape[0])
    # #     print(nocon.shape[0]+sud.shape[0])
    #     print('NoCon', round(np.mean(nocon.coef),4), round(np.mean(nocon['.025']),4), round(np.mean(nocon['.975']),4), 
    #           'AUD', round(np.mean(aud.coef),4), 
    #           'SUD', round(np.mean(sud.coef),4), 
    #           'Smoking', round(np.mean(tobacco.coef),4), 
    #           'AUD_SUD', round(np.mean(aud_sud.coef),4),
    #          'AUD_Smok', round(np.mean(aud_tobacco.coef),4), 
    #           'SUD_Smok', round(np.mean(sud_tobacco.coef),4), 
    #           'All', round(np.mean(aud_sud_tobacco.coef),4)
    #                       )
        
        fig, ax = plt.subplots()
        for ds,lab,col in zip([nocon, sud, tobacco, aud, sud_tobacco, aud_sud, aud_tobacco, aud_sud_tobacco],
                            ['No Controls','OUD','Smoking','AUD','OUD+Smok','OUD+AUD', 'Smok+AUD',  'All'],
                            ['white',      'red','yellow','blue','orange', 'purple', 'green',  'black']
                            ):
            p1 = ax.scatter(ds.coef, -np.log10(ds[pcol]), alpha=alpha, s=ds['markerscale'].values, 
                            edgecolor='k', linewidth=0.5,label=lab,c=col)
    else:
        nocon = df[(df.hx_aud==0) &  (df.hx_tobacco==0)]

        aud = df[(df.hx_aud==1) &  (df.hx_tobacco==0)]
        sud = df[(df.hx_aud==0) &  (df.hx_tobacco==0)]
        tobacco = df[(df.hx_aud==0) &  (df.hx_tobacco==1)]
        
        aud_sud = df[(df.hx_aud==1) &(df.hx_tobacco==0)]
        aud_tobacco = df[(df.hx_aud==1) &  (df.hx_tobacco==1)]
        sud_tobacco = df[(df.hx_aud==0) &  (df.hx_tobacco==1)]
        aud_sud_tobacco = df[(df.hx_aud==1)  & (df.hx_tobacco==1)]
        
        fig, ax = plt.subplots()
        for ds,lab,col in zip([nocon, tobacco, aud,  aud_tobacco, aud_sud_tobacco],
                            ['No Controls','Smoking','AUD', 'Smok+AUD',  'All'],
                            ['white',      'yellow','blue', 'green',  'black']
                            ):
            p1 = ax.scatter(ds.coef, -np.log10(ds[pcol]), alpha=alpha, s=ds['markerscale'].values, 
                            edgecolor='k', linewidth=0.5,label=lab,c=col)
    formatting(df, pcol, ax, p1, legend1title, markermin, markermax, xlab)
    
def aud_plot_by_controls(df, pcol,legend1title, xlab, alpha):
    df,markermin,markermax = marker_scale(df)
    
    nocon = df[(df.hx_sud_covar==0) & (df.hx_tobacco==0)]
    aud = df[(df.hx_sud_covar==0) & (df.hx_tobacco==0)]
    sud = df[(df.hx_sud_covar==1) & (df.hx_tobacco==0)]
    tobacco = df[(df.hx_sud_covar==0) & (df.hx_tobacco==1)]
    
    aud_sud = df[(df.hx_sud_covar==1) & (df.hx_tobacco==0)]
    aud_tobacco = df[(df.hx_sud_covar==0) & (df.hx_tobacco==1)]
    sud_tobacco = df[(df.hx_sud_covar==1) & (df.hx_tobacco==1)]
    aud_sud_tobacco = df[(df.hx_sud_covar==1) & (df.hx_tobacco==1)]

    print('NoCon', round(np.mean(nocon.coef),4), round(np.mean(nocon['.025']),4), round(np.mean(nocon['.975']),4), 
          'SUD', round(np.mean(sud.coef),4), 
          'Smoking', round(np.mean(tobacco.coef),4), 
          'SUD_Smok', round(np.mean(sud_tobacco.coef),4), 
                      )
    
    fig, ax = plt.subplots()
    for ds,lab,col in zip([nocon, sud, tobacco, sud_tobacco,],
                          ['No Controls','SUD','Smoking','SUD+Smok'],
                          ['white',      'red','blue','purple']
                         ):
        p1 = ax.scatter(ds.coef, -np.log10(ds[pcol]), alpha=alpha, s=ds['markerscale'].values, 
                        edgecolor='k', linewidth=0.5,label=lab,c=col)
    formatting(df, pcol, ax, p1, legend1title, markermin, markermax, xlab)
    

def plot_by_opioid_enroll(df,pcol,legend1title, xlab, alpha):
    df,markermin,markermax = marker_scale(df)
    op5 = df[(df.opioid_rx_enroll==5)]
    op10 = df[(df.opioid_rx_enroll==10)]
    op15 = df[(df.opioid_rx_enroll==15)]

    fig, ax = plt.subplots()
    for ds,lab,col in zip([op5, op10, op15],['5 Rx','10 Rx','15 Rx'], ['#00FF59','#FF5900','#5900FF']):
        p1 = ax.scatter(ds.coef, -np.log10(ds[pcol]), alpha=alpha, c=col,
                        edgecolor='k', linewidth=0.5, s=ds['markerscale'].values, label=lab)
        
    formatting(df, pcol, ax, p1, legend1title, markermin, markermax, xlab)
    
def plot_by_ncd_age_exclusion(df,pcol,legend1title, xlab, alpha):
    df,markermin,markermax = marker_scale(df)

    age45 = df[(df.ncd_age_threshold==45)]
    age55 = df[(df.ncd_age_threshold==55)]
    age65 = df[(df.ncd_age_threshold==65)]
    
    fig, ax = plt.subplots()
    for ds,lab,col in zip([age45,age55,age65], ['45','55','65'], ['#0069FF','#69FF00','#FF0069']):
        p1 = ax.scatter(ds.coef, -np.log10(ds[pcol]), alpha=alpha,
                        s=ds['markerscale'].values, edgecolor='k', linewidth=0.5, label=lab, c=col)

    formatting(df, pcol, ax, p1, legend1title, markermin, markermax, xlab)    
    
def plot_by_enroll_year(df,pcol,legend1title, xlab, alpha):
    df,markermin,markermax = marker_scale(df)

    enroll_years = sorted(list(set(df.start_enroll)))
    
    #plot each year's data in a loop
    fig, ax = plt.subplots()
    for ey in enroll_years:
        ey_df = df[(df.start_enroll==ey)]
        p1 = ax.scatter(ey_df.coef, -np.log10(ey_df[pcol]), alpha=alpha,
                        s=[float(x) for x in ey_df['markerscale'].values], edgecolor='k', linewidth=0.5,
                        label=f'{str(ey)}-{str(ey+2)}')
        
    formatting(df, pcol, ax, p1, legend1title, markermin, markermax, xlab)
        
        
def density_plot(df, pcol, legend1title, xlab, alpha):
    
    df,markermin,markermax = marker_scale(df)
    
    # set x,y data
    x = df.coef
    y = -np.log10(df[pcol])

    # Calculate the point density
    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)

    fig, ax = plt.subplots()
    p1 = ax.scatter(x, y, alpha=alpha, c=z, cmap=plt.cm.jet,s=df['markerscale'].values, edgecolor='k', linewidth=0.5,
                        label='Density')
    formatting(df, pcol, ax, p1, legend1title, markermin, markermax, xlab)


def final_plot(df, path, file_ehr):
    # if 'opioids/controlsLessThan3Opioids/prescription_count/binary_outcome/' in path:
    #     df = df[df.coef>0.99]


    if 'binary_exposure' in path:
        predictor = 'binaryExposure'
        if 'age_onset_ncd' in  path:
            outcome = 'ageOnsetNCD'
            xlab = 'Effect of OPRx (Yes/No) on Age of NCD Onset (Years)'#binary exposure, age of onset

        elif 'binary_outcome' in path:
            outcome = 'oddsratioNCD'
            xlab = 'Effect of OPRx (Yes/No) on Odds Ratio' #binary exposure, binary outcome
            
    elif 'prescription_count' in path:
        predictor = 'prescriptionCount'
        if 'age_onset_ncd' in  path:
            outcome = 'ageOnsetNCD'
            xlab = r'Effect of Individual OPRx on Age of NCD Onset (Years)' #prescription count, age of onset
        elif 'binary_outcome' in path:
            outcome = 'oddsratioNCD'
            xlab = r'Effect of Individual OPRx on Odds Ratio' #prescription count, binary outcome
    else:
        predictor = ''
        if 'binary_outcome' in path:
            outcome = 'oddsratioNCD'
            xlab = 'Effect of AUD (Yes/No) on Odds Ratio'
        elif 'age_onset_ncd' in path:
            outcome = 'ageOnsetNCD'
            xlab = 'Effect of AUD (Yes/No) on Age of NCD Onset (Years)'


    if '/aud/' in path:
        drug = 'AUD'
        alpha = 0.8
    else:
        drug = ''
        alpha = 0.2
        
    pcol = 'bh_p'

    colorby='enrollYear'
    plot_by_enroll_year(df,pcol,'Enrollment', xlab, alpha)
    plt.tight_layout()
    plt.savefig(f'{file_ehr}/figures/{drug}{predictor}_{outcome}_colorBy{colorby}.png', bbox_inches='tight', dpi=300)

    colorby='controlVars'
    if '/aud/' not in path:
        plot_by_controls(df,pcol,'Controls',xlab,alpha)
    else:
        aud_plot_by_controls(df,pcol,'Controls',xlab,alpha)
    plt.tight_layout()
    plt.savefig(f'{file_ehr}/figures/{drug}{predictor}_{outcome}_colorBy{colorby}.png', bbox_inches='tight', dpi=300)

    if '/aud/' not in path:
        colorby='OPRxEnroll'
        plot_by_opioid_enroll(df,pcol,'OPRx Enrollment', xlab,alpha)
        plt.tight_layout()
        plt.savefig(f'{file_ehr}/figures/{drug}{predictor}_{outcome}_colorBy{colorby}.png', bbox_inches='tight', dpi=300)

    colorby='NCDAgeExclusion'
    plot_by_ncd_age_exclusion(df,pcol,'NCD Age Exclusion',xlab,alpha)
    plt.tight_layout()
    plt.savefig(f'{file_ehr}/figures/{drug}{predictor}_{outcome}_colorBy{colorby}.png', bbox_inches='tight', dpi=300)

    colorby='Density'
    density_plot(df,pcol,'None',xlab,alpha)
    plt.tight_layout()
    plt.savefig(f'{file_ehr}/figures/{drug}{predictor}_{outcome}_colorBy{colorby}.png', bbox_inches='tight', dpi=300)

def update_figures(ehr, predictor, outcome):
    df, path, file_ehr = importdata(ehr, predictor, outcome)
    final_plot(df, path, file_ehr)
    
if __name__ == "__main__":
    update_figures('sinai', 'binary', 'binary')
    update_figures('sinai', 'binary', 'age_onset')
    update_figures('sinai', 'prescription', 'binary')
    update_figures('sinai', 'prescription', 'age_onset')
    update_figures('sinai', 'aud', 'binary')
    update_figures('sinai', 'aud', 'age_onset')
    update_figures('ukb', 'binary', 'binary')
    update_figures('ukb', 'binary', 'age_onset')
    update_figures('ukb', 'prescription', 'binary')
    update_figures('ukb', 'prescription', 'age_onset')
    update_figures('ukb', 'aud', 'binary')
    update_figures('ukb', 'aud', 'age_onset')