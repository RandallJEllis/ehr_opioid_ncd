import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pickle
import statsmodels.api as sm
import seaborn as sns
from collections import Counter
import scipy.stats as st
import sys
from scipy.stats import gaussian_kde

#MATPLOTLIB SETTINGS
plt.style.use('default')
plt.rcParams["font.weight"] = "bold"
plt.rcParams["font.size"] = 14
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = 'bold'

#HELPER FUNCTIONS FOR FORMATTING PLOTS
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
    '''Scale marker size based on the sample size of each experiment'''

    #min-max scale marker size based on total N of experiments for plotting and legend
    markermin = 5
    markermax = 100
    df['markerscale'] = (markermax-markermin) * ((df['total_N'] - min(df['total_N'])) / \
                                                   (max(df['total_N']) - min(df['total_N']))) + markermin
    return df,markermin,markermax

def add_legends(df, ax, p1, legend1_label, markermin, markermax):
    '''Add two legends: One showing the color code, the other showing the relationship between marker size and sample size'''

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
    '''
    1. Add dotted lines denoting median effect size and p-value
    2. Add solid lines demarcating the directionality of effect sizes and statistical significance
    3. Add axis labels
    4. Set axis limits
    '''

    #vertical dotted line showing median coef
    range_pvals = max(pvals) - min(pvals)
    plt.vlines(np.median(coefs),
                ymin=np.median(pvals)-(0.13*range_pvals), 
                ymax=np.median(pvals)+(0.13*range_pvals),
                linestyles='dashed', color='k')
    
    #horizontal dotted line showing median p-value  
    range_coefs = (max(coefs) - min(coefs))
    plt.hlines(np.median(pvals), 
                xmin=np.median(coefs)-(0.13*range_coefs),
                xmax=np.median(coefs)+(0.13*range_coefs),
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
    #if plotting effect sizes as odds ratios (i.e., outcome is binary diagnosis of NCD)
    if 'age_onset_ncd' not in path:
        if max(coefs) < 1:
            xlim_xmax = 1 + (0.1*coef_range)
            plt.hlines(y=-np.log10(0.05),xmin=hlines_xmin, xmax=1 + (0.07*coef_range), color='black')
        elif min(coefs) > 1:
            xlim_xmin = 1 - (0.1*coef_range)
            plt.hlines(y=-np.log10(0.05),xmin=1 - (0.07*coef_range), xmax=hlines_xmax, color='black')
        else:
            plt.hlines(y=-np.log10(0.05),xmin=hlines_xmin, xmax=hlines_xmax, color='black')
        plt.vlines(1, ymin=0, ymax=max(pvals)+0.5, color='black')

    #if plotting effect sizes as beta coefficients (i.e., outcome is age of onset of NCD)
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
    plt.ylim((-0.1,max(pvals)+1))

def add_title():
    #title
    res,rp = summ_stats(df.coef, -np.log10(df.bh_p))
    plt.title(f"RES: {res}\nRP: {rp}", size=12)
    
def formatting(df, pcol, ax, p1, legend1title, markermin, markermax, xlab):
    '''Wrap axis_lines_labels, add_legends, and add_title all into one function'''

    #axis lines for significance and division of increased/decreased risk
    axis_lines_labels(df.coef, -np.log10(df[pcol]), pcol, xlab)
    add_legends(df, ax, p1, legend1title, markermin, markermax)
    add_title()

#DIFFERENT TYPES OF PLOTS
def plot_by_controls(df,pcol,legend1title, xlab, alpha):
    '''Color code experiments by control variables, only for OPRx experiments'''

    df,markermin,markermax = marker_scale(df)
    
    nocon = df[(df.hx_aud==0) & (df.hx_sud_covar==0) & (df.hx_tobacco==0)]
    aud = df[(df.hx_aud==1) & (df.hx_sud_covar==0) & (df.hx_tobacco==0)]
    sud = df[(df.hx_aud==0) & (df.hx_sud_covar==1) & (df.hx_tobacco==0)]
    tobacco = df[(df.hx_aud==0) & (df.hx_sud_covar==0) & (df.hx_tobacco==1)]
    
    # aud_sud = df[(df.hx_aud==1) & (df.hx_sud_covar==1) & (df.hx_tobacco==0)]
    # aud_tobacco = df[(df.hx_aud==1) & (df.hx_sud_covar==0) & (df.hx_tobacco==1)]
    # sud_tobacco = df[(df.hx_aud==0) & (df.hx_sud_covar==1) & (df.hx_tobacco==1)]
    aud_sud_tobacco = df[(df.hx_aud==1) & (df.hx_sud_covar==1) & (df.hx_tobacco==1)]

    print('NoCon', round(np.mean(nocon.coef),4), round(np.mean(nocon['.025']),4), round(np.mean(nocon['.975']),4), 
          'AUD', round(np.mean(aud.coef),4), 
          'SUD', round(np.mean(sud.coef),4), 
          'Smoking', round(np.mean(tobacco.coef),4), 
        #   'AUD_SUD', round(np.mean(aud_sud.coef),4),
        #  'AUD_Smok', round(np.mean(aud_tobacco.coef),4), 
        #   'SUD_Smok', round(np.mean(sud_tobacco.coef),4), 
          'All', round(np.mean(aud_sud_tobacco.coef),4)
                      )
    
    fig, ax = plt.subplots()
    for ds,lab,col in zip([nocon, sud, tobacco, aud, 
                            #sud_tobacco, aud_sud, aud_tobacco, 
                            aud_sud_tobacco],
                          ['No Controls','OUD','Smoking','AUD',#'OUD+Smok','OUD+AUD', 'Smok+AUD',
                            'All'],
                          ['white','red','yellow','blue',#'orange', 'purple', 'green',
                            'black']
                         ):
        p1 = ax.scatter(ds.coef, -np.log10(ds[pcol]), alpha=alpha, s=ds['markerscale'].values, 
                        edgecolor='k', linewidth=0.5,label=lab,c=col)
    formatting(df, pcol, ax, p1, legend1title, markermin, markermax, xlab)
    
def aud_plot_by_controls(df, pcol,legend1title, xlab, alpha):
    '''Color code experiments by control variables, only for AUD experiments'''

    df,markermin,markermax = marker_scale(df)
    
    nocon = df[(df.hx_sud_covar==0) & (df.hx_tobacco==0)]
    sud = df[(df.hx_sud_covar==1) & (df.hx_tobacco==0)]
    tobacco = df[(df.hx_sud_covar==0) & (df.hx_tobacco==1)]
    sud_tobacco = df[(df.hx_sud_covar==1) & (df.hx_tobacco==1)]

    print('NoCon', round(np.mean(nocon.coef),4), round(np.mean(nocon['.025']),4), round(np.mean(nocon['.975']),4), 
          'SUD', round(np.mean(sud.coef),4), 
          'Smoking', round(np.mean(tobacco.coef),4), 
          'SUD_Smok', round(np.mean(sud_tobacco.coef),4), 
                      )
    
    fig, ax = plt.subplots()
    for ds,lab,col in zip([nocon, sud, tobacco, sud_tobacco,],
                          ['No Controls','OUD','Smoking','OUD+Smok'],
                          ['white',      'red','blue','purple']
                         ):
        p1 = ax.scatter(ds.coef, -np.log10(ds[pcol]), alpha=alpha, s=ds['markerscale'].values, 
                        edgecolor='k', linewidth=0.5,label=lab,c=col)
    formatting(df, pcol, ax, p1, legend1title, markermin, markermax, xlab)
    

def plot_by_opioid_enroll(df,pcol,legend1title, xlab, alpha):
    '''Color code experiments by the number of opioid prescriptions for enrollment'''

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
    '''Color code experiments by the age of exclusion for pre-existing NCD'''

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
    '''Color code experiments by enrollment period'''

    df,markermin,markermax = marker_scale(df)

    enroll_years = sorted(list(set(df.start_enroll)))
    
    #plot each year's data in a loop
    fig, ax = plt.subplots()
    for ey in enroll_years:
        ey_df = df[(df.start_enroll==ey)]
        p1 = ax.scatter(ey_df.coef, -np.log10(ey_df[pcol]), alpha=alpha,
                        s=ey_df['markerscale'].values, edgecolor='k', linewidth=0.5,
                        label=f'{str(ey)}-{str(ey+2)}')
        
    formatting(df, pcol, ax, p1, legend1title, markermin, markermax, xlab)
        
        
def density_plot(df, pcol, legend1title, xlab, alpha):
    '''Color code experiments by the density of where they are plotted'''

    df,markermin,markermax = marker_scale(df)
    
    # set x,y data
    x = df.coef
    y = -np.log10(df.bh_p)

    # Calculate the point density
    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)

    fig, ax = plt.subplots()
    p1 = ax.scatter(x, y, alpha=alpha, c=z, cmap=plt.cm.jet,s=df['markerscale'].values, edgecolor='k', linewidth=0.5,
                        label='Density')
    formatting(df, pcol, ax, p1, legend1title, markermin, markermax, xlab)



#import data for the different enrollment periods
for parent,child in zip(['opioids', 'aud'],['controlsLessThan3Opioids','controlsNoAUDDX']):
    if parent=='opioids':
        subfolders = ['binary_exposure/age_onset_ncd','binary_exposure/binary_outcome',
                        'prescription_count/age_onset_ncd', 'prescription_count/binary_outcome']
    elif parent=='aud':
        subfolders = ['age_onset_ncd','binary_outcome']

    for sf in subfolders:
        path = f'../voe_outputs/{parent}/{child}/{sf}/controlVarOUD/analyses/'

        datasets = []
        for enrollment_year in [2008,2009,2010,2011,2012,2013,2014]:
            ds = pd.read_csv(path + f'voe_{enrollment_year}_{enrollment_year+3}.csv')
            datasets.append(ds)
        df = pd.concat(datasets)
        print(df.shape)

        #check for duplicate experiments
        df = df.drop_duplicates()
        print(df.shape)

        '''
        Based on the DAG, we are making the following changes:
        1. Hepatitis C will not be controlled for (no path to or from opioid prescriptions)
        2. History of medication-assisted therapy will not be controlled for (creates cycle in DAG)
        '''
        if '/aud/' not in path:
            df = df[(df.hx_MAT==0)]
        print(df.shape) #will cut number of experiments in half

        # add FDR-corrected p-values
        # transform coefficients into odds ratios for the binary outcome experiments
        # add columns for total N
        # add percentage of each group with NCD
        if 'age_onset_ncd' not in path:
            df['coef'] = np.exp(df['coef'])
            df['.025'] = np.exp(df['.025'])
            df['.975'] = np.exp(df['.975'])
        df['bonferroni'] = sm.stats.multipletests(df['p'], alpha=0.05, method='bonferroni')[1]
        df['bh_p'] = sm.stats.multipletests(df['p'], alpha=0.05, method='fdr_bh')[1]
        df['total_N'] = df['control_N'] + df['opioid_N']
        df['opi_percent_ncd'] = 100 * (df['num_opioid_ncd'] / df['opioid_N'])
        df['con_percent_ncd'] = 100 * (df['num_control_ncd'] / df['control_N'])
        df.shape#, expts_enrollSUDpts_onlyControlAUDnotSUD.shape
        df = df.copy()

        # assign strings for predictors and outcomes for figure filnames and x-axis labels
        if 'binary_exposure' in path:
            predictor = 'binaryExposure'
            if 'age_onset_ncd' in  path:
                outcome = 'ageOnsetNCD'
                xlab = 'Effect of OPRx (Yes/No) on Age of NCD Onset (Years)'#binary exposure, age of onset
            else:
                outcome = 'oddsratioNCD'
                xlab = 'Effect of OPRx (Yes/No) on Odds Ratio' #binary exposure, binary outcome
        else:
            predictor = 'prescriptionCount'
            if 'age_onset_ncd' in  path:
                outcome = 'ageOnsetNCD'
                xlab = r'$\beta$, Effect of Individual OPRx on Age of NCD Onset (Years)' #prescription count, age of onset
            else:
                outcome = 'oddsratioNCD'
                xlab = r'$\beta$, Effect of Individual OPRx on Odds Ratio' #prescription count, binary outcome

        # assign strings for drug for figure filenames and opacity of markers
        if '/aud/' in path:
            drug = 'AUD_'
            alpha = 0.8
        else:
            drug = ''
            alpha = 0.2
            
        # plot by enrollment period 
        colorby='enrollYear'
        plot_by_enroll_year(df,'bh_p','Enrollment', xlab,alpha)
        plt.tight_layout()
        plt.savefig(f'../figures/final2/{drug}{predictor}_{outcome}_colorBy{colorby}.png', bbox_inches='tight', dpi=300)

        # plot by control variables
        colorby='controlVars'
        if '/aud/' not in path:
            plot_by_controls(df,'bh_p','Controls',xlab,alpha)
        else:
            aud_plot_by_controls(df,'bh_p','Controls',xlab,alpha)
        plt.tight_layout()
        plt.savefig(f'../figures/final2/{drug}{predictor}_{outcome}_colorBy{colorby}.png', bbox_inches='tight', dpi=300)

        # plot by OPRx criteria for enrollment (only for the opioid experiments)
        if '/aud/' not in path:
            colorby='OPRxEnroll'
            plot_by_opioid_enroll(df,'bh_p','OPRx Enrollment', xlab,alpha)
            plt.tight_layout()
            plt.savefig(f'../figures/final2/{drug}{predictor}_{outcome}_colorBy{colorby}.png', bbox_inches='tight', dpi=300)

        # plot by age for exclusion for pre-existing NCD
        colorby='NCDAgeExclusion'
        plot_by_ncd_age_exclusion(df,'bh_p','NCD Age Exclusion',xlab,alpha)
        plt.tight_layout()
        plt.savefig(f'../figures/final2/{drug}{predictor}_{outcome}_colorBy{colorby}.png', bbox_inches='tight', dpi=300)

        # plot by density of points
        colorby='Density'
        density_plot(df,'bh_p','None',xlab,alpha)
        plt.tight_layout()
        plt.savefig(f'../figures/final2/{drug}{predictor}_{outcome}_colorBy{colorby}.png', bbox_inches='tight', dpi=300)
