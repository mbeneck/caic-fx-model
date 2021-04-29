# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 20:13:03 2021

@author: eckmb
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.special import factorial
from scipy.stats import poisson
from sklearn.model_selection import KFold
import seaborn as sns


# generates DateTimeIndexes for November-April of specified years
def generateAvySeasonDates(first_year, last_year):
    all_dates = pd.DatetimeIndex([])
    for year in range(first_year, last_year+1):
        all_dates = all_dates.append(pd.date_range(str(year)+'/11/01', str(year+1)+'/04/30'))
        
    return all_dates

avy_obs = pd.read_csv('Data/CAIC_avalanches_2008_2021.csv')

# drop nans
avy_obs.dropna(subset=['Trigger'], inplace=True)

# drop D1's and D1.5's
avy_obs = avy_obs[~avy_obs['Dsize'].isin(['D1'])]

# Turn unknown triggers into naturals
avy_obs.loc[avy_obs['Trigger'] == 'U', 'Trigger'] = 'N'
avy_obs.loc[avy_obs['Trigger'].str.startswith('-'), 'Trigger'] = 'N'

# Aggregate all artificial triggers
avy_obs.loc[avy_obs['Trigger'].str.startswith('A'), 'Trigger'] = 'A'
# manually clean up some miscategorized zones
avy_obs.loc[2006, 'BC Zone'] = 'Southern San Juan'
avy_obs.loc[2007, 'BC Zone'] = 'Northern San Juan'
avy_obs.loc[2008, 'BC Zone'] = 'Northern San Juan'
avy_obs.loc[2311, 'BC Zone'] = 'Southern San Juan'
avy_obs.loc[2312, 'BC Zone'] = 'Northern San Juan'
avy_obs.loc[2313, 'BC Zone'] = 'Northern San Juan'

#count and pivot avalanches in each category
avy_obs_pivot = avy_obs.groupby(['Date', 'BC Zone', 'Trigger'])['#'].sum().unstack(fill_value=0).reset_index()
avy_obs_pivot['Date'] = pd.to_datetime(avy_obs_pivot['Date'])
avy_obs_pivot = avy_obs_pivot.set_index(['Date', 'BC Zone'])

# Reindex to avalanche season dates for period of interest
all_dates = generateAvySeasonDates(2010, 2019)
all_index = pd.MultiIndex.from_product([all_dates, avy_obs_pivot.index.get_level_values('BC Zone').unique()], names=['Date', 'BC Zone'])
avy_obs_pad = avy_obs_pivot.reindex(all_index, fill_value=0)
avy_obs_pad = avy_obs_pad.reorder_levels(['BC Zone', 'Date'])

scatters = plt.figure(constrained_layout=True, figsize=[11,8.5])
gs = scatters.add_gridspec(3,5)
all_ax= scatters.add_subplot(gs[0,:])
avy_obs_pad.plot(x='A', y='N', kind='scatter', ax=all_ax, title='All Zones')
for idx, (label, df) in enumerate(avy_obs_pad.groupby('BC Zone')):
    ax = scatters.add_subplot(gs[idx//5+1,idx % 5])
    df.plot(x='A', y='N', kind='scatter', ax=ax, title=label)
    
# generates n random data from a bivariate poisson
def bvpois(lambda_1, lambda_2, lambda_3, n):
    a = np.random.poisson(lambda_1, size=n)
    b = np.random.poisson(lambda_2, size=n)
    c = np.random.poisson(lambda_3, size=n)
    return np.column_stack((a+c, b+c))

def innersum(x, lambda_1, lambda_2, lambda_3):
    min_param = int(np.amin(x))
    innersum = int(0)
    for s in range(0, min_param+1):
        innersum += math.comb(x[0], s)*math.comb(x[1], s)*factorial(s)*(lambda_3/(lambda_1*lambda_2))**s
    return innersum

# evaluates density of bivariate poisson
def f_bvpois(x, lambda_1, lambda_2, lambda_3):
    inner_sum = np.apply_along_axis(innersum, 1, x, lambda_1, lambda_2, lambda_3)
    # print("Lambda 1: %f, Lambda 2: %f" % (lambda_1, lambda_2))
    # print(x[:,0])
    # print(x[:,1])
    return np.longdouble(math.exp(-(lambda_1 + lambda_2 + lambda_3))*np.float_power(lambda_1, x[:,0], dtype=np.longdouble)/np.uint64(factorial(x[:,0]))*np.float_power(lambda_2,x[:,1], dtype=np.longdouble)/np.uint64(factorial(x[:,1]))*inner_sum)

# convenience wrapper for vectorization
def f_bvpois_vec(lambdas, x):
    return f_bvpois(x, lambdas[0], lambdas[1], lambdas[2])

def em_estimate_bvpois(x, maxiter = 1000, epsilon=.001, start_guess = np.random.randint(1,10, size=3)*np.random.rand(3)):
    # expected value of hidden parameter
    x3 = np.zeros(np.shape(x)[0])
    x_nz = x[np.amin(x, axis=1)>0]
    x3[np.amin(x, axis=1)>0] = start_guess[2]*f_bvpois(x_nz-1, start_guess[0], start_guess[1], start_guess[2])/f_bvpois(x_nz, start_guess[0], start_guess[1], start_guess[2])
    for i in range(0, maxiter):
        # update parameter estimates (M-step)
        new_guess = np.array([np.sum(x[:,0]-x3)/np.shape(x)[0], np.sum(x[:,1]-x3)/np.shape(x)[0], np.sum(x3)/np.shape(x)[0]])
        if np.amax(np.abs(start_guess-new_guess)) < epsilon: break
        start_guess= new_guess
        
        #update x3 estimate again
        x3 = np.zeros(np.shape(x)[0])
        x_nz = x[np.amin(x, axis=1)>0]
        x3[np.amin(x, axis=1)>0] = start_guess[2]*f_bvpois(x_nz-1, start_guess[0], start_guess[1], start_guess[2])/f_bvpois(x_nz, start_guess[0], start_guess[1], start_guess[2])
    return new_guess

def x3_expected(x,lambda_1, lambda_2, lambda_3):
    min_param=int(np.amin(x))
    tot = 0
    for r in range(0, min_param+1):
        tot += r*poisson.pmf(x[0]-r, lambda_1)*poisson.pmf(x[1]-r, lambda_2)*poisson.pmf(r, lambda_3)
    return np.longdouble(tot)/f_bvpois(np.transpose(x[:,None]), lambda_1, lambda_2, lambda_3)

def calc_lambda_for_x(x, class_probs):
    return np.sum(np.multiply(class_probs,x), axis=0)/np.sum(class_probs, axis=0)

#takes in an i by k matrix of expected values (calculating by j)
def calc_lambda_from_x_exp(x_exp, class_probs):
    lambdas = np.empty((np.shape(class_probs)[1], 0))
    for t in range(0, np.shape(x_exp)[0]):
        lambdas = np.append(lambdas, calc_lambda_for_x(x_exp[t,:,:], class_probs)[:,None], axis=1)
    return lambdas
                    

# allow for specifiyng start guess and priors later... 
def em_cluster_bvpois(x, k, maxiter = 100, epsilon=.01, ind=False):
    priors=np.ones(k)/k
    start_guess=np.random.randint(1,10, size=(k,3))*np.random.rand(k,3)
    
    # calculate expected values
    x3 = np.empty((np.shape(x)[0],k))
    if(ind):
        x3 = np.zeros((np.shape(x)[0],k))
    else:
        for j in range(0, k):
            x3[:,j] = np.ravel(np.apply_along_axis(x3_expected, 1, x, start_guess[j,0], start_guess[j,1], start_guess[j,2]))
    x_exp = np.array([x[:,0][:,None]-x3, x[:,1][:,None]-x3, x3])
    
    loglikelihood = 0
    
    for i in range(0, maxiter):
        print("EM Round %i" %(i))
        # update parameter estimates (M-step)
        class_cond_probs = np.transpose(np.apply_along_axis(f_bvpois_vec, 1, start_guess, x))
        class_probs = priors*class_cond_probs/ np.sum(priors*class_cond_probs, axis=1)[:,None]
        loglikelihood = np.sum(np.log(np.sum(priors*class_cond_probs, axis=1)))
        new_priors = np.sum(class_probs, axis=0)/np.shape(x)[0]
        new_guess = calc_lambda_from_x_exp(x_exp, class_probs)
        print(new_guess)
        #new_guess = np.apply_along_axis(calc_lambda_from_x_exp, 0, x_exp, class_probs)
        
        
        if np.amax(np.abs(start_guess-new_guess)) < epsilon: break
        start_guess = new_guess
        priors = new_priors
        
        # calculate expected values again
        # calculate expected values
        x3 = np.empty((np.shape(x)[0],k))
        if(ind):
            x3 = np.zeros((np.shape(x)[0],k))
        else:
            for j in range(0, k):
                x3[:,j] = np.ravel(np.apply_along_axis(x3_expected, 1, x, start_guess[j,0], start_guess[j,1], start_guess[j,2]))
        x_exp = np.array([x[:,0][:,None]-x3, x[:,1][:,None]-x3, x3])
        
    if(ind):
        aic = 2*(3*k-1)-2*loglikelihood
    else:
        aic = 2*(4*k-1)-2*loglikelihood
    return new_priors, new_guess, loglikelihood, aic
    
# lambdas is a kx3 matrix of lambdas, priors is k-d, n # of data points
def generate_mixed_bvpois(priors, lambdas, n):
    x = np.empty((0,2), dtype='int')
    labels = np.empty(0, dtype = 'int')
    colors = ['g', 'yellow', 'orange', 'r', 'k']
    names = ['Low', 'Moderate', 'Considerable', 'High', 'Extreme']
    plt.figure()
    for idx, prior in reversed(list(enumerate(priors))):
        # ok, ok class creation isn't exactly random but whatever.
        newdata =  bvpois(lambdas[idx,:][0], lambdas[idx,:][1], lambdas[idx,:][2], int(round(n*prior)))
        x = np.append(x, newdata, axis=0)
        labels = np.append(labels, idx*np.ones(int(round(n*prior))).astype('int'))
        plt.scatter(newdata[:,0], newdata[:,1], c=colors[idx], label=names[idx], alpha=.2)
    plt.legend()
    plt.title('Simulated Bivariate Poisson Data')
    plt.xlabel('Artificial')
    plt.ylabel('Natural')
    
    return x, labels
    
def classify_mixed_bvpois(x, priors, lambdas):
    class_cond_probs = np.transpose(np.apply_along_axis(f_bvpois_vec, 1, lambdas, x))
    class_probs = priors*class_cond_probs/ np.sum(priors*class_cond_probs, axis=1)[:,None]
    labels =  np.argmax(class_probs, axis=1) 
    return labels, class_probs

def split_train_validate(data, split=.9):
    rand_vec = np.random.rand(data.shape[0]) <= split
    return data[rand_vec], data[~rand_vec]

def calc_loglikelihood(priors, lambdas, data):
    class_cond_probs = np.transpose(np.apply_along_axis(f_bvpois_vec, 1, lambdas, data))
    return np.sum(np.log(np.sum(priors*class_cond_probs, axis=1)))

# takes in a tuples return from em_clster_bvpois
def calc_xvalidated_loglikelihood(estimate, validate_data):
    return calc_loglikelihood(estimate[0], estimate[1], validate_data)

# takes in a list of tuples return from em_clster_bvpois
def calc_xvalidated_loglikelihoods(estimates, validate_data):
    loglikelihoods = np.empty(len(estimates), dtype='float64')
    for i, estimate in enumerate(estimates):
        loglikelihoods[i] = calc_loglikelihood(estimate[0], estimate[1], validate_data)
    return loglikelihoods

def calc_AICC(k, loglikelihood, data):
    return 2*(4*(k+1)-1)-2*loglikelihood+2*(k+1)**2+2*(k+1)/(data.shape[0]-(k+1)-1)
    
def calc_BIC(k, loglikelihood, data):
    return (4*(k+1)-1)*np.log(data.shape[0])-2*loglikelihood

def kfold_xval(data, splits, k):
    kf = KFold(n_splits=splits)
    data = data.to_numpy(dtype='int')
    LL_sum = 0
    for train_index, test_index in kf.split(data):
        model = em_cluster_bvpois(data[train_index], k, maxiter=50, epsilon=.01)
        LL_sum += calc_xvalidated_loglikelihood(model, data[test_index])
    return LL_sum/splits

def multi_kfold_xval(data, splits, ks):
    results = np.empty(ks.size, dtype='float64')
    for idx, k in enumerate(ks):
        results[idx] = kfold_xval(data, splits, k)
    return results

def classify_from_params_df(data, params, clusters):
    return classify_mixed_bvpois(data, params.iloc[clusters-1,:]['Priors'], params.iloc[clusters-1,:]['Lambdas'])

def plot_estimates_arbclass(region, data, labels, ax):
    ax.set_xlabel('Artificial')
    ax.set_ylabel('Natural')
    colors = ['r', 'b', 'g', 'c', 'y', 'm','k']
    classes = np.unique(labels).size

    for label in range(classes):
        count = data[labels == label].size
        class_name = 'Class %i, Count: %i' % (label, count)
        ax.scatter(data[labels == label]['A'], data[labels == label]['N'], c=colors[label], label=class_name, alpha=.2)
    
    ax.legend()
    ax.set_title(region + ', %i Classes' % (classes))
    return ax

def plot_estimated_dangerscale(region, data, labels, ax, classes):
    ax.set_xlabel('Artificial')
    ax.set_ylabel('Natural')
    names = {1:'Low', 2:'Moderate', 3:'Considerable', 4:'High', 5:'Extreme'}
    colors = ['g', 'yellow', 'orange', 'r', 'k']
    if(classes == 4):
        colors = ['g', 'yellow', 'r', 'k']
        names = {1:'Low', 2:'Moderate', 3:'High', 4:'Extreme'}
    
    for label in reversed(range(1, classes+1)):
        if(label != 0):
            count = data[labels == label].shape[0]
            class_name = names[label] + ', Count: %i' % (count)
            ax.scatter(data[labels == label]['A'], data[labels == label]['N'], c=colors[label-1], label=class_name, alpha=.3)
    
    ax.legend()
    ax.set_title(region + ', %i Classes' % (classes))
    return ax

nonzerodays = avy_obs_pad[np.amax(avy_obs_pad, axis=1)>0]
nonzerodays_by_zone = nonzerodays.groupby('BC Zone').size().reset_index(name='counts')

northern = avy_obs_pad.loc['Vail & Summit County']
southern = avy_obs_pad.loc['Northern San Juan']
central = avy_obs_pad.loc['Aspen']

clusters = np.arange(1,11)

northern_xval = multi_kfold_xval(northern, 3, clusters)
central_xval = multi_kfold_xval(central, 3, clusters)
southern_xval = multi_kfold_xval(southern, 3, clusters)


northern_est_params = []
for k in clusters:
    northern_est_params.append(em_cluster_bvpois(northern.to_numpy(dtype='int'), k, maxiter=160, epsilon=.01))

southern_est_params = []
for k in clusters:
    southern_est_params.append(em_cluster_bvpois(southern.to_numpy(dtype='int'), k, maxiter=160, epsilon=.01))
    
central_est_params = []
for k in clusters:
    central_est_params.append(em_cluster_bvpois(central.to_numpy(dtype='int'), k, maxiter=160, epsilon=.01))
    

northern_est_params_df = pd.DataFrame(northern_est_params, columns=['Priors', 'Lambdas', 'Log Likelihood', 'AIC'])
central_est_params_df = pd.DataFrame(central_est_params, columns=['Priors', 'Lambdas', 'Log Likelihood', 'AIC'])
southern_est_params_df = pd.DataFrame(southern_est_params, columns=['Priors', 'Lambdas', 'Log Likelihood', 'AIC'])

northern_est_params_df['X-Val Log Likelihood'] = northern_xval
central_est_params_df['X-Val Log Likelihood'] = central_xval
southern_est_params_df['X-Val Log Likelihood'] = southern_xval

northern_est_params_df['BIC'] = calc_BIC(northern_est_params_df.index.to_numpy(), northern_est_params_df['Log Likelihood'], northern)
southern_est_params_df['BIC'] = calc_BIC(southern_est_params_df.index.to_numpy(), southern_est_params_df['Log Likelihood'], southern)
central_est_params_df['BIC'] = calc_BIC(central_est_params_df.index.to_numpy(), central_est_params_df['Log Likelihood'], central)

northern_est_params_df.index += 1
southern_est_params_df.index += 1
central_est_params_df.index += 1

fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(8.5,11), sharex=False, sharey=False)
northern_est_params_df.iloc[0:6, :].plot(y=['AIC', 'BIC'], kind='line', ax = ax[0], sharex= False,xticks=np.arange(1,6, dtype='int'), title='Northern Mountains (Vail & Summit County)' )
northern_est_params_df.iloc[0:6, :].plot(y='X-Val Log Likelihood', kind='line', ax=ax[0], xticks=np.arange(1,6, dtype='int'), sharex= False, secondary_y=True)
central_est_params_df.iloc[0:6, :].plot(y=['AIC', 'BIC'], kind='line', ax = ax[1], sharex= False, xticks=np.arange(1,6, dtype='int'), title='Central Mountains (Aspen)' )
central_est_params_df.iloc[0:6, :].plot(y='X-Val Log Likelihood', kind='line', ax=ax[1], xticks=np.arange(1,6, dtype='int'), sharex= False, secondary_y=True)
southern_est_params_df.iloc[0:6, :].plot(y=['AIC', 'BIC'], kind='line', ax = ax[2], sharex= False, xticks=np.arange(1,6, dtype='int'), title='Southern Mountains (Northern San Juan)' )
southern_est_params_df.iloc[0:6, :].plot(y='X-Val Log Likelihood', kind='line', ax=ax[2], sharex= False, xticks=np.arange(1,6, dtype='int'), secondary_y=True)

for a in ax:
    a.set(xticks=np.arange(1,7, dtype='int'))
    a.set_xlabel('Clusters')
    a.set_ylabel('Information Criterion')
    a.right_ax.set_ylabel('Log Likelihood')
    
    l_lines, l_labels = a.get_legend_handles_labels()
    r_lines, r_labels = a.right_ax.get_legend_handles_labels()
    a.legend(l_lines+r_lines, l_labels+r_labels, loc='center right')
    
fig.tight_layout()

northern_6_labels, northern_6_class_probs = classify_from_params_df(northern.to_numpy(dtype='int'), northern_est_params_df, 6)
southern_6_labels, southern_6_class_probs = classify_from_params_df(southern.to_numpy(dtype='int'), southern_est_params_df, 6)
central_6_labels, central_6_class_probs = classify_from_params_df(central.to_numpy(dtype='int'), central_est_params_df, 6)


northern_5_labels, northern_5_class_probs = classify_from_params_df(northern.to_numpy(dtype='int'), northern_est_params_df, 5)
northern_4_labels, northern_4_class_probs = classify_from_params_df(northern.to_numpy(dtype='int'), northern_est_params_df, 4)
central_5_labels, central_5_class_probs = classify_from_params_df(central.to_numpy(dtype='int'), central_est_params_df, 5)
central_4_labels, central_4_class_probs = classify_from_params_df(central.to_numpy(dtype='int'), central_est_params_df, 4)
southern_5_labels, southern_5_class_probs = classify_from_params_df(southern.to_numpy(dtype='int'), southern_est_params_df, 5)
southern_4_labels, southern_4_class_probs = classify_from_params_df(southern.to_numpy(dtype='int'), southern_est_params_df, 4)

northern['5 Label'] = northern_5_labels
northern['4 Label'] = northern_4_labels
central['5 Label'] = central_5_labels
central['4 Label'] = central_4_labels
southern['5 Label'] = southern_5_labels
southern['4 Label'] = southern_4_labels
southern['6 Label'] = southern_6_labels
northern['6 Label'] = northern_6_labels
central['6 Label'] = central_6_labels

fig, ax = plt.subplots(nrows=3, ncols = 3, figsize=(8.5,11))

plot_estimates_arbclass('Northern', northern, northern_4_labels, ax[0,0])
plot_estimates_arbclass('Northern', northern, northern_5_labels, ax[0,1])
plot_estimates_arbclass('Northern', northern, northern_6_labels, ax[0,2])
plot_estimates_arbclass('Central', central, central_4_labels, ax[1,0])
plot_estimates_arbclass('Central', central, central_5_labels, ax[1,1])
plot_estimates_arbclass('Central', central, central_6_labels, ax[1,2])
plot_estimates_arbclass('Southern', southern, southern_4_labels, ax[2,0])
plot_estimates_arbclass('Southern', southern, southern_5_labels, ax[2,1])
plot_estimates_arbclass('Southern', southern, southern_6_labels, ax[2,2])

fig.tight_layout()

northern_fx = pd.read_csv('Data/z2data.csv')
northern_fx['Rating'] = northern_fx[['Above Danger', 'Near Danger', 'Below Danger']].max(axis=1)
northern_fx.groupby('Rating').agg('count')

central_fx = pd.read_csv('Data/z4data.csv')
central_fx['Rating'] = central_fx[['Above Danger', 'Near Danger', 'Below Danger']].max(axis=1)
central_fx.groupby('Rating').agg('count')

southern_fx = pd.read_csv('Data/z7data.csv')
southern_fx['Rating'] = southern_fx[['Above Danger', 'Near Danger', 'Below Danger']].max(axis=1)
southern_fx.groupby('Rating').agg('count')

northern_5_equiv = np.array([1, 5, 3, 4, 2])
central_5_equiv = np.array([2, 3, 4, 5, 1])
southern_5_equiv = np.array([3, 1, 5, 2, 4])
northern_4_equiv = np.array([4, 1, 3, 2])
central_4_equiv = np.array([3, 2, 4, 1])
southern_4_equiv = np.array([4, 3, 2, 1])

northern['5 Danger Scale'] = northern_5_equiv[northern_5_labels]
central['5 Danger Scale'] = central_5_equiv[central_5_labels]
southern['5 Danger Scale'] = southern_5_equiv[southern_5_labels]
northern['4 Danger Scale'] = northern_4_equiv[northern_4_labels]
central['4 Danger Scale'] = central_4_equiv[central_4_labels]
southern['4 Danger Scale'] = southern_4_equiv[southern_4_labels]

fig, ax = plt.subplots(nrows=3, ncols = 2, figsize=(8.5,11))
plot_estimated_dangerscale('Northern', northern, northern['4 Danger Scale'], ax[0,0], 4)
plot_estimated_dangerscale('Northern', northern, northern['5 Danger Scale'], ax[0,1], 5)
plot_estimated_dangerscale('Central', central, central['4 Danger Scale'], ax[1,0], 4)
plot_estimated_dangerscale('Central', central, central['5 Danger Scale'], ax[1,1], 5)
plot_estimated_dangerscale('Southern', southern, southern['4 Danger Scale'], ax[2,0], 4)
plot_estimated_dangerscale('Southern', southern, southern['5 Danger Scale'], ax[2,1], 5)
fig.tight_layout()

northern_fx['Date'] = pd.to_datetime(northern_fx['Date']).dt.date
central_fx['Date'] = pd.to_datetime(central_fx['Date']).dt.date
southern_fx['Date'] = pd.to_datetime(southern_fx['Date']).dt.date

northern.reset_index(inplace=True)
central.reset_index(inplace=True)
southern.reset_index(inplace=True)
northern['Date'] = pd.to_datetime(northern['Date']).dt.date
central['Date'] = pd.to_datetime(central['Date']).dt.date
southern['Date'] = pd.to_datetime(southern['Date']).dt.date

northern_merge = pd.merge(northern, northern_fx, on='Date')
central_merge = pd.merge(central, central_fx, on='Date')
southern_merge = pd.merge(southern, southern_fx, on='Date')


fig, ax = plt.subplots(nrows = 3, ncols = 2, figsize=(8.5,11))
plot_estimated_dangerscale('Northern', northern_merge, northern_merge['5 Danger Scale'], ax[0,0], 5)
plot_estimated_dangerscale('Northern', northern_merge, northern_merge['Rating_y'], ax[0,1], 5)
plot_estimated_dangerscale('Central', central_merge, central_merge['5 Danger Scale'], ax[1,0], 5)
plot_estimated_dangerscale('Central', central_merge, central_merge['Rating_y'], ax[1,1], 5)
plot_estimated_dangerscale('Southern', southern_merge, southern_merge['5 Danger Scale'], ax[2,0], 5)
plot_estimated_dangerscale('Southern', southern_merge, southern_merge['Rating_y'], ax[2,1], 5)
ax[0,0].set_title('Northern Cluster')
ax[1,0].set_title('Central Cluster')
ax[2,0].set_title('Southern Cluster')
ax[0,1].set_title('Northern Forecast')
ax[1,1].set_title('Central Forecast')
ax[2,1].set_title('Southern Forecast')
fig.tight_layout()

southern_conf = pd.crosstab(southern_merge['Rating_y'], southern_merge['5 Danger Scale'], rownames =['Forecast'], colnames=['Cluster'])
southern_conf.loc[5] = [0,0,0,0,0]
central_conf = pd.crosstab(central_merge['Rating_y'], central_merge['5 Danger Scale'], rownames =['Forecast'], colnames=['Cluster'])
northern_conf = pd.crosstab(northern_merge['Rating_y'], northern_merge['5 Danger Scale'], rownames =['Forecast'], colnames=['Cluster']).drop(0)

#confusion matrices from elder
elder_n_conf = pd.DataFrame(data=np.array([[79,36,6,0],[3,2,3,0],[0,1,4,0],[0,0,0,0]]).T, index=[1,2,4,5], columns=[1,2,4,5])
elder_c_conf = pd.DataFrame(data=np.array([[78,33,1,0],[2,2,4,0],[1,0,6,0],[0,2,5,0]]).T, index=[1,2,4,5], columns=[1,2,4,5])
elder_s_conf = pd.DataFrame(data=np.array([[81,27,9,0],[0,6,3,0],[1,1,2,1],[0,1,2,0]]).T, index=[1,2,4,5], columns=[1,2,4,5])
cols = ['BP Clustering', 'Elder & Armstrong']
rows = ['Northern', 'Central', 'Southern']
fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(8.5,11))
sns.heatmap(northern_conf/np.sum(northern_conf), fmt='.1%', annot=True, cmap='Blues', ax=axs[0,0], square=True)
sns.heatmap(central_conf/np.sum(central_conf), fmt='.1%', annot=True, cmap='Blues', ax=axs[1,0], square=True)
sns.heatmap(southern_conf/np.sum(southern_conf), fmt='.1%', annot=True, cmap='Blues', ax=axs[2,0], square=True)
sns.heatmap((elder_n_conf/np.sum(elder_n_conf)).fillna(0), fmt='.1%', annot=True, cmap='Blues', ax=axs[0,1], square=True)
sns.heatmap(elder_c_conf/np.sum(elder_c_conf), fmt='.1%', annot=True, cmap='Blues', ax=axs[1,1], square=True)
sns.heatmap(elder_s_conf/np.sum(elder_s_conf), fmt='.1%', annot=True, cmap='Blues', ax=axs[2,1], square=True)

for ax, col in zip(axs[0], cols):
    ax.set_title(col)
    
pad=5
for ax, row in zip(axs[:,0], rows):
    ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                xycoords=ax.yaxis.label, textcoords='offset points',
                size='large', ha='right', va='center')
    
for i in range(3):
    axs[i,0].set_xlabel('Cluster True Value')
    axs[i,1].set_xlabel('Expert True Value')
fig.tight_layout()
fig.subplots_adjust(left=0.15, top=0.95)


fig, ax = plt.subplots(nrows=1,ncols=4, figsize = (11, 4))
for idx, cov in enumerate([.01, .1, 1, 10]):
    rand_bvpois = bvpois(5, 15, cov, 1000)
    ax[idx].scatter(rand_bvpois[:,0], rand_bvpois[:,1], alpha=.2)
    ax[idx].set_xlabel('Artificial')
    ax[idx].set_ylabel('Natural')
    ax[idx].set_title('Cov: %.2f' % (cov))

fig.suptitle('Bivariate Poisson with Lambda 1: %i, Lambda 2: %i' % (5, 15))
fig.tight_layout(rect=(0,0,1,.85))

generate_mixed_bvpois(np.array([.75, .15, .05, .04, .01]), np.array([[0.1, 0.1, 0.1],[2,3,.1],[4,3, .2], [9,8,.3], [10,15,1]]), 1000)

fig, ax = plt.subplots(1)
northern_est_params_df.iloc[0:6, :].plot(y=['AIC', 'BIC'], kind='line', ax=ax, sharex= False,xticks=np.arange(1,6, dtype='int'), title='Northern Mountains (Vail & Summit County)' )
northern_est_params_df.iloc[0:6, :].plot(y='X-Val Log Likelihood', kind='line', ax=ax, xticks=np.arange(1,6, dtype='int'), sharex= False, secondary_y=True)
ax.set(xticks=np.arange(1,7, dtype='int'), xlabel='Clusters', ylabel='Information Criterion')
ax.right_ax.set_ylabel('Log Likelihood')
l_lines, l_labels = ax.get_legend_handles_labels()
r_lines, r_labels = ax.right_ax.get_legend_handles_labels()
ax.legend(l_lines+r_lines, l_labels+r_labels, loc='center right')

fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize=(11,5))
plot_estimated_dangerscale('Southern', southern_merge, southern_merge['5 Danger Scale'], ax[0], 5)
plot_estimated_dangerscale('Southern', southern_merge, southern_merge['Rating_y'], ax[1], 5)
ax[0].set_title('Southern Cluster')
ax[1].set_title('Southern Forecast')
fig.tight_layout()


fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(8.5,11))

