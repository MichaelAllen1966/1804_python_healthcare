# See https://docs.scipy.org/doc/scipy/reference/stats.html for dists



import matplotlib.pyplot as plt
import scipy
import scipy.stats
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd

# For full list of possible distributions to tets see 
# https://docs.scipy.org/doc/scipy-0.14.0/reference/stats.html#module-scipy.stats



size = 10000

# creating the dummy sample (using beta distribution)
y = np.random.lognormal(3, 1, size)

# y = pd.read_csv('data.csv').values

# Load data
from sklearn import datasets
data_set = datasets.load_breast_cancer()
y=data_set.data[:,0]

sc=StandardScaler() 
yy = y.reshape (-1,1)
sc.fit(yy)
y_std =sc.transform(yy)
y_std = y_std.flatten()
y_std
del yy



x = scipy.arange(len(y))



#y = np.linspace(-10,10,10)




dist_names = ['beta',
              'expon',
              'gamma',
              'lognorm',
              'norm',
              'pearson3',
              'triang',
              'uniform',
              'weibull_min', 
              'weibull_max']



# Set up empty lists to stroe results
chi_square = []
p_values = []

# Set up 50 bins for chi-square test
# Observed data will be approximately evenly distrubuted aross all bins
percentile_bins = np.linspace(0,100,51)
percentile_cutoffs = np.percentile(y_std, percentile_bins)
observed_frequency, bins = (np.histogram(y_std, bins=percentile_cutoffs))
observed_frequency_cumsum = np.cumsum(observed_frequency)

# Loop through candidate distributions

for distribution in dist_names:
    # Set up distribution and get fitted distribution parameters
    dist = getattr(scipy.stats, distribution)
    param = dist.fit(y_std)
    
    # Obtain the KS test P statistic, round it to 5 decimal places
    p = scipy.stats.kstest(y_std, distribution, args=param)[1]
    p = np.around(p, 5)
    p_values.append(p)    
    
    # Get expected counts in percentile bins
    # This is based on a 'cumulative distrubution function' (cdf)
    cdf_fitted = dist.cdf(percentile_cutoffs, *param[:-2], loc=param[-2], 
                          scale=param[-1])
    expected_frequency = []
    for bin in range(len(percentile_bins)-1):
        expected_cdf_area = cdf_fitted[bin+1] - cdf_fitted[bin]
        expected_frequency.append(expected_cdf_area)
        
    
    # calculate chi-squared 
    expected_frequency = np.array(expected_frequency)
    ss = sum (((expected_frequency - observed_frequency) ** 2) / observed_frequency)
    chi_square.append(ss)
        
# Collate results and sort by goodness of fit (best at top)

results = pd.DataFrame()
results['Distribution'] = dist_names
results['chi_square'] = chi_square
results['p_value'] = p_values
results.sort_values(['chi_square'], inplace=True)
    
# Report results

print ('\nDistributions sorted by goodness of fit:')
print ('----------------------------------------')
print (results)

# Plot best distributions

number_of_bins = 100
bin_cutoffs = np.linspace(np.percentile(y,0), np.percentile(y,99),number_of_bins)


number_distributions_to_plot = 3

print ('\nBest distributions:')
print ('-------------------')

h = plt.hist(y, bins = bin_cutoffs, color='0.75')

dist_names = results['Distribution'].iloc[0:number_distributions_to_plot]
parameters = []


for dist_name in dist_names:
    dist = getattr(scipy.stats, dist_name)
    # If distribution will accept a starting location, use mean of y
    param = dist.fit(y)
    parameters.append(param)
    pdf_fitted = dist.pdf(x, *param[:-2], loc=param[-2], scale=param[-1])
    scale_pdf = np.trapz (h[0], h[1][:-1]) / np.trapz (pdf_fitted, x)
    pdf_fitted *= scale_pdf
    plt.plot(pdf_fitted, label=dist_name)
    plt.xlim(0,np.percentile(y,99))
    del dist


plt.legend()
plt.show()

dist_parameters = pd.DataFrame()
dist_parameters['Distribution'] = (
        results['Distribution'].iloc[0:number_distributions_to_plot])
dist_parameters['Distribution parameters'] = parameters

print ('\nDistribution parameters:')
print ('------------------------')

for index, row in dist_parameters.iterrows():
    print ('\nDistribution:', row[0])
    print ('Parameters:', row[1] )
    



    
    
        