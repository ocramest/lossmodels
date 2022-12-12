# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 15:15:15 2022

@author: mesteban
"""

# Libraries: 
#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from KDEpy import FFTKDE as kdens
import seaborn as sns
import datetime
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 100)
#%%
# Data:
#%%
claims_data = pd.read_csv("Z:/02_RIESGOS_TECNICOS/PATRIMONIAL/siniestralidad/final/siniestralidad.csv", 
                          encoding = "latin-1",
                          dtype = {'asegurado': 'str', 'fecha': 'str', 'ramo': 'str',
                                   'subramo': 'str', 'estado': 'str', 'monto': 'float',
                                   'moneda': 'str', 'agregado': 'str', 'fuente': 'str'},
                          parse_dates = ['fecha'])

claims_data.head()
claims_data.info()
claims_data.shape
claims_data.describe()
#%%    
# Exchange rate:
# Price index:
# Current risks:
# Historical claims time period:
# Modeling:
client = 'IMSS'
risk = 'Terremoto y erupción volcánica'        
years_exposed = 12
years_insured = 1
simulations = 10000
t0 = datetime.datetime.now()
final = claims_data[(claims_data['asegurado']==client) & (claims_data['subramo'] == risk) & (claims_data['monto'] > 0)]
# Frequency:
claims_per_year = final.shape[0]*years_insured/years_exposed        
    # Severity:
        # Kernel:        
def rkernel(data, size, kernel = 'gaussian', bw = 'ISJ'):
    x = np.log(data.values)       
    kernel = kdens(kernel = kernel, bw=bw).fit(x)
    bdw = kernel.bw
    samp = np.exp(np.random.choice(x, size = size, replace = True) + 
                  np.random.normal(loc = 0, scale = bdw, size = size))
    return samp
               
    # Aggregate losses:
N = np.random.poisson(lam = claims_per_year, size = simulations)
X = list(map(lambda n: rkernel(final['monto'], size = n), N)) 
L = list(map(lambda x: sum(x), X))
t1 = datetime.datetime.now()
print(t1-t0)


# Function:
def aggloss(data, client, risk, years_exposed, years_insured = 1, simulations = 1000):
    final = data[(data['asegurado']==client) & (data['subramo'] == risk) & (data['monto'] > 0)]
    claims_per_year = final.shape[0]*years_insured/years_exposed
    N = np.random.poisson(lam = claims_per_year, size = simulations)
    X = list(map(lambda n: rkernel(final['monto'], size = n), N)) 
    L = list(map(lambda x: sum(x), X))
    return L
# Results:
#%%
VaR = np.quantile(L, .995)
print('P0.5:', np.quantile(L, 0.005)/1e3, '\n',
      'P25:', np.quantile(L, 0.25)/1e3, '\n',
      'P50:', np.quantile(L, 0.5)/1e3, '\n',
      'Media:', np.mean(L)/1e3, '\n',
      'P75:', np.quantile(L, 0.75)/1e3, '\n',
      'P99.5:', np.quantile(L, 0.995)/1e3, '\n',
      'CVaR:', np.mean([l for l in L if l > VaR])/1e3)

#%%
colors = ["#385623", "#c6e0b5", "#e9a083", "#e2be4c", "#7c9ebc", "#bfbfbf", "#984807", "#929292",
            "#d34c4c", "#ffe594", "#667da4"]
sns.set_style('whitegrid')
g = sns.kdeplot(L,
                color = ''.join(np.random.choice(colors, size = 1)),
                fill = True,)
g.set_xticklabels(['{:,.0f}'.format(x) for x in g.get_xticks()/1e6])
g.set(xlabel = 'Pérdida (MDP)', ylabel = 'Frecuencia relativa')
plt.axvline(x =  np.quantile(L, 0.995), color = 'black', linestyle = 'dashed', lw = 1)
plt.text(np.quantile(L, 0.997),0,'VaR',rotation=25)
plt.show(g)
#g.figure.savefig('gráfico.png', dpi = 500)
#%%

# For multiple risks insured:
#risks = claims_data['asegurado'] + '-' + claims_data['subramo']
#risks = list(set(risks))

risks = ['IMSS-Terremoto y erupción volcánica', 'IMSS-Misceláneos', 'SEMARNAT-Huracán y otros riesgos hidrometeorológicos']


losses = {}

for x in risks:
    losses[x] = aggloss(data = claims_data, client = x[0:x.rfind('-')], risk = x[x.rfind('-')+1:len(x)],
            years_exposed = 12, years_insured = 1/12, simulations = 100)

for l in losses:
    sns.set_style('whitegrid')
    g = sns.kdeplot(losses[l],
                    color = ''.join(np.random.choice(colors, size = 1)),
                    fill = True,)
    g.set_xticklabels(['{:,.0f}'.format(x) for x in g.get_xticks()/1e6])
    g.set(xlabel = 'Pérdida (MDP)', ylabel = 'Frecuencia relativa')
    plt.axvline(x =  np.quantile(losses[l], 0.995), color = 'black', linestyle = 'dashed', lw = 1)
    plt.text(np.quantile(losses[l], 0.9955),0,'VaR',rotation=0)
    plt.show(g)
    