"""
DESCRIPTION
    Script to create a nice metric plots using the model_history.csv file created by model_training_flow.py
    
AUTHOR:
    Magda Guglielmo <magda.guglielmo@csiro.au>
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
plt.rcParams['axes.labelcolor']='#575757'
plt.rcParams['xtick.color']='#707070'
plt.rcParams['ytick.color']='#707070'
plt.rcParams['axes.edgecolor']='#707070'
plt.rcParams['axes.prop_cycle']=cycler('color',['#0D4C87','#58a278','#397f58','#004B87','#00855B','#E67623','#E40028','#FFB61C','#9faee5','#6D2077','#142C3F',
'#1E22AA','#DC1894'])
plt.rcParams['xtick.minor.visible']=True
plt.rcParams['ytick.minor.visible']=True


df = pd.read_csv('model_history.csv')

df['epoch']  = df['epoch']+1

fig, ax = plt.subplots(2, 1, sharex=True)

ax[0].plot(df['epoch'],df['accuracy'],'-',label='Train')
ax[0].plot(df['epoch'],df['val_accuracy'],'-',label='Validation')
ax[0].vlines(30,0,2,linestyle='dashed',lw=1,color=(0.25,0.25,0.25),label='Selected model')
ax[0].legend(frameon=False)
ax[0].set_ylabel('Accuracy')
ax[0].set_xlim(0,70)
ax[0].set_ylim(0,1.05)
#
#ax[0].set_xticks(np.linspace(0,70,20))
ax[1].plot(df['epoch'],df['loss'],'-',label='Train')
ax[1].plot(df['epoch'],df['val_loss'],'-',label='Validation')
ax[1].set_ylabel('Categorical cross entropy')
ax[1].set_xlabel('Epoch')

ax[1].vlines(30,0,2,linestyle='dashed',lw=1,color=(0.25,0.25,0.25,1),label='Selected model')
ax[1].set_xlim(0,70)
ax[1].set_ylim(0,1.05)

plt.savefig('training_loss_acc.png')
