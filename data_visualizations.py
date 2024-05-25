#!/usr/bin/env python
# coding: utf-8

# In[19]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# In[31]:


df1 = pd.read_csv('./graphics/downy_no_aug.csv')
df2 = pd.read_csv('./graphics/healthy_no_aug.csv')
df3 = pd.read_csv('./graphics/powdery_no_aug.csv')
df4 = pd.read_csv('./graphics/all_classes_no_aug.csv')
df5 = pd.read_csv('./graphics/powdery_downy_no_aug.csv')

df1.rename(columns={'Step': 'Epoch', 'Value':'Loss'}, inplace=True)
df2.rename(columns={'Step': 'Epoch', 'Value':'Loss'}, inplace=True)
df3.rename(columns={'Step': 'Epoch', 'Value':'Loss'}, inplace=True)
df4.rename(columns={'Step': 'Epoch', 'Value':'Loss'}, inplace=True)
df5.rename(columns={'Step': 'Epoch', 'Value':'Loss'}, inplace=True)


# In[32]:


df1['key'] = 'downy_no_aug'
df2['key'] = 'healthy_no_aug'
df3['key'] = 'powdery_no_aug'
df4['key'] = 'all_classes_no_aug'
df5['key'] = 'powdery_downy_no_aug'


# In[33]:


df = pd.concat([df1, df2, df3, df4, df5])


# In[34]:


plt.figure(figsize=(12, 8))
sns.lineplot(data=df, x='Epoch', y='Loss', hue='key')
plt.show()


# In[46]:


no_aug = pd.read_csv('./graphics/powdery_no_aug.csv')
aug_500 = pd.read_csv('./graphics/powdery_aug_500.csv')

no_aug.rename(columns={'Step': 'Epoch', 'Value':'Loss'}, inplace=True)
aug_500.rename(columns={'Step': 'Epoch', 'Value':'Loss'}, inplace=True)
no_aug['key'] = 'powdery_no_aug'
aug_500['key'] = 'powdery_500_aug'

powdery_aug = pd.concat([no_aug, aug_500])

plt.figure(figsize=(12, 8))
sns.lineplot(data=powdery_aug, x='Epoch', y='Loss', hue='key')
plt.show()


# In[51]:


aug_1000 = pd.read_csv('./graphics/powdery_aug_1000.csv')
aug_1000['key'] = 'powdery_1000aug'
aug_1000.rename(columns={'Step': 'Epoch', 'Value':'Loss'}, inplace=True)
df_augs = pd.concat([no_aug,aug_500, aug_1000])
plt.figure(figsize=(12, 8))
sns.lineplot(data=df_augs, x='Epoch', y='Loss', hue='key')
plt.show()

