
# coding: utf-8

# In[2]:


import math, re, string, requests, json
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd

analyzer = SentimentIntensityAnalyzer()

data_listing=pd.read_csv('/Users/jaideep/Desktop/Airbnb Project Files/Data/new_data/reviews_merged.csv')
#data_listing=pd.read_csv('/Users/jaideep/Desktop/reviews_sample.csv')
df_list = pd.DataFrame(data_listing)

d = []
e=[]
f=[]
g=[]
h=[]
for i, row in df_list.iterrows():
    d.append(row['listing_id'])
    e.append(analyzer.polarity_scores(row['merged_reviews'])['compound'])
    f.append(analyzer.polarity_scores(row['merged_reviews'])['pos'])
    g.append(analyzer.polarity_scores(row['merged_reviews'])['neu'])
    h.append(analyzer.polarity_scores(row['merged_reviews'])['neg'])


df_vander = pd.DataFrame(
    {'id': d,
     'reviews_sen_comp': e,
     'reviews_sen_pos': f,
     'reviews_sen_neu': g,
     'reviews_sen_neg': h,
    })


# In[3]:


df_vander


# In[5]:


data_merged=pd.read_csv('/Users/jaideep/Desktop/Airbnb Project Files/Data/new_data/Airbnb_Merged_Data.csv')
df_merged = pd.DataFrame(data_merged).sort_values(by='id', ascending=True)
df_merged


# In[6]:


new_unique=pd.merge(df_merged,df_vander, on='id', how='outer')


del new_unique['Unnamed: 0']
new_unique


# In[7]:


new_unique.to_csv("/Users/jaideep/Desktop/Airbnb Project Files/Data/new_data/airbnb_sen_merged.csv", sep=',')

