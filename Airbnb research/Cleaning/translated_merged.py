
# coding: utf-8

# In[1]:


import sys 
reload(sys)  
sys.setdefaultencoding('utf8')
import pandas as pd
from textblob import TextBlob


data_listing=pd.read_csv('/Users/jaideep/Desktop/Airbnb Project Files/Data/main/reviews.csv')
#data_listing=pd.read_csv('/Users/jaideep/Desktop/reviews_sample.csv')
df_list = pd.DataFrame(data_listing)
df_list
df_new=df_list[['listing_id','comments']]
df_new

data_merged=pd.read_csv('/Users/jaideep/Desktop/Airbnb Project Files/Data/new_data/Airbnb_Merged_Data.csv')
df_merged = pd.DataFrame(data_merged)
df_merged
df_new_merged=df_merged[['id']]
df_new_merged.columns = ['listing_id']

new_unique=pd.merge(df_new,df_new_merged, on='listing_id', how='inner').sort_values(by='listing_id', ascending=True)

d = []
e=[]
for i, row in new_unique.iterrows():
    en_blob = TextBlob(str(row['comments']))
    if len(en_blob)>3:
        lan=en_blob.detect_language().encode("utf-8")
        print(i)
        if lan !='nl':
            if lan=='en':
                d.append(row['listing_id'])
                e.append(str(row['comments']))
            else:
                d.append(row['listing_id'])
                tanslated=en_blob.translate(to='en')
                e.append(str(tanslated)) 
df_sentiment = pd.DataFrame(
    {'listing_id': d,
     'comments': e
    })

df_sentiment


# In[2]:


df_sentiment.to_csv("/Users/jaideep/Desktop/Airbnb Project Files/Data/new_data/reviews_translated.csv", sep=',')

df_adwda=df_sentiment.groupby('listing_id', sort=False).comments.apply(''.join).reset_index(name='merged_reviews')
df_adwda

df_adwda.to_csv("/Users/jaideep/Desktop/Airbnb Project Files/Data/new_data/reviews_merged.csv", sep=',')

