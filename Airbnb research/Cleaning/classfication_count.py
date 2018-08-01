
# coding: utf-8

# In[1]:


import pandas as pd
data_listing=pd.read_csv('/Users/jaideep/Desktop/Airbnb Project Files/Data/new_data/Airbnb_cnn_classification_all_sorted.csv')
df_list = pd.DataFrame(data_listing)
df_list


# In[2]:


df_new=df_list[['id','list']]
#df_new=df_list
df_new


# In[5]:


df_grouped=df_new.groupby('list')['id'].apply(list)

df_indoor=pd.DataFrame(df_grouped['indoor'])
df_outdoor=pd.DataFrame(df_grouped['outdoor'])
df_indoor.columns = ['id']
df_outdoor.columns = ['id']


df_indoor['indoor'] = df_indoor.groupby('id')['id'].transform('count')
df_outdoor['outdoor'] = df_outdoor.groupby('id')['id'].transform('count')

df_indoor.drop_duplicates()
df_outdoor.drop_duplicates()

new_unique=pd.merge(df_indoor,df_outdoor, on='id', how='outer').drop_duplicates().fillna(0)
new_unique


# In[6]:


df_first=df_new.groupby('id', as_index=False).first()


df_first.reset_index()
df_first.columns = ['id','first_image']
df_check=df_first
for i in range(0,df_first.shape[0]):
    if df_first['first_image'][i]=="indoor":
        df_first['first_image'][i]=0
    else:
        df_first['first_image'][i]=1
        
df_first

total=df_new.groupby('id', as_index=False)['list'].size()

df_total=pd.DataFrame({'id':total.index, 'total_images':total.values})

df_total_first=pd.merge(df_first,df_total, on='id', how='outer')
df_total_first


# In[7]:


final_first=pd.merge(new_unique,df_total_first, on='id', how='outer')
final_first

data=pd.read_csv('/Users/jaideep/Desktop/Airbnb Project Files/Data/main/listings 2.csv')
df_data = pd.DataFrame(data)
#del df_data['Unnamed: 0']
df_data

final=pd.merge(final_first,df_data, on='id', how='inner')

final


# In[8]:


final.to_csv("/Users/jaideep/Desktop/Airbnb Project Files/Data/new_data/Airbnb_Merged_Data.csv", sep=',')

