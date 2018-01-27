
# coding: utf-8

# In[10]:


import pandas as pd
data_listing=pd.read_csv('/Users/jaideep/Desktop/Airbnb Project Files/Data/new_data/Airbnb_Merged_Data.csv')
df_list = pd.DataFrame(data_listing)


# In[11]:


df_list_id=df_list[["id"]]


# In[82]:


data_listing_cal=pd.read_csv('/Users/jaideep/Desktop/Airbnb Project Files/Data/main/calendar.csv')
df_list_cal = pd.DataFrame(data_listing_cal)
df_list_cal.columns = ['id','date','available','price']
df_list_cal


# In[83]:


df_total_first=pd.merge(df_list_id,df_list_cal, on='id', how='inner')
type(df_total_first["date"][0])


# In[84]:


df_total_first["Enc_date"]=pd.to_datetime(df_total_first["date"])


# In[114]:


# 30 days filter
df_date_filter=df_total_first[(df_total_first['Enc_date'] > '2017-04-02') & (df_total_first['Enc_date'] < '2017-05-03')]

df_date_filter[["available","id"]]

df_date_filter["available_30"]=pd.Series(map(lambda x: dict(t=1, f=0)[x],df_date_filter["available"].tolist()), df_date_filter.index)

df_date_filter=df_date_filter[["id","available_30"]]
df_date_filter=df_date_filter.groupby('id').sum().reset_index()


# In[115]:


#1 day filter
df_date_filter1=df_total_first[(df_total_first['Enc_date'] > '2017-04-02') & (df_total_first['Enc_date'] < '2017-04-04')]

df_date_filter1[["available","id"]]

df_date_filter1["available_1"]=pd.Series(map(lambda x: dict(t=1, f=0)[x],df_date_filter1["available"].tolist()), df_date_filter1.index)

df_date_filter1=df_date_filter1[["id","available_1"]]
df_date_filter1=df_date_filter1.groupby('id').sum().reset_index()


# In[118]:


#15 days filter
df_date_filter2=df_total_first[(df_total_first['Enc_date'] > '2017-04-02') & (df_total_first['Enc_date'] < '2017-04-18')]

df_date_filter2[["available","id"]]

df_date_filter2["available_15"]=pd.Series(map(lambda x: dict(t=1, f=0)[x],df_date_filter2["available"].tolist()), df_date_filter2.index)

df_date_filter2=df_date_filter2[["id","available_15"]]
df_date_filter2=df_date_filter2.groupby('id').sum().reset_index()
df_date_filter2


# In[119]:


#7 days filter
df_date_filter3=df_total_first[(df_total_first['Enc_date'] > '2017-04-02') & (df_total_first['Enc_date'] < '2017-04-10')]

df_date_filter3[["available","id"]]

df_date_filter3["available_7"]=pd.Series(map(lambda x: dict(t=1, f=0)[x],df_date_filter3["available"].tolist()), df_date_filter3.index)

df_date_filter3=df_date_filter3[["id","available_7"]]
df_date_filter3=df_date_filter3.groupby('id').sum().reset_index()
df_date_filter3


# In[113]:


new_unique=pd.merge(df_date_filter,df_date_filter1, on='id', how='inner')
new_unique


# In[120]:


new_unique1=pd.merge(df_date_filter2,df_date_filter3, on='id', how='inner')
new_unique1


# In[121]:


new_unique2=pd.merge(new_unique,new_unique1, on='id', how='inner')
new_unique2


# In[122]:


new_unique2.to_csv("/Users/jaideep/Desktop/Airbnb Project Files/Data/new_data/calendar_avail.csv", sep=',')

