
# coding: utf-8

# In[20]:


import pandas as pd


# In[21]:


data_listing=pd.read_csv('/Users/jaideep/Desktop/Airbnb Project Files/Data/new_data/airbnb_sen_merged.csv')
df_sen = pd.DataFrame(data_listing)
data_ohe=pd.read_csv('/Users/jaideep/Desktop/Airbnb Project Files/Data/new_data/OHE_Airbnb_Str_merge.csv')
df_ohe = pd.DataFrame(data_ohe)
data_read=pd.read_csv('/Users/jaideep/Desktop/Airbnb Project Files/Data/new_data/readability.csv')
df_read = pd.DataFrame(data_read)
data_avail=pd.read_csv('/Users/jaideep/Desktop/Airbnb Project Files/Data/new_data/calendar_avail.csv')
df_avail= pd.DataFrame(data_avail)


# In[22]:


df_sen


# In[24]:


df_merge4=pd.merge(df_sen,df_ohe, on='id', how='inner')
df_merge5=pd.merge(df_merge4,df_read, on='id', how='inner')
df_final_merge=pd.merge(df_merge5,df_avail, on='id', how='inner')


# In[25]:


del df_final_merge["Unnamed: 0_x"]
del df_final_merge["Unnamed: 0_y"]
del df_final_merge["listing_url"]
del df_final_merge["scrape_id"]
del df_final_merge["name_x"]
del df_final_merge["summary"]
del df_final_merge["space"]
del df_final_merge["description"]
del df_final_merge["experiences_offered"]
del df_final_merge["neighborhood_overview"]
del df_final_merge["notes"]
del df_final_merge["transit"]
del df_final_merge["access"]
del df_final_merge["interaction"]
del df_final_merge["house_rules"]
del df_final_merge["thumbnail_url"]
del df_final_merge["medium_url"]
del df_final_merge["picture_url"]
del df_final_merge["xl_picture_url"]
del df_final_merge["host_id"]
del df_final_merge["host_url"]
del df_final_merge["host_name"]
del df_final_merge["host_about"]
del df_final_merge["host_thumbnail_url"]
del df_final_merge["host_picture_url"]
del df_final_merge["street"]
del df_final_merge["host_verifications"]
del df_final_merge["amenities"]
del df_final_merge["calendar_updated"]
del df_final_merge["name_y"]
df_final_merge


# In[26]:


df_final_merge.to_csv("/Users/jaideep/Desktop/Airbnb Project Files/Data/new_data/airbnb_fin_merged.csv", sep=',')

