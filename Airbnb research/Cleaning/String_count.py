
# coding: utf-8

# In[2]:


import pandas as pd


# In[3]:


data_listing=pd.read_csv('/Users/jaideep/Desktop/Airbnb Project Files/Data/new_data/airbnb_sen_merged.csv')
df_list = pd.DataFrame(data_listing)
df_list
df_new=df_list[['id','name','host_about','summary','space','description','neighborhood_overview','transit','access','interaction','house_rules','notes']]
df_new


# In[4]:


len(df_new['name'][0]) - df_new['name'][0].count(' ')


# In[5]:


d=[]
e=[];f=[];g=[];h=[];i=[];j=[];k=[];l=[];m=[];n=[];o=[];p=[];q=[]
for a, row in df_new.iterrows():
    d.append(row['id'])
    e.append(len(row['name']) - row['name'].count(' '))
    f.append(len(str(row['host_about'])) - str(row['host_about']).count(' '))
    g.append(len(str(row['summary'])) - str(row['summary']).count(' '))
    h.append(len(str(row['space'])) - str(row['space']).count(' '))
    i.append(len(str(row['description'])) - str(row['description']).count(' '))
    j.append(len(str(row['neighborhood_overview'])) - str(row['neighborhood_overview']).count(' '))
    k.append(len(str(row['transit'])) - str(row['transit']).count(' '))
    l.append(len(str(row['access'])) - str(row['access']).count(' '))
    m.append(len(str(row['interaction'])) - str(row['interaction']).count(' '))
    n.append(len(str(row['house_rules'])) - str(row['house_rules']).count(' '))
    o.append(len(str(row['notes'])) - str(row['notes']).count(' '))
df_string_count = pd.DataFrame(
    {'id': d,
     'name_string_count': e,
     'host_about_string_count': f,
     'summary_string_count': g,
     'space_string_count': h,
     'desc_string_count': i,
     'neigh_string_count': j,
     'transit_string_count': k,
     'access_string_count': l,
     'interaction_string_count': m,
     'houseRules_string_count': n,
     'notes_string_count': o
    })

df_string_count


# In[6]:


df_string_count.to_csv("/Users/jaideep/Desktop/Airbnb Project Files/Data/new_data/strng_count.csv", sep=',')

