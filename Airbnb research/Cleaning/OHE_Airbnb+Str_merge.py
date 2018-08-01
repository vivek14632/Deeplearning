
# coding: utf-8

# In[1]:


import pandas as pd
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


# In[2]:


data_listing=pd.read_csv('/Users/jaideep/Desktop/Airbnb Project Files/Data/new_data/airbnb_sen_merged.csv')
df_imported = pd.DataFrame(data_listing)
df_new=df_imported[["host_verifications","id"]]
d=[];e=[]
for i, row in df_new.iterrows():
    arr=[]
    arr = row["host_verifications"][1:-1].replace("'", "").replace(" ", "").split(",")
    for j in range(0,len(arr)):
        d.append(row['id'])
        e.append(arr[j])
df_list = pd.DataFrame(
    {'id': d,
     'list': e
    })  
#OHE by scikit
# data = df_list["list"]
# values = array(df_list["list"])
# # binary encode
# label_encoder = LabelEncoder()
# integer_encoded = label_encoder.fit_transform(values)
# onehot_encoder = OneHotEncoder(sparse=False)
# integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
# onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

# df_OHE=pd.DataFrame(data=onehot_encoded[0:,0:])
# df_OHE["id"]=df_list["id"]
# df_OHE.columns = ['fb_veri_host','google_veri_host','govt_id_veri_host','jumio_veri_host','linkedin_veri_host','manual_offline_veri_host','manual_online_veri_host',' offline_gov_id_veri','phone_veri_host','reviews_veri_host','work_email_veri_host','email_veri_host','id']
# df_OHE=df_OHE.groupby(['id']).sum()
# df_OHE=df_OHE.reset_index(drop=False)
# df_OHE
# df_list

# pandas dummies method for OHE
df_OHE = pd.get_dummies(df_list, columns=['list']).groupby(['id']).sum()
del df_OHE["list_"]
df_OHE=df_OHE.reset_index(drop=False)
df_OHE


# In[3]:


df_OHE.to_csv("/Users/jaideep/Desktop/Airbnb Project Files/Data/new_data/ohe1.csv", sep=',')


# In[4]:


df_new1=df_imported[["amenities","id"]]
d=[];e=[]
for i, row in df_new1.iterrows():
    arr=[]
    arr = row["amenities"][1:-1].replace('"', '').replace(" ", "-").replace("\xe2\x80\x99s", "").replace("\xe2\x80\x99n", "").split(",")
    for j in range(0,len(arr)):
        if arr[j]!="translation-missing:-en.hosting_amenity_49" and arr[j] !="translation-missing:-en.hosting_amenity_50":
            d.append(row['id'])
            e.append(arr[j])
df_output = pd.DataFrame(
    {'id': d,
     'ame': e
    })  
# #OHE
# data1 = df_output["list"]
# values1 = array(df_output["list"])
# # binary encode
# label_encoder1 = LabelEncoder()
# integer_encoded1 = label_encoder.fit_transform(values1)
# print(integer_encoded1)
# onehot_encoder1 = OneHotEncoder(sparse=False)
# integer_encoded1 = integer_encoded1.reshape(len(integer_encoded1), 1)
# onehot_encoded1 = onehot_encoder.fit_transform(integer_encoded1)

# df_OHE1=pd.DataFrame(data=onehot_encoded1[0:,0:])
# df_OHE1["id"]=df_output["id"]
# #df_OHE1.columns = ['fb_veri_host','google_veri_host','govt_id_veri_host','jumio_veri_host','linkedin_veri_host','manual_offline_veri_host','manual_online_veri_host',' offline_gov_id_veri','phone_veri_host','reviews_veri_host','work_email_veri_host','email_veri_host','id']
# df_OHE1=df_OHE1.groupby(['id']).sum()
# df_OHE1


df_output["ame"].unique()


# In[5]:


df_ame = pd.get_dummies(df_output, columns=['ame']).groupby(['id']).sum()
del df_ame["ame_"]
df_ame=df_ame.reset_index(drop=False)
df_ame


# In[6]:


df_ame.to_csv("/Users/jaideep/Desktop/Airbnb Project Files/Data/new_data/ohe2.csv", sep=',')


# In[7]:


df_new2=df_imported[["calendar_updated","id"]]

d=[];e=[]
for i, row in df_new2.iterrows():
    arr=[]
    arr=row["calendar_updated"].split(" ")
    if len(arr)>2:
        if str(arr[1])=="months":
            d.append(row['id'])
            e.append(int(arr[0])*30)
        elif str(arr[1])=="week" or str(arr[1])=="weeks":
            if str(arr[0])=='a':
                d.append(row['id'])
                e.append(7)
            else:
                d.append(row['id'])
                e.append(int(arr[0])*7)
        else:
            d.append(row['id'])
            e.append(int(arr[0])) 
    elif str(arr[0])=="today":
        d.append(row['id'])
        e.append(0)
    elif str(arr[0])=="yesterday":
        d.append(row['id'])
        e.append(1)
    else:
        d.append(row['id'])
        e.append("")
            
df_output = pd.DataFrame(
    {'id': d,
     'calendar_updated(days-ago)': e
    }) 
df_output


# In[8]:


df_merge1=pd.merge(df_ame,df_OHE, on='id', how='inner')
df_merge2=pd.merge(df_merge1,df_output, on='id', how='inner')
df_merge2


# In[23]:


data_str=pd.read_csv('/Users/jaideep/Desktop/Airbnb Project Files/Data/new_data/strng_count.csv')
df_str_imp = pd.DataFrame(data_str)
df_str_imp
df_merge3=pd.merge(df_merge2,df_str_imp, on='id', how='inner')
df_merge3


# In[24]:


df_merge3.to_csv("/Users/jaideep/Desktop/Airbnb Project Files/Data/new_data/OHE_Airbnb_Str_merge.csv", sep=',')


# In[25]:


df_merge2.columns


# In[9]:


#for stata encoding
for column in df_merge2:
    stuff=" tostring ame_airconditioning, generate(ame_airconditioning_temp) \n encode ame_airconditioning_temp,gen(ame_airconditioning_temp1) \n drop ame_airconditioning ame_airconditioning_temp \n rename ame_airconditioning_temp1 ame_airconditioning \n tab ame_airconditioning"
    print(stuff.replace("ame_airconditioning",str(column).lower()).replace("-","").replace("/","").replace("(s)","s"))

