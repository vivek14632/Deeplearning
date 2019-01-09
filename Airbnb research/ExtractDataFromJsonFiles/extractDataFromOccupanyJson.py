# Purpose: Extract data from Occupancy Json
#!/usr/bin/env python
# coding: utf-8

# In[5]:


import glob
import json
import os
import csv
import pandas as pd
mycsvdir =os.getcwd()

col = ['listing_id','date','available','local_price']
csv_file = "new.csv" 

def writeHeader():
    f = open(csv_file, "w")
    writer = csv.DictWriter(f, fieldnames=col,lineterminator='\n')
    writer.writeheader()
    f.close()
writeHeader()

jsonfiles = glob.glob(os.path.join(mycsvdir, '*.json'))
try:
    for jsonfile in jsonfiles:
        with open(jsonfile) as f:
            data = json.load(f)
            dict_data = data["calendar_months"]
        def writeTocsv(row):
            with open(csv_file, 'a') as csvFile:
                writer = csv.writer(csvFile,lineterminator='\n')
                writer.writerow(row)
            csvFile.close()
        def parseData(): 
            for i in range(len(dict_data)):
                month= dict_data[i]
                listing_id= month["listing_id"]
                days = month["days"]
                date = []
                local_price=[]
                available =[]
                for i in range(len(days)):
                    available.append(days[i]["available"])
                    date.append(days[i]["date"])
                    local_price.append(days[i]["price"]["local_price"])   
            row = [listing_id,date,available,local_price]
            writeTocsv(row)
        parseData()
except:
    print("Exception")


# In[ ]:




