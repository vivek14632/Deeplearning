import json
from pprint import pprint

with open('10009100_full.json') as f:
    data = json.load(f)

pprint(data)

bdata = data["bootstrapData"]["all_top_destinations"]

import csv
import os
def WriteDictToCSV(csv_file,csv_columns,dict_data):
    csv_file = "C:/Users/Saniya Islam/Desktop/new2.csv"
    csv_columns = ['url','name']
    with open(csv_file, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for data in bdata:
            writer.writerow(data)
    return
WriteDictToCSV(csv_file,csv_columns,data)             

