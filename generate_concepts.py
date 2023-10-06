import pandas as pd
import csv
 

df = pd.read_csv('train.tsv', sep='\t')
df = pd.DataFrame(df)
df = df.values.tolist()
 #printing data

# name of csv file  
filename = "positive.csv"
    
# writing to csv file  
with open(filename, 'w') as csvfile:  
    # creating a csv writer object  
    csvwriter = csv.writer(csvfile)  
    csvwriter.writerow(['Text'])  
    for i in range(len(df)):
        if df[i][2]>0:
            print(df[i][1])
            csvwriter.writerows([[df[i][1]]]) 
        
filename = "neutral.csv"
    
# writing to csv file  
with open(filename, 'w') as csvfile:  
    # creating a csv writer object  
    csvwriter = csv.writer(csvfile)  
    csvwriter.writerow(['Text'])  
    for i in range(len(df)):
        if df[i][2]==0:
            print(df[i][1])
            csvwriter.writerows([[df[i][1]]]) 

