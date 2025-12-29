import pandas as pd
import numpy as np
import sys
from .transfer_target_data_cleaning import top_or_others
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FINAL_STATS = ROOT / "DataSources"/"Processed"/"Final_Stats_2000_2025.csv"
CLEANED_FINAL_STATS = ROOT / "DataSources"/"Processed"/"Cleaned_Final_Stats.csv"

#this will become part of the clean python file
df=pd.read_csv(FINAL_STATS).iloc[:,1:]
for c in [('Top_Left','League_Left'),('Top_Joined','League_Joined')]:
    df[c[0]]=df[c[1]].apply(lambda x: 1 if x in ['Premier League','LaLiga','Bundesliga','Serie A','Ligue 1'] else 0)
df=df[(df['Top_Left']==1) | (df['Top_Joined']==1)].iloc[:,:-2]
df['League_Joined']=df['League_Joined'].map(top_or_others)
print('New CSV file generated')
df.to_csv(CLEANED_FINAL_STATS,index=False)