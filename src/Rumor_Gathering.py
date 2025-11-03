import pandas as pd
import numpy as np
import re
import time
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from urllib.parse import urljoin, urlparse
import hashlib
from functools import lru_cache
import random
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import sys
sys.path.append('..\..\src')
from PY_Class_Def import find_player_id_by_search,get_player_rumours_archive

df=pd.read_csv(r'..\DataSources\Processed\Cleaned_Final_Stats_test.csv')



pn_df=[]
name_player=list(df['Player'].unique())

for np in name_player:
    player_name=np
    player_id_=find_player_id_by_search(player_name)

    pn_df.append(
        {
            'Player':player_name,
            'Player-ID':int(player_id_) if player_id_ else 'No ID'
        }
    )

name_id=pd.DataFrame(pn_df)

rumor_df=pd.DataFrame()
for id in list(name_id['Player-ID']):
    archive_rumor = get_player_rumours_archive(id, tld="de", debug=True)
    archive_rumor['Player']=name_id[name_id['Player-ID'].eq(id)]['Player'].iloc[0]
    rumor_df=pd.concat([rumor_df,archive_rumor],axis=0)
rumor_df.to_csv('..\DataSources\Processed\Rumor_Overview.csv',index=False)

club_league_df=df[['Club_Joined','League_Joined']].drop_duplicates()
club_league_df.rename(columns={'Club_Joined':'club_name'},inplace=True)

df_rumor=pd.read_csv(r'..\DataSources\Processed\Rumor_Overview.csv').drop(['probability_raw','page','probability_pct','club_id'],axis=1)
df_rumor['last_source_dt'] = pd.to_datetime(df_rumor['last_source_dt'], errors='coerce')
df_rumor['Performance_Year']=df_rumor['last_source_dt'].dt.year
#df_rumor['Performance_Month']=df_rumor['last_source_dt'].dt.month
df_rumor.drop(['last_source','last_answer','last_source_dt',],axis=1,inplace=True)

df_final=pd.merge(df_rumor,club_league_df,on='club_name',how='left')

df_rumor_agg=df_final.groupby(
    [
        'Player', 'Performance_Year', 'League_Joined'
    ]
    ).size().unstack(fill_value=0).rename(columns=lambda c: f'{c}_Rumors').reset_index().rename_axis(columns=' ')

df_final=pd.merge(df,df_rumor_agg,on=['Player','Performance_Year'],how='left').fillna(0)
df_final.to_csv('..\DataSources\Processed\Cleaned_Final_Stats_w_Rumors.csv',index=False)