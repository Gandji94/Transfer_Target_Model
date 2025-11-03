import pandas as pd
import numpy as np
import time, random, re, requests
from bs4 import BeautifulSoup, Comment
from requests.adapters import HTTPAdapter, Retry
import sys
sys.path.append('..\..\src')
from PY_Class_Def import all_trans_window,trans_value,player_stats,limited_time,age_transformer,clean_league_joined_aut,clean_league_left_aut,_playwright_get_html,make_fbref_session,fetch_html,_table_tag_to_df,read_fbref_table,player_stats_update,log,fix_min,best_nation_series,player_stats_update_final,to_iso3,nation_from_name,_norm_name
from Transfer_Target_Data_Cleaning import top_or_others,league_class_trans
import logging

#setting up logging info
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)

#webscrapper for Transfermarket.de
try:
    all_trans_df=all_trans_window(2000)
    logging.info(f'2000 done')
except Exception as ex:
    logging.error(f'During the 2000 Transfer Scrapping an error occured: {repr(ex)}')

try:
    year_run=[]
    for x in range(2001,2025):
        loop_df=all_trans_window(x)
        logging.info(f'{x} done')
        all_trans_df=pd.concat([all_trans_df,loop_df],ignore_index=True)
        year_run.append(x)
    logging.info('Alle done')

    all_trans_df['Transfer_Fee']=all_trans_df['Transfer_Fee'].apply(trans_value)

    all_trans_df.to_csv('..\DataSources\Processed\All_Trans_2000_2025_test.csv')
    logging.info('Transfer done\n')
except Exception as ex:
    logging.error()

#data gathering for FBREF
#Premier League Stats from 2017-2025
try:
    if __name__ == "__main__":
        # Run older seasons first; current season last (stricter protection)
        years = ['2024-2025','2023-2024','2022-2023','2021-2022','2020-2021','2019-2020','2018-2019','2017-2018']

        dfs = []
        for y in years:
            logging.info(f"=== Season {y} ===")
            base = f"https://fbref.com/en/comps/9/{y}"
            df_y = player_stats_update_final(
                f"{base}/stats/{y}-Premier-League-Stats",
                f"{base}/defense/{y}-Premier-League-Stats",
                f"{base}/passing/{y}-Premier-League-Stats",
                f"{base}/passing_types/{y}-Premier-League-Stats",
                f"{base}/possession/{y}-Premier-League-Stats",
                f"{base}/gca/{y}-Premier-League-Stats",
                f"{base}/keepers/{y}-Premier-League-Stats",
                f"{base}/keepersadv/{y}-Premier-League-Stats",
                "Premier League"
            )
            df_y['Year'] = y.split('-')[1]  # e.g., '2025' for '2024-2025'
            dfs.append(df_y)
            time.sleep(random.uniform(0.5, 1.0))

        df_pl = pd.concat(dfs, ignore_index=True)
        #df_pl['Nation']=df_pl['Nation'][0][0:3]
        #df_pl['Pos'].replace({0:'GK'},inplace=True)
        df_pl.drop(['Pos'],axis=1,inplace=True)
        logging.info(f"Done. Combined rows: {len(df_pl)}")
    df_pl.rename(columns={'Year':'Performance_Year'},inplace=True)
    df_pl.to_csv('..\DataSources\Raw\Premier_League_Stats_2017_2025_test.csv')
    logging.info('PL Done\n')
except Exception as ex:
    logging.error(f'During the Premier League Stats collection an Error occured: {repr(ex)}')

#La Liga Stats from 2017-2025
try:
    if __name__ == "__main__":
        # Run older seasons first; current season last (stricter protection)
        years = ['2024-2025','2023-2024','2022-2023','2021-2022','2020-2021','2019-2020','2018-2019','2017-2018']

        dfs = []
        for y in years:
            logging.info(f"=== Season {y} ===")
            base = f"https://fbref.com/en/comps/12/{y}"
            df_y = player_stats_update_final(
                f"{base}/stats/{y}-La-Liga-Stats",
                f"{base}/defense/{y}-La-Liga-Stats",
                f"{base}/passing/{y}-La-Liga-Stats",
                f"{base}/passing_types/{y}-La-Liga-Stats",
                f"{base}/possession/{y}-La-Liga-Stats",
                f"{base}/gca/{y}-La-Liga-Stats",
                f"{base}/keepers/{y}-La-Liga-Stats",
                f"{base}/keepersadv/{y}-La-Liga-Stats",
                "La Liga"
            )
            df_y['Year'] = y.split('-')[1]  # e.g., '2025' for '2024-2025'
            dfs.append(df_y)
            time.sleep(random.uniform(0.5, 1.0))

        df_esp = pd.concat(dfs, ignore_index=True)
        #df_esp['Nation']=df_esp['Nation'][0][0:3]
        #df_esp['Pos'].replace({0:'GK'},inplace=True)
        df_esp.drop(['Pos'],axis=1,inplace=True)
        logging.info(f"Done. Combined rows: {len(df_esp)}")
    df_esp.rename(columns={'Year':'Performance_Year'},inplace=True)
    df_esp.to_csv('..\DataSources\Raw\La_Liga_Stats_2017_2025_test.csv')
    logging.info('LaLiga Done\n')
except Exception as ex:
    logging.error(f'During the LaLiga Stats collection an Error occured: {repr(ex)}')

#Serie A Stats from 2017-2025
try:
    if __name__ == "__main__":
        # Run older seasons first; current season last (stricter protection)
        years = ['2024-2025','2023-2024','2022-2023','2021-2022','2020-2021','2019-2020','2018-2019','2017-2018']

        dfs = []
        for y in years:
            logging.info(f"=== Season {y} ===")
            base = f"https://fbref.com/en/comps/11/{y}"
            df_y = player_stats_update_final(
                f"{base}/stats/{y}-Serie-A-Stats",
                f"{base}/defense/{y}-Serie-A-Stats",
                f"{base}/passing/{y}-Serie-A-Stats",
                f"{base}/passing_types/{y}-Serie-A-Stats",
                f"{base}/possession/{y}-Serie-A-Stats",
                f"{base}/gca/{y}-Serie-A-Stats",
                f"{base}/keepers/{y}-Serie-A-Stats",
                f"{base}/keepersadv/{y}-Serie-A-Stats",
                "Serie A"
            )
            df_y['Year'] = y.split('-')[1]  # e.g., '2025' for '2024-2025'
            dfs.append(df_y)
            time.sleep(random.uniform(0.5, 1.0))

        df_seriea = pd.concat(dfs, ignore_index=True)
        #df_seriea['Nation']=df_seriea['Nation'][0][0:3]
        #df_seriea['Pos'].replace({0:'GK'},inplace=True)
        df_seriea.drop(['Pos'],axis=1,inplace=True)
        logging.info(f"Done. Combined rows: {len(df_seriea)}")
    df_seriea.rename(columns={'Year':'Performance_Year'},inplace=True)
    df_seriea.to_csv('..\DataSources\Raw\Serie_A_Stats_2017_2025_test.csv')
    logging.info('SerieA Done\n')
except Exception as ex:
    logging.error(f'During the Serie A Stats collection an Error occured: {repr(ex)}')


#BundesLiga Stats from 2017-2025
try:
    if __name__ == "__main__":
        # Run older seasons first; current season last (stricter protection)
        years = ['2024-2025','2023-2024','2022-2023','2021-2022','2020-2021','2019-2020','2018-2019','2017-2018']

        dfs = []
        for y in years:
            logging.info(f"=== Season {y} ===")
            base = f"https://fbref.com/en/comps/20/{y}"
            df_y = player_stats_update_final(
                f"{base}/stats/{y}-Bundesliga-Stats",
                f"{base}/defense/{y}-Bundesliga-Stats",
                f"{base}/passing/{y}-Bundesliga-Stats",
                f"{base}/passing_types/{y}-Bundesliga-Stats",
                f"{base}/possession/{y}-Bundesliga-Stats",
                f"{base}/gca/{y}-Bundesliga-Stats",
                f"{base}/keepers/{y}-Bundesliga-Stats",
                f"{base}/keepersadv/{y}-Bundesliga-Stats",
                "BundesLiga"
            )
            df_y['Year'] = y.split('-')[1]  # e.g., '2025' for '2024-2025'
            dfs.append(df_y)
            time.sleep(random.uniform(0.5, 1.0))

        df_BL = pd.concat(dfs, ignore_index=True)
        #df_BL['Nation']=df_BL['Nation'][0][0:3]
        #df_BL['Pos'].replace({0:'GK'},inplace=True)
        df_BL.drop(['Pos'],axis=1,inplace=True)
        logging.info(f"Done. Combined rows: {len(df_BL)}")
    df_BL.rename(columns={'Year':'Performance_Year'},inplace=True)
    df_BL.to_csv('..\DataSources\Raw\BundesLiga_Stats_2017_2025_test.csv')
    logging.info('Bundesliga Done\n')
except Exception as ex:
    logging.error(f'During the Bundesliga Stats collection an Error occured: {repr(ex)}')

#Ligue1 Stats from 2017-2025
try:
    if __name__ == "__main__":
        # Run older seasons first; current season last (stricter protection)
        years = ['2024-2025','2023-2024','2022-2023','2021-2022','2020-2021','2019-2020','2018-2019','2017-2018']

        dfs = []
        for y in years:
            logging.info(f"=== Season {y} ===")
            base = f"https://fbref.com/en/comps/13/{y}"
            df_y = player_stats_update_final(
                f"{base}/stats/{y}-Ligue-1-Stats",
                f"{base}/defense/{y}-Ligue-1-Stats",
                f"{base}/passing/{y}-Ligue-1-Stats",
                f"{base}/passing_types/{y}-Ligue-1-Stats",
                f"{base}/possession/{y}-Ligue-1-Stats",
                f"{base}/gca/{y}-Ligue-1-Stats",
                f"{base}/keepers/{y}-Ligue-1-Stats",
                f"{base}/keepersadv/{y}-Ligue-1-Stats",
                "Ligue 1"
            )
            df_y['Year'] = y.split('-')[1]  # e.g., '2025' for '2024-2025'
            dfs.append(df_y)
            time.sleep(random.uniform(0.5, 1.0))

        df_L1 = pd.concat(dfs, ignore_index=True)
        #df_L1['Nation']=df_L1['Nation'][0][0:3]
        #df_L1['Pos'].replace({0:'GK'},inplace=True)
        df_L1.drop(['Pos'],axis=1,inplace=True)
        logging.info(f"Done. Combined rows: {len(df_L1)}")
    df_L1.rename(columns={'Year':'Performance_Year'},inplace=True)
    df_L1.to_csv('..\DataSources\Raw\Ligue1_Stats_2017_2025_test.csv')
    logging.info('L1 Done\n')
except Exception as ex:
    logging.error(f'During the Ligue 1 Stats collection an Error occured: {repr(ex)}')

logging.info('Combining all Leagues')
try:
    df_final=pd.concat([df_pl,df_esp],axis=0)
    for df in [df_seriea,df_BL,df_L1]:
        df_final=pd.concat([df_final,df],axis=0)
    #df_final.drop([ 'Gls.1','Ast.1','G+A.1','G-PK.1','Pos','Rk'])
    #df_final.rename({'Main_Pos':'Pos'})
    df_final.to_csv('..\DataSources\Processed\Player_Stats_2017_2025_test.csv')


    #Combining datasets
    df_player_stats=pd.read_csv(r'..\DataSources\Processed\Player_Stats_2017_2025_test.csv').iloc[:,1:]
    df_player_stats['Min'] = df_player_stats['Min'].apply(fix_min)
    df_player_stats.drop(['Gls.1','Ast.1','G+A.1','G-PK.1','Rk','Rk.1','90s.1','Sec_Pos','Matches'],axis=1,inplace=True)
    df_player_stats.rename(columns={'Main_Pos':'Pos','Launch%':'Launch%_Passes','Launch%.1':'Launch%_Goal_Kicks','AvgLen':'AvgLen_Passes','AvgLen.1':'AvgLen_Goal_Kicks',
                                    'Att.1':'Goal_Kick_Att','Save%.1':'Penalty_Kick_Saves'},inplace=True)
    #test=test[(test['Sec_Pos']!='GK')]

    rev=list(df_player_stats.columns)
    #rev.remove('Sec_Pos')
    df_player_stats=pd.read_csv(r'..\DataSources\Processed\Player_Stats_2017_2025_test.csv').iloc[:,1:]
    df_player_stats['Min'] = df_player_stats['Min'].apply(fix_min)
    df_player_stats.drop(['Gls.1','Ast.1','G+A.1','G-PK.1','Rk','Rk.1','90s.1','Sec_Pos','Matches'],axis=1,inplace=True)
    df_player_stats.rename(columns={'Main_Pos':'Pos','Launch%':'Launch%_Passes','Launch%.1':'Launch%_Goal_Kicks','AvgLen':'AvgLen_Passes','AvgLen.1':'AvgLen_Goal_Kicks',
                                    'Att.1':'Goal_Kick_Att','Save%.1':'Penalty_Kick_Saves'},inplace=True)
    #df_player_stats=df_player_stats[(df_player_stats['Sec_Pos']!='-') & (df_player_stats['Pos']!='-')]
    goal_agg=list(df_player_stats)[5:-2]
    for r in ['Pos']:
        goal_agg.remove(r)
    df_GK=df_player_stats[(df_player_stats['Pos']=='GK')].groupby(['Player','Nation','Pos','Squad','Age','Born','League','Performance_Year'])[goal_agg].sum().reset_index()
    df_GK=df_GK[rev]


    df_player_stats=pd.concat([df_player_stats,df_GK],axis=0)
    #grouping the selected features and getting the average based on the grouping features
    df_player_stats['Limited_Time']=df_player_stats.apply(limited_time,axis=1)
    df_player_stats['Age_Num']=df_player_stats['Age'].copy()
    df_player_stats['Age']=df_player_stats['Age'].apply(age_transformer)
    #df_player_stats.drop(['Sec_Pos'],axis=1,inplace=True)
    group=list(df_player_stats.columns)[:5]
    group.append(df_player_stats.columns[-4])
    group.append(df_player_stats.columns[25])
    by=list(df_player_stats.columns)[5:-4]
    by.remove('Pos')
    for c in df_player_stats.columns[-3:]:
        by.append(c)
    mean_player_stats=df_player_stats.groupby(group)[by].mean().reset_index()

    group=list(df_player_stats.columns)[:5]
    group.append(df_player_stats.columns[-4])
    group.append(df_player_stats.columns[25])
    by=list(df_player_stats.columns)[-3]
    max_year=df_player_stats.groupby(group)[by].max().to_frame().reset_index()

    df_stats_new=pd.merge(mean_player_stats,max_year,on=list(max_year.columns)[:-1],how='left')
    df_stats_new.drop(['Performance_Year_x'],axis=1,inplace=True)
    df_stats_new.rename(columns={'Performance_Year_y':'Transfer_Window'},inplace=True)
    col=df_stats_new.pop('Pos')
    df_stats_new.insert(3,'Pos',col)
    #data set with all leagues
    df_stats_new.to_csv('..\DataSources\Processed\Player_Stats_Mean_2017_2025_test.csv')
    logging.info('Player_Stats_Mean_2017_2025 Done\n')
except Exception as ex:
    logging.error(f'During League Stats Combination Phase an Error occured: {repr(ex)}')
try:
    #creating mean stats
    df_mean_stats=pd.read_csv(r'..\DataSources\Processed\Player_Stats_Mean_2017_2025_test.csv').iloc[:,1:]

    #transfer dataframe
    df_trans=pd.read_csv(r'..\DataSources\Processed\All_Trans_2000_2025_test.csv').iloc[:,1:]
    df_trans.drop(['Age'],axis=1,inplace=True)
    #appyling trans_value
    #df_trans['Transfer_Fee']=df_trans['Transfer_Fee'].apply(trans_value)

    df_stats=pd.merge(df_mean_stats,df_trans,on=['Player','Transfer_Window'],how='left').dropna()
    #just to get actaull transfer fees
    #df_stats=df_stats[df_stats['Transfer_Fee']!=0]
    #df_stats['Pos'].replace({'MF,DF':'DF'},inplace=True)
    df_stats['League'].replace({'La Liga':'LaLiga'},inplace=True)
    df_stats['League_Left'].replace({'Vereinigte Staaten':'MLS','Liga NOS':'Liga Portugal'},inplace=True)
    df_stats['League_Joined'].replace({'Vereinigte Staaten':'MLS','Liga NOS':'Liga Portugal'},inplace=True)
    df_stats['League_Left'] = df_stats.apply(clean_league_left_aut, axis=1)
    df_stats['League_Joined'] = df_stats.apply(clean_league_joined_aut, axis=1)
    df_stats.drop(['Born'],axis=1,inplace=True)
    df_stats['Nation'] = df_stats['Nation'].apply(to_iso3)
    #build helper from names
    PLACEHOLDERS = {"", "0", "0.0", "-", "NA", "N/A", "<NA>"}
    df_stats["Nation_from_name"] = df_stats["Player"].map(nation_from_name)
    #make sure Nation is pandas string dtype so NA stays NA (not 'nan' text)
    df_stats["Nation"] = df_stats["Nation"].astype("string")
    #replace placeholder-looking tokens with NA (if any survived upstream)
    df_stats["Nation"] = df_stats["Nation"].where(~df_stats["Nation"].str.upper().isin(PLACEHOLDERS), pd.NA)
    #fill missing from name map and keep as string dtype
    df_stats["Nation"] = df_stats["Nation"].fillna(df_stats["Nation_from_name"]).astype("string")
    #drop helper
    df_stats.drop(columns=["Nation_from_name"], inplace=True)

    df_stats.to_csv('..\DataSources\Processed\Final_Stats_2000_2025_test.csv')
    logging.info('Final_Stats_2000_2025 Done')

    df=pd.read_csv(r"..\DataSources\Processed\Final_Stats_2000_2025_test.csv").iloc[:,1:]
    for c in [('Top_Left','League_Left'),('Top_Joined','League_Joined')]:
        df[c[0]]=df[c[1]].apply(lambda x: 1 if x in ['Premier League','LaLiga','Bundesliga','Serie A','Ligue 1','Eredivisie','Liga Portugal'] else 0)
    df=df[(df['Top_Left']==1) | (df['Top_Joined']==1)].iloc[:,:-2]
    df['League_Joined']=df['League_Joined'].map(top_or_others)
    df.to_csv("..\DataSources\Processed\Cleaned_Final_Stats_test.csv",index=False)
except Exception as ex:
    logging.error(f'Could not create the final cleaned file {repr(ex)}')