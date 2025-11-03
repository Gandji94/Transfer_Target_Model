#-----------------------------------------------------------------------------------------------------------------------------------------------------
#Transfer Model to predict classes python file for def and classes
#-----------------------------------------------------------------------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import re
import os
import time, random, re, requests
from bs4 import BeautifulSoup, Comment
from requests.adapters import HTTPAdapter, Retry
from scipy.stats import median_abs_deviation
from difflib import get_close_matches
from unidecode import unidecode
from scipy.stats import zscore
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score,make_scorer,log_loss,brier_score_loss
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler,LabelEncoder
from sklearn.utils.class_weight import compute_class_weight,compute_sample_weight
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectFromModel
import shap
import warnings
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
from copy import deepcopy
from sklearn.utils import check_random_state
from sklearn.base import clone


#python file is in the repositroy as well => Transfers_Window
def all_trans_window(year):
    def transfer_window(url):
      tfyear=str(year)
      #setting up the header
      headers={'User-Agent':'Mozilla/5.0 (X11;Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.106 Safari/537.36'}
      #setting up the header
      url=url
      data=requests.get(url, headers=headers).text
      soup=BeautifulSoup(data,'html.parser')

      #player
      players_step = soup.find_all('td', {'class': 'hauptlink'})

      Player = []
      for player in players_step:
          player_link = player.find('a', {'title': True})
          if player_link and '/spieler/' in player_link['href']:
              Player.append(player_link)

      Players=[player['title'] for player in Player]

      clubs = soup.find_all('td', {'class': 'hauptlink'})

      Club = []
      for club in clubs:
          club_link = club.find('a', {'title': True})
          if club_link and '/verein/' in club_link['href']:
              Club.append(club_link)
          elif 'pausiert' in club.get_text(strip=True):
              Club.append('pausiert')
          else:
              vereinslos_link = club.find('a', {'title': 'Vereinslos'})
              if vereinslos_link:
                  Club.append('No Club')
      # Extract titles or use string values directly
      Clubs = [club['title'] if hasattr(club, 'attrs') and 'title' in club.attrs else club for club in Club]
      #selecting the odd numbers to get the teams that the player left
      Club_left=Clubs[::2]
      #selecting even number to get the teams that the player has joined
      Club_joined=Clubs[1::2]

      #transfer fee
      transfers=soup.find_all('td', class_=lambda x: x and x.startswith('rechts'))
      transfers=transfers[1::2]
      #transfer_values=[int(transfer.get_text(strip=True).split(' ')[0][:-3])*1000000 for transfer in transfers]
      transfer_values=[transfer.get_text(strip=True) for transfer in transfers]
      transfer_values

      age_elements =soup.find_all('td', {'class': 'zentriert'})
      ages = []
      #getting only the numeric values
      for td in age_elements:
          age = td.get_text(strip=True)
          #only numeric values
          if age.isdigit() or age=='-':
              ages.append(age)
      age=ages[1::2]

      league = soup.find_all('table', {'class': 'inline-table'})
      lea = []

      for l in league:
          league_links = l.find_all('a', {'href': True, 'title': True})
          found_league = False

          for ll in league_links:
              if '/wettbewerb/' in ll['href']:
                  if ll['title'] == "Unbekannt":
                      lea.append('Other League')
                  else:
                      lea.append(ll['title'])
                  found_league = True
                  break  # Stop after finding the first valid league

          if not found_league:
              # Check for "Unbekannt" link explicitly
              unbekannt_link = l.find('a', {'title': 'Unbekannt'})
              if unbekannt_link:
                  lea.append('Other League')
              else:
                  # Check for image tag with title attribute
                  img_tag = l.find('img', {'class': 'flaggenrahmen'})
                  if img_tag and 'title' in img_tag.attrs:
                      lea.append(img_tag['title'])  # Add the title value from the img tag
                  else:
                      # Check for "Vereinslos" link
                      vereinslos_link = l.find('a', {'title': 'Vereinslos'})
                      if vereinslos_link:
                          lea.append('No League')
                      else:
                          # Check for "pausiert" text in the td element
                          pausiert_td = l.find('td', {'class': 'hauptlink'})
                          if pausiert_td and 'pausiert' in pausiert_td.get_text(strip=True):
                              lea.append('pausiert')
                          else:
                            sperre_link=l.find('a',{'title':'Sperre'})
                            if sperre_link:
                              lea.append('Sperre')

      # Separate the leagues into those left and those joined
      league_left = lea[::2]
      league_joined = lea[1::2]

      #creating the data frame
      test_df=pd.DataFrame({'Player':Players,
                          'Age':age,
                          'Club_Left':Club_left,
                          'League_Left':league_left,
                          'Club_Joined':Club_joined,
                          'League_Joined':league_joined,
                          'Transfer_Fee':transfer_values,
                          'Transfer_Window':tfyear})
      #how to add the decimal serpators
      #test_df['Transfer_Fee']=test_df['Transfer_Fee'].apply(lambda x: '{:,}'.format(x).replace(',','.'))
      return test_df

    headers={'User-Agent':'Mozilla/5.0 (X11;Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.106 Safari/'}

    url='https://www.transfermarkt.de/transfers/saisontransfers/statistik/top/plus/1/galerie/0?saison_id={}&transferfenster=alle&land_id=&ausrichtung=&spielerposition_id=&altersklasse=&leihe='.format(year)
    data=requests.get(url, headers=headers).text
    soup=BeautifulSoup(data,'html.parser')

    # Find the total number of pages
    pag_num = soup.find_all('a', {'class': 'tm-pagination__link'})
    for link in pag_num:
        if 'title' in link.attrs and link['title'].startswith("Zur letzten Seite"):
            max_page = int(link['title'].split(' ')[-1][:-1])
    # Initialize lists to store data

    all_data=transfer_window(url)

    page_num=[]
    for loop in range(2, max_page + 1):
        # Get transfer data from the current page
        url_loop = 'https://www.transfermarkt.de/transfers/saisontransfers/statistik/top/plus/1/galerie/0?saison_id={}&transferfenster=alle&land_id=&ausrichtung=&spielerposition_id=&altersklasse=&leihe=&page={}'.format(year,loop)
        page_data = transfer_window(url_loop)
        #stacking the data frames on each other, ignore_index=True => ignoring the seperate indecies and creat a new one
        all_data = pd.concat([all_data, page_data], ignore_index=True)
        page_num.append(loop)
    all_data['Transfer_Window']=all_data['Transfer_Window'].astype(int)
    all_data['Performance_Year']=all_data['Transfer_Window'].astype(int)-1
    return all_data


#-----------------------------------------------------------------------------------------------------------------------------------------------------
# creating a def function to create a dataframe which contain all the categories that we need for the holding six analysis
#-----------------------------------------------------------------------------------------------------------------------------------------------------
def player_stats(stats,tkl,passing,passing_style,poss,goal,gkl,gkal,league):

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from bs4 import BeautifulSoup
    from bs4 import Comment
    import requests

    #to avoid
    def string_int_transform(x):
        try:
            return int(str(x).split('-')[0])
        except(ValueError, IndexError):
            return None

    url_stats = stats
    rep = requests.get(url_stats)
    soup = BeautifulSoup(rep.text, 'html.parser')
    
    # Find all comments in the HTML
    comments = soup.find_all(string=lambda text: isinstance(text, Comment))
    
    basic_stats = []
    # Extract tables from comments
    for each in comments:
        if 'table' in each:
            try:
                basic_stats.append(pd.read_html(each)[0])
            except:
                continue
    
    # Assuming the first table is the one needed
    df_basic = basic_stats[0]
    df_basic.columns=df_basic.columns.droplevel(level=0)
    df_basic=df_basic[df_basic['Rk']!='Rk']
    df_basic1=df_basic.iloc[:,26:-1]
    df_basic=df_basic.iloc[:,:-6]
    df_basic['Nation']=df_basic['Nation'].astype(str).apply(lambda x: x.split(' ')[0].upper() if x else 'No Country')
    df_basic['Main_Pos']=df_basic['Pos'].astype(str).apply(lambda x: x.split(',')[0] if x else '-')
    df_basic['Sec_Pos']=df_basic['Pos'].astype(str).apply(lambda x: x.split(',')[-1] if x else '-')
    for x in list(df_basic1.columns):
        df_basic1['{}_per_90'.format(x)]=df_basic1[x]
    df_basic1=df_basic1.iloc[:,10:]
    df_basic=pd.concat([df_basic,df_basic1],axis=1)
    
    df_basic=df_basic[['Player', 'Nation','Main_Pos', 'Sec_Pos','Squad', 'Age', 'Born',
                      'MP', 'Starts','Min', '90s', 'Gls', 'Ast', 'G+A', 'G-PK', 'PK', 'PKatt', 'CrdY',
                       'CrdR', 'xG', 'npxG', 'xAG', 'npxG+xAG', 'PrgC', 'PrgP', 'PrgR', 'Gls',
                       'Ast', 'G+A', 'G-PK', 'G+A-PK', 'Gls_per_90',
                       'Ast_per_90', 'G+A_per_90', 'G-PK_per_90', 'G+A-PK_per_90', 'xG_per_90',
                       'xAG_per_90', 'xG+xAG_per_90', 'npxG_per_90', 'npxG+xAG_per_90']]
    df_basic['Age']=df_basic['Age'].apply(string_int_transform)
    df_basic['Born']=df_basic['Born'].apply(string_int_transform)
    time.sleep(10)
    
    url_tkl = tkl
    response = requests.get(url_tkl)

    soup = BeautifulSoup(response.text, 'html.parser')

    comments = soup.find_all(string=lambda text: isinstance(text, Comment))

    tables_tackel = []
    for each in comments:
        if 'table' in each:
            try:
                tables_tackel.append(pd.read_html(each)[0])
            except:
                continue

    df_tackel=tables_tackel[0]
    df_tackel.columns=df_tackel.columns.droplevel(level=0)
    df_tackel=df_tackel
    df_tackel['Age']=df_tackel['Age'].apply(string_int_transform)
    df_tackel['Born']=df_tackel['Born'].apply(string_int_transform)
    df_tackel['Nation']=df_tackel['Nation'].fillna('No_Country')
    df_tackel['Total Tackels']=df_tackel.iloc[:,8]
    df_tackel['Nation']=df_tackel['Nation'].astype(str).apply(lambda x: x.split(' ')[0].upper() if x else 'No Country')
    df_tackel['Main_Pos']=df_tackel['Pos'].astype(str).apply(lambda x: x.split(',')[0] if x else '-')
    df_tackel['Sec_Pos']=df_tackel['Pos'].astype(str).apply(lambda x: x.split(',')[-1] if x else '-')
    df_tackel['Dribblers_Tackeld']=df_tackel.iloc[:,13]
    df_tackel['Dribblers_Tackel_Att']=df_tackel.iloc[:,14]
    df_tackel['Dribblers_Tackel%']=df_tackel.iloc[:,15]
    df_tackel=df_tackel[['Player', 'Nation','Main_Pos', 'Sec_Pos','Squad', 'Age', 'Born', 'Total Tackels',
           'TklW', 'Def 3rd', 'Mid 3rd', 'Att 3rd', 'Dribblers_Tackeld', 'Dribblers_Tackel_Att',
           'Dribblers_Tackel%','Lost','Blocks', 'Sh', 'Pass', 'Int', 'Tkl+Int', 'Clr', 'Err']]
    time.sleep(10)

    # getting the passing stats
    url_pass = passing
    response = requests.get(url_pass)

    soup = BeautifulSoup(response.text, 'html.parser')

    comments = soup.find_all(string=lambda text: isinstance(text, Comment))

    tables_passing = []
    for each in comments:
        if 'table' in each:
            try:
                tables_passing.append(pd.read_html(each)[0])
            except:
                continue

    df_passing=tables_passing[0]
    df_passing.columns=df_passing.columns.droplevel(level=0)
    df_passing=df_passing
    df_passing['Age']=df_passing['Age'].apply(string_int_transform)
    df_passing['Born']=df_passing['Born'].apply(string_int_transform)
    df_passing['Nation']=df_passing['Nation'].fillna('No_Country')
    df_passing['Nation']=df_passing['Nation'].astype(str).apply(lambda x: x.split(' ')[0].upper() if x else 'No Country')
    df_passing['Main_Pos']=df_passing['Pos'].astype(str).apply(lambda x: x.split(',')[0] if x else '-')
    df_passing['Sec_Pos']=df_passing['Pos'].astype(str).apply(lambda x: x.split(',')[-1] if x else '-')
    df_passing['Total_Cmp']=df_passing.iloc[:,8]
    df_passing['Total_Att']=df_passing.iloc[:,9]
    df_passing['Total_Cmp%']=df_passing.iloc[:,10]
    df_passing['Short_Cmp']=df_passing.iloc[:,13]
    df_passing['Short_Att']=df_passing.iloc[:,14]
    df_passing['Short_Cmp%']=df_passing.iloc[:,15]
    df_passing['Medium_Cmp']=df_passing.iloc[:,16]
    df_passing['Medium_Att']=df_passing.iloc[:,17]
    df_passing['Medium_Cmp%']=df_passing.iloc[:,18]
    df_passing['Long_Cmp']=df_passing.iloc[:,19]
    df_passing['Long_Att']=df_passing.iloc[:,20]
    df_passing['Long_Cmp%']=df_passing.iloc[:,21]
    df_passing=df_passing[['Player', 'Nation', 'Main_Pos', 'Sec_Pos','Squad','Age','Born',
                                 'Total_Cmp','Total_Att','Total_Cmp%','TotDist','PrgDist',
                                 'Short_Cmp','Short_Att','Short_Cmp%',
                                 'Medium_Cmp','Medium_Att','Medium_Cmp%','Long_Cmp','Long_Att','Long_Cmp%','Ast', 'xAG', 'xA',
                                 'A-xAG', 'KP', '1/3','PPA', 'CrsPA', 'PrgP']]
    time.sleep(10)

    # getting the passing style
    url_passing_style = passing_style
    response = requests.get(url_passing_style)

    soup = BeautifulSoup(response.text, 'html.parser')

    comments = soup.find_all(string=lambda text: isinstance(text, Comment))

    tables_passing_style = []
    for each in comments:
        if 'table' in each:
            try:
                tables_passing_style.append(pd.read_html(each)[0])
            except:
                continue

    df_passing_style=tables_passing_style[0]
    df_passing_style.columns=df_passing_style.columns.droplevel(level=0)
    df_passing_style=df_passing_style
    df_passing_style['Age']=df_passing_style['Age'].apply(string_int_transform)
    df_passing_style['Born']=df_passing_style['Born'].apply(string_int_transform)
    df_passing_style['Nation']=df_passing_style['Nation'].fillna('No_Country')
    df_passing_style['Nation']=df_passing_style['Nation'].astype(str).apply(lambda x: x.split(' ')[0].upper() if x else 'No Country')
    df_passing_style['Main_Pos']=df_passing_style['Pos'].astype(str).apply(lambda x: x.split(',')[0] if x else '-')
    df_passing_style['Sec_Pos']=df_passing_style['Pos'].astype(str).apply(lambda x: x.split(',')[-1] if x else '-')
    df_passing_style['Passes_in_Gameflow']=df_passing_style['Live']
    df_passing_style['Passes_Out_of_Gameflow']=df_passing_style['Dead']
    df_passing_style['Freekick']=df_passing_style['FK']
    df_passing_style['Throughball']=df_passing_style['TB']
    df_passing_style['40_yds_pass']=df_passing_style['Sw']
    df_passing_style['Crosses']=df_passing_style['Crs']
    df_passing_style['Corner_Kicks']=df_passing_style['CK']
    df_passing_style['Corner_Kicks_In']=df_passing_style['In']
    df_passing_style['Corner_Kicks_Out']=df_passing_style['Out']
    df_passing_style['Corner_Kicks_Straight']=df_passing_style['Str']
    df_passing_style['Passes_Offside']=df_passing_style['Off']
    df_passing_style['Passes_Blocked']=df_passing_style['Blocks']
    df_passing_style=df_passing_style[['Player', 'Nation','Main_Pos', 'Sec_Pos','Squad', 'Age', 'Born','Passes_in_Gameflow', 'Passes_Out_of_Gameflow', 'Freekick','Throughball', '40_yds_pass',
                                             'Crosses', 'Corner_Kicks','Corner_Kicks_In', 'Corner_Kicks_Out', 'Corner_Kicks_Straight','Passes_Offside',
                                             'Passes_Blocked']]
    time.sleep(10)

    # getting the possesions stats
    url_poss = poss
    response = requests.get(url_poss)

    soup = BeautifulSoup(response.text, 'html.parser')

    comments = soup.find_all(string=lambda text: isinstance(text, Comment))

    tables_possesion = []
    for each in comments:
        if 'table' in each:
            try:
                tables_possesion.append(pd.read_html(each)[0])
            except:
                continue

    df_possesion=tables_possesion[0]
    df_possesion.columns=df_possesion.columns.droplevel(level=0)
    df_possesion=df_possesion
    df_possesion['Age']=df_possesion['Age'].apply(string_int_transform)
    df_possesion['Born']=df_possesion['Born'].apply(string_int_transform)
    df_possesion['Nation']=df_possesion['Nation'].fillna('No_Country')
    df_possesion['Nation']=df_possesion['Nation'].astype(str).apply(lambda x: x.split(' ')[0].upper() if x else 'No Country')
    df_possesion['Main_Pos']=df_possesion['Pos'].astype(str).apply(lambda x: x.split(',')[0] if x else '-')
    df_possesion['Sec_Pos']=df_possesion['Pos'].astype(str).apply(lambda x: x.split(',')[-1] if x else '-')
    df_possesion['Touches_Def_Pen_Area']=df_possesion['Def Pen']
    df_possesion['Touches_Def_3rd_Area']=df_possesion['Def 3rd']
    df_possesion['Touches_Mid_3rd_Area']=df_possesion['Mid 3rd']
    df_possesion['Touches_Att_3rd_Area']=df_possesion['Att 3rd']
    df_possesion['Touches_Att_Pen_Area']=df_possesion['Att Pen']
    df_possesion['Live_Touches_in_Game']=df_possesion['Live']
    df_possesion['Dribbling_Att']=df_possesion['Att']
    df_possesion['Dribbling_Succ']=df_possesion['Succ']
    df_possesion['Dribbling_Succ%']=df_possesion['Succ%']
    df_possesion['Tackeld_Dribbling']=df_possesion['Tkld']
    df_possesion['Tackeld_Dribbling%']=df_possesion['Tkld%']
    df_possesion['Total_Carry_Distance']=df_possesion['TotDist']
    df_possesion['Total_Progressive_Carry_Distance']=df_possesion['PrgDist']
    df_possesion['Total_Carries_in_1/3']=df_possesion['1/3']
    df_possesion['Total_Carries_in_Penalty_Area']=df_possesion['CPA']
    df_possesion['Miscontrols_Carries']=df_possesion['Mis']
    df_possesion['Dispossed_Carries']=df_possesion['Dis']
    df_possesion['Passes_Received']=df_possesion['Rec']
    df_possesion['Progressive_Passes_Received']=df_possesion['PrgR']
    df_possesion=df_possesion[['Player', 'Nation','Main_Pos', 'Sec_Pos','Squad', 'Age', 'Born','Touches',
                                    'Touches_Def_Pen_Area', 'Touches_Def_3rd_Area','Touches_Mid_3rd_Area', 'Touches_Att_3rd_Area', 'Touches_Att_Pen_Area','Live_Touches_in_Game',
                                     'Dribbling_Att', 'Dribbling_Succ','Dribbling_Succ%', 'Tackeld_Dribbling', 'Tackeld_Dribbling%',
                                     'Total_Carry_Distance', 'Total_Progressive_Carry_Distance','Total_Carries_in_1/3', 'Total_Carries_in_Penalty_Area',
                                     'Miscontrols_Carries', 'Dispossed_Carries', 'Passes_Received','Progressive_Passes_Received']]
    time.sleep(10)

    # getting the goals stats
    url = goal
    response = requests.get(url)

    soup = BeautifulSoup(response.text, 'html.parser')

    comments = soup.find_all(string=lambda text: isinstance(text, Comment))

    tables_goals = []
    for each in comments:
        if 'table' in each:
            try:
                tables_goals.append(pd.read_html(each)[0])
            except:
                continue

    df_goals=tables_goals[0]
    df_goals.columns=df_goals.columns.droplevel(level=0)
    df_goals=df_goals
    df_goals['Age']=df_goals['Age'].apply(string_int_transform)
    df_goals['Born']=df_goals['Born'].apply(string_int_transform)
    df_goals['Nation']=df_goals['Nation'].fillna('No_Country')
    df_goals['Nation']=df_goals['Nation'].astype(str).apply(lambda x: x.split(' ')[0].upper() if x else 'No Country')
    df_goals['Main_Pos']=df_goals['Pos'].astype(str).apply(lambda x: x.split(',')[0] if x else '-')
    df_goals['Sec_Pos']=df_goals['Pos'].astype(str).apply(lambda x: x.split(',')[-1] if x else '-')
    df_goals['Shot_Creating_Action']=df_goals['SCA']
    df_goals['Shot_Creating_Action_90']=df_goals['SCA90']
    df_goals['Live_Passes_lead_Shot_Att']=df_goals.iloc[:,10]
    df_goals['Dead_Passes_lead_Shot_Att']=df_goals.iloc[:,11]
    df_goals['Shot_Att_after_Dribbling']=df_goals.iloc[:,12]
    df_goals['Shot_lead_to_Shot_Att']=df_goals.iloc[:,13]
    df_goals['Foul_drawn_lead_Shot_Att']=df_goals.iloc[:,14]
    df_goals['Def_Action_lead_Shot_Att']=df_goals.iloc[:,15]
    df_goals['Live_Pass_lead_Goal']=df_goals.iloc[:,18]
    df_goals['Dead_Pass_lead_Goal']=df_goals.iloc[:,19]
    df_goals['Goal_after_Dribbling']=df_goals.iloc[:,20]
    df_goals['Shot_lead_Goal']=df_goals.iloc[:,21]
    df_goals['Foul_drawn_lead_Goal']=df_goals.iloc[:,22]
    df_goals['Def_Action_lead_Goal']=df_goals.iloc[:,23]
    df_goals=df_goals[['Player', 'Nation','Main_Pos', 'Sec_Pos','Squad', 'Age', 'Born',
                            'Shot_Creating_Action', 'Shot_Creating_Action_90','Live_Passes_lead_Shot_Att', 'Dead_Passes_lead_Shot_Att',
                             'Shot_Att_after_Dribbling', 'Shot_lead_to_Shot_Att','Foul_drawn_lead_Shot_Att', 'Def_Action_lead_Shot_Att', 'GCA', 'GCA90',
                            'Live_Pass_lead_Goal', 'Dead_Pass_lead_Goal','Goal_after_Dribbling', 'Shot_lead_Goal', 'Foul_drawn_lead_Goal','Def_Action_lead_Goal']]
    time.sleep(10)

    #getting basic goalkeeper statistics
    url = gkl
    rep = requests.get(url)
    soup = BeautifulSoup(rep.text, 'html.parser')

    # Find all comments in the HTML
    comments = soup.find_all(string=lambda text: isinstance(text, Comment))

    gk = []
    # Extract tables from comments
    for each in comments:
        if 'table' in each:
            try:
                gk.append(pd.read_html(each)[0])
            except:
                continue

    # Assuming the first table is the one needed
    df_gk = gk[0]
    df_gk.columns=df_gk.columns.droplevel(level=0)
    df_gk=df_gk[df_gk['Rk']!='Rk']
    df_gk['Age']=df_gk['Age'].apply(string_int_transform)
    df_gk['Born']=df_gk['Born'].apply(string_int_transform)
    #lambda x: x.split(' ')[0].upper() if x => that if x just indicates that the variable contains a value which sets it True
    df_gk['Nation']=df_gk['Nation'].astype(str).apply(lambda x: x.split(' ')[0].upper() if x else 'No Country')
    df_gk['Main_Pos']=df_gk['Pos']
    df_gk['Sec_Pos']='-'
    df_gk.drop(['Matches','90s','Rk'], axis=1, inplace=True)
    df_gk.fillna(0,inplace=True)
    df_gk=df_gk[['Player', 'Nation', 'Main_Pos', 'Sec_Pos','Squad', 'Age', 'Born', 'MP', 'Starts','Min', 'GA', 'GA90', 'SoTA', 'Saves', 'Save%', 'W', 'D', 'L', 'CS',
          'CS%', 'PKatt', 'PKA', 'PKsv', 'PKm', 'Save%']]
    col=list(df_gk.columns)

    for c in col:
        if c in df_gk.columns[:4]:
            df_gk.loc[:, c] = df_gk[c].astype(str)
        else:
            try:
                df_gk.loc[:, c] = df_gk[c].astype(float)
            except ValueError:
                continue  # Handle columns that cannot be converted to int
    time.sleep(10)

    #getting advanced goalkeeper statistics
    url = gkal
    rep = requests.get(url)
    soup = BeautifulSoup(rep.text, 'html.parser')

    # Find all comments in the HTML
    comments = soup.find_all(string=lambda text: isinstance(text, Comment))

    gka = []
    # Extract tables from comments
    for each in comments:
        if 'table' in each:
            try:
                gka.append(pd.read_html(each)[0])
            except:
                continue

    # Assuming the first table is the one needed
    df_gka = gka[0]
    df_gka.columns=df_gka.columns.droplevel(level=0)
    df_gka=df_gka[df_gka['Rk']!='Rk']
    df_gka['Age']=df_gka['Age'].apply(string_int_transform)
    df_gka['Born']=df_gka['Born'].apply(string_int_transform)
    df_gka['Nation']=df_gka['Nation'].astype(str).apply(lambda x: x.split(' ')[0].upper() if x else 'No Country')
    df_gka['Main_Pos']=df_gka['Pos']
    df_gka['Sec_Pos']='-'
    df_gka.drop(['Matches','90s'], axis=1, inplace=True)
    df_gka.fillna(0,inplace=True)
    df_gka=df_gka[['Player',
    'Nation',
    'Main_Pos',
    'Sec_Pos',
    'Squad',
    'Age',
    'Born',
    'GA',
    'PKA',
    'FK',
    'CK',
    'OG',
    'PSxG',
    'PSxG/SoT',
    'PSxG+/-',
    '/90',
    'Cmp',
    'Att',
    'Cmp%',
    'Att (GK)',
    'Thr',
    'Launch%',
    'AvgLen',
    'Att',
    'Launch%',
    'AvgLen',
    'Opp',
    'Stp',
    'Stp%',
    '#OPA',
    '#OPA/90',
    'AvgDist']]
    col=list(df_gka.columns)

    for c in col:
        if c in df_gka.columns[:5]:
            df_gka.loc[:, c] = df_gka[c].astype(str)
        else:
            try:
                df_gka.loc[:, c] = df_gka[c].astype(float)
            except ValueError:
                continue  # Handle columns that cannot be converted to int
    time.sleep(10)

    #getting the stats of the midfielder
    df_list=[df_basic,df_tackel,df_passing,df_possesion,df_goals]
    df_final=df_list[0]
    for x in df_list:
        df_final=pd.merge(df_final,x,on=['Player','Nation','Main_Pos','Sec_Pos','Squad','Age','Born'],how='inner')

    #getting the goalkeeper stats and concat them on the midfielder stats
    df_finalgk=pd.merge(df_gk,df_gka,on=['Player','Nation','Main_Pos','Sec_Pos','Squad','Age','Born'],how='outer')

    df_final=pd.merge(df_final,df_finalgk,on=['Player','Nation','Main_Pos','Sec_Pos','Squad','Age','Born'],how='outer')
    #df_final=pd.merge(df_final,df_basic,on=['Player','Nation','Main_Pos','Sec_Pos','Squad','Age','Born'],how='outer')

    df_final.reset_index(drop=True, inplace=True)
    df_final=df_final.drop_duplicates()
    df_final['League']=league

    #these three lines of code help us to drop the unwanted duplicates of the column "Sec_Pos"
    lst=[*range(0,df_final.shape[1])]
    lst.pop(59)
    df_final=df_final.iloc[:,lst]

    df_final=df_final.fillna(0)
    df_final=df_final.reset_index()
    df_final=df_final.iloc[:,1:]
    #getting rid of all the duplicate columns, that end with a _y
    drop_col=list(df_final.columns[df_final.columns.str.endswith('_y')])
    df_final=df_final.drop(drop_col, axis=1)

    for new in list(df_final.columns):
        if new.endswith('_x'):
            df_final.rename(columns={new:new.split('_')[0]},inplace=True)
        else:
            df_final.rename(columns={new:new})

    return df_final


#-----------------------------------------------------------------------------------------------------------------------------------------------------
# transforming string values into corrosponding numeric value
#-----------------------------------------------------------------------------------------------------------------------------------------------------
def trans_value(x):
    x=str(x)
    if 'Mio' in x and not 'Leihgebühr:' in x and 'Leihe' not in x:
        return int(float(x.split(' ')[0].replace(',','.'))*1000000)
        #return int(float(phase1_mio.split(':')[1].replace(',','.'))*1000000)
    elif 'Tsd' in x and not 'Leihgebühr:' in x and 'Leihe' not in x:
        return int(float(x.split(' ')[0].replace(',','.'))*1000)
    else:
        return 0


#-----------------------------------------------------------------------------------------------------------------------------------------------------
# creating a def function to create a dataframe which contain all the categories that we need for the holding six analysis
#-----------------------------------------------------------------------------------------------------------------------------------------------------
def player_stats(stats,tkl,passing,passing_style,poss,goal,gkl,gkal,league):

    #to avoid
    def string_int_transform(x):
        try:
            return int(str(x).split('-')[0])
        except(ValueError, IndexError):
            return None

    url_stats = stats
    rep = requests.get(url_stats)
    soup = BeautifulSoup(rep.text, 'html.parser')
    
    # Find all comments in the HTML
    comments = soup.find_all(string=lambda text: isinstance(text, Comment))
    
    basic_stats = []
    # Extract tables from comments
    for each in comments:
        if 'table' in each:
            try:
                basic_stats.append(pd.read_html(each)[0])
            except:
                continue
    
    # Assuming the first table is the one needed
    df_basic = basic_stats[0]
    df_basic.columns=df_basic.columns.droplevel(level=0)
    df_basic=df_basic[df_basic['Rk']!='Rk']
    df_basic1=df_basic.iloc[:,26:-1]
    df_basic=df_basic.iloc[:,:-6]
    df_basic['Nation']=df_basic['Nation'].astype(str).apply(lambda x: x.split(' ')[0].upper() if x else 'No Country')
    df_basic['Main_Pos']=df_basic['Pos'].astype(str).apply(lambda x: x.split(',')[0] if x else '-')
    df_basic['Sec_Pos']=df_basic['Pos'].astype(str).apply(lambda x: x.split(',')[-1] if x else '-')
    for x in list(df_basic1.columns):
        df_basic1['{}_per_90'.format(x)]=df_basic1[x]
    df_basic1=df_basic1.iloc[:,10:]
    df_basic=pd.concat([df_basic,df_basic1],axis=1)
    
    df_basic=df_basic[['Player', 'Nation','Main_Pos', 'Sec_Pos','Squad', 'Age', 'Born',
                      'MP', 'Starts','Min', '90s', 'Gls', 'Ast', 'G+A', 'G-PK', 'PK', 'PKatt', 'CrdY',
                       'CrdR', 'xG', 'npxG', 'xAG', 'npxG+xAG', 'PrgC', 'PrgP', 'PrgR', 'Gls',
                       'Ast', 'G+A', 'G-PK', 'G+A-PK', 'Gls_per_90',
                       'Ast_per_90', 'G+A_per_90', 'G-PK_per_90', 'G+A-PK_per_90', 'xG_per_90',
                       'xAG_per_90', 'xG+xAG_per_90', 'npxG_per_90', 'npxG+xAG_per_90']]
    df_basic['Age']=df_basic['Age'].apply(string_int_transform)
    df_basic['Born']=df_basic['Born'].apply(string_int_transform)
    time.sleep(10)
    
    url_tkl = tkl
    response = requests.get(url_tkl)

    soup = BeautifulSoup(response.text, 'html.parser')

    comments = soup.find_all(string=lambda text: isinstance(text, Comment))

    tables_tackel = []
    for each in comments:
        if 'table' in each:
            try:
                tables_tackel.append(pd.read_html(each)[0])
            except:
                continue

    df_tackel=tables_tackel[0]
    df_tackel.columns=df_tackel.columns.droplevel(level=0)
    df_tackel=df_tackel
    df_tackel['Age']=df_tackel['Age'].apply(string_int_transform)
    df_tackel['Born']=df_tackel['Born'].apply(string_int_transform)
    df_tackel['Nation']=df_tackel['Nation'].fillna('No_Country')
    df_tackel['Total Tackels']=df_tackel.iloc[:,8]
    df_tackel['Nation']=df_tackel['Nation'].astype(str).apply(lambda x: x.split(' ')[0].upper() if x else 'No Country')
    df_tackel['Main_Pos']=df_tackel['Pos'].astype(str).apply(lambda x: x.split(',')[0] if x else '-')
    df_tackel['Sec_Pos']=df_tackel['Pos'].astype(str).apply(lambda x: x.split(',')[-1] if x else '-')
    df_tackel['Dribblers_Tackeld']=df_tackel.iloc[:,13]
    df_tackel['Dribblers_Tackel_Att']=df_tackel.iloc[:,14]
    df_tackel['Dribblers_Tackel%']=df_tackel.iloc[:,15]
    df_tackel=df_tackel[['Player', 'Nation','Main_Pos', 'Sec_Pos','Squad', 'Age', 'Born', 'Total Tackels',
           'TklW', 'Def 3rd', 'Mid 3rd', 'Att 3rd', 'Dribblers_Tackeld', 'Dribblers_Tackel_Att',
           'Dribblers_Tackel%','Lost','Blocks', 'Sh', 'Pass', 'Int', 'Tkl+Int', 'Clr', 'Err']]
    time.sleep(10)

    # getting the passing stats
    url_pass = passing
    response = requests.get(url_pass)

    soup = BeautifulSoup(response.text, 'html.parser')

    comments = soup.find_all(string=lambda text: isinstance(text, Comment))

    tables_passing = []
    for each in comments:
        if 'table' in each:
            try:
                tables_passing.append(pd.read_html(each)[0])
            except:
                continue

    df_passing=tables_passing[0]
    df_passing.columns=df_passing.columns.droplevel(level=0)
    df_passing=df_passing
    df_passing['Age']=df_passing['Age'].apply(string_int_transform)
    df_passing['Born']=df_passing['Born'].apply(string_int_transform)
    df_passing['Nation']=df_passing['Nation'].fillna('No_Country')
    df_passing['Nation']=df_passing['Nation'].astype(str).apply(lambda x: x.split(' ')[0].upper() if x else 'No Country')
    df_passing['Main_Pos']=df_passing['Pos'].astype(str).apply(lambda x: x.split(',')[0] if x else '-')
    df_passing['Sec_Pos']=df_passing['Pos'].astype(str).apply(lambda x: x.split(',')[-1] if x else '-')
    df_passing['Total_Cmp']=df_passing.iloc[:,8]
    df_passing['Total_Att']=df_passing.iloc[:,9]
    df_passing['Total_Cmp%']=df_passing.iloc[:,10]
    df_passing['Short_Cmp']=df_passing.iloc[:,13]
    df_passing['Short_Att']=df_passing.iloc[:,14]
    df_passing['Short_Cmp%']=df_passing.iloc[:,15]
    df_passing['Medium_Cmp']=df_passing.iloc[:,16]
    df_passing['Medium_Att']=df_passing.iloc[:,17]
    df_passing['Medium_Cmp%']=df_passing.iloc[:,18]
    df_passing['Long_Cmp']=df_passing.iloc[:,19]
    df_passing['Long_Att']=df_passing.iloc[:,20]
    df_passing['Long_Cmp%']=df_passing.iloc[:,21]
    df_passing=df_passing[['Player', 'Nation', 'Main_Pos', 'Sec_Pos','Squad','Age','Born',
                                 'Total_Cmp','Total_Att','Total_Cmp%','TotDist','PrgDist',
                                 'Short_Cmp','Short_Att','Short_Cmp%',
                                 'Medium_Cmp','Medium_Att','Medium_Cmp%','Long_Cmp','Long_Att','Long_Cmp%','Ast', 'xAG', 'xA',
                                 'A-xAG', 'KP', '1/3','PPA', 'CrsPA', 'PrgP']]
    time.sleep(10)

    # getting the passing style
    url_passing_style = passing_style
    response = requests.get(url_passing_style)

    soup = BeautifulSoup(response.text, 'html.parser')

    comments = soup.find_all(string=lambda text: isinstance(text, Comment))

    tables_passing_style = []
    for each in comments:
        if 'table' in each:
            try:
                tables_passing_style.append(pd.read_html(each)[0])
            except:
                continue

    df_passing_style=tables_passing_style[0]
    df_passing_style.columns=df_passing_style.columns.droplevel(level=0)
    df_passing_style=df_passing_style
    df_passing_style['Age']=df_passing_style['Age'].apply(string_int_transform)
    df_passing_style['Born']=df_passing_style['Born'].apply(string_int_transform)
    df_passing_style['Nation']=df_passing_style['Nation'].fillna('No_Country')
    df_passing_style['Nation']=df_passing_style['Nation'].astype(str).apply(lambda x: x.split(' ')[0].upper() if x else 'No Country')
    df_passing_style['Main_Pos']=df_passing_style['Pos'].astype(str).apply(lambda x: x.split(',')[0] if x else '-')
    df_passing_style['Sec_Pos']=df_passing_style['Pos'].astype(str).apply(lambda x: x.split(',')[-1] if x else '-')
    df_passing_style['Passes_in_Gameflow']=df_passing_style['Live']
    df_passing_style['Passes_Out_of_Gameflow']=df_passing_style['Dead']
    df_passing_style['Freekick']=df_passing_style['FK']
    df_passing_style['Throughball']=df_passing_style['TB']
    df_passing_style['40_yds_pass']=df_passing_style['Sw']
    df_passing_style['Crosses']=df_passing_style['Crs']
    df_passing_style['Corner_Kicks']=df_passing_style['CK']
    df_passing_style['Corner_Kicks_In']=df_passing_style['In']
    df_passing_style['Corner_Kicks_Out']=df_passing_style['Out']
    df_passing_style['Corner_Kicks_Straight']=df_passing_style['Str']
    df_passing_style['Passes_Offside']=df_passing_style['Off']
    df_passing_style['Passes_Blocked']=df_passing_style['Blocks']
    df_passing_style=df_passing_style[['Player', 'Nation','Main_Pos', 'Sec_Pos','Squad', 'Age', 'Born','Passes_in_Gameflow', 'Passes_Out_of_Gameflow', 'Freekick','Throughball', '40_yds_pass',
                                             'Crosses', 'Corner_Kicks','Corner_Kicks_In', 'Corner_Kicks_Out', 'Corner_Kicks_Straight','Passes_Offside',
                                             'Passes_Blocked']]
    time.sleep(10)

    # getting the possesions stats
    url_poss = poss
    response = requests.get(url_poss)

    soup = BeautifulSoup(response.text, 'html.parser')

    comments = soup.find_all(string=lambda text: isinstance(text, Comment))

    tables_possesion = []
    for each in comments:
        if 'table' in each:
            try:
                tables_possesion.append(pd.read_html(each)[0])
            except:
                continue

    df_possesion=tables_possesion[0]
    df_possesion.columns=df_possesion.columns.droplevel(level=0)
    df_possesion=df_possesion
    df_possesion['Age']=df_possesion['Age'].apply(string_int_transform)
    df_possesion['Born']=df_possesion['Born'].apply(string_int_transform)
    df_possesion['Nation']=df_possesion['Nation'].fillna('No_Country')
    df_possesion['Nation']=df_possesion['Nation'].astype(str).apply(lambda x: x.split(' ')[0].upper() if x else 'No Country')
    df_possesion['Main_Pos']=df_possesion['Pos'].astype(str).apply(lambda x: x.split(',')[0] if x else '-')
    df_possesion['Sec_Pos']=df_possesion['Pos'].astype(str).apply(lambda x: x.split(',')[-1] if x else '-')
    df_possesion['Touches_Def_Pen_Area']=df_possesion['Def Pen']
    df_possesion['Touches_Def_3rd_Area']=df_possesion['Def 3rd']
    df_possesion['Touches_Mid_3rd_Area']=df_possesion['Mid 3rd']
    df_possesion['Touches_Att_3rd_Area']=df_possesion['Att 3rd']
    df_possesion['Touches_Att_Pen_Area']=df_possesion['Att Pen']
    df_possesion['Live_Touches_in_Game']=df_possesion['Live']
    df_possesion['Dribbling_Att']=df_possesion['Att']
    df_possesion['Dribbling_Succ']=df_possesion['Succ']
    df_possesion['Dribbling_Succ%']=df_possesion['Succ%']
    df_possesion['Tackeld_Dribbling']=df_possesion['Tkld']
    df_possesion['Tackeld_Dribbling%']=df_possesion['Tkld%']
    df_possesion['Total_Carry_Distance']=df_possesion['TotDist']
    df_possesion['Total_Progressive_Carry_Distance']=df_possesion['PrgDist']
    df_possesion['Total_Carries_in_1/3']=df_possesion['1/3']
    df_possesion['Total_Carries_in_Penalty_Area']=df_possesion['CPA']
    df_possesion['Miscontrols_Carries']=df_possesion['Mis']
    df_possesion['Dispossed_Carries']=df_possesion['Dis']
    df_possesion['Passes_Received']=df_possesion['Rec']
    df_possesion['Progressive_Passes_Received']=df_possesion['PrgR']
    df_possesion=df_possesion[['Player', 'Nation','Main_Pos', 'Sec_Pos','Squad', 'Age', 'Born','Touches',
                                    'Touches_Def_Pen_Area', 'Touches_Def_3rd_Area','Touches_Mid_3rd_Area', 'Touches_Att_3rd_Area', 'Touches_Att_Pen_Area','Live_Touches_in_Game',
                                     'Dribbling_Att', 'Dribbling_Succ','Dribbling_Succ%', 'Tackeld_Dribbling', 'Tackeld_Dribbling%',
                                     'Total_Carry_Distance', 'Total_Progressive_Carry_Distance','Total_Carries_in_1/3', 'Total_Carries_in_Penalty_Area',
                                     'Miscontrols_Carries', 'Dispossed_Carries', 'Passes_Received','Progressive_Passes_Received']]
    time.sleep(10)

    # getting the goals stats
    url = goal
    response = requests.get(url)

    soup = BeautifulSoup(response.text, 'html.parser')

    comments = soup.find_all(string=lambda text: isinstance(text, Comment))

    tables_goals = []
    for each in comments:
        if 'table' in each:
            try:
                tables_goals.append(pd.read_html(each)[0])
            except:
                continue

    df_goals=tables_goals[0]
    df_goals.columns=df_goals.columns.droplevel(level=0)
    df_goals=df_goals
    df_goals['Age']=df_goals['Age'].apply(string_int_transform)
    df_goals['Born']=df_goals['Born'].apply(string_int_transform)
    df_goals['Nation']=df_goals['Nation'].fillna('No_Country')
    df_goals['Nation']=df_goals['Nation'].astype(str).apply(lambda x: x.split(' ')[0].upper() if x else 'No Country')
    df_goals['Main_Pos']=df_goals['Pos'].astype(str).apply(lambda x: x.split(',')[0] if x else '-')
    df_goals['Sec_Pos']=df_goals['Pos'].astype(str).apply(lambda x: x.split(',')[-1] if x else '-')
    df_goals['Shot_Creating_Action']=df_goals['SCA']
    df_goals['Shot_Creating_Action_90']=df_goals['SCA90']
    df_goals['Live_Passes_lead_Shot_Att']=df_goals.iloc[:,10]
    df_goals['Dead_Passes_lead_Shot_Att']=df_goals.iloc[:,11]
    df_goals['Shot_Att_after_Dribbling']=df_goals.iloc[:,12]
    df_goals['Shot_lead_to_Shot_Att']=df_goals.iloc[:,13]
    df_goals['Foul_drawn_lead_Shot_Att']=df_goals.iloc[:,14]
    df_goals['Def_Action_lead_Shot_Att']=df_goals.iloc[:,15]
    df_goals['Live_Pass_lead_Goal']=df_goals.iloc[:,18]
    df_goals['Dead_Pass_lead_Goal']=df_goals.iloc[:,19]
    df_goals['Goal_after_Dribbling']=df_goals.iloc[:,20]
    df_goals['Shot_lead_Goal']=df_goals.iloc[:,21]
    df_goals['Foul_drawn_lead_Goal']=df_goals.iloc[:,22]
    df_goals['Def_Action_lead_Goal']=df_goals.iloc[:,23]
    df_goals=df_goals[['Player', 'Nation','Main_Pos', 'Sec_Pos','Squad', 'Age', 'Born',
                            'Shot_Creating_Action', 'Shot_Creating_Action_90','Live_Passes_lead_Shot_Att', 'Dead_Passes_lead_Shot_Att',
                             'Shot_Att_after_Dribbling', 'Shot_lead_to_Shot_Att','Foul_drawn_lead_Shot_Att', 'Def_Action_lead_Shot_Att', 'GCA', 'GCA90',
                            'Live_Pass_lead_Goal', 'Dead_Pass_lead_Goal','Goal_after_Dribbling', 'Shot_lead_Goal', 'Foul_drawn_lead_Goal','Def_Action_lead_Goal']]
    time.sleep(10)

    #getting basic goalkeeper statistics
    url = gkl
    rep = requests.get(url)
    soup = BeautifulSoup(rep.text, 'html.parser')

    # Find all comments in the HTML
    comments = soup.find_all(string=lambda text: isinstance(text, Comment))

    gk = []
    # Extract tables from comments
    for each in comments:
        if 'table' in each:
            try:
                gk.append(pd.read_html(each)[0])
            except:
                continue

    # Assuming the first table is the one needed
    df_gk = gk[0]
    df_gk.columns=df_gk.columns.droplevel(level=0)
    df_gk=df_gk[df_gk['Rk']!='Rk']
    df_gk['Age']=df_gk['Age'].apply(string_int_transform)
    df_gk['Born']=df_gk['Born'].apply(string_int_transform)
    #lambda x: x.split(' ')[0].upper() if x => that if x just indicates that the variable contains a value which sets it True
    df_gk['Nation']=df_gk['Nation'].astype(str).apply(lambda x: x.split(' ')[0].upper() if x else 'No Country')
    df_gk['Main_Pos']=df_gk['Pos']
    df_gk['Sec_Pos']='-'
    df_gk.drop(['Matches','90s','Rk'], axis=1, inplace=True)
    df_gk.fillna(0,inplace=True)
    df_gk=df_gk[['Player', 'Nation', 'Main_Pos', 'Sec_Pos','Squad', 'Age', 'Born', 'MP', 'Starts','Min', 'GA', 'GA90', 'SoTA', 'Saves', 'Save%', 'W', 'D', 'L', 'CS',
          'CS%', 'PKatt', 'PKA', 'PKsv', 'PKm', 'Save%']]
    col=list(df_gk.columns)

    for c in col:
        if c in df_gk.columns[:4]:
            df_gk.loc[:, c] = df_gk[c].astype(str)
        else:
            try:
                df_gk.loc[:, c] = df_gk[c].astype(float)
            except ValueError:
                continue  # Handle columns that cannot be converted to int
    time.sleep(10)

    #getting advanced goalkeeper statistics
    url = gkal
    rep = requests.get(url)
    soup = BeautifulSoup(rep.text, 'html.parser')

    # Find all comments in the HTML
    comments = soup.find_all(string=lambda text: isinstance(text, Comment))

    gka = []
    # Extract tables from comments
    for each in comments:
        if 'table' in each:
            try:
                gka.append(pd.read_html(each)[0])
            except:
                continue

    # Assuming the first table is the one needed
    df_gka = gka[0]
    df_gka.columns=df_gka.columns.droplevel(level=0)
    df_gka=df_gka[df_gka['Rk']!='Rk']
    df_gka['Age']=df_gka['Age'].apply(string_int_transform)
    df_gka['Born']=df_gka['Born'].apply(string_int_transform)
    df_gka['Nation']=df_gka['Nation'].astype(str).apply(lambda x: x.split(' ')[0].upper() if x else 'No Country')
    df_gka['Main_Pos']=df_gka['Pos']
    df_gka['Sec_Pos']='-'
    df_gka.drop(['Matches','90s'], axis=1, inplace=True)
    df_gka.fillna(0,inplace=True)
    df_gka=df_gka[['Player',
    'Nation',
    'Main_Pos',
    'Sec_Pos',
    'Squad',
    'Age',
    'Born',
    'GA',
    'PKA',
    'FK',
    'CK',
    'OG',
    'PSxG',
    'PSxG/SoT',
    'PSxG+/-',
    '/90',
    'Cmp',
    'Att',
    'Cmp%',
    'Att (GK)',
    'Thr',
    'Launch%',
    'AvgLen',
    'Att',
    'Launch%',
    'AvgLen',
    'Opp',
    'Stp',
    'Stp%',
    '#OPA',
    '#OPA/90',
    'AvgDist']]
    col=list(df_gka.columns)

    for c in col:
        if c in df_gka.columns[:5]:
            df_gka.loc[:, c] = df_gka[c].astype(str)
        else:
            try:
                df_gka.loc[:, c] = df_gka[c].astype(float)
            except ValueError:
                continue  # Handle columns that cannot be converted to int
    time.sleep(10)

    #getting the stats of the midfielder
    df_list=[df_basic,df_tackel,df_passing,df_possesion,df_goals]
    df_final=df_list[0]
    for x in df_list:
        df_final=pd.merge(df_final,x,on=['Player','Nation','Main_Pos','Sec_Pos','Squad','Age','Born'],how='inner')

    #getting the goalkeeper stats and concat them on the midfielder stats
    df_finalgk=pd.merge(df_gk,df_gka,on=['Player','Nation','Main_Pos','Sec_Pos','Squad','Age','Born'],how='outer')

    df_final=pd.merge(df_final,df_finalgk,on=['Player','Nation','Main_Pos','Sec_Pos','Squad','Age','Born'],how='outer')
    #df_final=pd.merge(df_final,df_basic,on=['Player','Nation','Main_Pos','Sec_Pos','Squad','Age','Born'],how='outer')

    df_final.reset_index(drop=True, inplace=True)
    df_final=df_final.drop_duplicates()
    df_final['League']=league

    #these three lines of code help us to drop the unwanted duplicates of the column "Sec_Pos"
    lst=[*range(0,df_final.shape[1])]
    lst.pop(59)
    df_final=df_final.iloc[:,lst]

    df_final=df_final.fillna(0)
    df_final=df_final.reset_index()
    df_final=df_final.iloc[:,1:]
    #getting rid of all the duplicate columns, that end with a _y
    drop_col=list(df_final.columns[df_final.columns.str.endswith('_y')])
    df_final=df_final.drop(drop_col, axis=1)

    for new in list(df_final.columns):
        if new.endswith('_x'):
            df_final.rename(columns={new:new.split('_')[0]},inplace=True)
        else:
            df_final.rename(columns={new:new})

    return df_final


#-----------------------------------------------------------------------------------------------------------------------------------------------------
#row just a varibale, we tell it to perform on a row level with axis=1 in apply()
#-----------------------------------------------------------------------------------------------------------------------------------------------------
def clean_league_left_aut(row):
    if row['Club_Left'] in ['SK Sturm Graz', 'Red Bull Salzburg','SK Rapid Wien'] and row['League_Left'] == 'Bundesliga':
        return 'Österreichische Bundesliga'
    return row['League_Left']


#-----------------------------------------------------------------------------------------------------------------------------------------------------
#row just a varibale, we tell it to perform on a row level with axis=1 in apply()
#-----------------------------------------------------------------------------------------------------------------------------------------------------
def clean_league_joined_aut(row):
    if row['Club_Joined'] in ['SK Sturm Graz', 'Red Bull Salzburg','SK Rapid Wien'] and row['League_Joined'] == 'Bundesliga':
        return 'Österreichische Bundesliga'
    return row['League_Joined']


#-----------------------------------------------------------------------------------------------------------------------------------------------------
#creating a RobustZscore, which is better for more skewed data, works well even if the data is not perfectly normally distributed
#-----------------------------------------------------------------------------------------------------------------------------------------------------
def RobustZ_Score(x):
    median=np.median(x)
    mad=median_abs_deviation(x,scale='normal')
    #when the mad is zero, we use a default value
    if mad==0:
        mad=1e-6
    return (x-median)/mad


#-----------------------------------------------------------------------------------------------------------------------------------------------------
#transforming the string values into float values
#-----------------------------------------------------------------------------------------------------------------------------------------------------
def trans_value(x):
    """
    Parameter
    ----------------------------------------------------------------
    x => each row will be passed

    Outcome
    ----------------------------------------------------------------
    string values will be turned into float values
    """
    x=str(x)
    if 'Mio' in x and not 'Leihgebühr:' in x and 'Leihe' not in x:
        return int(float(x.split(' ')[0].replace(',','.'))*1000000)
        #return int(float(phase1_mio.split(':')[1].replace(',','.'))*1000000)
    elif 'Tsd' in x and not 'Leihgebühr:' in x and 'Leihe' not in x:
        return int(float(x.split(' ')[0].replace(',','.'))*1000)
    else:
        return 0


#-----------------------------------------------------------------------------------------------------------------------------------------------------
#creating new age groups
#-----------------------------------------------------------------------------------------------------------------------------------------------------
def age_transformer(x):
    if x < 20 and x >= 16:
        return 'Teenager'
    elif x < 25 and x >= 20:
        return 'Early Twenties'
    elif x < 30 and x >= 25:
        return 'Late Twenties'
    elif x < 35 and x >=30:
        return 'Early Thirties'
    else:
        return 'Late Thirties'


#-----------------------------------------------------------------------------------------------------------------------------------------------------
#checking if a player was injured in a season
#-----------------------------------------------------------------------------------------------------------------------------------------------------
##x_row works as a placeholder for the dataframe
def limited_time(x_row):
    league = x_row['League']
    mp = x_row['MP']

    if league == 'Premier League' and mp < 38 * 0.35:
        return 1
    elif league == 'La Liga' and mp < 38 * 0.35:
        return 1
    elif league == 'Serie A' and mp < 38 * 0.35:
        return 1
    elif league == 'BundesLiga' and mp < 34 * 0.35:
        return 1
    elif league == 'Ligue 1' and mp < 34 * 0.35:
        return 1
    elif league == 'Série A Brasil' and mp < 38 * 0.35:
        return 1
    elif league == 'Eredivisie' and mp < 34 * 0.35:
        return 1
    elif league == 'Primeira Liga' and mp < 34 * 0.35:
        return 1
    elif league == 'Liga MX' and mp < 34 * 0.35:
        return 1
    elif league == 'Belgian Pro League' and mp < 34 * 0.35:
        return 1
    elif league == '2. BundesLiga' and mp < 34 * 0.35:
        return 1
    elif league == 'Ligue 2' and mp < 34 * 0.35:
        return 1
    elif league == 'Serie B' and mp < 38 * 0.35:
        return 1
    elif league == 'Segunda División' and mp < 42 * 0.35:
        return 1
    elif league == 'MLS' and mp < 34 * 0.35:
        return 1
    else:
        return 0
    

# -------------------- knobs (tune speed vs reliability) --------------------
FORCE_MODE            = "robust"   # "fast" | "robust" | None -> auto (current season robust, others fast)
SCRAPER_PAUSE_FAST    = (0.7, 1.2)   # sleep after successful request in FAST mode
SCRAPER_PAUSE_ROBUST  = (1.8, 3.0)   # sleep after successful request in ROBUST mode
REQ_TIMEOUT_FAST      = 10
REQ_TIMEOUT_ROBUST    = 18
USE_PLAYWRIGHT        = False  # set True only if you installed Playwright

def log(msg): print(f"[fbref] {msg}", flush=True)

# ---------------- optional helpers (auto-detected; safe if missing) ----------------
try:
    import requests_cache
    requests_cache.install_cache("fbref_cache", expire_after=24*3600)
except Exception:
    pass

try:
    import browser_cookie3 as bc3
    HAS_BROWSER_COOKIES = True
except Exception:
    HAS_BROWSER_COOKIES = False

try:
    import cloudscraper
    HAS_CLOUDSCRAPER = True
except Exception:
    HAS_CLOUDSCRAPER = False

try:
    from curl_cffi import requests as curlreq
    HAS_CURLCFFI = True
except Exception:
    HAS_CURLCFFI = False

#printing different url outputs
def log(msg): print(f"[fbref] {msg}", flush=True)

# --------------------- optional last-resort fetch (Playwright) ---------------------
def _playwright_get_html(url: str, ua: str) -> str | None:
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            ctx = browser.new_context(user_agent=ua)
            page = ctx.new_page()
            page.goto(url, wait_until="networkidle", timeout=30000)
            html = page.content()
            browser.close()
            return html
    except Exception:
        return None


# -------------------------------- session + headers --------------------------------
def make_fbref_session() -> requests.Session:
    uas = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.5 Safari/605.1.15",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    ]
    s = requests.Session()
    s.headers.update({
        "User-Agent": random.choice(uas),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Referer": "https://fbref.com/en/comps/9/Premier-League-Stats",
    })
    retries = Retry(
        total=3,
        backoff_factor=0.6,
        status_forcelist=(403, 408, 429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
    )
    s.mount("https://", HTTPAdapter(max_retries=retries))

    # Load cookies from your real browser for fbref.com (helps with 403)
    if HAS_BROWSER_COOKIES:
        try:
            cj = bc3.load(domain_name="fbref.com")
            try: s.cookies.update(cj)
            except Exception: s.cookies.update({c.name: c.value for c in cj})
        except Exception:
            pass

    # Preflight to set site cookies
    try:
        s.get("https://fbref.com/", timeout=10); time.sleep(0.5)
        s.get("https://fbref.com/en/comps/9/Premier-League-Stats", timeout=10); time.sleep(0.5)
    except requests.RequestException:
        pass
    return s


# --------------------------------- fast/robust fetch ---------------------------------
def fetch_html(url: str, session: requests.Session, pause=None, mode="fast") -> str:
    """
    FAST: requests only (+ small UA rotate). No cloudscraper, no playwright.
    ROBUST: curl_cffi -> requests(+UA rotate) -> cloudscraper -> (optional) playwright.
    """
    t0 = time.time()
    pause = pause or (SCRAPER_PAUSE_FAST if mode == "fast" else SCRAPER_PAUSE_ROBUST)
    req_timeout = REQ_TIMEOUT_FAST if mode == "fast" else REQ_TIMEOUT_ROBUST
    tried = []

    # 1) curl_cffi first (robust)
    if mode == "robust" and HAS_CURLCFFI:
        try:
            with curlreq.Session(impersonate="chrome124") as cs:
                # carry cookies
                try:
                    for c in session.cookies:
                        cs.cookies.set(c.name, c.value, domain=c.domain or "fbref.com")
                except Exception:
                    pass
                r = cs.get(url, timeout=req_timeout)
                tried.append(f"curl_cffi:{r.status_code}")
                if r.status_code == 200:
                    time.sleep(random.uniform(*pause)); log(f"{url} ✓ curl_cffi {time.time()-t0:.2f}s")
                    return r.text
                # quick second try
                r = cs.get(url, timeout=req_timeout, impersonate="chrome125")
                tried.append(f"curl_cffi2:{r.status_code}")
                if r.status_code == 200:
                    time.sleep(random.uniform(*pause)); log(f"{url} ✓ curl_cffi2 {time.time()-t0:.2f}s")
                    return r.text
        except Exception:
            tried.append("curl_cffi:err")

    # 2) requests
    r = session.get(url, timeout=req_timeout)
    tried.append(f"req:{r.status_code}")
    if r.status_code == 200:
        time.sleep(random.uniform(*pause)); log(f"{url} ✓ requests {time.time()-t0:.2f}s")
        return r.text

    # 3) rotate UA + retry
    if r.status_code == 403:
        session.headers["User-Agent"] = make_fbref_session().headers["User-Agent"]
        time.sleep(0.7)
        r = session.get(url, timeout=req_timeout)
        tried.append(f"ua-rotate:{r.status_code}")
        if r.status_code == 200:
            time.sleep(random.uniform(*pause)); log(f"{url} ✓ UA-rotate {time.time()-t0:.2f}s")
            return r.text

    # 4) cloudscraper (robust)
    if mode == "robust" and r.status_code == 403 and HAS_CLOUDSCRAPER:
        try:
            scraper = cloudscraper.create_scraper(browser={'browser':'chrome','platform':'windows','mobile':False})
            try:
                for c in session.cookies:
                    scraper.cookies.set(c.name, c.value, domain=c.domain or "fbref.com")
            except Exception:
                pass
            r2 = scraper.get(url, timeout=req_timeout)
            tried.append(f"cloudscraper:{r2.status_code}")
            if r2.status_code == 200:
                time.sleep(random.uniform(*pause)); log(f"{url} ✓ cloudscraper {time.time()-t0:.2f}s")
                return r2.text
        except Exception:
            tried.append("cloudscraper:err")

    # 5) playwright (robust, optional)
    if mode == "robust" and USE_PLAYWRIGHT:
        html = _playwright_get_html(url, session.headers.get("User-Agent", ""))
        tried.append("playwright:" + ("ok" if html else "err"))
        if html:
            time.sleep(random.uniform(*pause)); log(f"{url} ✓ playwright {time.time()-t0:.2f}s")
            return html

    log(f"{url} ✗ failed in {time.time()-t0:.2f}s via {'>'.join(tried)}")
    r.raise_for_status()


# ----------------------- Pure-BeautifulSoup table → DataFrame -----------------------
def _table_tag_to_df(table_tag) -> pd.DataFrame:
    """
    Convert a <table> tag to DataFrame using only html.parser.
    Uses the LAST <thead> row as headers and reads <tbody> rows.
    Skips repeated header rows (cells[0] == 'Rk') inside <tbody>.
    """
    # headers
    header_cells = []
    thead = table_tag.find("thead")
    if thead:
        hdr_rows = thead.find_all("tr")
        if hdr_rows:
            last_hdr = hdr_rows[-1]
            header_cells = [th.get_text(strip=True) for th in last_hdr.find_all(["th", "td"])]

    # body
    tbody = table_tag.find("tbody") or table_tag
    rows = []
    for tr in tbody.find_all("tr"):
        cells = tr.find_all(["th", "td"])
        if not cells:
            continue
        first_text = cells[0].get_text(strip=True)
        if first_text == "Rk":
            continue
        rows.append([td.get_text(strip=True) for td in cells])

    if not rows:
        return pd.DataFrame()

    # align cols
    max_len = max(len(r) for r in rows)
    cols = header_cells if header_cells else [f"col_{i}" for i in range(max_len)]
    if len(cols) + 1 == max_len:
        cols = ["Rk"] + cols
    elif len(cols) != max_len:
        if len(cols) < max_len:
            cols += [f"col_{i}" for i in range(len(cols), max_len)]
        else:
            cols = cols[:max_len]

    df = pd.DataFrame(rows, columns=cols)
    return df


def read_fbref_table(url: str, table_id: str, session: requests.Session, referer: str | None = None, mode="fast") -> pd.DataFrame:
    """
    Read FBref table by id without lxml/html5lib:
      - fetch HTML
      - find <table id=...> in visible markup; else scan commented HTML
      - convert that <table> into a DataFrame
    """
    if referer:
        session.headers["Referer"] = referer
    html = fetch_html(url, session=session, mode=mode)

    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", id=table_id)

    if table is None:
        for c in soup.find_all(string=lambda t: isinstance(t, Comment)):
            inner = BeautifulSoup(c, "html.parser")
            table = inner.find("table", id=table_id)
            if table is not None:
                break

    if table is None:
        raise ValueError(f"No tables found for id='{table_id}' at {url}")

    df = _table_tag_to_df(table)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(0)
    return df


#fixing Nationality
def extract_fifa3(x):
    if pd.isna(x): return np.nan
    s = str(x).strip()
    m = re.search(r'\b([A-Za-z]{3})\b', s)  # ENG, fra, Ger, etc.
    if m:
        return m.group(1).upper()
    # fallback: first token's first 3 letters
    t = re.sub(r'[^A-Za-z\s]', ' ', s).strip()
    if not t: return np.nan
    first = t.split()[0]
    return first[:3].upper() if first else np.nan


# ============================== your function (Option B applied) ==============================
def player_stats_update(stats, tkl, passing, passing_style, poss, goal, gkl, gkal, league):
    session = make_fbref_session()
    hub = "https://fbref.com/en/comps/9/Premier-League-Stats"

    # decide mode: robust for current season, fast for older (unless FORCE_MODE set)
    def decide_mode(stats_url: str) -> str:
        if FORCE_MODE in ("fast", "robust"):
            return FORCE_MODE
        return "robust" if re.search(r"/2024-2025/", stats_url) else "fast"

    mode = decide_mode(stats)
    # -------- Option B: force ROBUST for GK pages even if season is old -------
    mode_gk = "robust" if mode == "fast" else mode

    def string_int_transform(x):
        try:
            return int(str(x).split('-')[0])
        except (ValueError, IndexError):
            return None

    # ---------------- fetch tables ----------------
    df_basic   = read_fbref_table(stats,         "stats_standard",    session, referer=hub,   mode=mode)
    if 'Rk' in df_basic.columns: df_basic = df_basic[df_basic['Rk'] != 'Rk']

    df_tackel  = read_fbref_table(tkl,           "stats_defense",     session, referer=stats, mode=mode)
    df_passing = read_fbref_table(passing,       "stats_passing",     session, referer=stats, mode=mode)
    df_passing_style = read_fbref_table(passing_style,"stats_passing_types", session, referer=passing, mode=mode)
    df_possesion = read_fbref_table(poss,        "stats_possession",  session, referer=stats, mode=mode)
    df_goals   = read_fbref_table(goal,          "stats_gca",         session, referer=stats, mode=mode)

    # ---- GK tables forced to ROBUST (Option B) ----
    df_gk      = read_fbref_table(gkl,           "stats_keeper",      session, referer=hub,   mode=mode_gk)
    if 'Rk' in df_gk.columns: df_gk = df_gk[df_gk['Rk'] != 'Rk']
    df_gka     = read_fbref_table(gkal,          "stats_keeper_adv",  session, referer=gkl,   mode=mode_gk)

    # ---------------- cleaning (unchanged) ----------------
    df_basic1 = df_basic.iloc[:, 26:-1].copy()
    df_basic  = df_basic.iloc[:, :-6].copy()
    #df_basic['Nation']   = df_basic['Nation'].astype(str).apply(lambda x: x.split(' ')[0].upper() if x else 'No Country')
    df_basic['Main_Pos'] = df_basic['Pos'].astype(str).apply(lambda x: x.split(',')[0] if x else '-')
    df_basic['Sec_Pos']  = df_basic['Pos'].astype(str).apply(lambda x: x.split(',')[-1] if x else '-')
    for x in list(df_basic1.columns):
        df_basic1[f'{x}_per_90'] = df_basic1[x]
    df_basic1 = df_basic1.iloc[:, 10:]
    df_basic['Age']  = df_basic['Age'].apply(string_int_transform)
    df_basic['Born'] = df_basic['Born'].apply(string_int_transform)
    df_basic = pd.concat([df_basic, df_basic1], axis=1)

    # defense
    df_tackel['Age']  = df_tackel['Age'].apply(string_int_transform)
    df_tackel['Born'] = df_tackel['Born'].apply(string_int_transform)
    #df_tackel['Nation'] = df_tackel['Nation'].fillna('No_Country')
    #df_tackel['Nation'] = df_tackel['Nation'].astype(str).apply(lambda x: x.split(' ')[0].upper() if x else 'No Country')
    df_tackel['Main_Pos'] = df_tackel['Pos'].astype(str).apply(lambda x: x.split(',')[0] if x else '-')
    df_tackel['Sec_Pos']  = df_tackel['Pos'].astype(str).apply(lambda x: x.split(',')[-1] if x else '-')
    df_tackel['Total Tackels']        = df_tackel.iloc[:, 8]
    df_tackel['Dribblers_Tackeld']    = df_tackel.iloc[:, 13]
    df_tackel['Dribblers_Tackel_Att'] = df_tackel.iloc[:, 14]
    df_tackel['Dribblers_Tackel%']    = df_tackel.iloc[:, 15]
    df_tackel = df_tackel[['Player','Nation','Main_Pos','Sec_Pos','Squad','Age','Born','Total Tackels',
                           'TklW','Def 3rd','Mid 3rd','Att 3rd','Dribblers_Tackeld','Dribblers_Tackel_Att',
                           'Dribblers_Tackel%','Lost','Blocks','Sh','Pass','Int','Tkl+Int','Clr','Err']]

    # passing
    df_passing['Age']  = df_passing['Age'].apply(string_int_transform)
    df_passing['Born'] = df_passing['Born'].apply(string_int_transform)
    #df_passing['Nation'] = df_passing['Nation'].fillna('No_Country')
    #df_passing['Nation'] = df_passing['Nation'].astype(str).apply(lambda x: x.split(' ')[0].upper() if x else 'No Country')
    df_passing['Main_Pos'] = df_passing['Pos'].astype(str).apply(lambda x: x.split(',')[0] if x else '-')
    df_passing['Sec_Pos']  = df_passing['Pos'].astype(str).apply(lambda x: x.split(',')[-1] if x else '-')
    df_passing['Total_Cmp']  = df_passing.iloc[:, 8]
    df_passing['Total_Att']  = df_passing.iloc[:, 9]
    df_passing['Total_Cmp%'] = df_passing.iloc[:, 10]
    df_passing['Short_Cmp']  = df_passing.iloc[:, 13]
    df_passing['Short_Att']  = df_passing.iloc[:, 14]
    df_passing['Short_Cmp%'] = df_passing.iloc[:, 15]
    df_passing['Medium_Cmp'] = df_passing.iloc[:, 16]
    df_passing['Medium_Att'] = df_passing.iloc[:, 17]
    df_passing['Medium_Cmp%']= df_passing.iloc[:, 18]
    df_passing['Long_Cmp']   = df_passing.iloc[:, 19]
    df_passing['Long_Att']   = df_passing.iloc[:, 20]
    df_passing['Long_Cmp%']  = df_passing.iloc[:, 21]
    df_passing = df_passing[['Player','Nation','Main_Pos','Sec_Pos','Squad','Age','Born',
                             'Total_Cmp','Total_Att','Total_Cmp%','TotDist','PrgDist',
                             'Short_Cmp','Short_Att','Short_Cmp%','Medium_Cmp','Medium_Att','Medium_Cmp%',
                             'Long_Cmp','Long_Att','Long_Cmp%','Ast','xAG','xA','A-xAG','KP','1/3','PPA','CrsPA','PrgP']]

    # passing types
    df_passing_style['Age']  = df_passing_style['Age'].apply(string_int_transform)
    df_passing_style['Born'] = df_passing_style['Born'].apply(string_int_transform)
    #df_passing_style['Nation'] = df_passing_style['Nation'].fillna('No_Country')
    #df_passing_style['Nation'] = df_passing_style['Nation'].astype(str).apply(lambda x: x.split(' ')[0].upper() if x else 'No Country')
    df_passing_style['Main_Pos'] = df_passing_style['Pos'].astype(str).apply(lambda x: x.split(',')[0] if x else '-')
    df_passing_style['Sec_Pos']  = df_passing_style['Pos'].astype(str).apply(lambda x: x.split(',')[-1] if x else '-')
    df_passing_style['Passes_in_Gameflow']     = df_passing_style['Live']
    df_passing_style['Passes_Out_of_Gameflow'] = df_passing_style['Dead']
    df_passing_style['Freekick']               = df_passing_style['FK']
    df_passing_style['Throughball']            = df_passing_style['TB']
    df_passing_style['40_yds_pass']            = df_passing_style['Sw']
    df_passing_style['Crosses']                = df_passing_style['Crs']
    df_passing_style['Corner_Kicks']           = df_passing_style['CK']
    df_passing_style['Corner_Kicks_In']        = df_passing_style['In']
    df_passing_style['Corner_Kicks_Out']       = df_passing_style['Out']
    df_passing_style['Corner_Kicks_Straight']  = df_passing_style['Str']
    df_passing_style['Passes_Offside']         = df_passing_style['Off']
    df_passing_style['Passes_Blocked']         = df_passing_style['Blocks']
    df_passing_style = df_passing_style[['Player','Nation','Main_Pos','Sec_Pos','Squad','Age','Born',
                                         'Passes_in_Gameflow','Passes_Out_of_Gameflow','Freekick','Throughball',
                                         '40_yds_pass','Crosses','Corner_Kicks','Corner_Kicks_In','Corner_Kicks_Out',
                                         'Corner_Kicks_Straight','Passes_Offside','Passes_Blocked']]

    # possession
    df_possesion['Age']  = df_possesion['Age'].apply(string_int_transform)
    df_possesion['Born'] = df_possesion['Born'].apply(string_int_transform)
    #df_possesion['Nation'] = df_possesion['Nation'].fillna('No_Country')
    #df_possesion['Nation'] = df_possesion['Nation'].astype(str).apply(lambda x: x.split(' ')[0].upper() if x else 'No Country')
    df_possesion['Main_Pos'] = df_possesion['Pos'].astype(str).apply(lambda x: x.split(',')[0] if x else '-')
    df_possesion['Sec_Pos']  = df_possesion['Pos'].astype(str).apply(lambda x: x.split(',')[-1] if x else '-')
    df_possesion['Touches_Def_Pen_Area']   = df_possesion['Def Pen']
    df_possesion['Touches_Def_3rd_Area']   = df_possesion['Def 3rd']
    df_possesion['Touches_Mid_3rd_Area']   = df_possesion['Mid 3rd']
    df_possesion['Touches_Att_3rd_Area']   = df_possesion['Att 3rd']
    df_possesion['Touches_Att_Pen_Area']   = df_possesion['Att Pen']
    df_possesion['Live_Touches_in_Game']   = df_possesion['Live']
    df_possesion['Dribbling_Att']          = df_possesion['Att']
    df_possesion['Dribbling_Succ']         = df_possesion['Succ']
    df_possesion['Dribbling_Succ%']        = df_possesion['Succ%']
    df_possesion['Tackeld_Dribbling']      = df_possesion['Tkld']
    df_possesion['Tackeld_Dribbling%']     = df_possesion['Tkld%']
    df_possesion['Total_Carry_Distance']   = df_possesion['TotDist']
    df_possesion['Total_Progressive_Carry_Distance'] = df_possesion['PrgDist']
    df_possesion['Total_Carries_in_1/3']   = df_possesion['1/3']
    df_possesion['Total_Carries_in_Penalty_Area'] = df_possesion['CPA']
    df_possesion['Miscontrols_Carries']    = df_possesion['Mis']
    df_possesion['Dispossed_Carries']      = df_possesion['Dis']
    df_possesion['Passes_Received']        = df_possesion['Rec']
    df_possesion['Progressive_Passes_Received'] = df_possesion['PrgR']
    df_possesion = df_possesion[['Player','Nation','Main_Pos','Sec_Pos','Squad','Age','Born','Touches',
                                 'Touches_Def_Pen_Area','Touches_Def_3rd_Area','Touches_Mid_3rd_Area',
                                 'Touches_Att_3rd_Area','Touches_Att_Pen_Area','Live_Touches_in_Game',
                                 'Dribbling_Att','Dribbling_Succ','Dribbling_Succ%','Tackeld_Dribbling',
                                 'Tackeld_Dribbling%','Total_Carry_Distance','Total_Progressive_Carry_Distance',
                                 'Total_Carries_in_1/3','Total_Carries_in_Penalty_Area','Miscontrols_Carries',
                                 'Dispossed_Carries','Passes_Received','Progressive_Passes_Received']]

    # gca
    df_goals['Age']  = df_goals['Age'].apply(string_int_transform)
    df_goals['Born'] = df_goals['Born'].apply(string_int_transform)
    #df_goals['Nation'] = df_goals['Nation'].fillna('No_Country')
    #df_goals['Nation'] = df_goals['Nation'].astype(str).apply(lambda x: x.split(' ')[0].upper() if x else 'No Country')
    df_goals['Main_Pos'] = df_goals['Pos'].astype(str).apply(lambda x: x.split(',')[0] if x else '-')
    df_goals['Sec_Pos']  = df_goals['Pos'].astype(str).apply(lambda x: x.split(',')[-1] if x else '-')
    df_goals['Shot_Creating_Action']     = df_goals['SCA']
    df_goals['Shot_Creating_Action_90']  = df_goals['SCA90']
    df_goals['Live_Passes_lead_Shot_Att'] = df_goals.iloc[:,10]
    df_goals['Dead_Passes_lead_Shot_Att'] = df_goals.iloc[:,11]
    df_goals['Shot_Att_after_Dribbling']  = df_goals.iloc[:,12]
    df_goals['Shot_lead_to_Shot_Att']     = df_goals.iloc[:,13]
    df_goals['Foul_drawn_lead_Shot_Att']  = df_goals.iloc[:,14]
    df_goals['Def_Action_lead_Shot_Att']  = df_goals.iloc[:,15]
    df_goals['Live_Pass_lead_Goal']       = df_goals.iloc[:,18]
    df_goals['Dead_Pass_lead_Goal']       = df_goals.iloc[:,19]
    df_goals['Goal_after_Dribbling']      = df_goals.iloc[:,20]
    df_goals['Shot_lead_Goal']            = df_goals.iloc[:,21]
    df_goals['Foul_drawn_lead_Goal']      = df_goals.iloc[:,22]
    df_goals['Def_Action_lead_Goal']      = df_goals.iloc[:,23]
    df_goals = df_goals[['Player','Nation','Main_Pos','Sec_Pos','Squad','Age','Born',
                         'Shot_Creating_Action','Shot_Creating_Action_90','Live_Passes_lead_Shot_Att',
                         'Dead_Passes_lead_Shot_Att','Shot_Att_after_Dribbling','Shot_lead_to_Shot_Att',
                         'Foul_drawn_lead_Shot_Att','Def_Action_lead_Shot_Att','GCA','GCA90',
                         'Live_Pass_lead_Goal','Dead_Pass_lead_Goal','Goal_after_Dribbling','Shot_lead_Goal',
                         'Foul_drawn_lead_Goal','Def_Action_lead_Goal']]

    # keeper basic
    df_gk['Age']  = df_gk['Age'].apply(string_int_transform)
    df_gk['Born'] = df_gk['Born'].apply(string_int_transform)
    #df_gk['Nation']=df_gk['Nation'].astype(str).apply(lambda x: x.split(' ')[0].upper() if x else 'No Country')
    df_gk['Main_Pos']=df_gk['Pos']; df_gk['Sec_Pos']='-'
    df_gk.drop(['Matches','90s','Rk'], axis=1, errors='ignore', inplace=True)
    df_gk.fillna(0, inplace=True)
    if df_gk.columns.duplicated().any():
        df_gk = df_gk.loc[:, ~df_gk.columns.duplicated()]
    df_gk = df_gk[['Player','Nation','Main_Pos','Sec_Pos','Squad','Age','Born','MP','Starts','Min',
                   'GA','GA90','SoTA','Saves','Save%','W','D','L','CS','CS%','PKatt','PKA','PKsv','PKm']]
    for c in df_gk.columns:
        if c not in ['Player','Nation','Main_Pos','Sec_Pos','Squad','Pos']:
            df_gk[c] = pd.to_numeric(df_gk[c], errors='coerce')

    # keeper adv
    df_gka['Age']  = df_gka['Age'].apply(string_int_transform)
    df_gka['Born'] = df_gka['Born'].apply(string_int_transform)
    #df_gka['Nation']=df_gka['Nation'].astype(str).apply(lambda x: x.split(' ')[0].upper() if x else 'No Country')
    df_gka['Main_Pos']=df_gka['Pos']; df_gka['Sec_Pos']='-'
    df_gka.drop(['Matches','90s'], axis=1, errors='ignore', inplace=True)
    df_gka.fillna(0, inplace=True)
    if df_gka.columns.duplicated().any():
        df_gka = df_gka.loc[:, ~df_gka.columns.duplicated()]
    for c in df_gka.columns:
        if c not in ['Player','Nation','Main_Pos','Sec_Pos','Squad','Pos']:
            df_gka[c] = pd.to_numeric(df_gka[c], errors='coerce')

    # Replace any earlier "split(' ')[0].upper()" logic. Do this BEFORE merges and BEFORE any .fillna(0)
    df_basic['Nation'] = df_basic['Nation'].map(extract_fifa3)

    # Drop Nation from all other frames so it can't spoil the join
    for d in [df_tackel, df_passing, df_passing_style, df_possesion, df_goals, df_gk, df_gka]:
        if 'Nation' in d.columns:
            d.drop(columns=['Nation'], inplace=True)

    # --- Merge WITHOUT Nation in the keys ---
    merge_keys = ['Player','Main_Pos','Sec_Pos','Squad','Age','Born']

    df_list  = [df_basic, df_tackel, df_passing, df_possesion, df_goals]
    df_final = df_list[0]
    for x in df_list[1:]:
        df_final = pd.merge(df_final, x, on=merge_keys, how='left')

    df_finalgk = pd.merge(df_gk, df_gka, on=merge_keys, how='outer')
    df_final   = pd.merge(df_final, df_finalgk, on=merge_keys, how='outer')

    df_final['League'] = league

    # keep your original pop(59), but guard it
    if df_final.shape[1] > 59:
        cols = list(range(df_final.shape[1]))
        try:
            cols.pop(59)
            df_final = df_final.iloc[:, cols]
        except Exception:
            pass

    df_final = df_final.fillna(0).infer_objects(copy=False).reset_index(drop=True)

    # drop *_y and rename *_x
    drop_col = [c for c in df_final.columns if c.endswith('_y')]
    if drop_col:
        df_final = df_final.drop(columns=drop_col)
    df_final = df_final.rename(columns={c: c[:-2] for c in df_final.columns if c.endswith('_x')})

    return df_final

def fix_min(x):
    if pd.isna(x):
        return x
    s = str(x).strip()
    s = s.replace(',', '')            # remove all commas
    return pd.to_numeric(s, errors='coerce')

ISO2_TO_FIFA3 = {
    # Europe
    'GB':'ENG','UK':'ENG','EN':'ENG','IE':'IRL','IR':'IRN',  # 'IR' is Iran (guard below for Ireland)
    'SCO':'SCO','WAL':'WAL','NIR':'NIR',  # already FIFA3, included for safety
    'DE':'GER','FR':'FRA','IT':'ITA','ES':'ESP','PT':'POR','NL':'NED','BE':'BEL',
    'AT':'AUT','CH':'SUI','DK':'DEN','SE':'SWE','NO':'NOR','FI':'FIN','IS':'ISL',
    'PL':'POL','CZ':'CZE','SK':'SVK','HU':'HUN','RO':'ROU','BG':'BUL','GR':'GRE','TR':'TUR',
    'UA':'UKR','RU':'RUS','KZ':'KAZ','AM':'ARM','GE':'GEO','AZ':'AZE','BY':'BLR',
    'AL':'ALB','BA':'BIH','RS':'SRB','ME':'MNE','MK':'MKD','SI':'SVN','XK':'KOS','LU':'LUX','LI':'LIE',

    # Africa
    'DZ':'ALG','MA':'MAR','TN':'TUN','EG':'EGY','LY':'LBY',
    'NG':'NGA','GH':'GHA','CI':'CIV','SN':'SEN','CM':'CMR','ML':'MLI','NE':'NER',
    'BF':'BFA','BJ':'BEN','TG':'TOG','GN':'GUI','SL':'SLE','LR':'LBR','GM':'GAM',
    'GQ':'EQG','GA':'GAB','CD':'COD','CG':'CGO','KE':'KEN','UG':'UGA','TZ':'TAN',
    'ET':'ETH','SO':'SOM','DJ':'DJI','RW':'RWA','BI':'BDI','ZA':'RSA','ZW':'ZIM',
    'ZM':'ZAM','MW':'MWI','MZ':'MOZ','NA':'NAM','BW':'BOT','LS':'LES','SZ':'SWZ',
    'AO':'ANG','MR':'MTN','CV':'CPV','ST':'STP','KM':'COM','SC':'SEY','MU':'MRI',

    # Americas
    'US':'USA','CA':'CAN','MX':'MEX',
    'AR':'ARG','BR':'BRA','UY':'URU','CL':'CHI','CO':'COL','EC':'ECU','PE':'PER','PY':'PAR','BO':'BOL','VE':'VEN',

    # Asia / Oceania
    'JP':'JPN','KR':'KOR','CN':'CHN','AU':'AUS','NZ':'NZL',
    'IR':'IRN','IQ':'IRQ','SA':'KSA','AE':'UAE','QA':'QAT','JO':'JOR','LB':'LBN','IL':'ISR',
    'TR':'TUR','AM':'ARM','AZ':'AZE','GE':'GEO','KZ':'KAZ','KG':'KGZ','TJ':'TJK','TM':'TKM','UZ':'UZB',
    'AF':'AFG','PK':'PAK','BD':'BAN','LK':'SRI','NP':'NEP','BT':'BHU','MM':'MYA',
    'TH':'THA','LA':'LAO','KH':'CAM','VN':'VIE','MY':'MAS','SG':'SIN','ID':'IDN','PH':'PHI',
    'TW':'TPE','HK':'HKG','MO':'MAC'
}

# Known FIFA-3 codes (so we can reject garbage like 'ESE', 'DKD')
KNOWN_FIFA3 = {
    'ENG','SCO','WAL','NIR','IRL','NED','GER','FRA','ESP','POR','ITA','DEN','SWE','NOR','FIN','ISL',
    'BEL','SUI','AUT','POL','CZE','SVK','HUN','ROU','BUL','GRE','TUR','UKR','RUS','KAZ','ARM','AZE','GEO',
    'SRB','CRO','BIH','SVN','MKD','MNE','ALB','KOS','LUX','LIE',
    'ALG','MAR','TUN','EGY','LBY','NGA','GHA','CIV','SEN','CMR','MLI','NER','BFA','BEN','TOG','GUI','SLE',
    'LBR','GAM','EQG','GAB','COD','CGO','KEN','UGA','TAN','ETH','SOM','DJI','RWA','BDI','RSA','ZIM','ZAM',
    'MWI','MOZ','NAM','BOT','LES','SWZ','ANG','MTN','CPV','STP','COM','SEY','MRI',
    'USA','CAN','MEX','ARG','BRA','URU','CHI','COL','ECU','PER','PAR','BOL','VEN',
    'JPN','KOR','CHN','AUS','NZL','IRN','IRQ','KSA','UAE','QAT','JOR','LBN','ISR',
    'KGZ','TJK','TKM','UZB','AFG','PAK','BAN','SRI','NEP','BHU','MYA','THA','LAO','CAM','VIE','MAS','SIN','IDN','PHI',
    'TPE','HKG','MAC'
}

def extract_fifa3(x):
    """
    Normalize FBref 'Nation' to a FIFA-3 code.
    - Accepts valid FIFA-3 tokens (ENG, DEN, ESP...).
    - Fixes artifacts like 'ESE','DKD' (from duplicated ISO-2: 'ESES','DKDK').
    - Converts ISO-2 (ES, DK, AR, GH...) to FIFA-3 via mapping.
    - Light fallback for plain country names.
    """
    if pd.isna(x):
        return np.nan
    s = str(x).strip().upper()

    # 1) If there's a clean 3-letter token and it's a known FIFA code → return it
    for t in re.findall(r'\b([A-Z]{3})\b', s):
        if t in KNOWN_FIFA3:
            return t
        # Fix XYX garbage like ESE, DKD (likely from 'ESES','DKDK' truncated)
        if len(t) == 3 and t[0] == t[2] and t[:2] in ISO2_TO_FIFA3:
            return ISO2_TO_FIFA3[t[:2]]

    # 2) Handle duplicated ISO-2 patterns: 'ESES','DKDK','ARAR', etc.
    m = re.search(r'\b([A-Z]{2})\1\b', s)  # XYXY
    if m:
        iso2 = m.group(1)
        if iso2 in ISO2_TO_FIFA3:
            return ISO2_TO_FIFA3[iso2]

    # 3) Plain ISO-2 tokens
    for t2 in re.findall(r'\b([A-Z]{2})\b', s):
        # Disambiguate Ireland vs Iran if someone wrote 'IR'
        if t2 == 'IR':
            return 'IRL'  # prefer Ireland in PL context
        if t2 in ISO2_TO_FIFA3:
            return ISO2_TO_FIFA3[t2]

    # 4) Fallback: try full country names (minimal list — extend as needed)
    name = re.sub(r'[^A-Z\s]', ' ', s).strip().lower()
    NAME_TO_FIFA3 = {
        'denmark':'DEN','spain':'ESP','mali':'MLI','ghana':'GHA','uzbekistan':'UZB','argentina':'ARG',
        'england':'ENG','scotland':'SCO','wales':'WAL','northern ireland':'NIR','ireland':'IRL',
        'germany':'GER','france':'FRA','italy':'ITA','portugal':'POR','netherlands':'NED','belgium':'BEL',
        'romania':'ROU','greece':'GRE','turkey':'TUR','austria':'AUT','switzerland':'SUI',
        'poland':'POL','czech republic':'CZE','slovakia':'SVK','hungary':'HUN',
        'brazil':'BRA','uruguay':'URU','chile':'CHI','colombia':'COL','ecuador':'ECU','peru':'PER','paraguay':'PAR',
        'japan':'JPN','korea republic':'KOR','south korea':'KOR','china':'CHN','australia':'AUS','new zealand':'NZL',
        'iran':'IRN','iraq':'IRQ','saudi arabia':'KSA','united arab emirates':'UAE','qatar':'QAT','israel':'ISR','lebanon':'LBN',
        'algeria':'ALG','morocco':'MAR','tunisia':'TUN','egypt':'EGY','cameroon':'CMR','senegal':'SEN','cote d ivoire':'CIV',
        'nigeria':'NGA'
    }
    if name in NAME_TO_FIFA3:
        return NAME_TO_FIFA3[name]

    # give up
    return np.nan


def best_nation_series(df: pd.DataFrame):
    """
    Return a Series of the best Nation-like column in df, mapped to FIFA-3, or None.
    Picks the Nation* column with the highest cardinality; if mapping collapses
    to all-NaN/constant, it tries the next-best candidate.
    """
    cols = [c for c in df.columns if c.lower().startswith('nation')]
    if not cols:
        return None

    # rank candidates by pre-map uniqueness (highest first)
    ranked = sorted(cols, key=lambda c: df[c].nunique(dropna=True), reverse=True)

    best = None
    best_unique = -1
    for c in ranked:
        s = df[c].map(extract_fifa3)
        u = s.nunique(dropna=True)
        if u > best_unique:
            best, best_unique = s, u
        if u > 1:  # good enough
            return s

    # fall back to the best we saw (may be constant or NaN-only)
    return best

def player_stats_update_2(stats, tkl, passing, passing_style, poss, goal, gkl, gkal, league):
    session = make_fbref_session()
    hub = "https://fbref.com/en/comps/9/Premier-League-Stats"

    # decide mode: robust for current season, fast for older (unless FORCE_MODE set)
    def decide_mode(stats_url: str) -> str:
        if FORCE_MODE in ("fast", "robust"):
            return FORCE_MODE
        return "robust" if re.search(r"/2024-2025/", stats_url) else "fast"

    mode = decide_mode(stats)
    mode_gk = "robust" if mode == "fast" else mode

    def string_int_transform(x):
        try:
            return int(str(x).split('-')[0])
        except (ValueError, IndexError):
            return None

    # ---------------- fetch tables ----------------
    df_basic   = read_fbref_table(stats,         "stats_standard",    session, referer=hub,   mode=mode)
    if 'Rk' in df_basic.columns: df_basic = df_basic[df_basic['Rk'] != 'Rk']

    df_tackel  = read_fbref_table(tkl,           "stats_defense",     session, referer=stats, mode=mode)
    df_passing = read_fbref_table(passing,       "stats_passing",     session, referer=stats, mode=mode)
    df_passing_style = read_fbref_table(passing_style,"stats_passing_types", session, referer=passing, mode=mode)
    df_possesion = read_fbref_table(poss,        "stats_possession",  session, referer=stats, mode=mode)
    df_goals   = read_fbref_table(goal,          "stats_gca",         session, referer=stats, mode=mode)

    # ---- GK tables forced to ROBUST (Option B) ----
    df_gk      = read_fbref_table(gkl,           "stats_keeper",      session, referer=hub,   mode=mode_gk)
    if 'Rk' in df_gk.columns: df_gk = df_gk[df_gk['Rk'] != 'Rk']
    df_gka     = read_fbref_table(gkal,          "stats_keeper_adv",  session, referer=gkl,   mode=mode_gk)

    # === NATIONALITY FIX (extract early & preserve) ===========================
    # Prefer your helper; fall back to mapping if needed.
    try:
        nat_series = best_nation_series(df_basic)  # returns FIFA-3 Series if you kept our helper
    except NameError:
        nat_series = None

    if nat_series is not None:
        df_basic['Nation'] = nat_series
    else:
        # fallback if helper isn't present
        if 'Nation' in df_basic.columns:
            df_basic['Nation'] = df_basic['Nation'].map(extract_fifa3)
        else:
            df_basic['Nation'] = None
    # =========================================================================

    # ---------------- cleaning (unchanged core) ----------------
    # slice aux/per90 block first, keep the original 'Nation' we just fixed
    df_basic1 = df_basic.iloc[:, 26:-1].copy()
    df_basic  = df_basic.iloc[:, :-6].copy()

    df_basic['Main_Pos'] = df_basic['Pos'].astype(str).apply(lambda x: x.split(',')[0] if x else '-')
    df_basic['Sec_Pos']  = df_basic['Pos'].astype(str).apply(lambda x: x.split(',')[-1] if x else '-')
    for x in list(df_basic1.columns):
        df_basic1[f'{x}_per_90'] = df_basic1[x]
    df_basic1 = df_basic1.iloc[:, 10:]
    df_basic['Age']  = df_basic['Age'].apply(string_int_transform)
    df_basic['Born'] = df_basic['Born'].apply(string_int_transform)
    df_basic = pd.concat([df_basic, df_basic1], axis=1)

    # defense
    df_tackel['Age']  = df_tackel['Age'].apply(string_int_transform)
    df_tackel['Born'] = df_tackel['Born'].apply(string_int_transform)
    df_tackel['Main_Pos'] = df_tackel['Pos'].astype(str).apply(lambda x: x.split(',')[0] if x else '-')
    df_tackel['Sec_Pos']  = df_tackel['Pos'].astype(str).apply(lambda x: x.split(',')[-1] if x else '-')
    df_tackel['Total Tackels']        = df_tackel.iloc[:, 8]
    df_tackel['Dribblers_Tackeld']    = df_tackel.iloc[:, 13]
    df_tackel['Dribblers_Tackel_Att'] = df_tackel.iloc[:, 14]
    df_tackel['Dribblers_Tackel%']    = df_tackel.iloc[:, 15]
    df_tackel = df_tackel[['Player','Nation','Main_Pos','Sec_Pos','Squad','Age','Born','Total Tackels',
                           'TklW','Def 3rd','Mid 3rd','Att 3rd','Dribblers_Tackeld','Dribblers_Tackel_Att',
                           'Dribblers_Tackel%','Lost','Blocks','Sh','Pass','Int','Tkl+Int','Clr','Err']]

    # passing
    df_passing['Age']  = df_passing['Age'].apply(string_int_transform)
    df_passing['Born'] = df_passing['Born'].apply(string_int_transform)
    df_passing['Main_Pos'] = df_passing['Pos'].astype(str).apply(lambda x: x.split(',')[0] if x else '-')
    df_passing['Sec_Pos']  = df_passing['Pos'].astype(str).apply(lambda x: x.split(',')[-1] if x else '-')
    df_passing['Total_Cmp']  = df_passing.iloc[:, 8]
    df_passing['Total_Att']  = df_passing.iloc[:, 9]
    df_passing['Total_Cmp%'] = df_passing.iloc[:, 10]
    df_passing['Short_Cmp']  = df_passing.iloc[:, 13]
    df_passing['Short_Att']  = df_passing.iloc[:, 14]
    df_passing['Short_Cmp%'] = df_passing.iloc[:, 15]
    df_passing['Medium_Cmp'] = df_passing.iloc[:, 16]
    df_passing['Medium_Att'] = df_passing.iloc[:, 17]
    df_passing['Medium_Cmp%']= df_passing.iloc[:, 18]
    df_passing['Long_Cmp']   = df_passing.iloc[:, 19]
    df_passing['Long_Att']   = df_passing.iloc[:, 20]
    df_passing['Long_Cmp%']  = df_passing.iloc[:, 21]
    df_passing = df_passing[['Player','Nation','Main_Pos','Sec_Pos','Squad','Age','Born',
                             'Total_Cmp','Total_Att','Total_Cmp%','TotDist','PrgDist',
                             'Short_Cmp','Short_Att','Short_Cmp%','Medium_Cmp','Medium_Att','Medium_Cmp%',
                             'Long_Cmp','Long_Att','Long_Cmp%','Ast','xAG','xA','A-xAG','KP','1/3','PPA','CrsPA','PrgP']]

    # passing types
    df_passing_style['Age']  = df_passing_style['Age'].apply(string_int_transform)
    df_passing_style['Born'] = df_passing_style['Born'].apply(string_int_transform)
    df_passing_style['Main_Pos'] = df_passing_style['Pos'].astype(str).apply(lambda x: x.split(',')[0] if x else '-')
    df_passing_style['Sec_Pos']  = df_passing_style['Pos'].astype(str).apply(lambda x: x.split(',')[-1] if x else '-')
    df_passing_style['Passes_in_Gameflow']     = df_passing_style['Live']
    df_passing_style['Passes_Out_of_Gameflow'] = df_passing_style['Dead']
    df_passing_style['Freekick']               = df_passing_style['FK']
    df_passing_style['Throughball']            = df_passing_style['TB']
    df_passing_style['40_yds_pass']            = df_passing_style['Sw']
    df_passing_style['Crosses']                = df_passing_style['Crs']
    df_passing_style['Corner_Kicks']           = df_passing_style['CK']
    df_passing_style['Corner_Kicks_In']        = df_passing_style['In']
    df_passing_style['Corner_Kicks_Out']       = df_passing_style['Out']
    df_passing_style['Corner_Kicks_Straight']  = df_passing_style['Str']
    df_passing_style['Passes_Offside']         = df_passing_style['Off']
    df_passing_style['Passes_Blocked']         = df_passing_style['Blocks']
    df_passing_style = df_passing_style[['Player','Nation','Main_Pos','Sec_Pos','Squad','Age','Born',
                                         'Passes_in_Gameflow','Passes_Out_of_Gameflow','Freekick','Throughball',
                                         '40_yds_pass','Crosses','Corner_Kicks','Corner_Kicks_In','Corner_Kicks_Out',
                                         'Corner_Kicks_Straight','Passes_Offside','Passes_Blocked']]

    # possession
    df_possesion['Age']  = df_possesion['Age'].apply(string_int_transform)
    df_possesion['Born'] = df_possesion['Born'].apply(string_int_transform)
    df_possesion['Main_Pos'] = df_possesion['Pos'].astype(str).apply(lambda x: x.split(',')[0] if x else '-')
    df_possesion['Sec_Pos']  = df_possesion['Pos'].astype(str).apply(lambda x: x.split(',')[-1] if x else '-')
    df_possesion['Touches_Def_Pen_Area']   = df_possesion['Def Pen']
    df_possesion['Touches_Def_3rd_Area']   = df_possesion['Def 3rd']
    df_possesion['Touches_Mid_3rd_Area']   = df_possesion['Mid 3rd']
    df_possesion['Touches_Att_3rd_Area']   = df_possesion['Att 3rd']
    df_possesion['Touches_Att_Pen_Area']   = df_possesion['Att Pen']
    df_possesion['Live_Touches_in_Game']   = df_possesion['Live']
    df_possesion['Dribbling_Att']          = df_possesion['Att']
    df_possesion['Dribbling_Succ']         = df_possesion['Succ']
    df_possesion['Dribbling_Succ%']        = df_possesion['Succ%']
    df_possesion['Tackeld_Dribbling']      = df_possesion['Tkld']
    df_possesion['Tackeld_Dribbling%']     = df_possesion['Tkld%']
    df_possesion['Total_Carry_Distance']   = df_possesion['TotDist']
    df_possesion['Total_Progressive_Carry_Distance'] = df_possesion['PrgDist']
    df_possesion['Total_Carries_in_1/3']   = df_possesion['1/3']
    df_possesion['Total_Carries_in_Penalty_Area'] = df_possesion['CPA']
    df_possesion['Miscontrols_Carries']    = df_possesion['Mis']
    df_possesion['Dispossed_Carries']      = df_possesion['Dis']
    df_possesion['Passes_Received']        = df_possesion['Rec']
    df_possesion['Progressive_Passes_Received'] = df_possesion['PrgR']
    df_possesion = df_possesion[['Player','Nation','Main_Pos','Sec_Pos','Squad','Age','Born','Touches',
                                 'Touches_Def_Pen_Area','Touches_Def_3rd_Area','Touches_Mid_3rd_Area',
                                 'Touches_Att_3rd_Area','Touches_Att_Pen_Area','Live_Touches_in_Game',
                                 'Dribbling_Att','Dribbling_Succ','Dribbling_Succ%','Tackeld_Dribbling',
                                 'Tackeld_Dribbling%','Total_Carry_Distance','Total_Progressive_Carry_Distance',
                                 'Total_Carries_in_1/3','Total_Carries_in_Penalty_Area','Miscontrols_Carries',
                                 'Dispossed_Carries','Passes_Received','Progressive_Passes_Received']]

    # gca
    df_goals['Age']  = df_goals['Age'].apply(string_int_transform)
    df_goals['Born'] = df_goals['Born'].apply(string_int_transform)
    df_goals['Main_Pos'] = df_goals['Pos'].astype(str).apply(lambda x: x.split(',')[0] if x else '-')
    df_goals['Sec_Pos']  = df_goals['Pos'].astype(str).apply(lambda x: x.split(',')[-1] if x else '-')
    df_goals['Shot_Creating_Action']     = df_goals['SCA']
    df_goals['Shot_Creating_Action_90']  = df_goals['SCA90']
    df_goals['Live_Passes_lead_Shot_Att'] = df_goals.iloc[:,10]
    df_goals['Dead_Passes_lead_Shot_Att'] = df_goals.iloc[:,11]
    df_goals['Shot_Att_after_Dribbling']  = df_goals.iloc[:,12]
    df_goals['Shot_lead_to_Shot_Att']     = df_goals.iloc[:,13]
    df_goals['Foul_drawn_lead_Shot_Att']  = df_goals.iloc[:,14]
    df_goals['Def_Action_lead_Shot_Att']  = df_goals.iloc[:,15]
    df_goals['Live_Pass_lead_Goal']       = df_goals.iloc[:,18]
    df_goals['Dead_Pass_lead_Goal']       = df_goals.iloc[:,19]
    df_goals['Goal_after_Dribbling']      = df_goals.iloc[:,20]
    df_goals['Shot_lead_Goal']            = df_goals.iloc[:,21]
    df_goals['Foul_drawn_lead_Goal']      = df_goals.iloc[:,22]
    df_goals['Def_Action_lead_Goal']      = df_goals.iloc[:,23]
    df_goals = df_goals[['Player','Nation','Main_Pos','Sec_Pos','Squad','Age','Born',
                         'Shot_Creating_Action','Shot_Creating_Action_90','Live_Passes_lead_Shot_Att',
                         'Dead_Passes_lead_Shot_Att','Shot_Att_after_Dribbling','Shot_lead_to_Shot_Att',
                         'Foul_drawn_lead_Shot_Att','Def_Action_lead_Shot_Att','GCA','GCA90',
                         'Live_Pass_lead_Goal','Dead_Pass_lead_Goal','Goal_after_Dribbling','Shot_lead_Goal',
                         'Foul_drawn_lead_Goal','Def_Action_lead_Goal']]

    # keeper basic
    df_gk['Age']  = df_gk['Age'].apply(string_int_transform)
    df_gk['Born'] = df_gk['Born'].apply(string_int_transform)
    df_gk['Main_Pos']=df_gk['Pos']; df_gk['Sec_Pos']='-'
    df_gk.drop(['Matches','90s','Rk'], axis=1, errors='ignore', inplace=True)
    df_gk.fillna(0, inplace=True)
    if df_gk.columns.duplicated().any():
        df_gk = df_gk.loc[:, ~df_gk.columns.duplicated()]
    df_gk = df_gk[['Player','Nation','Main_Pos','Sec_Pos','Squad','Age','Born','MP','Starts','Min',
                   'GA','GA90','SoTA','Saves','Save%','W','D','L','CS','CS%','PKatt','PKA','PKsv','PKm']]
    for c in df_gk.columns:
        if c not in ['Player','Nation','Main_Pos','Sec_Pos','Squad','Pos']:
            df_gk[c] = pd.to_numeric(df_gk[c], errors='coerce')

    # keeper adv
    df_gka['Age']  = df_gka['Age'].apply(string_int_transform)
    df_gka['Born'] = df_gka['Born'].apply(string_int_transform)
    df_gka['Main_Pos']=df_gka['Pos']; df_gka['Sec_Pos']='-'
    df_gka.drop(['Matches','90s'], axis=1, errors='ignore', inplace=True)
    df_gka.fillna(0, inplace=True)
    if df_gka.columns.duplicated().any():
        df_gka = df_gka.loc[:, ~df_gka.columns.duplicated()]
    for c in df_gka.columns:
        if c not in ['Player','Nation','Main_Pos','Sec_Pos','Squad','Pos']:
            df_gka[c] = pd.to_numeric(df_gka[c], errors='coerce')

    # === NATIONALITY FIX: prevent other frames from overwriting it on merge ===
    for d in [df_tackel, df_passing, df_passing_style, df_possesion, df_goals, df_gk, df_gka]:
        if 'Nation' in d.columns:
            d.drop(columns=['Nation'], inplace=True)
    # ========================================================================

    merge_keys = ['Player','Main_Pos','Sec_Pos','Squad','Age','Born']

    df_list  = [df_basic, df_tackel, df_passing, df_possesion, df_goals]
    df_final = df_list[0]
    for x in df_list[1:]:
        df_final = pd.merge(df_final, x, on=merge_keys, how='left')

    df_finalgk = pd.merge(df_gk, df_gka, on=merge_keys, how='outer')
    df_final   = pd.merge(df_final, df_finalgk, on=merge_keys, how='outer')

    # add league as its own column (distinct from Nation)
    df_final['League'] = league

    # keep your original pop(59), but guard it
    if df_final.shape[1] > 59:
        cols = list(range(df_final.shape[1]))
        try:
            cols.pop(59)
            df_final = df_final.iloc[:, cols]
        except Exception:
            pass

    # === NATIONALITY FIX: only fill numerics with 0; leave Nation as text =====
    import numpy as np
    num_cols = df_final.select_dtypes(include=[np.number]).columns
    df_final[num_cols] = df_final[num_cols].fillna(0)

    # optional: tidy object cols (but do NOT overwrite Nation)
    df_final['Main_Pos'] = df_final['Main_Pos'].fillna('-')
    df_final['Sec_Pos']  = df_final['Sec_Pos'].fillna('-')
    # ========================================================================

    df_final = df_final.infer_objects(copy=False).reset_index(drop=True)

    # drop *_y and rename *_x
    drop_col = [c for c in df_final.columns if c.endswith('_y')]
    if drop_col:
        df_final = df_final.drop(columns=drop_col)
    df_final = df_final.rename(columns={c: c[:-2] for c in df_final.columns if c.endswith('_x')})

    return df_final



def player_stats_update_final(stats, tkl, passing, passing_style, poss, goal, gkl, gkal, league):
    session = make_fbref_session()

    # ---------- mode ----------
    def decide_mode(stats_url: str) -> str:
        if 'FORCE_MODE' in globals() and FORCE_MODE in ("fast", "robust"):
            return FORCE_MODE
        return "robust" if "/2024-2025/" in stats_url else "fast"

    mode = decide_mode(stats)
    mode_gk = "robust" if mode == "fast" else mode

    # ---------- helpers ----------
    def _fetch(url, table_id, referer, mode_):
        df = read_fbref_table(url, table_id, session, referer=referer, mode=mode_)
        if df is None or len(df) == 0:
            raise RuntimeError(f"No rows for {table_id} from {url} (referer={referer})")
        return df

    def string_int_transform(x):
        try:
            return int(str(x).split('-')[0])
        except (ValueError, IndexError):
            try:
                return int(float(x))
            except Exception:
                return None

    def ensure_merge_keys(df: pd.DataFrame) -> pd.DataFrame:
        if 'Player' not in df.columns: df['Player'] = pd.NA
        if 'Squad'  not in df.columns: df['Squad']  = pd.NA
        df['Age']  = df['Age'].apply(string_int_transform)  if 'Age'  in df.columns else None
        df['Born'] = df['Born'].apply(string_int_transform) if 'Born' in df.columns else None
        if 'Main_Pos' not in df.columns:
            if 'Pos' in df.columns: df['Main_Pos'] = df['Pos'].astype(str).str.split(',').str[0]
            else:                    df['Main_Pos'] = '-'
        if 'Sec_Pos' not in df.columns:
            if 'Pos' in df.columns: df['Sec_Pos'] = df['Pos'].astype(str).str.split(',').str[-1]
            else:                    df['Sec_Pos'] = '-'
        return df

    def _derive_fifa3(df_with_nation: pd.DataFrame) -> pd.Series:
        """Robustly get FIFA-3 code from FBref 'Nation' text like 'engENG', 'ESPSPAIN', '🇪🇸ESP', etc."""
        import unicodedata

        ISO2_TO_FIFA3 = {
            "ES":"ESP","NL":"NED","PT":"POR","DK":"DEN","AR":"ARG","UY":"URU","PL":"POL",
            "GH":"GHA","NG":"NGA","SK":"SVK","CM":"CMR","ME":"MNE","MA":"MAR","IS":"ISL",
            "IT":"ITA","EN":"ENG","GB":"GBR","DE":"GER","FR":"FRA","BR":"BRA","BE":"BEL",
            "CH":"SUI","AT":"AUT","SE":"SWE","NO":"NOR","FI":"FIN","IE":"IRL","SC":"SCO",
            "WA":"WAL","NI":"NIR","CZ":"CZE","HR":"CRO","RS":"SRB","BA":"BIH","AL":"ALB",
            "RO":"ROU","BG":"BUL","GR":"GRE","TR":"TUR","HU":"HUN","UA":"UKR","RU":"RUS",
            "JP":"JPN","KR":"KOR","MX":"MEX","US":"USA","CA":"CAN","CL":"CHI","CO":"COL",
            "EC":"ECU","PE":"PER","BO":"BOL","PY":"PAR","VE":"VEN","TN":"TUN","DZ":"ALG",
            "EG":"EGY","CI":"CIV","SN":"SEN","CD":"COD","ZA":"RSA","AU":"AUS","NZ":"NZL",
            "SI":"SVN","LT":"LTU","LV":"LVA","EE":"EST","GE":"GEO","AM":"ARM","MK":"MKD"
        }
        BROKEN_FIX = {
            # FBref sometimes concatenates two 3-letter chunks; if we catch the wrong one, fix here.
            "ESE":"ESP","NLN":"NED","PTP":"POR","DKD":"DEN","ARA":"ARG","UYU":"URU","PLP":"POL",
            "GHG":"GHA","NGN":"NGA","SKS":"SVK","CMC":"CMR","MEM":"MNE","MAM":"MAR","ISI":"ISL",
            "ITI":"ITA","MLM":"MLI","SES":"SWE","IEI":"IRL","BRB":"BRA","DEG":"GER","BEB":"BEL",
            "XKK":"KVX","FRF":"FRA","TRT":"TRU","SNS":"SEN"
        }
        NAME_TO_FIFA3 = {
            "SPAIN":"ESP","ENGLAND":"ENG","ITALY":"ITA","NETHERLANDS":"NED","MONTENEGRO":"MNE",
            "MOROCCO":"MAR","ICELAND":"ISL","PORTUGAL":"POR","DENMARK":"DEN","ARGENTINA":"ARG",
            "URUGUAY":"URU","POLAND":"POL","GHANA":"GHA","NIGERIA":"NGA","SLOVAKIA":"SVK",
            "CAMEROON":"CMR","SCOTLAND":"SCO","WALES":"WAL","NORTHERN IRELAND":"NIR",
            "IRELAND":"IRL","MALI":"MLI","SLOVENIA":"SVN","LITHUANIA":"LTU","LATVIA":"LVA",
            "ESTONIA":"EST","GEORGIA":"GEO","ARMENIA":"ARM","NORTH MACEDONIA":"MKD",
            "UNITED STATES":"USA","UNITED KINGDOM":"GBR","ENGLAND":"ENG"
        }

        if 'Nation' not in df_with_nation.columns:
            return pd.Series(pd.NA, index=df_with_nation.index, dtype=pd.StringDtype())

        def norm_letters(x: str) -> str:
            if pd.isna(x): return ""
            x = str(x)
            x = unicodedata.normalize('NFKD', x).encode('ascii', 'ignore').decode('ascii')
            x = re.sub(r'[^A-Za-z]+', '', x).upper()  # keep letters, smash everything else
            return x

        def norm_words(x: str) -> str:
            if pd.isna(x): return ""
            x = str(x)
            x = unicodedata.normalize('NFKD', x).encode('ascii', 'ignore').decode('ascii')
            x = re.sub(r'[^A-Za-z ]+', ' ', x).upper()
            x = re.sub(r'\s+', ' ', x).strip()
            return x

        s_letters = df_with_nation['Nation'].map(norm_letters)
        s_words   = df_with_nation['Nation'].map(norm_words)

        out = []
        for letters, words in zip(s_letters, s_words):
            # 1) prefer any 3-letter block; take the LAST one (e.g., 'engENG' -> ['ENG'] -> ENG)
            blocks = re.findall(r'[A-Z]{3}', letters or '')
            if blocks:
                c = blocks[-1]
                out.append(BROKEN_FIX.get(c, c))
                continue
            # 2) look for any 2-letter token we can convert
            tokens = words.split()
            code2 = next((t for t in reversed(tokens) if len(t) == 2 and t in ISO2_TO_FIFA3), None)
            if code2:
                out.append(ISO2_TO_FIFA3[code2]); continue
            # 3) country names
            name_hit = next((code for name, code in NAME_TO_FIFA3.items() if name in words), None)
            out.append(name_hit if name_hit else pd.NA)

        return pd.Series(out, index=df_with_nation.index, dtype=pd.StringDtype())

    # ---------- fetch ----------
    df_basic   = _fetch(stats,   "stats_standard",    referer=stats, mode_=mode)
    if 'Rk' in df_basic.columns: df_basic = df_basic[df_basic['Rk'] != 'Rk']
    df_tackel  = _fetch(tkl,     "stats_defense",     referer=stats, mode_=mode)
    df_passing = _fetch(passing, "stats_passing",     referer=stats, mode_=mode)
    df_passing_style = _fetch(passing_style, "stats_passing_types", referer=passing, mode_=mode)
    df_possesion = _fetch(poss,  "stats_possession",  referer=stats, mode_=mode)
    df_goals   = _fetch(goal,    "stats_gca",         referer=stats, mode_=mode)
    df_gk      = _fetch(gkl,     "stats_keeper",      referer=gkl,   mode_=mode_gk)
    if 'Rk' in df_gk.columns: df_gk = df_gk[df_gk['Rk'] != 'Rk']
    df_gka     = _fetch(gkal,    "stats_keeper_adv",  referer=gkl,   mode_=mode_gk)

    # ---------- GK nation map BEFORE dropping ----------
    if 'Nation' in df_gk.columns:
        gk_nat_tmp = df_gk[['Player','Squad','Nation']].copy()
    else:
        # rare fallback
        gk_nat_tmp = df_basic[['Player','Squad']].copy()
        gk_nat_tmp['Nation'] = pd.NA

    gk_nat_tmp['Nation'] = _derive_fifa3(gk_nat_tmp)
    gk_nat_map = (gk_nat_tmp.dropna(subset=['Nation'])
                            .drop_duplicates(subset=['Player','Squad']))

    # ---------- nationality on basic ----------
    df_basic['Nation'] = _derive_fifa3(df_basic)
    nation_freeze = df_basic['Nation'].copy()

    # ---------- basic cleaning ----------
    df_basic1 = df_basic.iloc[:, 26:-1].copy()
    df_basic  = df_basic.iloc[:, :-6].copy()
    df_basic['Main_Pos'] = df_basic['Pos'].astype(str).apply(lambda x: x.split(',')[0] if x else '-')
    df_basic['Sec_Pos']  = df_basic['Pos'].astype(str).apply(lambda x: x.split(',')[-1] if x else '-')
    for x in list(df_basic1.columns):
        df_basic1[f'{x}_per_90'] = df_basic1[x]
    df_basic1 = df_basic1.iloc[:, 10:]
    df_basic['Age']  = df_basic['Age'].apply(string_int_transform)
    df_basic['Born'] = df_basic['Born'].apply(string_int_transform)
    df_basic = pd.concat([df_basic, df_basic1], axis=1)
    df_basic['Nation'] = nation_freeze  # restore

    # ---------- other frames ----------
    # defense
    df_tackel['Age']  = df_tackel['Age'].apply(string_int_transform)
    df_tackel['Born'] = df_tackel['Born'].apply(string_int_transform)
    df_tackel['Main_Pos'] = df_tackel['Pos'].astype(str).apply(lambda x: x.split(',')[0] if x else '-')
    df_tackel['Sec_Pos']  = df_tackel['Pos'].astype(str).apply(lambda x: x.split(',')[-1] if x else '-')
    df_tackel['Total Tackels']        = df_tackel.iloc[:, 8]
    df_tackel['Dribblers_Tackeld']    = df_tackel.iloc[:, 13]
    df_tackel['Dribblers_Tackel_Att'] = df_tackel.iloc[:, 14]
    df_tackel['Dribblers_Tackel%']    = df_tackel.iloc[:, 15]
    df_tackel = df_tackel[['Player','Nation','Main_Pos','Sec_Pos','Squad','Age','Born','Total Tackels',
                           'TklW','Def 3rd','Mid 3rd','Att 3rd','Dribblers_Tackeld','Dribblers_Tackel_Att',
                           'Dribblers_Tackel%','Lost','Blocks','Sh','Pass','Int','Tkl+Int','Clr','Err']]

    # passing
    df_passing['Age']  = df_passing['Age'].apply(string_int_transform)
    df_passing['Born'] = df_passing['Born'].apply(string_int_transform)
    df_passing['Main_Pos'] = df_passing['Pos'].astype(str).apply(lambda x: x.split(',')[0] if x else '-')
    df_passing['Sec_Pos']  = df_passing['Pos'].astype(str).apply(lambda x: x.split(',')[-1] if x else '-')
    df_passing['Total_Cmp']  = df_passing.iloc[:, 8]
    df_passing['Total_Att']  = df_passing.iloc[:, 9]
    df_passing['Total_Cmp%'] = df_passing.iloc[:, 10]
    df_passing['Short_Cmp']  = df_passing.iloc[:, 13]
    df_passing['Short_Att']  = df_passing.iloc[:, 14]
    df_passing['Short_Cmp%'] = df_passing.iloc[:, 15]
    df_passing['Medium_Cmp'] = df_passing.iloc[:, 16]
    df_passing['Medium_Att'] = df_passing.iloc[:, 17]
    df_passing['Medium_Cmp%']= df_passing.iloc[:, 18]
    df_passing['Long_Cmp']   = df_passing.iloc[:, 19]
    df_passing['Long_Att']   = df_passing.iloc[:, 20]
    df_passing['Long_Cmp%']  = df_passing.iloc[:, 21]
    df_passing = df_passing[['Player','Nation','Main_Pos','Sec_Pos','Squad','Age','Born',
                             'Total_Cmp','Total_Att','Total_Cmp%','TotDist','PrgDist',
                             'Short_Cmp','Short_Att','Short_Cmp%','Medium_Cmp','Medium_Att','Medium_Cmp%',
                             'Long_Cmp','Long_Att','Long_Cmp%','Ast','xAG','xA','A-xAG','KP','1/3','PPA','CrsPA','PrgP']]

    # possession
    df_possesion['Age']  = df_possesion['Age'].apply(string_int_transform)
    df_possesion['Born'] = df_possesion['Born'].apply(string_int_transform)
    df_possesion['Main_Pos'] = df_possesion['Pos'].astype(str).apply(lambda x: x.split(',')[0] if x else '-')
    df_possesion['Sec_Pos']  = df_possesion['Pos'].astype(str).apply(lambda x: x.split(',')[-1] if x else '-')
    df_possesion['Touches_Def_Pen_Area']   = df_possesion['Def Pen']
    df_possesion['Touches_Def_3rd_Area']   = df_possesion['Def 3rd']
    df_possesion['Touches_Mid_3rd_Area']   = df_possesion['Mid 3rd']
    df_possesion['Touches_Att_3rd_Area']   = df_possesion['Att 3rd']
    df_possesion['Touches_Att_Pen_Area']   = df_possesion['Att Pen']
    df_possesion['Live_Touches_in_Game']   = df_possesion['Live']
    df_possesion['Dribbling_Att']          = df_possesion['Att']
    df_possesion['Dribbling_Succ']         = df_possesion['Succ']
    df_possesion['Dribbling_Succ%']        = df_possesion['Succ%']
    df_possesion['Tackeld_Dribbling']      = df_possesion['Tkld']
    df_possesion['Tackeld_Dribbling%']     = df_possesion['Tkld%']
    df_possesion['Total_Carry_Distance']   = df_possesion['TotDist']
    df_possesion['Total_Progressive_Carry_Distance'] = df_possesion['PrgDist']
    df_possesion['Total_Carries_in_1/3']   = df_possesion['1/3']
    df_possesion['Total_Carries_in_Penalty_Area'] = df_possesion['CPA']
    df_possesion['Miscontrols_Carries']    = df_possesion['Mis']
    df_possesion['Dispossed_Carries']      = df_possesion['Dis']
    df_possesion['Passes_Received']        = df_possesion['Rec']
    df_possesion['Progressive_Passes_Received'] = df_possesion['PrgR']
    df_possesion = df_possesion[['Player','Nation','Main_Pos','Sec_Pos','Squad','Age','Born','Touches',
                                 'Touches_Def_Pen_Area','Touches_Def_3rd_Area','Touches_Mid_3rd_Area',
                                 'Touches_Att_3rd_Area','Touches_Att_Pen_Area','Live_Touches_in_Game',
                                 'Dribbling_Att','Dribbling_Succ','Dribbling_Succ%','Tackeld_Dribbling',
                                 'Tackeld_Dribbling%','Total_Carry_Distance','Total_Progressive_Carry_Distance',
                                 'Total_Carries_in_1/3','Total_Carries_in_Penalty_Area','Miscontrols_Carries',
                                 'Dispossed_Carries','Passes_Received','Progressive_Passes_Received']]

    # gca
    df_goals['Age']  = df_goals['Age'].apply(string_int_transform)
    df_goals['Born'] = df_goals['Born'].apply(string_int_transform)
    df_goals['Main_Pos'] = df_goals['Pos'].astype(str).apply(lambda x: x.split(',')[0] if x else '-')
    df_goals['Sec_Pos']  = df_goals['Pos'].astype(str).apply(lambda x: x.split(',')[-1] if x else '-')
    df_goals['Shot_Creating_Action']     = df_goals['SCA']
    df_goals['Shot_Creating_Action_90']  = df_goals['SCA90']
    df_goals['Live_Passes_lead_Shot_Att'] = df_goals.iloc[:,10]
    df_goals['Dead_Passes_lead_Shot_Att'] = df_goals.iloc[:,11]
    df_goals['Shot_Att_after_Dribbling']  = df_goals.iloc[:,12]
    df_goals['Shot_lead_to_Shot_Att']     = df_goals.iloc[:,13]
    df_goals['Foul_drawn_lead_Shot_Att']  = df_goals.iloc[:,14]
    df_goals['Def_Action_lead_Shot_Att']  = df_goals.iloc[:,15]
    df_goals['Live_Pass_lead_Goal']       = df_goals.iloc[:,18]
    df_goals['Dead_Pass_lead_Goal']       = df_goals.iloc[:,19]
    df_goals['Goal_after_Dribbling']      = df_goals.iloc[:,20]
    df_goals['Shot_lead_Goal']            = df_goals.iloc[:,21]
    df_goals['Foul_drawn_lead_Goal']      = df_goals.iloc[:,22]
    df_goals['Def_Action_lead_Goal']      = df_goals.iloc[:,23]
    df_goals = df_goals[['Player','Nation','Main_Pos','Sec_Pos','Squad','Age','Born',
                         'Shot_Creating_Action','Shot_Creating_Action_90','Live_Passes_lead_Shot_Att',
                         'Dead_Passes_lead_Shot_Att','Shot_Att_after_Dribbling','Shot_lead_to_Shot_Att',
                         'Foul_drawn_lead_Shot_Att','Def_Action_lead_Shot_Att','GCA','GCA90',
                         'Live_Pass_lead_Goal','Dead_Pass_lead_Goal','Goal_after_Dribbling','Shot_lead_Goal',
                         'Foul_drawn_lead_Goal','Def_Action_lead_Goal']]

    # ---------- guarantee merge keys ----------
    df_basic     = ensure_merge_keys(df_basic)
    df_tackel    = ensure_merge_keys(df_tackel)
    df_passing   = ensure_merge_keys(df_passing)
    df_possesion = ensure_merge_keys(df_possesion)
    df_goals     = ensure_merge_keys(df_goals)
    df_gk        = ensure_merge_keys(df_gk)
    df_gka       = ensure_merge_keys(df_gka)
    df_gk['Main_Pos']  = 'GK'; df_gk['Sec_Pos']  = '-'
    df_gka['Main_Pos'] = 'GK'; df_gka['Sec_Pos'] = '-'

    # ---------- prevent Nation overwrite ----------
    for d in [df_tackel, df_passing, df_passing_style, df_possesion, df_goals, df_gk, df_gka]:
        if 'Nation' in d.columns:
            d.drop(columns=['Nation'], inplace=True)

    # ---------- merge ----------
    merge_keys = ['Player','Main_Pos','Sec_Pos','Squad','Age','Born']

    df_list  = [df_basic, df_tackel, df_passing, df_possesion, df_goals]
    df_final = df_list[0]
    for x in df_list[1:]:
        df_final = pd.merge(df_final, x, on=merge_keys, how='left')

    df_finalgk = pd.merge(df_gk, df_gka, on=merge_keys, how='outer')
    df_final   = pd.merge(df_final, df_finalgk, on=merge_keys, how='outer')

    df_final['League'] = league

    if df_final.shape[1] > 59:
        cols = list(range(df_final.shape[1]))
        try:
            cols.pop(59)
            df_final = df_final.iloc[:, cols]
        except Exception:
            pass

    # --- keep Nation as string; fill only numerics ---
    if 'Nation' not in df_final.columns:
        df_final['Nation'] = pd.Series(pd.NA, index=df_final.index, dtype=pd.StringDtype())
    else:
        df_final['Nation'] = df_final['Nation'].astype(pd.StringDtype())

    # GK nation backfill
    df_final = df_final.merge(gk_nat_map, on=['Player','Squad'], how='left', suffixes=('', '_gkNat'))
    df_final['Nation'] = df_final['Nation'].fillna(df_final['Nation_gkNat']).astype(pd.StringDtype())
    if 'Nation_gkNat' in df_final.columns:
        df_final.drop(columns=['Nation_gkNat'], inplace=True)

    num_cols = df_final.select_dtypes(include=[np.number]).columns
    df_final.loc[:, num_cols] = df_final.loc[:, num_cols].fillna(0)

    df_final['Main_Pos'] = df_final['Main_Pos'].fillna('-')
    df_final['Sec_Pos']  = df_final['Sec_Pos'].fillna('-')

    df_final = df_final.infer_objects(copy=False).reset_index(drop=True)

    drop_col = [c for c in df_final.columns if c.endswith('_y')]
    if drop_col:
        df_final = df_final.drop(columns=drop_col)
    df_final = df_final.rename(columns={c: c[:-2] for c in df_final.columns if c.endswith('_x')})
    df_final=df_final.fillna(0.0)
    df_final['Pos'].replace({0.0:'GK'},inplace=True)

    return df_final


# --- canonical ISO3 set (common + Kosovo) ---
ISO3_SET = {
    'AFG','ALB','DZA','AND','AGO','ARG','ARM','AUS','AUT','AZE','BHS','BHR','BGD','BRB','BLR','BEL','BLZ','BEN','BTN',
    'BOL','BIH','BWA','BRA','BRN','BGR','BFA','BDI','KHM','CMR','CAN','CPV','CAF','TCD','CHL','CHN','COL','COM','COG',
    'COD','CRI','CIV','HRV','CUB','CYP','CZE','DNK','DJI','DMA','DOM','ECU','EGY','SLV','GNQ','ERI','EST','SWZ','ETH',
    'FJI','FIN','FRA','GAB','GMB','GEO','DEU','GHA','GRC','GRD','GTM','GIN','GNB','GUY','HTI','HND','HUN','ISL','IND',
    'IDN','IRN','IRQ','IRL','ISR','ITA','JAM','JPN','JOR','KAZ','KEN','KIR','PRK','KOR','KWT','KGZ','LAO','LVA','LBN',
    'LSO','LBR','LBY','LIE','LTU','LUX','MDG','MWI','MYS','MDV','MLI','MLT','MHL','MRT','MUS','MEX','FSM','MDA','MCO',
    'MNG','MNE','MAR','MOZ','MMR','NAM','NRU','NPL','NLD','NZL','NIC','NER','NGA','MKD','NOR','OMN','PAK','PLW','PAN',
    'PNG','PRY','PER','PHL','POL','PRT','QAT','ROU','RUS','RWA','KNA','LCA','VCT','WSM','SMR','STP','SAU','SEN','SRB',
    'SYC','SLE','SGP','SVK','SVN','SLB','SOM','ZAF','SSD','ESP','LKA','SDN','SUR','SWE','CHE','SYR','TJK','TZA','THA',
    'TLS','TGO','TON','TTO','TUN','TUR','TKM','TUV','UGA','UKR','ARE','GBR','USA','URY','UZB','VUT','VEN','VNM','YEM',
    'ZMB','ZWE','XKX'
}

# --- FIFA3 -> ISO3 (only where they differ or are special) ---
FIFA3_TO_ISO3 = {
    'GER':'DEU','NED':'NLD','SUI':'CHE','GRE':'GRC','KSA':'SAU','UAE':'ARE','URU':'URY','PAR':'PRY','CHI':'CHL',
    'ENG':'GBR','SCO':'GBR','WAL':'GBR','NIR':'GBR','ALG':'DZA','RSA':'ZAF','KVX':'XKX',
    # many are identical (ESP->ESP, FRA->FRA, ITA->ITA, etc.), let pass-through handle those.
}

# --- ‘broken’ triples you showed + a few very common OCR-like errors ---
BROKEN_FIX = {
    'ESE':'ESP','NLN':'NED','PTP':'POR','DKD':'DEN','ARA':'ARG','UYU':'URU','PLP':'POL','GHG':'GHA','NGN':'NGA',
    'SKS':'SVK','CMC':'CMR','MEM':'MNE','MAM':'MAR','ISI':'ISL','ITI':'ITA','MLM':'MLI','SES':'SWE','IEI':'IRL',
    'BRB':'BRA','DEG':'GER','BEB':'BEL','XKK':'KVX','FRF':'FRA','TRT':'TUR','SNS':'SEN',
    # junk you listed – best-effort guesses:
    'AUA':'AUT','UZU':'UZB','RSS':'RUS','ATA':'ITA','CVC':'CIV','CZC':'CZE','NON':'NOR','GWG':'GNB','ROR':'ROU',
    'CLC':'CHL','USU':'USA','IRI':'IRN','CHS':'CHE','UAU':'UKR','PEP':'PER','PYP':'PRY','HRC':'HRV','ECE':'ECU',
    'ALA':'ALB','CDC':'COD','MDM':'MDA','GMG':'GMB','TNT':'TTO','CIC':'CIV','COC':'COG','BFB':'BFA','JMJ':'JAM',
    'ZAR':'COD','GAG':'GAB','AOA':'AGO','TDC':'TCD','CRC':'CRI','NZN':'NZL','GPG':'PNG','CWC':'CUW','CAC':'CAN',
    'MXM':'MEX','JPJ':'JPN','VEV':'VEN','RUR':'RUS','HUH':'HUN','BAB':'BIH','IDI':'IND','MKM':'MKD','TGT':'TGO',
    'CFC':'CAF','GEG':'GEO','PHP':'PHL','BDB':'BDI','AMA':'ARM','SRS':'SRB','FIF':'FIN','ZWZ':'ZWE','DOD':'DOM',
    'BGB':'BGR','ILI':'IRL','MZM':'MOZ','KEK':'KEN','EGE':'EGY','KMC':'KHM','BMB':'BMU','GQE':'GNQ','SIS':'SVN',
    'CGC':'COG','EEE':'EST','KNS':'KNA','GDG':'GRD'
}

# optional: names -> FIFA3 first (then we convert to ISO3)
NAME_TO_FIFA3 = {
    'SPAIN':'ESP','ENGLAND':'ENG','ITALY':'ITA','NETHERLANDS':'NED','MONTENEGRO':'MNE','MOROCCO':'MAR',
    'ICELAND':'ISL','PORTUGAL':'POR','DENMARK':'DEN','ARGENTINA':'ARG','URUGUAY':'URU','POLAND':'POL',
    'GHANA':'GHA','NIGERIA':'NGA','SLOVAKIA':'SVK','CAMEROON':'CMR','SCOTLAND':'SCO','WALES':'WAL',
    'NORTHERN IRELAND':'NIR','IRELAND':'IRL','MALI':'MLI','SLOVENIA':'SVN','LITHUANIA':'LTU','LATVIA':'LVA',
    'ESTONIA':'EST','GEORGIA':'GEO','ARMENIA':'ARM','NORTH MACEDONIA':'MKD','UNITED STATES':'USA',
    'UNITED KINGDOM':'GBR'
}

# build a unified lookup space for fuzzy matching
CANON_KEYS = set(ISO3_SET) | set(FIFA3_TO_ISO3.keys()) | set(BROKEN_FIX.keys())

def _to_iso3_from_fifa3(code3: str):
    """Map a 3-letter token that might be FIFA or ISO to ISO3."""
    if code3 in ISO3_SET:
        return code3
    if code3 in FIFA3_TO_ISO3:
        return FIFA3_TO_ISO3[code3]
    # pass-through for identical FIFA/ISO triples (e.g., ESP, FRA, ITA, BRA, POR, GHA…)
    return code3 if code3 in ISO3_SET else None

def to_iso3(x):
    """Normalize anything like 'ENG', 'ESE', 'AUA', 'us', 'United States' → ISO3 (e.g., 'GBR','ESP','AUT','USA')."""
    if pd.isna(x):
        return pd.NA
    s = str(x).strip().upper()
    if s in {"", "0", "0.0", "-", "NA", "N/A", "<NA>"}:
        return pd.NA

    # keep first token if cell had extra text
    s = re.split(r"[\/\s]+", s)[0]

    # 1) direct fixes
    if s in BROKEN_FIX:
        s = BROKEN_FIX[s]

    # 2) names → FIFA3 → ISO3
    if s in NAME_TO_FIFA3:
        s = NAME_TO_FIFA3[s]

    # 3) 2-letter → ISO3 via a minimal map (ISO2→ISO3)
    ISO2_TO_ISO3 = {
        'US':'USA','GB':'GBR','UK':'GBR','EN':'GBR','DE':'DEU','FR':'FRA','ES':'ESP','IT':'ITA','PT':'PRT','NL':'NLD',
        'CH':'CHE','SE':'SWE','NO':'NOR','DK':'DNK','FI':'FIN','IE':'IRL','PL':'POL','CZ':'CZE','AT':'AUT','BE':'BEL',
        'IS':'ISL','HR':'HRV','RS':'SRB','BA':'BIH','AL':'ALB','RO':'ROU','BG':'BGR','GR':'GRC','TR':'TUR','HU':'HUN',
        'UA':'UKR','RU':'RUS','JP':'JPN','KR':'KOR','MX':'MEX','CA':'CAN','CL':'CHL','CO':'COL','EC':'ECU','PE':'PER',
        'BO':'BOL','PY':'PRY','VE':'VEN','TN':'TUN','DZ':'DZA','EG':'EGY','CI':'CIV','SN':'SEN','CD':'COD','ZA':'ZAF',
        'AU':'AUS','NZ':'NZL','SI':'SVN','LT':'LTU','LV':'LVA','EE':'EST','GE':'GEO','AM':'ARM','MK':'MKD','NG':'NGA',
        'GH':'GHA','CM':'CMR','MA':'MAR','SA':'SAU','AE':'ARE','UY':'URY'
    }
    if len(s) == 2 and s in ISO2_TO_ISO3:
        return ISO2_TO_ISO3[s]

    # 4) 3-letter: FIFA or ISO or garbage
    if len(s) == 3:
        # canonical pass
        iso = _to_iso3_from_fifa3(s)
        if iso in ISO3_SET:
            return iso

        # fuzzy attempt among known keys
        # (restrict candidates to those sharing the first letter to reduce mistakes)
        candidates = [k for k in CANON_KEYS if len(k) == 3 and k[0] == s[0]]
        guess = get_close_matches(s, candidates, n=1, cutoff=0.67)
        if guess:
            g = guess[0]
            # apply the same canonicalization path
            if g in BROKEN_FIX:
                g = BROKEN_FIX[g]
            iso = _to_iso3_from_fifa3(g)
            if iso in ISO3_SET:
                return iso

    # if we reach here, we couldn't map confidently
    return pd.NA

iso3_by_player = {
    "Abdoulaye Touré":"GIN",
    "Aboubakar Kamara":"MRT",
    "Adrien Silva":"PRT",
    "Ahmad Benali":"LBY",
    "Ahmed Kutucu":"TUR",
    "Aiham Ousou":"SWE",
    "Alberth Elis":"HND",
    "Amadou Diawara":"GIN",
    "Anastasios Donis":"GRC",
    "Andreas Christensen":"DNK",
    "Andreas Cornelius":"DNK",
    "André Gomes":"PRT",
    "André Silva":"PRT",
    "Aurélio Buta":"AGO",
    "Berkay Özcan":"TUR",
    "Bertuğ Yıldırım":"TUR",
    "Billy Ketkeophomphone":"LAO",
    "Bruma":"PRT",
    "Bruno Jordão":"PRT",
    "Cengiz Ünder":"TUR",
    "Cenk Tosun":"TUR",
    "Cenk Özkacar":"TUR",
    "Charalampos Lykogiannis":"GRC",
    "Christian Eriksen":"DNK",
    "Christian Nørgaard":"DNK",
    "Christophe Hérelle":"FRA",
    "Christos Tzolis":"GRC",
    "Cristiano Ronaldo":"PRT",
    "Cuco Martina":"CUW",
    "Cédric Soares":"PRT",
    "Daniel Carriço":"PRT",
    "Daniel Iversen":"DNK",
    "Daniel Podence":"PRT",
    "Daniel Wass":"DNK",
    "Danilo Pereira":"PRT",
    "Diego Moreira":"PRT",
    "Diogo Jota":"PRT",
    "Domingos Duarte":"PRT",
    "Domingos Quina":"GNB",
    "Doğan Alemdar":"TUR",
    "Emirhan İlkhan":"TUR",
    "Emre Çolak":"TUR",
    "Enes Ünal":"TUR",
    "Filip Jørgensen":"DNK",
    "Francisco Trincão":"PRT",
    "François Kamano":"GIN",
    "Fábio Silva":"PRT",
    "Gedson Fernandes":"PRT",
    "Gelson Martins":"PRT",
    "George Baldock":"GRC",
    "Gonçalo Guedes":"PRT",
    "Gonçalo Paciência":"PRT",
    "Grégoire Defrel":"FRA",
    "Hakan Çalhanoğlu":"TUR",
    "Ilaix Moriba":"GIN",
    "Iuri Medeiros":"PRT",
    "Ivan Cavaleiro":"PRT",
    "Jacob Bruun Larsen":"DNK",
    "Jacob Rasmussen":"DNK",
    "Jannik Vestergaard":"DNK",
    "Jens Odgaard":"DNK",
    "Jens Stryger Larsen":"DNK",
    "Jesper Lindstrøm":"DNK",
    "Joachim Andersen":"DNK",
    "Joakim Mæhle":"DNK",
    "Jonas Lössl":"DNK",
    "Joshua Brenet":"NLD",
    "José Fonte":"PRT",
    "Jota":"PRT",
    "João Cancelo":"PRT",
    "João Félix":"PRT",
    "João Moutinho":"PRT",
    "João Mário":"PRT",
    "João Palhinha":"PRT",
    "Jules Keita":"GIN",
    "Jürgen Locadia":"NLD",
    "Kaan Ayhan":"TUR",
    "Kasper Dolberg":"DNK",
    "Kasper Schmeichel":"DNK",
    "Kenan Karaman":"TUR",
    "Kenneth Zohore":"DNK",
    "Konstantinos Mavropanos":"GRC",
    "Kévin Rodrigues":"CPV",
    "Luisinho":"PRT",
    "Lukas Lerager":"DNK",
    "Luís Maximiano":"PRT",
    "Mads Bech Sørensen":"DNK",
    "Mads Roerslev":"DNK",
    "Marcus Ingvartsen":"DNK",
    "Martin Braithwaite":"DNK",
    "Mathias Jensen":"DNK",
    "Mathias Pereira Lage":"PRT",
    "Merih Demiral":"TUR",
    "Mikkel Damsgaard":"DNK",
    "Mikkel Desler":"DNK",
    "Mikkel Kaufmann":"DNK",
    "Morgan Guilavogui":"GIN",
    "Morten Hjulmand":"DNK",
    "Mouctar Diakhaby":"GIN",
    "Naby Keïta":"GIN",
    "Nahki Wells":"BMU",
    "Nani":"PRT",
    "Nuno Mendes":"PRT",
    "Nuno Tavares":"PRT",
    "Nuri Şahin":"TUR",
    "Nélson Semedo":"PRT",
    "Odysseas Vlachodimos":"GRC",
    "Okay Yokuşlu":"TUR",
    "Oliver Abildgaard":"DNK",
    "Oliver Christensen":"DNK",
    "Orestis Karnezis":"GRC",
    "Ozan Kabak":"TUR",
    "Ozan Tufan":"TUR",
    "Panagiotis Retsos":"GRC",
    "Pantelis Hatzidiakos":"GRC",
    "Papa Ndiaga Yade":"SEN",
    "Paulo Oliveira":"PRT",
    "Pedro Neto":"PRT",
    "Pedro Pereira":"PRT",
    "Pedro Rebocho":"PRT",
    "Philip Billing":"DNK",
    "Pione Sisto":"DNK",
    "Rafael Leão":"PRT",
    "Raphaël Guerreiro":"PRT",
    "Rasmus Højlund":"DNK",
    "Rasmus Kristensen":"DNK",
    "Renato Sanches":"PRT",
    "Renato Veiga":"PRT",
    "Riza Durmisi":"DNK",
    "Rony Lopes":"PRT",
    "Rui Fonte":"PRT",
    "Rui Patrício":"PRT",
    "Rui Silva":"PRT",
    "Rúben Neves":"PRT",
    "Rúben Semedo":"PRT",
    "Rúben Vezo":"PRT",
    "Rúben Vinagre":"PRT",
    "Salih Özcan":"TUR",
    "Serhou Guirassy":"GIN",
    "Simon Falette":"GIN",
    "Sinan Gümüş":"TUR",
    "Sokratis Papastathopoulos":"GRC",
    "Sory Kaba":"GIN",
    "Steve Mounié":"BEN",
    "Sérgio Oliveira":"PRT",
    "Thomas Delaney":"DNK",
    "Tiago Djaló":"PRT",
    "Tiago Tomás":"PRT",
    "Umut Bozok":"TUR",
    "Vitinha":"PRT",
    "Wahid Faghir":"DNK",
    "William Osula":"DNK",
    "Yunus Mallı":"TUR",
    "Yusuf Yazıcı":"TUR",
    "Zeki Çelik":"TUR",
    "Çağlar Söyüncü":"TUR",
    "Ömer Toprak":"TUR",
}

def _norm_name(s: str) -> str:
    # lower, strip accents, collapse spaces
    return re.sub(r"\s+", " ", unidecode(str(s)).strip().lower())

# build a normalized-key map so "Joao Felix" and "João Félix" both work
_iso3_map_norm = {_norm_name(k): v for k, v in iso3_by_player.items()}

def nation_from_name(name):
    key = _norm_name(name)
    return _iso3_map_norm.get(key)

#-------------------------------------------------------------------------------------------------------------------------
# checking if the data is overlapping or not, based on that PCA, UMAP or t-SNE will be used
#-------------------------------------------------------------------------------------------------------------------------
def check_pca_separation(X_pca, n_clusters=2):
    """Use Davies-Bouldin Index to check if PCA keeps separable structure."""
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=101, n_init=10)
    cluster_labels = kmeans.fit_predict(X_pca)

    dbi = davies_bouldin_score(X_pca, cluster_labels)
    print(f"Davies-Bouldin Index: {dbi:.4f}")

    if dbi < 0.6:
        print("PCA structure is good (Use PCA).")
    else:
        print("Data is overlapping (Consider UMAP/t-SNE).")


#-------------------------------------------------------------------------------------------------------------------------
# Feature Selector based on Importance for classification models
#-------------------------------------------------------------------------------------------------------------------------
class Feature_Selector_Classification(BaseEstimator,TransformerMixin):
    def __init__(self,threshold,random_state=101):
        self.threshold=threshold
        self.random_state=random_state
        self.selector_=None
        self.selected_features_=None
        self.feature_importance_=None
        
    def fit(self,X,y):
        self.model_=RandomForestClassifier(random_state=self.random_state,n_jobs=-1,class_weight='balanced')
        self.model_.fit(X,y)
        
        self.selector_=SelectFromModel(self.model_,threshold=self.threshold,prefit=True)
        selected_mask=self.selector_.get_support()
        selected_features=X.columns[selected_mask].tolist()
        
        self.selected_features_=list(set(selected_features))
        self.feature_importance_=pd.Series(self.model_.feature_importances_,index=X.columns)
        
        return self
    
    def transform(self, X):
        return X[self.selected_features_]
        
    def get_selected_features(self):
        return self.selected_features_
        
    def get_selected_importances(self):
        return self.feature_importance_.loc[self.selected_features_]
    

#-------------------------------------------------------------------------------------------------------------------------
# def function to check if file is existing or not; if not existing create new one; if existing add new input
#-------------------------------------------------------------------------------------------------------------------------
def save_model_performance(performance_df,file_path):
    if os.path.exists(file_path):
        df_path = pd.read_csv(file_path)
        new_df = pd.concat([df_path, performance_df], axis=0)
        new_df.to_csv(file_path, index=True)
    else:
        performance_df.to_csv(file_path, index=True)


#-------------------------------------------------------------------------------------------------------------------------
# class for creating club weights
#-------------------------------------------------------------------------------------------------------------------------
class ClubWeightadder(BaseEstimator,TransformerMixin):
    def __init__(self, columns, top_club_weights, non_top_weight=0.5):
        self.columns = columns  # List of columns
        self.top_club_weights = top_club_weights
        self.non_top_weight = non_top_weight

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            #if the input is a list or tuple and has a length of 2
            X[f'{col}_Weight']=X[col].apply(lambda x: self.top_club_weights.get(x, self.non_top_weight))
        return X
    

#-------------------------------------------------------------------------------------------------------------------------
# class transforming individual club names into top & non-top clubs
#-------------------------------------------------------------------------------------------------------------------------
class TopClubTransformer(BaseEstimator, TransformerMixin):
    #list can contain mutliple tuples
    def __init__(self, columns=[('Club','League')]):
        self.columns = columns
        self.top_clubs = [
            'FC Bayern München', 'Borussia Dortmund', 'RasenBallsport Leipzig', 'Bayer 04 Leverkusen',
            'Inter Mailand', 'AC Mailand', 'Juventus Turin', 'SSC Neapel', 'FC Liverpool', 'Manchester City',
            'Tottenham Hotspur', 'Newcastle United', 'FC Arsenal', 'Manchester United', 'FC Chelsea',
            'FC Paris Saint-Germain', 'AS Monaco', 'Olympique Marseille', 'FC Barcelona', 'Real Madrid',
            'Atlético Madrid', 'FC Porto', 'Benfica Lissabon', 'Sporting Lissabon', 'Galatasaray',
            'Fenerbahce', 'Ajax Amsterdam'
        ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        #here we loop through the different entered column names
        #we do not need to add the col name before X, because we use the different col names on a row level
        for col,col2 in self.columns:
            X[col] = X.apply(lambda x: f'Top Club - {x[col2]}' if x[col] in self.top_clubs else f'Non-Top Club -  {x[col2]}',axis=1)
        return X
    

#-------------------------------------------------------------------------------------------------------------------------
# class creating string values for leagues, which allow us group values; less granular values
#-------------------------------------------------------------------------------------------------------------------------
class TopLeagueTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns=['League']):
        self.columns = columns
        self.mapping = {
            'Serie A': 'Top League', 'LaLiga': 'Top League', 'Bundesliga': 'Top League',
            'Premier League': 'Top League', 'Ligue 1': 'Top League',
            'LALIGA HYPERMOTI': 'Other Small League', 'LaLiga 1|2|3': 'Other Small League',
            'LaLiga SmartBank': 'Other Small League', 'Primera Federación - Gr. I': 'Other Small League',
            'Primera Federación - Gr. II': 'Other Small League', 'Segunda Federación - Gr. III': 'Other Small League',
            '2ª B - Grupo I': 'Other Small League', '2ª B - Grupo III': 'Other Small League',
            'Championship': 'Other Small League', 'Premier League 2': 'Other Small League',
            'League One': 'Other Small League', 'Serie B': 'Other Small League', 'Serie C-A': 'Other Small League',
            'Serie C-B': 'Other Small League', 'Serie D - I': 'Other Small League',
            'Serie A Segunda Etapa': 'Other Small League', 'Serie C-C': 'Other Small League',
            'Primavera 1': 'Other Small League',
            '2. Bundesliga': 'Other Small League', '3. Liga': 'Other Small League',
            'Ligue 2': 'Other Small League', 'Championnat National': 'Other Small League',
            'National 2 - Grp. C': 'Other Small League', 'National 2 - Grp. D': 'Other Small League',
            'N3 - Paris IdF': 'Other Small League',
            'Série B': 'Other Small League',
            'Liga Sabseg': 'Other Small League', 'LigaPro': 'Other Small League',
            'Keuken Kampioen Divisie': 'Other Small League',
            'Proximus League': 'Other Small League',
            'Ascenso MX Cl.': 'Other Small League', 'Copa de la Liga': 'Other Small League',
            'Liga DIMAYOR II': 'Other Small League'
        }

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        #here we loop through the different entered column names
        for col in self.columns:
            X[col] = X[col].apply(lambda x: self.mapping.get(x, 'Non-Top League'))
        return X
    

class LeagueWeightEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns, top_league_weights, non_top_weight=0.5):
        self.columns = columns  # List of columns
        self.top_league_weights = top_league_weights
        self.non_top_weight = non_top_weight

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            X[f'{col}_Weight']=X[col].apply(lambda x: self.top_league_weights.get(x,self.non_top_weight))
        return X


#-------------------------------------------------------------------------------------------------------------------------
# creating new features
#-------------------------------------------------------------------------------------------------------------------------
class NewFeatureCreate(BaseEstimator, TransformerMixin):
    def __init__(self,origin_drop=True):
        self.origin_drop=origin_drop
        pass

    def fit(self,X,y=None):
        return self

    def transform(self,X):
        df=X.copy()
        #Goal Keeper
        def save_goal(row):
            if row['90s']<4:
                return (row['Save%']-row['PSxG/SoT'])/(row['90s']+4)
            else:
                return (row['Save%']-row['PSxG/SoT'])/(row['90s'])
        df['Save%_vs_xG_90s']=df.apply(save_goal,axis=1)

        def sse(row):
            psxg=row['PSxG']
            ga=row['GA']
            time=row['90s']

            if (psxg+ga) == 0 or time == 0 or ga == 0:
                return 0
            
            rate=((psxg+ga)/ga)

            if time <4:
                rate/=time+4
            else:
                rate/=time

            return rate
        df['Shoot_Stopping_Eff_90s']=df.apply(sse,axis=1)

        z_cmp=zscore(df['Cmp%'])
        z_avg_len_pass=zscore(df['AvgLen_Passes'])
        z_launch=zscore(df['Launch%_Passes'])
        df['Composite']=(z_cmp+z_avg_len_pass+z_launch)/3

        #Defender
        def adjusted_tackle_rate(row):
            tackles = row['Total Tackels']
            #minutes = row['90s']
            won = row['TklW']
            
            if tackles == 0: #or minutes == 0:
                return 0

            rate = won / tackles

            if tackles < 10: #and minutes < 4:
                rate *= (tackles / 10)
                #rate /= (minutes + 4)  # slight penalization for low minutes
            else:
                rate #  # normalize per 90s

            return rate
        df['Tkl_Success_Rate'] = df.apply(adjusted_tackle_rate, axis=1)

        def adjusted_error_rate(row):
            if row['Clr'] < 10:
                return (row['Err']/row['Clr']) * (row['Clr'] / 10) if row['Clr'] != 0 else 0
            else:
                return row['Err']/row['Clr']
        df['Err_Rate']=df.apply(adjusted_error_rate,axis=1)

        df['Clear_Rate_90']=df['Clr']/df['90s'].replace(0, np.nan)

        def pro_inv(row):
            if row['90s'] < 4:
                return (row['PrgP']+row['PrgC']+row['PrgR'])/(row['90s']+4)
            else:
                return (row['PrgP']+row['PrgC']+row['PrgR'])/row['90s']
        df['Progressive_Involv_90s']=df.apply(pro_inv,axis=1)

        def ts(row):
            tt=row['Total Tackels']
            b=row['Blocks']
            int=row['Int']
            time=row['90s']

            if tt==0 or b==0 or int==0:
                return 0
            
            rate=(row['Touches_Def_Pen_Area']+row['Touches_Def_3rd_Area'])/(tt+b+int)

            if time < 4:
                rate/=(time+4)
            else:
                rate/=time
            return rate
        df['Touch_Stability_90s']=df.apply(ts,axis=1)

        #Midfielder
        df['Prog_Passing_Eff']=df['PrgDist']/df['Total_Att'].replace(0, np.nan)

        def adjusted_creative_rate(row):
            if row['Pass'] < 100:
                return (row['Shot_Creating_Action']/row['Pass']) * (row['Pass']/100) if row['Pass'] != 0 else 0
            else:
                return row['Shot_Creating_Action']/row['Pass']
        df['Creative_Rate']=df.apply(adjusted_creative_rate,axis=1)

        def adjusted_dual_threat(row):
            if row['xAG'] == 0:
                return row['npxG']/(row['xAG']+0.01) if row['xAG'] != 0 else 0
            else:
                return row['npxG']/row['xAG']
        df['Dual_Threat']=df.apply(adjusted_dual_threat,axis=1)

        def adjusted_ball_retention_index(row):
            touches = row['Touches']
            minutes = row['90s']
            
            if touches == 0 or minutes == 0:
                return 0
            
            raw_loss_rate = (row['Miscontrols_Carries'] + row['Dispossed_Carries']) / touches

            # Sample size correction for low touches
            if touches < 100:
                raw_loss_rate *= (touches / 100)

            # Normalize per 90 minutes
            if minutes < 4:
                return raw_loss_rate / (minutes + 4)
            else:
                return raw_loss_rate / minutes
        df['Ball_Retention_Index_90s'] = df.apply(adjusted_ball_retention_index, axis=1)

        #Forward
        def adjusted_fin_eff_xG(row):
            if row['xG'] == 0:
                return row['Gls']/(row['xG'] + 0.1)
            elif row['Gls'] == 0:
                return (row['Gls'] + 0.1)/row['xG']
            else:
                return row['Gls'] / row['xG']
        df['Finishing_Eff_xG'] = df.apply(adjusted_fin_eff_xG,axis=1)

        def adjusted_fin_eff_npxG(row):
            if row['npxG'] == 0:
                return row['Gls']/(row['npxG'] + 0.5)
            elif row['Gls'] == 0:
                return (row['Gls'] + 0.5)/row['npxG']
            else:
                return row['Gls'] / row['npxG']
        df['Finishing_Eff_npxG'] = df.apply(adjusted_fin_eff_npxG,axis=1)

        def exp_con(row):
            if row['90s']<4:
                return (row['xG'] + row['xAG'])/(row['90s']+4)
            else:
                return (row['xG'] + row['xAG'])/row['90s']
        df['Expected_Contribution_90s'] = df.apply(exp_con,axis=1)

        def adjusted_xCon_per_Touch(row):
            touches = row['Touches']
            minutes = row['90s']
            xcon = row['xG'] + row['xAG']  # Define Expected_Contribution

            if touches == 0 or minutes == 0:
                return 0

            # Base contribution per touch
            rate = xcon / touches

            # Downweight low-touch samples
            if touches < 30:
                rate *= (touches / 30)

            # Normalize per 90 mins, smoothing for small sample sizes
            adjusted_minutes = minutes if minutes >= 4 else (minutes + 4)
            return rate / adjusted_minutes
        df['xContrib_per_Touch_90s'] = df.apply(adjusted_xCon_per_Touch,axis=1)

        df['Goal_Involvement_per90'] = (df['Gls'] + df['Ast']) / df['90s'].replace(0, np.nan)

        def adjusted_dribbling_success_rate(row):
            shots = row['Sh']
            dribble_shots = row['Shot_Att_after_Dribbling']
            minutes = row['90s']

            if minutes == 0:
                return 0

            # Shot sample correction
            if shots < 5:
                rate = dribble_shots / (shots + 5)
            else:
                rate = dribble_shots / shots

            # Normalize per 90 minutes, with smoothing for low minutes
            if minutes < 4:
                return rate / (minutes + 4)
            else:
                return rate / minutes
        df['dribbles_completed_per90'] = df.apply(adjusted_dribbling_success_rate,axis=1)

        df['Shots_On_Target_per90']=(df['Sh'] + df['Gls'])/df['90s'].replace(0, np.nan)

        def adjusted_self_created_chance_rate(row):
            shots = row['Sh']
            dribble_shots = row['Shot_Att_after_Dribbling']
            minutes = row['90s']

            if minutes == 0:
                return 0

            # Smooth for low shot volume
            if shots < 5:
                rate = dribble_shots / (shots + 5)
            else:
                rate = dribble_shots / shots

            # Normalize per 90 minutes with smoothing for low minutes
            if minutes < 4:
                return rate / (minutes + 4)
            else:
                return rate / minutes
        df['Self_Created_Chance_Rate_90s'] = df.apply(adjusted_self_created_chance_rate,axis=1)

        def touch_mid_3rd_90s(row):
            if row['90s'] < 4:
                return row['Touches_Mid_3rd_Area']/(row['90s']+4)
            else:
                return row['Touches_Mid_3rd_Area']/row['90s']
        df['Touches_Mid_3rd_Area_90s']=df.apply(touch_mid_3rd_90s,axis=1)

        def touch_attack_3rd_90s(row):
            if row['90s'] < 4:
                return row['Touches_Att_3rd_Area']/(row['90s']+4)
            else:
                return row['Touches_Att_3rd_Area']/row['90s']
        df['Touches_Att_3rd_Area_90s']=df.apply(touch_attack_3rd_90s,axis=1)

        def touch_attack_pen_90s(row):
            if row['90s'] < 4:
                return row['Touches_Att_Pen_Area']/(row['90s']+4)
            else:
                return row['Touches_Att_Pen_Area']/row['90s']
        df['Touches_Att_Pen_Area_90s']=df.apply(touch_attack_pen_90s,axis=1)

        def dribb_succ_90s(row):
            if row['90s'] < 4:
                return row['Dribbling_Succ']/(row['90s']+4)
            else:
                return row['Dribbling_Succ']/row['90s']
        df['Dribbling_Succ_90s']=df.apply(dribb_succ_90s,axis=1)

        def prog_pass_rec_90s(row):
            if row['90s'] < 4:
                return row['Progressive_Passes_Received']/(row['90s']+4)
            else:
                return row['Progressive_Passes_Received']/row['90s']
        df['Progressive_Passes_Received_90s']=df.apply(prog_pass_rec_90s,axis=1)

        def kp_90s(row):
            if row['90s'] < 4:
                return row['KP']/(row['90s']+4)
            else:
                return row['KP']/row['90s']
        df['KP_90s']=df.apply(kp_90s,axis=1)

        def one_third_90s(row):
            if row['90s'] < 4:
                return row['1/3']/(row['90s']+4)
            else:
                return row['1/3']/row['90s']
        df['1/3_90s']=df.apply(one_third_90s,axis=1)

        def prgp_90s(row):
            if row['90s'] < 4:
                return row['PrgP']/(row['90s']+4)
            else:
                return row['PrgP']/row['90s']
        df['PrgP_90s']=df.apply(prgp_90s,axis=1)

        def xa_90s(row):
            if row['90s'] < 4:
                return row['xA']/(row['90s']+4)
            else:
                return row['xA']/row['90s']
        df['xA_90s']=df.apply(xa_90s,axis=1)

        def a_xag_90s(row):
            if row['90s'] < 4:
                return row['A-xAG']/(row['90s']+4)
            else:
                return row['A-xAG']/row['90s']
        df['A-xAG_90s']=df.apply(a_xag_90s,axis=1)

        def total_carries_13_90s(row):
            if row['90s'] < 4:
                return row['Total_Carries_in_1/3']/(row['90s']+4)
            else:
                return row['Total_Carries_in_1/3']/row['90s']
        df['Total_Carries_in_1/3_90s']=df.apply(total_carries_13_90s,axis=1)

        def tklw_90s(row):
            if row['90s'] < 4:
                return row['TklW']/(row['90s']+4)
            else:
                return row['TklW']/row['90s']
        df['TklW_90s']=df.apply(tklw_90s,axis=1)

        def int_90s(row):
            if row['90s'] < 4:
                return row['Int']/(row['90s']+4)
            else:
                return row['Int']/row['90s']
        df['Int_90s']=df.apply(int_90s,axis=1)

        def clr_90s(row):
            if row['90s'] < 4:
                return row['Clr']/(row['90s']+4)
            else:
                return row['Clr']/row['90s']
        df['Clr_90s']=df.apply(clr_90s,axis=1)

        def tt_90s(row):
            if row['90s'] < 4:
                return row['Total Tackels']/(row['90s']+4)
            else:
                return row['Total Tackels']/row['90s']
        df['Total Tackels_90s']=df.apply(tt_90s,axis=1)

        def blocks_90s(row):
            if row['90s'] < 4:
                return row['Blocks']/(row['90s']+4)
            else:
                return row['Blocks']/row['90s']
        df['Blocks_90s']=df.apply(blocks_90s,axis=1)

        def tpcd_90s(row):
            if row['90s'] < 4:
                return row['Total_Progressive_Carry_Distance']/(row['90s']+4)
            else:
                return row['Total_Progressive_Carry_Distance']/row['90s']
        df['Total_Progressive_Carry_Distance_90s']=df.apply(tpcd_90s,axis=1)

        def touch_def_3rd_90s(row):
            if row['90s'] < 4:
                return row['Touches_Def_3rd_Area']/(row['90s']+4)
            else:
                return row['Touches_Def_3rd_Area']/row['90s']
        df['Touches_Def_3rd_Area_90s']=df.apply(touch_def_3rd_90s,axis=1)

        def pks_90s(row):
            if row['90s'] < 4:
                return row['Penalty_Kick_Saves']/(row['90s']+4)
            else:
                return row['Penalty_Kick_Saves']/row['90s']
        df['Penalty_Kick_Saves_90s']=df.apply(pks_90s,axis=1)

        def psxg_90s(row):
            if row['90s'] < 4:
                return row['PSxG']/(row['90s']+4)
            else:
                return row['PSxG']/row['90s']
        df['PSxG_90s']=df.apply(psxg_90s,axis=1)

        def alp_90s(row):
            if row['90s'] < 4:
                return row['AvgLen_Passes']/(row['90s']+4)
            else:
                return row['AvgLen_Passes']/row['90s']
        df['AvgLen_Passes_90s']=df.apply(alp_90s,axis=1)

        #addressing division through zero
        ##replacing the division through zero (np.inf and -np.inf gets replaced with NaN) with NaN values
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        ##replacing the NaN values with 0.0
        df.fillna(0,inplace=True)

        if self.origin_drop:
            df=df.drop(['PSxG/SoT','PSxG','GA','TklW','Total Tackels','Err','Clr','90s','PrgC','PrgR','Touches_Def_Pen_Area','Touches_Def_3rd_Area',
                        'Blocks','Int','PrgDist','Total_Att','Shot_Creating_Action','xAG','Miscontrols_Carries','Dispossed_Carries','Touches','xG','Gls','npxG',
                        'xAG','Ast','Shot_Att_after_Dribbling','Sh','G+A','GCA','G+A-PK','#OPA','Short_Cmp','Short_Att','Medium_Cmp','Medium_Att',
                        'Long_Cmp','Long_Att','Dribbling_Att','Dribbling_Succ','Tackeld_Dribbling','G-PK','AvgLen_Passes','Penalty_Kick_Saves','CS',
                        'Total_Progressive_Carry_Distance','Blocks','Total_Carries_in_1/3','A-xAG','xA','PrgP','KP','1/3','Progressive_Passes_Received','Touches_Att_Pen_Area',
                        'Touches_Att_3rd_Area','Touches_Mid_3rd_Area'],axis=1)
        return df
    
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
    

#-------------------------------------------------------------------------------------------------------------------------
# allocating weight position specific and position specific features
#-------------------------------------------------------------------------------------------------------------------------
class PositionScore(BaseEstimator, TransformerMixin):
    def __init__(self, pos,balanced_classes=True,model_params=None,random_state=101,shap_sample=2000):
        #setting up all parameters for the class
        self.pos = pos
        self.balanced_classes = balanced_classes
        self.model_params = model_params #or {}
        self.random_state = random_state
        self.shap_sample = shap_sample  # subsample rows for SHAP for speed
        self.weights = None
        self.scaler_ = None
        self.used_features_ = None
        self.score_col_ = None

    def fit(self, X, y=None):
        df = X.copy()
        #the classes of the y train set, we need to encode them
        le = LabelEncoder()
        y_enc = le.fit_transform(pd.Series(y))
        self.classes_ = le.classes_  # optional: store mapping if you ever need it

        feature_map = {
                'FW': ['Gls_per_90','Ast_per_90','xG_per_90','xAG_per_90','Dribbling_Succ_90s','Progressive_Passes_Received_90s','Touches_Att_Pen_Area_90s','Shot_Creating_Action_90','Finishing_Eff_xG','Finishing_Eff_npxG','Expected_Contribution_90s','xContrib_per_Touch_90s','Goal_Involvement_per90','dribbles_completed_per90','Shots_On_Target_per90','Self_Created_Chance_Rate_90s','GCA90','Club_Left_Weight','League_Left_Weight'],
                'MF': ['Ast_per_90','xAG_per_90','KP_90s','1/3_90s','PrgP_90s','dribbles_completed_per90','Total_Cmp%','xA_90s','A-xAG_90s','Touches_Mid_3rd_Area_90s','Total_Carries_in_1/3_90s','Shot_Creating_Action_90','GCA90','Prog_Passing_Eff','Creative_Rate','Dual_Threat','Ball_Retention_Index_90s','Club_Left_Weight','League_Left_Weight'],
                'DF': ['TklW_90s','Int_90s','Clr_90s','Total Tackels_90s','Blocks_90s','Touches_Def_3rd_Area_90s','Tkl_Success_Rate','Err_Rate','Clear_Rate_90','Progressive_Involv_90s','Touch_Stability_90s','Long_Cmp%','Total_Progressive_Carry_Distance_90s','Club_Left_Weight','League_Left_Weight'],
                'GK': ['GA90','Save%','CS%','Penalty_Kick_Saves_90s','PSxG_90s','Launch%_Passes','#OPA/90','Save%_vs_xG_90s','Shoot_Stopping_Eff_90s','Composite','Club_Left_Weight','League_Left_Weight']
            }

        #raising an error when the position input is not in the feature_map dict is given
        if self.pos not in feature_map:
            raise ValueError("Position must be one of: 'FW', 'MF', 'DF', 'GK'.")

        #filtering the dict based on the position
        required_features = feature_map[self.pos]
        #we add the position values to a list, based on, if they exists
        available_features = [f for f in required_features if f in df.columns]
        #missing will contain feature names that are not contained in both sets
        missing = set(required_features) - set(available_features)
        #showing position(s) and position
        if missing:
            warnings.warn(f"Missing features for position {self.pos}: {missing}")

        #making sure that input_df is only containing numerical features
        input_df = df[available_features].select_dtypes(include=[np.number]).copy()
        if input_df.empty:
            raise ValueError("No numeric features available for the model after filtering.")
        #self.used_features_ will contain all available features
        self.used_features_ = list(input_df.columns)

        #getting all avaiable classes as a count
        n_classes=pd.Series(y).nunique()

        base_params = dict(
            objective='multi:softprob',      # good for multiclass + SHAP
            num_class=n_classes,
            eval_metric='mlogloss',
            n_estimators=500,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            tree_method='hist',
            n_jobs=-1,                       # use 'nthread' if your xgboost is old
        )

        # Safe merge: if self.model_params is None, this just does nothing
        base_params.update(self.model_params or {})

        model = XGBClassifier(**base_params)
        #computing class weights to address class imbalance
        sample_weight = None
        if self.balanced_classes:
            sample_weight = compute_sample_weight(class_weight='balanced', y=y)
        #with the class weights we can fit 
        model.fit(input_df, y_enc, sample_weight=sample_weight)

        # SHAP (subsample rows for speed if large)
        X_shap = input_df
        #when the self.shap_sample exists and the row number of the dataframe is bigger as self.shap_sample
        if self.shap_sample and len(input_df) > self.shap_sample:
            X_shap = input_df.sample(self.shap_sample, random_state=self.random_state)

        explainer = shap.TreeExplainer(model)
        sv = explainer(X_shap)

        # Handle both new and old SHAP formats
        values = sv.values
        if isinstance(values, list):  # old: list of arrays per class
            values = np.stack(values, axis=-1)  # (n_samples, n_features, n_classes)

        # Importance = mean(|SHAP|) over samples and classes
        #axis=0 → samples; axis=1 → classes; axis=2 → features
        #getting the average between samples and features
        importances = np.mean(np.abs(values), axis=(0, 2))  # -> (n_features,)

        # Normalize to weights
        eps = 1e-12
        weights = importances / (importances.sum() + eps)
        #with zip() we can turn two values to more dict friendly input, key:values
        self.weights_ = dict(zip(self.used_features_, weights))

        # Fit scaler on the raw score (training data)
        #based on self.used_features_ we get the weights in teh same order of the features
        ##multipling based on the weights with the corosponding features with all its values in teh column
        raw_score = X[self.used_features_].mul([self.weights_[f] for f in self.used_features_], axis=1).sum(axis=1).values.reshape(-1, 1)
        #scale the results but inside the fitting, otherwise would cause leakage, we need to perform it within fit; axis=1 within mul() ->
        self.scaler_ = RobustScaler().fit(raw_score)

        # Persist a nice column name
        position_name_map = {'FW': 'Offense','MF': 'Creativity','DF': 'Defending','GK': 'Goalkeeping'}
        self.score_col_ = f"Performance_Score_{position_name_map.get(self.pos, self.pos)}"

        return self

    def transform(self, X):
        if self.weights_ is None or self.scaler_ is None:
            raise RuntimeError("Fit must be called before transform.")

        df = X.copy()
        valid_feats = [f for f in self.used_features_ if f in df.columns]
        if len(valid_feats) < len(self.used_features_):
            missing = sorted(set(self.used_features_) - set(valid_feats))
            warnings.warn(f"Missing features during transform: {missing}")

        #multipling weigthes with feature column
        raw = df[valid_feats].mul([self.weights_[f] for f in valid_feats], axis=1).sum(axis=1).values.reshape(-1, 1)
        df[self.score_col_] = self.scaler_.transform(raw)

        return df
    
#-------------------------------------------------------------------------------------------------------------------------
# creating a log loss, which makes sure that all classes are addressed
#-------------------------------------------------------------------------------------------------------------------------
def logloss_addressing_all_classes(y_true, y_proba,class_sort):
    return log_loss(y_true, y_proba, labels=class_sort)


#-------------------------------------------------------------------------------------------------------------------------
# creating a log loss class, which makes sure that all classes are addressed
#-------------------------------------------------------------------------------------------------------------------------
class FixedLabelLogLoss:
    def __init__(self, labels, eps=1e-15):
        self.labels = np.array(labels)
        self.eps = eps
        # map label -> index in global order
        self._global_idx = {c: i for i, c in enumerate(self.labels)}

    #using __call__, when you want to call afunction
    def __call__(self, estimator, X, y):
        proba = estimator.predict_proba(X)              # (n, k_fold)
        est_labels = estimator.classes_                 # length k_fold

        # Build full (n, K) matrix in global label order
        ##create an empty matrix filled with 0s, with as many rows as samples and as many columns as global classes
        proba_full = np.zeros((proba.shape[0], len(self.labels)), dtype=float)
        #build a dictionary mapping each class label that appeared in this fold’s training set
        est_idx = {c: i for i, c in enumerate(est_labels)}
        #c -> each class
        #j_fold -> find its column index in the fold
        #j_global -> find its global index in the full label list
        for c, j_fold in est_idx.items():
            j_global = self._global_idx[c]
            proba_full[:, j_global] = proba[:, j_fold]

        # (optional) clip tiny values for numerical stability
        #np.clip(arr, low, high) forces every element in arr to lie between low and high
        ##values smaller than low and bigger than high, will be clipped
        proba_full = proba_full + self.eps
        # Rows should already sum to ~1, but normalize defensively:
        proba_full /= proba_full.sum(axis=1, keepdims=True)

        return -log_loss(y, proba_full, labels=self.labels)
    

#-------------------------------------------------------------------------------------------------------------------------
# creating a brier score, which can be passed in tree based models for early stopping
#-------------------------------------------------------------------------------------------------------------------------
def brier_eval_binary(y_true, y_pred):#y_pred, dataset):
    #y_true = dataset.get_label()
    y_prob = 1.0 / (1.0 + np.exp(-y_pred))  # LightGBM gives raw scores
    score = brier_score_loss(y_true, y_prob)#, pos_label=1)
    return 'brier', score, False  # lower is better
    
#-------------------------------------------------------------------------------------------------------------------------
# creating a brier score, which is for XGBoost
#-------------------------------------------------------------------------------------------------------------------------
def brier_eval_xgb(preds, dtrain):
    y_true = dtrain.get_label()        # preds are probabilities for class 1 with binary:logistic
    score = brier_score_loss(y_true, preds, pos_label=1)
    return 'brier', score

#-------------------------------------------------------------------------------------------------------------------------
# creating the input data, for predictions
#-------------------------------------------------------------------------------------------------------------------------
#who_is_that_player(p,cj,lj) -> slightly modified for classification, we just use player, league_joined (will be predicted) and club_joined (not relevant)
def who_is_that_player(p):
    df_mean_stats=pd.read_csv(r"..\..\DataSources\Processed\Player_Stats_Mean_2017_2025_test.csv").iloc[:,1:]
    player_pred=pd.read_csv(r"..\..\DataSources\Processed\Player_Stats_Mean_2017_2025_test.csv").iloc[:,1:]
    df_trans=pd.read_csv(r"..\..\DataSources\Processed\All_Trans_2000_2025_test.csv").iloc[:,1:]
    df_trans.drop(['Age'],axis=1,inplace=True)
    #df_trans['Transfer_Fee']=df_trans['Transfer_Fee'].apply(trans_value)

    df_stats=pd.merge(df_mean_stats,df_trans,on=['Player','Transfer_Window'],how='left').dropna()
    #just to get actaull transfer fees
    #df_stats=df_stats[df_stats['Transfer_Fee']!=0]
    df_stats=df_stats.drop(['Player','Squad','League','Transfer_Fee','Born','Club_Joined','League_Joined'],axis=1)
    player=player_pred[(player_pred['Player']==p) & (player_pred['Transfer_Window']==player_pred['Transfer_Window'].max())].copy()
    player.rename(columns={'Squad':'Club_Left','League':'League_Left'},inplace=True)
    #player['Club_Joined']=cj
    #player['League_Joined']=lj
    player['Performance_Year']=player['Transfer_Window']-1
    col_names=list(df_stats.columns)
    player=player[col_names]
    return player


#-------------------------------------------------------------------------------------------------------------------------
# building a session for the webscrapper
#-------------------------------------------------------------------------------------------------------------------------
# 1) Build a robust session (reuse yours or replace it)
def build_session() -> requests.Session:
    s = requests.Session()
    # Retries on 429/5xx and some transient network errors
    retry = Retry(
        total=6,
        connect=3,
        read=3,
        status=6,
        backoff_factor=0.6,                 # 0.6, 1.2, 2.4, ...
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]), # only GET in our scraper
        raise_on_status=False,
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=50, pool_maxsize=50)
    s.mount("https://", adapter)
    s.mount("http://", adapter)

    s.headers.update({
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
        "Accept-Language": "de-DE,de;q=0.9,en;q=0.8",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Connection": "keep-alive",
    })
    return s

# Use this once at startup
SESSION = build_session()

# 2) A safe fetch wrapper used by ALL your scraping functions
def fetch(
    url: str,
    *,
    referer: str | None = None,
    connect_timeout: float = 5.0,
    read_timeout: float = 25.0,
    tries: int = 3,
    backoff: float = 1.4,
) -> requests.Response:
    """
    GET with extra retries for ReadTimeouts/ConnectTimeouts beyond urllib3 Retry.
    Raises the last exception if all tries fail.
    """
    last_err = None
    headers = {}
    if referer:
        headers["Referer"] = referer

    for attempt in range(1, tries + 1):
        try:
            resp = SESSION.get(url, headers=headers, timeout=(connect_timeout, read_timeout), allow_redirects=True)
            # Retry on 429/5xx here too (even though adapter retries)
            if resp.status_code in (429, 500, 502, 503, 504):
                last_err = requests.HTTPError(f"HTTP {resp.status_code} for {url}")
                # backoff then retry
                sleep = (backoff ** (attempt - 1)) + random.uniform(0, 0.4)
                time.sleep(sleep)
                continue
            resp.raise_for_status()
            return resp
        except (requests.ReadTimeout, requests.ConnectTimeout, requests.ChunkedEncodingError) as e:
            last_err = e
            sleep = (backoff ** (attempt - 1)) + random.uniform(0, 0.4)
            time.sleep(sleep)
            continue
        except requests.RequestException as e:
            # Non-retryable (DNS, SSL, etc.) – bubble up after short delay
            last_err = e
            break

    # All attempts failed
    if last_err:
        raise last_err
    raise RuntimeError(f"Failed to GET {url} for unknown reasons.")


#-------------------------------------------------------------------------------------------------------------------------
# creating a webscrapper to get the player-id on transfermarket website, either by name or url
#-------------------------------------------------------------------------------------------------------------------------
#two def functions to get the player-id, by either using the url or player name
SESSION = build_session()

def extract_player_id_from_url(url: str) -> str | None:
    m = re.search(r"/spieler/(\d+)", url)
    return m.group(1) if m else None

#NOTE -> by default it scrapps .de switch to to .com, .co.uk, .es etc.
def find_player_id_by_search(name: str, tld: str = "de", pause: float = 1.0) -> str | None:
    """Scrape Transfermarkt quick search results and return the first 'Spieler' hit ID."""
    url = f"https://www.transfermarkt.{tld}/schnellsuche/ergebnis/schnellsuche?query={requests.utils.quote(name)}"
    r = fetch(url, read_timeout=25.0)
    if r.status_code != 200:
        return None
    time.sleep(pause)  # be polite

    soup = BeautifulSoup(r.text, "html.parser")
    # Look for a link that contains '/spieler/<id>' (profil or main)
    for a in soup.select('a[href*="/spieler/"]'):
        m = re.search(r"/spieler/(\d+)", a.get("href", ""))
        if m:
            return m.group(1)
    return None



#-------------------------------------------------------------------------------------------------------------------------
# scrapping the rumors archive table and getting every available table page
#-------------------------------------------------------------------------------------------------------------------------
def _parse_rumours_page(html: str) -> pd.DataFrame:
    """
    Parse exactly the 'Gerüchtearchiv' table(s) from a rumours page (any page number).
    Returns a DataFrame with columns:
      ['club_name','club_id','last_source','last_answer','probability_raw']
    """
    soup = BeautifulSoup(html, "html.parser")

    def _parse_table(table) -> list[dict]:
        rows_out = []
        if not table:
            return rows_out
        for tr in table.select("tbody tr"):
            tds = tr.find_all("td")
            if len(tds) < 5:
                continue

            # interested club (2nd td)
            club_link = tds[1].select_one("a[href*='/verein/']")
            club_name = club_link.get_text(strip=True) if club_link else tds[1].get_text(strip=True)
            club_href = club_link.get("href", "") if club_link else ""
            m = re.search(r"/verein/(\d+)", club_href or "")
            club_id = m.group(1) if m else None

            # dates and probability
            last_source = tds[2].get_text(strip=True)
            last_answer = tds[3].get_text(strip=True)
            prob_text   = tds[4].get_text(strip=True).replace("–", "-")

            rows_out.append({
                "club_name": safe_strip(club_name),
                "club_id":   safe_strip(club_id),
                "last_source": safe_strip(last_source),
                "last_answer": safe_strip(last_answer),
                "probability_raw": safe_strip(prob_text),
            })
        return rows_out

    # 1) Prefer tables inside a box whose <h2> == "Gerüchtearchiv"
    all_rows = []
    for h2 in soup.find_all("h2"):
        if safe_strip(h2.get_text()).casefold() == "gerüchtearchiv":
            box = h2.find_parent(class_="box")
            if not box:
                continue
            tables = box.select("table.items")
            for tbl in tables:
                all_rows.extend(_parse_table(tbl))

    if all_rows:
        return pd.DataFrame(all_rows, columns=["club_name","club_id","last_source","last_answer","probability_raw"])

    # 2) Fallback: try pandas.read_html to find a table with the expected header
    try:
        dfs = pd.read_html(html, extract_links="body")
        for df in dfs:
            if "Interessierter Verein" in df.columns:
                # unpack tuple cells from extract_links="body"
                def unpack(cell):
                    if isinstance(cell, tuple):
                        return cell[0], cell[1]
                    return cell, None

                rows = []
                for _, row in df.iterrows():
                    club_text, club_href = unpack(row.get("Interessierter Verein"))
                    last_source, _ = unpack(row.get("Letzter Quelleneintrag"))
                    last_answer, _ = unpack(row.get("Letzte Antwort"))
                    prob_text, _  = unpack(row.get("Wechselwahrscheinlichkeit"))

                    club_id = None
                    if club_href:
                        m = re.search(r"/verein/(\d+)", club_href or "")
                        if m:
                            club_id = m.group(1)

                    rows.append({
                        "club_name": safe_strip(club_text),
                        "club_id":   safe_strip(club_id),
                        "last_source": safe_strip(last_source),
                        "last_answer": safe_strip(last_answer),
                        "probability_raw": safe_strip(prob_text),
                    })
                if rows:
                    return pd.DataFrame(rows, columns=["club_name","club_id","last_source","last_answer","probability_raw"])
    except ValueError:
        pass  # no tables found by read_html

    # 3) Final generic fallback: first table.items on the page
    generic_tbl = soup.select_one("table.items")
    if generic_tbl:
        rows = _parse_table(generic_tbl)
        return pd.DataFrame(rows, columns=["club_name","club_id","last_source","last_answer","probability_raw"])

    # Empty stable schema
    return pd.DataFrame(columns=["club_name","club_id","last_source","last_answer","probability_raw"])


def safe_strip(x):
    """
    Normalize a cell value to a clean string.
    - Unpacks (text, href) tuples from pandas.read_html(extract_links="body")
    - Handles None / NaN / floats
    """
    if isinstance(x, tuple):
        x = x[0]
    if x is None:
        return ""
    try:
        # handle pandas/numpy NaN without importing numpy explicitly
        if isinstance(x, float) and pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x).strip()

def _to_dt(s) -> pd.Timestamp | None:
    """
    Parse Transfermarkt date strings (e.g., '11.06.2025') to Timestamp.
    Returns None if empty or '-' or unparsable.
    """
    s = safe_strip(s)
    if not s or s == "-":
        return None
    ts = pd.to_datetime(s, dayfirst=True, errors="coerce")
    return None if pd.isna(ts) else ts

def _parse_prob(x) -> float | None:
    """
    Parse probability strings like '75%' or '-' to a float (0..100) or None.
    """
    s = safe_strip(x).replace("%", "").replace(",", ".")
    if not s or s == "-":
        return None
    try:
        return float(s)
    except ValueError:
        return None

def _clean(s: str) -> str:
    """Collapse whitespace and trim."""
    return re.sub(r"\s+", " ", s).strip()

@lru_cache(maxsize=1024)
def get_club_name(club_id: str | int, tld: str = "de", pause: float = 0.5) -> str | None:
    """
    Resolve a club_id to its display name by scraping the club rumours page.
    Strategies: header <h1>, og:title, <title>, breadcrumb.
    Cached for speed/stability.
    """
    url = f"https://www.transfermarkt.{tld}/x/geruechte/verein/{club_id}"
    r = SESSION.get(url, timeout=20)
    if r.status_code != 200:
        return None
    time.sleep(pause)

    soup = BeautifulSoup(r.text, "html.parser")

    # 1) Main H1 headline
    h1 = soup.select_one(".data-header__headline-container h1")
    if h1 and _clean(h1.text):
        return _clean(h1.text)

    # 2) og:title (strip suffix like " – Gerüchte ...")
    og = soup.find("meta", attrs={"property": "og:title"})
    if og and og.get("content"):
        title = re.sub(r"\s*[-|–]\s*Gerüchte.*$", "", og["content"], flags=re.IGNORECASE)
        title = _clean(title)
        if title:
            return title

    # 3) <title>
    title_tag = soup.find("title")
    if title_tag and title_tag.text:
        title = re.sub(r"\s*[-|–]\s*Gerüchte.*$", "", title_tag.text, flags=re.IGNORECASE)
        title = _clean(title)
        if title:
            return title

    # 4) Breadcrumb fallback
    bc = soup.select_one(".tm-breadcrumb a[href*='/verein/']")
    if bc and _clean(bc.text):
        return _clean(bc.text)

    return None


def _find_archive_grid_info(html: str) -> tuple[str|None, str|None]:
    """
    Return (slug_path, grid_id) for the Gerüchtearchiv grid.
    slug_path example: '/nick-woltemade/geruechte/spieler/182913'
    grid_id example: 'yw2'
    """
    soup = BeautifulSoup(html, "html.parser")

    # slug from the grid's keys 'title' (most reliable)
    slug_path = None
    keys = soup.select_one("div.grid-view div.keys") or soup.select_one("div.keys")
    if keys and keys.has_attr("title"):
        slug_path = keys["title"].strip()

    # find the specific grid-view that contains the archive table
    grid = None
    for gv in soup.select("div.grid-view"):
        if gv.select_one("table.items thead th") and (gv.select_one("div.keys") or keys):
            # ensure this grid has the 'Interessierter Verein' header (archive table)
            head_txt = " ".join(th.get_text(" ", strip=True) for th in gv.select("thead th"))
            if "Interessierter Verein" in head_txt:
                grid = gv
                break
    grid_id = grid.get("id") if grid else None

    return slug_path, grid_id

def get_player_rumours_archive(
    player_id: str,
    tld: str = "de",
    pause: float = 0.7,
    fill_club_names: bool = True,
    max_pages: int | None = None,
    debug: bool = False,
) -> pd.DataFrame:
    base = f"https://www.transfermarkt.{tld}"

    def empty_df():
        df = pd.DataFrame(columns=["club_name","club_id","last_source","last_answer","probability_raw","page"])
        df["last_source_dt"] = pd.NaT
        df["last_answer_dt"] = pd.NaT
        df["probability_pct"] = pd.Series(dtype="float")
        return df

    # --- fetch page 1 (normal HTML) ---
    url_p1 = f"{base}/x/geruechte/spieler/{player_id}"
    r = SESSION.get(url_p1, timeout=20, allow_redirects=True)
    if r.status_code != 200:
        if debug: print(f"GET p1 failed: {r.status_code}")
        return empty_df()
    html1 = r.text
    time.sleep(pause)

    # parse page 1
    df_all = []
    df1 = _parse_rumours_page(html1)
    if not df1.empty:
        df1["page"] = 1
        df_all.append(df1)

    # discover slug and grid id for AJAX
    slug_path, grid_id = _find_archive_grid_info(html1)
    if debug: print("slug_path:", slug_path, "grid_id:", grid_id)

    # detect last page from the pager on page 1 (if present)
    def get_last_page(html: str) -> int:
        soup = BeautifulSoup(html, "html.parser")
        pager = soup.select_one("div.grid-view .tm-pagination") or soup.select_one(".tm-pagination")
        if not pager:
            return 1
        nums = []
        for a in pager.select("a.tm-pagination__link"):
            txt = (a.text or "").strip()
            if txt.isdigit():
                nums.append(int(txt))
            else:
                m = re.search(r"/page/(\d+)$", a.get("href") or "")
                if m:
                    nums.append(int(m.group(1)))
        return max(nums) if nums else 1

    last_page = get_last_page(html1)
    if max_pages is not None:
        last_page = min(last_page, max_pages)

    if debug: print("last_page:", last_page)

    # If no slug or grid id or only one page -> done
    if not slug_path or not grid_id or last_page <= 1:
        # finalize types/sort
        if not df_all:
            return empty_df()
        df = pd.concat(df_all, ignore_index=True)
        df["last_source_dt"] = df["last_source"].map(_to_dt)
        df["last_answer_dt"] = df["last_answer"].map(_to_dt)
        df["probability_pct"] = df["probability_raw"].map(_parse_prob)
        if fill_club_names:
            need = df["club_name"].astype(str).str.strip().eq("") & df["club_id"].astype(str).str.strip().ne("")
            ids = df.loc[need, "club_id"].astype(str).unique()
            if len(ids):
                mp = {cid: get_club_name(cid, tld=tld) for cid in ids}
                df.loc[need, "club_name"] = df.loc[need, "club_id"].astype(str).map(mp).fillna(df.loc[need, "club_name"])
        return df.sort_values(["last_answer_dt","last_source_dt","page"], ascending=[False,False,True], na_position="last").reset_index(drop=True)

    # --- fetch pages 2..last via AJAX endpoint ---
    # Pattern: /<slug>/ajax/<grid_id>/page/<n>
    for n in range(2, last_page + 1):
        ajax_url = urljoin(base, f"{slug_path}/ajax/{grid_id}/page/{n}")
        if debug: print("GET", ajax_url)
        r = SESSION.get(ajax_url, timeout=20, headers={"Referer": urljoin(base, slug_path)}, allow_redirects=True)
        if r.status_code != 200:
            if debug: print(" -> status", r.status_code)
            break
        html = r.text
        # the AJAX response is a fragment that still contains the table; parse it
        dfp = _parse_rumours_page(html)
        if dfp.empty:
            if debug: print(" -> empty page, stopping")
            break
        dfp["page"] = n
        df_all.append(dfp)
        time.sleep(pause)

    # finalize
    if not df_all:
        return empty_df()
    df = pd.concat(df_all, ignore_index=True)
    df["last_source_dt"] = df["last_source"].map(_to_dt)
    df["last_answer_dt"] = df["last_answer"].map(_to_dt)
    df["probability_pct"] = df["probability_raw"].map(_parse_prob)
    if fill_club_names:
        need = df["club_name"].astype(str).str.strip().eq("") & df["club_id"].astype(str).str.strip().ne("")
        ids = df.loc[need, "club_id"].astype(str).unique()
        if len(ids):
            mp = {cid: get_club_name(cid, tld=tld) for cid in ids}
            df.loc[need, "club_name"] = df.loc[need, "club_id"].astype(str).map(mp).fillna(df.loc[need, "club_name"])
    return df.sort_values(["last_answer_dt","last_source_dt","page"], ascending=[False,False,True], na_position="last").reset_index(drop=True)


#-----------------------------------------------------------------------------------------------------------------------------------------------------
#Manual learnig_curve to align class weights in classification model
#-----------------------------------------------------------------------------------------------------------------------------------------------------

def learning_curve_weighted(
    estimator,
    X,
    y,
    groups,
    cv,                           # e.g. StratifiedGroupKFold(...)
    train_sizes=np.linspace(0.1,1.0,10),
    scoring=None,                 # None -> neg_log_loss(proba) with provided labels
    labels=None,                  # pass np.arange(num_class) for log_loss stability
    sample_weight_full=None,      # array-like, shape (n_samples,)
    sample_weight_param='XGBoost Classifier__sample_weight', #default setting, depending on the model, change naming
    extra_fit_params=None,        # dict of static fit params (e.g. eval_metric)
    use_early_stopping=False,     # if True, we’ll pass eval_set per fold
    random_state=101
):
    """
    Returns:
      train_sizes_abs: array of absolute train sizes used
      train_scores:    array shape (len(train_sizes), n_splits)
      val_scores:      array shape (len(train_sizes), n_splits)
    """
    rng = check_random_state(random_state)
    train_sizes = np.array(train_sizes, dtype=float)

    #due to the fact that we use it for XGBoost mulitclass problem, we just set up log_loss
    #default scorer: neg_log_loss on proba with fixed label order (recommended)
    if scoring is None:
        if labels is None:
            raise ValueError("For default scoring, provide `labels` (encoded label order).")
        scorer = make_scorer(log_loss, needs_proba=True, greater_is_better=False, labels=labels)
    else:
        from sklearn.metrics import get_scorer
        scorer = get_scorer(scoring)

    #collect split indices once => group aware cv-splitting
    splits = list(cv.split(X, y, groups))
    n_splits = len(splits)

    # precompute absolute sizes per split (convert fractions -> counts based on fold size)
    fold_train_sizes_abs = []
    for tr_idx, _ in splits:
        n_tr = len(tr_idx)
        sizes_abs = np.unique(np.maximum(1, (train_sizes * n_tr).astype(int)))
        fold_train_sizes_abs.append(sizes_abs)

    # shape outputs like sklearn: [len(train_sizes_unified), n_splits]
    # For simplicity, we’ll unify sizes across folds using the sizes derived from the FIRST fold:
    train_sizes_abs = fold_train_sizes_abs[0]
    train_scores = np.zeros((len(train_sizes_abs), n_splits))
    val_scores   = np.zeros((len(train_sizes_abs), n_splits))

    for s, (tr_idx, va_idx) in enumerate(splits):
        X_tr_all, y_tr_all = X.iloc[tr_idx], y[tr_idx]
        X_va, y_va = X.iloc[va_idx], y[va_idx]

        # reproducible shuffle within the training fold (keeps class strata intact if your CV already stratified)
        perm = rng.permutation(len(tr_idx))
        X_tr_all = X_tr_all.iloc[perm]
        y_tr_all = y_tr_all[perm]
        sw_tr_all = None
        if sample_weight_full is not None:
            sw_tr_all = sample_weight_full[tr_idx][perm]

        # compute sizes for this fold and map to unified index positions
        sizes_abs_fold = fold_train_sizes_abs[s]
        # map each size in unified list to the closest <= size in this fold
        size_map = [sizes_abs_fold[np.searchsorted(sizes_abs_fold, t, side='right')-1] for t in train_sizes_abs]

        for i, n_tr_use in enumerate(size_map):
            X_tr = X_tr_all.iloc[:n_tr_use]
            y_tr = y_tr_all[:n_tr_use]
            fit_params = {}

            if sw_tr_all is not None:
                fit_params[sample_weight_param] = sw_tr_all[:n_tr_use]

            # per-fold early stopping using validation fold
            if use_early_stopping:
                fit_params.update({
                    'XGBoost Classifier__eval_set': [(X_va, y_va)],
                    'XGBoost Classifier__verbose': False,
                })

            if extra_fit_params:
                # static extras (e.g., callbacks, eval_metric)
                fit_params.update(extra_fit_params)

            est = clone(estimator)

            # fit
            est.fit(X_tr, y_tr, **fit_params)

            # score
            tr_score = scorer(est, X_tr, y_tr)
            va_score = scorer(est, X_va, y_va)
            train_scores[i, s] = tr_score
            val_scores[i,  s]  = va_score

    return train_sizes_abs, train_scores, val_scores