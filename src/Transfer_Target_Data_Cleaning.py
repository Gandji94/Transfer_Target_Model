import pandas as pd
import numpy as np

def top_or_others_more(x):
    """
    Function: Allows us to quickly bin into top leagues, which we will leave as they are,
    when they are not top leagues, it will return 'Other Leagues'

    Parameter: x; represents each row

    Output: On column level, transform column based on described function (see 'Function')
    """
    if x in ['Premier League','LaLiga','Bundesliga','Serie A','Ligue 1']:
        return x
    elif x in ["Liga Portugal", "Eredivisie", "Süper Lig", "Jupiler Pro League","Super League","Super League 1","Premiership","efbet Liga","SuperSport HNL","Allsvenskan","Ekstraklasa","Fortuna Liga","Protathlima Cyta",
               "Österreichische Bundesliga","Superliga","Super Liga Srbije"]:
        return 'Smaller Top European Leagues'
    elif x in ["Serie B","Ligue 2","2. Bundesliga","LaLiga2","Championship","Proximus League","2ª División"]:
        return 'Secon Tier Leagues'
    else:
        return 'Other Leagues'
    

def top_or_others(x):
    """
    Function: Allows us to quickly bin into top leagues, which we will leave as they are,
    when they are not top leagues, it will return 'Other Leagues'

    Parameter: x; represents each row

    Output: On column level, transform column based on described function (see 'Function')
    """
    if x in ['Premier League','LaLiga','Bundesliga','Serie A','Ligue 1']:
        return x
    else:
        return 'Other Leagues'
    
def league_class_trans(x):
    """
    Function: Allows us to quickly bin into top leagues, which we will leave as they are,
    when they are not top leagues, it will return 'Other Leagues'

    Parameter: x; represents each row

    Output:
    """
    if x in ['Premier League','LaLiga','Bundesliga','Serie A','Ligue 1']:
        return x
    elif x in ["Liga Portugal", "Eredivisie", "Süper Lig","JPL"]:
        return "Smaller European Leagues"
    else:
        return "Other Leagues"