import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import sys
#sys.path.append('..\..\src')
from .PY_Class_Def import who_is_that_player
import ipywidgets as widgets
from IPython.display import display
from flask import Flask,request,jsonify
from sklearn.base import BaseEstimator,TransformerMixin
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MODEL_BINARY = ROOT / "Models" / "Final_Model" / "Binary"
MODEL_MULTI = ROOT / "Models" / "Final_Model" / "Multiclass"
DATA_PROCESSED_PLAYERS_MEAN = ROOT / "DataSources" / "Processed" / "Player_Stats_Mean_2017_2025.csv"
DATA_PROCESSED_TRANSFERS = ROOT / "DataSources" / "Processed" / "All_Trans_2000_2025.csv"
DATA_PROCESSED_FINAL_STATS = ROOT / "DataSources" / "Processed" / "Cleaned_Final_Stats.csv"
DATA_PROCESSED_RUMORS = ROOT / "DataSources" / "Processed" / "Rumor_Overview.csv"
MODEL = ROOT / "Models" / "Final_Model"
BINARY_PRED = ROOT / "dbt_league_pred" / "seeds" / "batch_binary_pred.csv"
BINARY_PRED_SMALL = ROOT / "DataSources" / "Prediction" / "Binary" / "batch_binary_pred_small.csv"
MULTI_PRED = ROOT / "dbt_league_pred"/ "seeds" / "batch_multiclass_pred.csv"
MULTI_PRED_SMALL = ROOT / "dbt_league_pred"/ "seeds" / "batch_multiclass_pred_small.csv"

##################### continoue tomorrow ###########################

class batch_prediction(BaseEstimator,TransformerMixin):
    #binary total model
    binary_total_model=joblib.load(MODEL_BINARY/'binary_final_model_total_data_5_11_2025.pkl')

    #multiclass total model
    multiclass_total_model=joblib.load(MODEL_MULTI/'multiclass_final_model_total_data_8_11_2025.pkl')
    
    #translation for y labels
    classes_=joblib.load(MODEL_MULTI/'multiclass_translation.dict')

    df_mean_stats=pd.read_csv(DATA_PROCESSED_PLAYERS_MEAN).iloc[:,1:]

    #list with player names
    player_lst=sorted(list(df_mean_stats['Player'].unique()))

    #best probability threshold for 
    best_prob_value=joblib.load(MODEL_BINARY/'best_thresh_binary.dict')

    def __init__(self):
        self.player_=None
    
    def who_is_that_player(self,player_):
        #binary columns
        binary_columns=joblib.load(MODEL_BINARY/'binary_model_columns.joblib')
        #multiclass columns
        multiclass_columns=joblib.load(MODEL_MULTI/'multiclass_model_columns.joblib')

        df_mean_stats=pd.read_csv(DATA_PROCESSED_PLAYERS_MEAN).iloc[:,1:]
        player_pred=pd.read_csv(DATA_PROCESSED_PLAYERS_MEAN).iloc[:,1:]
        df_trans=pd.read_csv(DATA_PROCESSED_TRANSFERS).iloc[:,1:]
        df_trans.drop(['Age'],axis=1,inplace=True)
        df_cf=pd.read_csv(DATA_PROCESSED_FINAL_STATS)
        df_rumor=pd.read_csv(DATA_PROCESSED_RUMORS).drop(['probability_raw','page','probability_pct','club_id'],axis=1)

        df_stats=pd.merge(df_mean_stats,df_trans,on=['Player','Transfer_Window'],how='left').dropna()
        df_stats=df_stats.drop(['Squad','League','Transfer_Fee','Born','Club_Joined','League_Joined'],axis=1)
        player_=player_pred[(player_pred['Player']==player_)]
        max_date=player_['Transfer_Window'].max()
        player=player_[player_['Transfer_Window'].eq(max_date)]
        player.rename(columns={'Squad':'Club_Left','League':'League_Left'},inplace=True)
        player['Performance_Year']=player['Transfer_Window']-1
        col_names=list(df_stats.columns)
        player=player[col_names]

        club_league_df=df_cf[['Club_Joined','League_Joined']].drop_duplicates()
        club_league_df.rename(columns={'Club_Joined':'club_name'},inplace=True)

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

        player_final=pd.merge(player,df_rumor_agg,on=['Player','Performance_Year'],how='left').drop(['Player'],axis=1).fillna(0)

        #column check between the different models (binary and multiclass)
        if not np.array_equal(binary_columns,multiclass_columns):
            raise ValueError('Columns do not match between binary and multiclass model')
        else:
            standard_columns=binary_columns
        
        #check if the newly created dataframe columns match match with the prerequesits for the binary and multiclass model
        #if list(player_final.columns)!=standard_columns:
        if not np.array_equal(list(player_final.columns),standard_columns):
            raise ValueError('Columnes do not match!')
        else:
            #making sure that the generated dataframe has the same column order as the dataframe the model was trained on
            player_final=player_final.reindex(columns=standard_columns)
            return player_final


    
    def batch_binary_class(self,player_):
        binary_total_model=joblib.load(MODEL_BINARY/'binary_final_model_total_data_5_11_2025.pkl')
        #we need to call a already defined def functions with self.
        player_stats=self.who_is_that_player(player_)

        binary_pred_classes=binary_total_model.predict_proba(player_stats)[0]
        pos_class=binary_pred_classes[1]

        binary_dict={'Player':player_,'Top-League':pos_class,'Non-Top-League':(1-pos_class)}
        batch_binary_df=pd.DataFrame.from_dict(binary_dict,orient='index').transpose()
        return batch_binary_df 
    

    
    def binary_batch_preds(self,player_):
        #best probability threshold for 
        best_prob_value=joblib.load(MODEL_BINARY/'best_thresh_binary.dict')
        # --- Normalize input to a list ---
        #when we do not pass any value we need to set player_=None in the function, this will allow us to trigger the list of all players
        if player_ is None:
            player_lst=[]
        #if a single name is passed as string -> wrap into list
        elif isinstance(player_, str):
            player_lst = [player_]
        #if it's already a list/tuple/Series/ndarray -> turn into list
        elif isinstance(player_, (list, tuple)):
            player_lst = list(player_)
        else:
            #raising an error when 
            raise TypeError(f"player_ must be str or list, got {type(player_)}")

        #when no player(s) are given, code will use all available player as default, and create the CSV for the dbt runs
        if not player_lst:
            df_mean_stats = pd.read_csv(DATA_PROCESSED_PLAYERS_MEAN).iloc[:, 1:]
            player_lst = sorted(df_mean_stats['Player'].unique())
            player_id=1
            player_id_lst=[player_id+i for i in range(len(player_lst))]
            batch_df_player=pd.DataFrame(player_id_lst).rename(columns={0:'Player_ID'})

            binary_batch_df = pd.DataFrame()
            for p in player_lst:
                player_binary_stats = self.batch_binary_class(p)
                if binary_batch_df.empty:
                    binary_batch_df = player_binary_stats
                else:
                    binary_batch_df = pd.concat([binary_batch_df, player_binary_stats],axis=0,ignore_index=True)
            
            binary_batch_df=pd.concat([batch_df_player,binary_batch_df],axis=1)
            binary_batch_df['Top-League_Check']=binary_batch_df['Top-League'].apply(lambda x: 1 if x > best_prob_value['appropriate_prob_thresh'] else 0)
            binary_batch_df['Creation_Timestamp']=datetime.now()
            binary_batch_df.to_csv(BINARY_PRED,index=False)
            return binary_batch_df

        #when we pass just one value as a string
        if len(player_lst) == 1:
            p = player_lst[0]
            player_binary_stats = self.batch_binary_class(p)
            player_binary_stats['Top-League Threshold'] = best_prob_value['appropriate_prob_thresh']
            return player_binary_stats

        #when we pass multipel player in a list
        binary_batch_df = pd.DataFrame()
        for p in player_lst:
            player_binary_stats = self.batch_binary_class(p)
            player_binary_stats['Top-League Threshold'] = best_prob_value['appropriate_prob_thresh']
            if binary_batch_df.empty:
                binary_batch_df = player_binary_stats
            else:
                binary_batch_df = pd.concat([binary_batch_df, player_binary_stats],axis=0,ignore_index=True)

        binary_batch_df.to_csv(BINARY_PRED_SMALL,index=False)
        return binary_batch_df
    


    def batch_multi_class(self,player_):
        multiclass_total_model=joblib.load(MODEL_MULTI/"multiclass_final_model_total_data_8_11_2025.pkl")
        #translation for y labels
        classes_=joblib.load(MODEL_MULTI/'multiclass_translation.dict')

        player_stats=self.who_is_that_player(player_)
        multiclass_pred_classes=multiclass_total_model.predict_proba(player_stats)[0]

        reveresed_dict={v:k for k,v in classes_.items()}
        clol_league_names=[reveresed_dict[c] for c in range(5)]
        col_df={clol_league_names[i]:float(multiclass_pred_classes[i]) for i in range(5)}
        batch_multi_class=pd.DataFrame.from_dict(col_df,orient='index').transpose()
        batch_multi_class.insert(0,'Player',player_)
        return batch_multi_class
    


    def multiclass_batch_predict(self,player_):
        if player_ is None:
            player_=[]
        elif isinstance(player_,str):
            player_=[player_]
        elif isinstance(player_,(tuple,list)):
            player_=list(player_)
        else:
            raise TypeError(f'player_ must be str or list, got {type(player_)}')
        
        if not player_:
            player_=pd.read_csv(BINARY_PRED)
            player_id_=player_[['Player_ID','Player']]

            top_league_player=list(player_[(player_['Top-League_Check'].eq(1))]['Player'].unique())
            batch_df_player = pd.DataFrame()
            for p in top_league_player:
                player_stats_=self.batch_multi_class(p)
                if batch_df_player.empty:
                    batch_df_player=player_stats_
                else:
                    batch_df_player=pd.concat([batch_df_player,player_stats_],axis=0)
            batch_df_player['Creation_Timestamp']=datetime.now()
            batch_df_player=player_id_.merge(batch_df_player,how='left',on='Player').dropna()
            batch_df_player.to_csv(MULTI_PRED,index=False)
            return batch_df_player

        if len(player_) == 1:
            p=player_[0]
            player_stats_=self.batch_multi_class(p)
            return player_stats_
        
        multi_player=player_
        batch_df_player = pd.DataFrame()
        for p in multi_player:
            player_stats_=self.batch_multi_class(p)
            if batch_df_player.empty:
                batch_df_player=player_stats_
            else:
                batch_df_player=pd.concat([batch_df_player,player_stats_],axis=0)
        batch_df_player.to_csv(MULTI_PRED_SMALL,index=False)
        return batch_df_player