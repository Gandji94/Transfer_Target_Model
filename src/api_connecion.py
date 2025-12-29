import pandas as pd
import numpy as np
import joblib
import sys
#sys.path.append('..\..\src')
from .PY_Class_Def import who_is_that_player
import ipywidgets as widgets
from IPython.display import display
from flask import Flask,request,jsonify
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


#binary total model
binary_total_model=joblib.load(MODEL_BINARY/'binary_final_model_total_data.pkl')
#binary columns
binary_columns=joblib.load(MODEL_BINARY/'binary_model_columns.joblib')

#multiclass total model
multiclass_total_model=joblib.load(MODEL_MULTI/'multiclass_final_model_total_data.pkl')
#translation for y labels
classes_=joblib.load(MODEL_MULTI/'multiclass_translation.dict')
#multiclass columns
multiclass_columns=joblib.load(MODEL_MULTI/'multiclass_model_columns.joblib')

df_mean_stats=pd.read_csv(DATA_PROCESSED_PLAYERS_MEAN).iloc[:,1:]

#list with player names
player_lst=sorted(list(df_mean_stats['Player'].unique()))

#best probability threshold for 
best_prob_value=joblib.load(MODEL_BINARY/'best_thresh_binary.dict')

#create Flask App
app=Flask(__name__)

@app.route('/predict',methods=['Post'])
def predict():
    """
    Expects JSON like:
    {
        "player_name":"Tom Bischof"
    }
    """
    #-------------------------------------------------- Setting up the API ------------------------------------------------
    data = request.get_json()
    #when no request or value for player_name is given, we will run into an error
    if not data or 'player_name' not in data:
        return jsonify({"error": "JSON body must contain 'player_name'"}),400
    #here we allocate the input value to our variable
    pn=data['player_name']
    try:
        player_stats=who_is_that_player(pn)
    except Exception as e:
        return jsonify({"error":str(e)}),500
    #--------------------------------------- Making Binary Prediction ----------------------------------------------------
    binary_pred_classes=binary_total_model.predict_proba(player_stats)[0]
    pos_class=binary_pred_classes[1]
    threshold_binary=best_prob_value['appropriate_prob_thresh']

    #output to the request
    response = {
        "player": pn,
        "binary_top_league_prob": pos_class,
        "binary_threshold": threshold_binary,
        "binary_prediction": "Top League" if pos_class > threshold_binary else "Non-Top League"
    }
    #---------------------------------------- Making Multiclass Prediction -------------------------------------------------
    if pos_class > threshold_binary:
        multiclass_pred_classes=multiclass_total_model.predict_proba(player_stats)[0]

        #---------------------------------------------------- Overview --------------------------------------------------
        prob_out=[multiclass_pred_classes[i] for i in range(len(multiclass_pred_classes))]

        ink_classes={v:k for k,v in classes_.items()}

        result_overview={ink_classes[i]:float(p) for i,p in enumerate(prob_out)}
        #--------------------------------------------------- Most Likely League -----------------------------------------
        max_prob=float(np.max((multiclass_pred_classes)))
        ink_classes={v:k for k,v in classes_.items()}

        result_max={float(p):ink_classes[i] for i,p in enumerate(prob_out)}[max_prob]
        try:
            current_league = player_stats["League_Left"].iloc[0]
        except Exception:
            current_league = None

        if current_league:
            if result_max.lower() == str(current_league).lower():
                msg=f"{pn} is most likely to stay in the {result_max}"
            else:
                msg=f"{pn} is most likely to join the {result_max}"
        else:
            msg=f"{pn} is most likely to be in the {result_max}"
        response.update(
            {
                'multiclass_probs':result_overview,
                'multiclass_top_league':result_max,
                'message':msg
            }
        )
    else:
        response['message']=f'{pn} is predicted for a Non-Top League'
    return jsonify(response),200

if __name__ == '__main__':
    app.run(host='127.0.0.1',port=5000,debug=True)

#http://localhost:5000/predict => that needs to be entered into postman
#host="127.0.0.1" => can only be ran on local machine
#calling the api_connection file
#python -m src.api_connection