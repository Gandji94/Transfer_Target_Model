from .PY_Class_Def import add_missing_class_rows,entropy_from_proba,max_confidence,multiclass_margin
from pathlib import Path
import pandas as pd
import numpy as np
from evidently import Report, Dataset, DataDefinition, BinaryClassification, MulticlassClassification
from evidently.presets import DataDriftPreset, ClassificationPreset
from evidently.metrics import ValueDrift
import joblib
import sys
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.preprocessing import LabelEncoder
from math import floor
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

ROOT = Path(__file__).resolve().parent.parent

DATA_STATS_RUMORS = ROOT / "DataSources" / "Processed" / "Cleaned_Final_Stats_w_Rumors.csv"
MODEL_TRAIN_BINARY_X_TRAIN = ROOT / "DataSources" / "Model_Data" / "Binary" / "Train_Test" / "X_Train_Data.CSV"
MODEL_TRAIN_BINARY_y_TRAIN = ROOT / "DataSources" / "Model_Data" / "Binary" / "Train_Test" / "y_Train_Data.CSV"
MODEL_TRAIN_MULTI_X_TRAIN = ROOT / "DataSources" / "Model_Data" / "Multiclass" / "Train_Test" / "X_Train_Data.CSV"
MODEL_TRAIN_MULTI_y_TRAIN = ROOT / "DataSources" / "Model_Data" / "Multiclass" / "Train_Test" / "y_Train_Data.CSV"
MODEL_BINARY = ROOT / "Models" / "Final_Model" / "Binary"
MODEL_MULTI = ROOT / "Models" / "Final_Model" / "Multiclass"
#EXPERIEMNTS = ROOT / "Experiments"
#EXPERIEMNTS.mkdir(parents=True, exist_ok=True)

try:
    new_data = pd.read_csv(DATA_STATS_RUMORS)
    top_5={'Premier League', 'Ligue 1', 'Serie A','Bundesliga', 'LaLiga'}
    new_data['Other_or_Top']=new_data['League_Joined'].isin(top_5).astype(int)
    binary_model=joblib.load(MODEL_BINARY/'binary_final_model_total_data_5_11_2025.pkl')
    binary_trhesh = joblib.load(MODEL_BINARY/'best_thresh_binary.dict')
    multi_model=joblib.load(MODEL_MULTI/'multiclass_final_model_total_data_8_11_2025.pkl')
    #we get avaiable classes
    le=LabelEncoder()
except Exception:
    logging.error('Issue(s) loading data and models')

#new data does not need to be split, it will just be transformed
new_data_binary_ = new_data.copy()
new_data_binary = new_data_binary_.drop(['Player','Squad','League','League_Joined','Club_Joined','Transfer_Fee','Other_or_Top'],axis=1)
new_data_target_binary = new_data_binary_['Other_or_Top']

new_data_multiclass_ = new_data_binary_[new_data_binary_['Other_or_Top'].eq(1)]
new_data_multiclass = new_data_multiclass_.drop(['Player','Squad','League','League_Joined','Club_Joined','Transfer_Fee','Other_or_Top'],axis=1)
new_data_target_multiclass = new_data_multiclass_['League_Joined']
trans_new_target_multi = le.fit_transform(new_data_target_multiclass)

#old data will be just seperated into train and test
##note because I got carried away and saved the csv files via joblib's dump, therefore we need to load it via load
old_train_binary = joblib.load(MODEL_TRAIN_BINARY_X_TRAIN)
old_target_train_binary = joblib.load(MODEL_TRAIN_BINARY_y_TRAIN)

old_train_multi = joblib.load(MODEL_TRAIN_MULTI_X_TRAIN)
old_target_train_multi = joblib.load(MODEL_TRAIN_MULTI_y_TRAIN)

#next we load the models with the existing pipeline structure and cut it down so we only have the transformation steps in the pipeline
#by adding estimator, we can select different steps in the pipeline
#--------------------------------------------------- Binary Data --------------------------------------------------------------------
#we need to call "calibrated_classifiers_" directly otherwise we would just get an unfitted clone
ccv=binary_model.calibrated_classifiers_[0].estimator
check_is_fitted(ccv)
binary_transform = ccv[:-2]
trans_new_binary = binary_transform.transform(new_data_binary)
trans_old_train_binary = binary_transform.transform(old_train_binary)

#--------------------------------------------------- Multiclass Data --------------------------------------------------------------------
#we need to call "calibrated_classifiers_" directly otherwise we would just get an unfitted clone
ccv=multi_model.calibrated_classifiers_[0].estimator
check_is_fitted(ccv)
multi_transform = ccv[:-2]
trans_new_multi = multi_transform.transform(new_data_multiclass)
trans_old_train_multi = multi_transform.transform(old_train_multi)

logging.info('------------------------------------------------- X Feature Data Drift -------------------------------------------------')
logging.info('------------------------------------------------- Binary Report -------------------------------------------------')
logging.info('------------------------------------------------- Raw Binary Data Drift -------------------------------------------------\n')
try:
    report = Report(metrics=[DataDriftPreset(
        #js => Jensen-Shannon will be used for categorical and binary values
        cat_method = 'jensenshannon', cat_threshold = 0.1,
        num_method='mannw', num_threshold=0.05,
        #threshold for data drift, when surpassed, then column will be considered as drifted
        drift_share=0.2

    )])
    report_binary_drift = report.run(reference_data=old_train_binary, current_data=new_data_binary)
    report_binary_drift.save_html('Binary_Raw_Data_Drift_Detection.html')
    logging.info("Saved successfully")
except Exception:
    logging.error('Issue with generating report')

logging.info('------------------------------------------------- Transformed Binary Data Drift -------------------------------------------------\n')
try:
    report = Report(metrics=[DataDriftPreset(
        #js => Jensen-Shannon will be used for categorical and binary values
        cat_method = 'jensenshannon', cat_threshold = 0.1,
        num_method='psi', num_threshold=0.05,
        #threshold for data drift, when surpassed, then column will be considered as drifted
        drift_share=0.2

    )])
    report_binary_drift = report.run(reference_data=trans_old_train_binary, current_data=trans_new_binary)
    report_binary_drift.save_html('Binary_Transformed_Data_Drift_Detection.html')
    logging.info("Saved successfully")
except Exception:
    logging.error(f'Issue with generating report')

logging.info('------------------------------------------------- Binary Data Drift Done -------------------------------------------------\n')

logging.info('------------------------------------------------- Multiclass Report -------------------------------------------------')
logging.info('------------------------------------------------- Raw Multiclass Data Drift -------------------------------------------------\n')

try:
    report = Report(metrics=[DataDriftPreset(
        #js => Jensen-Shannon will be used for categorical and binary values
        cat_method = 'jensenshannon', cat_threshold = 0.1,
        num_method='mannw', num_threshold=0.05,
        #threshold for data drift, when surpassed, then column will be considered as drifted
        drift_share=0.2

    )])
    report_binary_drift = report.run(reference_data=trans_old_train_multi, current_data=trans_new_multi)
    report_binary_drift.save_html('Multiclass_Raw_Data_Drift_Detection.html')
    logging.info("Saved successfully")
except Exception:
    logging.error(f'Issue with generating report')

logging.info('------------------------------------------------- Transformed Multiclass Data Drift -------------------------------------------------\n')

try:
    report = Report(metrics=[DataDriftPreset(
        #js => Jensen-Shannon will be used for categorical and binary values
        cat_method = 'jensenshannon', cat_threshold = 0.1,
        num_method='psi', num_threshold=0.05,
        drift_share=0.2

    )])
    report_multiclass_drift = report.run(reference_data=trans_old_train_multi, current_data=trans_new_multi)
    report_multiclass_drift.save_html('Multiclass_Transformed_Data_Drift_Detection.html')
    logging.info("Saved successfully")
except Exception:
    logging.error('Issue with generating report')

logging.info('------------------------------------------------- Multi Data Drift Done -------------------------------------------------\n')

logging.info('------------------------------------------------- Drift in Predictions -------------------------------------------------')
logging.info('------------------------------------------------- Setting up Predictions -------------------------------------------------\n')
try:
    binary_old_pred = binary_model.predict_proba(old_train_binary)
    binary_new_pred = binary_model.predict_proba(new_data_binary)

    #with ravel() we flatten the array, so we can check if there is a drift
    #in class 0 and class 1
    binary_old_pred_df = pd.DataFrame({
        "p_class_1": binary_old_pred[:,1]
    })
    binary_new_pred_df = pd.DataFrame({
        "p_class_1": binary_new_pred[:,1]
    })
except Exception:
    logging.error(f'Issues making predictions for the binary model')
try:
    multi_old_pred = multi_model.predict_proba(old_train_multi)
    multi_new_pred = multi_model.predict_proba(new_data_multiclass)

    #with ravel() we flatten the array, so we can check if there is a drift
    #in all 5 classes
    multi_old_pred_df = pd.DataFrame({f'p_class_{i}':multi_old_pred[:,i] for i in range(0,len(le.classes_))})

    multi_new_pred_df = pd.DataFrame({f'p_class_{i}':multi_new_pred[:,i] for i in range(0,len(le.classes_))})
except Exception:
    logging.error(f'Issues making predictions for the multiclass model')

logging.info('------------------------------------------------- Binary Model Drift in Prediction -------------------------------------------------\n')
try:
    report = Report(metrics=[DataDriftPreset(
    #wasserstein distance is used for numerical features
    #it works well for continuous probability outputs and
    #is sensitive to subtle distribution and calibration shifts
    num_method = 'wasserstein', cat_threshold = 0.1,
    #threshold for data drift, when surpassed, then column will be considered as drifted
    #for the binary class we select a drif_share of 1.0, this means we need a shift of a whole class
    #to trigger a drift notification
    drift_share=1.0),
    #THIS COLUMN IS OPTIONAL, WE ONLY CHECK ONE COLUMN
    ValueDrift(column="p_class_1", method="wasserstein", threshold=0.1)
    ])
    report_binary_drift = report.run(reference_data=binary_old_pred_df, current_data=binary_new_pred_df)
    report_binary_drift.save_html('Drift in Binary Prediction.html')
    logging.info("Saved successfully")
except Exception:
    logging.error('Issue creating the binary prediction drift report')
logging.info('------------------------------------------------- Multiclass Model Drift in Prediction -------------------------------------------------\n')
try:
    #in this example we take half of all avaiable classes and floor it
    #in our example we would trigger an alert, when 2 of 5 classes shifted
    min_number_of_classes_drift = floor((len(le.classes_)*0.5))
    num_of_classes = len(le.classes_)
    multi_drift_share = min_number_of_classes_drift/num_of_classes

    report = Report(metrics=[DataDriftPreset(
        #wasserstein distance is used for numerical features
        #it works well for continuous probability outputs and
        #is sensitive to subtle distribution and calibration shifts
        num_method = 'wasserstein', cat_threshold = 0.1,
        #threshold for data drift, when surpassed, then column will be considered as drifted
        drift_share=multi_drift_share

    )])
    report_multi_drift = report.run(reference_data=multi_old_pred_df, current_data=multi_new_pred_df)
    report_multi_drift.save_html('Drift in Multi Prediction.html')
    logging.info("Saved successfully")
except Exception:
    logging.error('Issue creating the multiclass prediction drift report')
logging.info('------------------------------------------------- Binary Lable Drift -------------------------------------------------\n')
try:
    #building prediction dataframe
    old_binary_proba = binary_model.predict_proba(old_train_binary)[:, 1]
    old_binary_pred  = (old_binary_proba >= binary_trhesh["appropriate_prob_thresh"]).astype(int)

    old_pred_df = pd.DataFrame({
        "target": old_target_train_binary,
        "prob_class_1": old_binary_proba,
        "pred": old_binary_pred,
    })

    new_binary_proba = binary_model.predict_proba(new_data_binary)[:, 1]
    new_binary_pred  = (new_binary_proba >= binary_trhesh["appropriate_prob_thresh"]).astype(int)

    new_pred_df = pd.DataFrame({
        "target": new_data_target_binary,
        "prob_class_1": new_binary_proba,
        "pred": new_binary_pred,
    })

    #here we setup the data definition
    data_def = DataDefinition(
        classification=[
            BinaryClassification(
                target="target",
                prediction_labels="pred",        #hard label column
                prediction_probas="prob_class_1",#probability column
                pos_label=1
            )
        ],
        categorical_columns=["target", "pred"]  #help Evidently treat them correctly
    )

    #creating a dataset
    ref_ds = Dataset.from_pandas(old_pred_df, data_definition=data_def)
    cur_ds = Dataset.from_pandas(new_pred_df, data_definition=data_def)

    report = Report([ClassificationPreset()], include_tests=True)
    binary_quality_drift = report.run(current_data=cur_ds, reference_data=ref_ds)
    binary_quality_drift.save_html("Binary_Quality_Drift.html")
    logging.info("Saved successfully")
except Exception:
    logging.error('Issue creating the binary lable drift report')
logging.info('------------------------------------------------- Multiclass Lable Drift -------------------------------------------------\n')
try:
    K = len(le.classes_)
    proba_cols = [str(i) for i in range(K)]  # proba column names are strings

    #the def-function is added to the "PY_Class_Def" file
    def add_missing_class_rows(df: pd.DataFrame, proba_cols, K: int):
        """
        Add 1 dummy row per missing target class to avoid Evidently division-by-zero
        when a class has 0 support in `target`.
        IMPORTANT: labels must be strings to match proba_cols.
        """
        present = set(df["target"].unique())          # strings
        missing = sorted(set(proba_cols) - present, key=int)

        if not missing:
            return df

        extra = []
        for c in missing:
            row = {col: 0.0 for col in proba_cols}
            row[c] = 1.0
            row["pred"] = c
            row["target"] = c
            extra.append(row)

        return pd.concat([df, pd.DataFrame(extra)], ignore_index=True)


    # -------------------- OLD (reference) --------------------
    old_multi_proba = multi_model.predict_proba(old_train_multi)
    old_multi_pred  = np.argmax(old_multi_proba, axis=1)

    old_pred_df = pd.DataFrame(old_multi_proba, columns=proba_cols)
    old_pred_df["pred"] = old_multi_pred
    old_pred_df["target"] = old_target_train_multi

    # FORCE CONSISTENT LABEL SPACE: everything as string labels "0","1",...
    old_pred_df["pred"] = old_pred_df["pred"].astype(int).astype(str)
    old_pred_df["target"] = old_pred_df["target"].astype(int).astype(str)

    old_pred_df = add_missing_class_rows(old_pred_df, proba_cols, K)


    # -------------------- NEW (current) --------------------
    new_multi_proba = multi_model.predict_proba(new_data_multiclass)
    new_multi_pred  = np.argmax(new_multi_proba, axis=1)

    new_pred_df = pd.DataFrame(new_multi_proba, columns=proba_cols)
    new_pred_df["pred"] = new_multi_pred
    new_pred_df["target"] = trans_new_target_multi

    # FORCE CONSISTENT LABEL SPACE: everything as string labels "0","1",...
    new_pred_df["pred"] = new_pred_df["pred"].astype(int).astype(str)
    new_pred_df["target"] = new_pred_df["target"].astype(int).astype(str)

    new_pred_df = add_missing_class_rows(new_pred_df, proba_cols, K)


    # -------------------- Evidently Definition --------------------
    multi_data_def = DataDefinition(
        classification=[
            MulticlassClassification(
                target="target",
                prediction_labels="pred",
                prediction_probas=proba_cols
            )
        ],
        categorical_columns=["target", "pred"]
    )

    ref_ds = Dataset.from_pandas(old_pred_df, data_definition=multi_data_def)
    cur_ds = Dataset.from_pandas(new_pred_df, data_definition=multi_data_def)

    report = Report([ClassificationPreset()], include_tests=True)
    multi_quality_drift = report.run(current_data=cur_ds, reference_data=ref_ds)
    multi_quality_drift.save_html("Multi_Quality_Drift.html")
    logging.info("Saved successfully")
except Exception:
    logging.error('Issue creating the binary lable drift report')
logging.info('------------------------------------------------- Binary Entropy, Max_Confidence & Margin Drift -------------------------------------------------\n')
try:
    # old/new probabilities
    old_p1 = binary_model.predict_proba(old_train_binary)[:, 1]
    new_p1 = binary_model.predict_proba(new_data_binary)[:, 1]

    # convert to (n,2) to compute entropy consistently
    old_P = np.vstack([1 - old_p1, old_p1]).T
    new_P = np.vstack([1 - new_p1, new_p1]).T

    ref_df = pd.DataFrame({
        "p_class_1": old_p1,
        "entropy": entropy_from_proba(old_P),
        "max_conf": max_confidence(old_P),
        "margin": np.abs(old_p1 - 0.5)
    })

    cur_df = pd.DataFrame({
        "p_class_1": new_p1,
        "entropy": entropy_from_proba(new_P),
        "max_conf": max_confidence(new_P),
        "margin": np.abs(new_p1 - 0.5)
    })

    report = Report([
        DataDriftPreset(num_method="wasserstein", drift_share=0.25),  # 1/4 columns drifting triggers dataset drift
        ValueDrift(column="p_class_1", method="wasserstein", threshold=0.1),
        ValueDrift(column="entropy", method="wasserstein", threshold=0.1),
        ValueDrift(column="max_conf", method="wasserstein", threshold=0.1),
        ValueDrift(column="margin", method="wasserstein", threshold=0.1),
    ])
    binary_all_signals = report.run(cur_df, ref_df)
    binary_all_signals.save_html("Binary_AllSignals_Drift.html")
    logging.info("Saved successfully")
except Exception:
    logging.error(f'Issue creating the binary Entropy, Max_Confidence & Margin drift report')
logging.info('------------------------------------------------- Multiclass Entropy, Max_Confidence & Margin Drift -------------------------------------------------\n')
try:
    # reference (old)
    old_P = multi_model.predict_proba(old_train_multi)  # shape (n, K)

    # current (new)
    new_P = multi_model.predict_proba(new_data_multiclass)

    K = old_P.shape[1]
    proba_cols = [f"p_class_{i}" for i in range(K)]

    ref_multi = pd.DataFrame(old_P, columns=proba_cols)
    cur_multi = pd.DataFrame(new_P, columns=proba_cols)

    # derived signals
    ref_multi["entropy"] = entropy_from_proba(old_P)
    cur_multi["entropy"] = entropy_from_proba(new_P)

    ref_multi["max_conf"] = max_confidence(old_P)
    cur_multi["max_conf"] = max_confidence(new_P)

    ref_multi["margin"] = multiclass_margin(old_P)   # pmax - p2
    cur_multi["margin"] = multiclass_margin(new_P)

    # predicted label frequency drift (regime change)
    ref_multi["pred_label"] = np.argmax(old_P, axis=1).astype(str)
    cur_multi["pred_label"] = np.argmax(new_P, axis=1).astype(str)

    all_cols = proba_cols + ["entropy", "max_conf", "margin", "pred_label"]

    report = Report([
        DataDriftPreset(
            num_method="wasserstein",
            cat_method="jensenshannon",
            cat_threshold=0.1,
            drift_share=0.3
        ),
        ValueDrift(column="entropy", method="wasserstein", threshold=0.1),
        ValueDrift(column="max_conf", method="wasserstein", threshold=0.1),
        ValueDrift(column="margin", method="wasserstein", threshold=0.1),
    ])
    multiclass_allsignal = report.run(cur_multi[all_cols], ref_multi[all_cols])
    multiclass_allsignal.save_html("Multi_AllSignals_Drift.html")
    logging.info("Saved successfully")
except Exception:
    logging.error('Issue creating the multiclass Entropy, Max_Confidence & Margin drift report')

logging.info('All reports are created.')
#how to run it
#python -m src.Checking_for_data_drift