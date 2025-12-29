def run_multiclass_training():
    #---------------------------------------------------------------------------------- Importing libaries ----------------------------------------------------------------------------------------------------------------------------
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split,RepeatedStratifiedKFold,cross_val_score,StratifiedGroupKFold,learning_curve,LearningCurveDisplay,PredefinedSplit
    from sklearn.metrics import make_scorer,f1_score,log_loss,classification_report,confusion_matrix,ConfusionMatrixDisplay,top_k_accuracy_score,brier_score_loss,roc_auc_score,roc_curve,RocCurveDisplay,auc,recall_score,precision_score
    from sklearn.pipeline import Pipeline
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.feature_selection import SelectFromModel
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder,RobustScaler
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.preprocessing import label_binarize
    from sklearn.base import clone
    from lightgbm import LGBMClassifier
    from lightgbm.callback import early_stopping
    import xgboost
    from xgboost import XGBClassifier
    from feature_engine.encoding import RareLabelEncoder,OneHotEncoder,MeanEncoder
    from category_encoders import CatBoostEncoder
    import optuna
    from optuna.samplers import TPESampler,CmaEsSampler
    from optuna.pruners import SuccessiveHalvingPruner,HyperbandPruner
    from optuna.integration import LightGBMPruningCallback,XGBoostPruningCallback
    import optuna.visualization.matplotlib as optuna_matplotlib
    import joblib
    from joblib import Parallel, delayed, Memory
    import time
    import datetime
    import logging
    from pathlib import Path
    import sys
    #sys.path.append('..\..\src')
    from .PY_Class_Def import Feature_Selector_Classification,save_model_performance,ClubWeightadder,TopClubTransformer,LeagueWeightEncoder,TopLeagueTransformer,NewFeatureCreate,PositionScore,FixedLabelLogLoss,who_is_that_player,brier_eval_binary,brier_eval_xgb,learning_curve_weighted

    #---------------------------------------------------------------------------------- setting up the loging ----------------------------------------------------------------------------------------------------------------------------
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    #---------------------------------------------------------------------------------- loading the data ----------------------------------------------------------------------------------------------------------------------------
    ROOT = Path(__file__).resolve().parent.parent
    
    DATA_STATS_RUMORS = ROOT / "DataSources" / "Processed" / "Cleaned_Final_Stats_w_Rumors.csv"
    MODEL_TRAIN_TEST_MULTI = ROOT / "DataSources" / "Model_Data" / "Multiclass" / "Train_Test"
    MODEL_WHOLE_DATA_MULTI = ROOT / "DataSources" / "Model_Data" / "Multiclass" / "Whole_Data"
    MODEL_PARAMS_MODELS_MULTI = ROOT / "Models" / "Final_Model" / "Multiclass"

    #---------------------------------------------------------------------------------- loading the data ----------------------------------------------------------------------------------------------------------------------------
    logging.info('Loading the data set')
    try:
        df=pd.read_csv(DATA_STATS_RUMORS)
        top_5={'Premier League', 'Ligue 1', 'Serie A','Bundesliga', 'LaLiga'}
        df['Other_or_Top']=df['League_Joined'].isin(top_5).astype(int)
    except Exception as ex:
        logging.ERROR(f'The table was not able to be loaded {ex}')

    #---------------------------------------------------------------------------------- setting up X & y features ----------------------------------------------------------------------------------------------------------------------------
    logging.info('Filtering dataset for only top league transfers')
    try:
        df_top=df[df['Other_or_Top'].eq(1)]
    except Exception as ex:
        logging.ERROR(f'The tabel could not be generated {ex}')

    logging.info('Splitting dataset into X and y fearures; and setiing up grouped cross validation')
    X=df_top.drop(['Player','Squad','League','League_Joined','Club_Joined','Transfer_Fee','Other_or_Top'],axis=1).copy()
    y=df_top['League_Joined']
    #strong indicator to idenitfy data points and address duplicates so the model cannot learn from 
    players = df_top['Player'].str.strip().str.lower()

    #for the group CV
    #we need to create a identifier, which aussres that duplicated values are addressed in the cross validation
    #take the first split as train vs test
    #the smaller the n_splits, more evenlly the data split
    ##check the code below, to see how much data is allocated
    sgf_multi_outer = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=101)
    train_idx, test_idx = next(sgf_multi_outer.split(X, y, groups=players))
    X_train_all, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train_all, y_test = y.iloc[train_idx], y.iloc[test_idx]
    groups_train_all = players.iloc[train_idx].reset_index(drop=True)

    #the smaller the n_splits, more evenlly the data split
    ##check the code below, to see how much data is allocated
    sgf_multi_val = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=202)
    fit_idx, val_idx = next(sgf_multi_val.split(X_train_all, y_train_all, groups_train_all))
    X_train, X_val = X_train_all.iloc[fit_idx], X_train_all.iloc[val_idx]
    y_train, y_val = y_train_all.iloc[fit_idx], y_train_all.iloc[val_idx]
    groups_fit = groups_train_all.iloc[fit_idx].reset_index(drop=True)

    #we fit it on all the training data, so the encoder is familir with all possible labels
    le = LabelEncoder().fit(y_train_all)
    y_train_enc = le.transform(y_train)
    y_val_enc = le.transform(y_val)
    y_test_enc = le.transform(y_test)

    #setting up with the global class overview, so we can allow the model and evaluation metric to get an idea for the overall classes avaiable
    labels_global = le.classes_
    num_class=len(le.classes_)

    # 5) Inner CV for HPO (group-aware on FIT only)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=101)
    # -> use: for tr_idx, val_idx in cv.split(X_fit, y_fit, groups_fit): ...
    #    and keep all preprocessing inside your Pipeline per fold

    #creating a make_scorer, which addresses all classes
    log_loss_score=FixedLabelLogLoss(labels_global)

    #---------------------------------------------------------------------------------------- training the best model ----------------------------------------------------------------------------------------------------------------------------
    logging.info('Starting broad hyperparameter tuning')

    def stage1(trial):
        nation_tol=trial.suggest_float('nation_tol',0.001,0.01)
        club_left_tol=trial.suggest_float('club_left_tol',0.001,0.01)
        #league_left_tol=trial.suggest_float('league_left_tol',0.001,0.01)
        feature_select_thresh=trial.suggest_float('feature_select_thresh',0.0001,0.0085)
        

        # --- CatBoost-only params (no num_leaves; no max_leaves unless Lossguide)
        xgb_params = {
            #'booster': trial.suggest_categorical('booster', ['gbtree','dart']),
            'booster': 'gbtree',
            #'multi:softmax' -> looks for accuracy; 'multi:softprob' looks for probability
            'objective': 'multi:softprob', #trial.suggest_categorical('objective',['multi:softmax','multi:softprob']),
            #'n_estimators':8000,
            'num_class': num_class,
            'tree_method':trial.suggest_categorical('tree_method',['hist','approx']),
            'gamma': trial.suggest_float('gamma', 2.0, 7.0, log=True),
            'eta': trial.suggest_float('eta', 0.005, 0.012, log=True),
            'max_delta_step': trial.suggest_int('max_delta_step', 6, 10),
            'min_child_weight': trial.suggest_float('min_child_weight', 10.0, 40.0, log=True),
            'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise','lossguide']),
            'subsample': trial.suggest_float('subsample', 0.7, 0.95),
            'colsample_bytree': trial.suggest_float('colsample_bytree',0.6, 0.8, log=True),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel',0.6, 0.8, log=True),
            'colsample_bynode': trial.suggest_float('colsample_bynode',0.7, 0.9, log=True),
            'alpha': trial.suggest_float('alpha',0.5, 2.5, log=True),
            'lambda': trial.suggest_float('lambda',25.0, 85.0, log=True),
            'eval_metric':'mlogloss',
            'nthread':-1,
            'verbosity':0,
            'seed':101

        }
        if xgb_params['tree_method'] in ['hist','approx']:
            xgb_params['max_bin'] = trial.suggest_int('max_bin',64,84)
        if xgb_params['booster'] == 'dart':
            xgb_params['sample_type']= trial.suggest_categorical('sample_type',['uniform','weighted'])
            xgb_params['normalize_type']= trial.suggest_categorical('normalize_type',['tree','forest'])
            xgb_params['rate_drop']= trial.suggest_float('rate_drop',0.1,0.8)
            xgb_params['one_drop']= trial.suggest_int('one_drop',0,1)
            xgb_params['skip_drop']= trial.suggest_float('skip_drop',0.1,0.8)
        if xgb_params['grow_policy']=='depthwise':
            xgb_params['max_depth']=trial.suggest_int('max_depth',2,3)
        elif xgb_params['grow_policy']=='lossguide':
            xgb_params['max_leaves']=trial.suggest_int('max_leaves',12,28)
            xgb_params['max_depth']=0

        fold_losses,fold_ns,best_trees = [],[],[]
        #need to encode y features, otherwise XGBoost is not able to prrocess it

        # -------- NEW: per-class boost multipliers (around 1.0), tuned by Optuna --------
        # one multiplier per encoded class (use the global LabelEncoder `le` you already fit on y_train)
        m = np.array([trial.suggest_float(f"boost_{cls}", 0.8, 1.4) for cls in le.classes_], dtype=float)

        # optional: slightly tighter range for the dominant class in the TRAIN set
        dom = np.argmax(np.bincount(le.transform(y_train)))   # dominant encoded label in TRAIN
        m[dom] = trial.suggest_float(f"boost_{le.classes_[dom]}", 0.7, 1.2)

        # normalize so the average boost is ~1 (keeps overall scale stable)
        m = m / m.mean()

        # keep for later (so you can reuse best multipliers after the study)
        trial.set_user_attr("class_boosts", m.tolist())
        #------------------------------------------- Splitting training set into train and val, so we can use early stopping in xgboost ------------------------------------

        for tr_idx,val_idx in cv.split(X_train,y_train,groups_fit):
            X_tr,X_val=X_train.iloc[tr_idx],X_train.iloc[val_idx]
            y_tr,y_val=y_train.iloc[tr_idx],y_train.iloc[val_idx]

            base_pipe_xgb=Pipeline(
            [
                ('NewFeatures',NewFeatureCreate()),
                ('Club_weight', ClubWeightadder(columns=['Club_Left'],top_club_weights={'Manchester City': 27,'Real Madrid': 26,'FC Bayern München': 25,'FC Paris Saint-Germain': 24,'Inter Mailand': 23,'FC Barcelona': 22,
                                                    'FC Arsenal': 21,'FC Liverpool': 20,'Atlético Madrid': 19,'Borussia Dortmund': 18,'Juventus Turin': 17,'AC Mailand': 16,'Manchester United': 15,'RasenBallsport Leipzig': 14,
                                                    'FC Chelsea': 13,'SSC Neapel': 12,'Newcastle United': 11,'Bayer 04 Leverkusen': 10,'AS Monaco': 9,'Olympique Marseille': 8,'Ajax Amsterdam': 7,'FC Porto': 6,'Benfica Lissabon': 5,
                                                    'Sporting Lissabon': 4,'Galatasaray': 3,'Fenerbahce': 2,'Tottenham Hotspur': 1},non_top_weight=0.05)),
                ('TopClubs',TopClubTransformer(columns=[('Club_Left','League_Left')])),
                ('League_Weight',LeagueWeightEncoder(top_league_weights={'Premier League': 5,'LaLiga': 4,'Bundesliga': 3,'Serie A': 2,'Ligue 1': 1 },columns=['League_Left'])),
                ('TopLeague',TopLeagueTransformer(['League_Left'])),
                ('Nation Binning',RareLabelEncoder(variables='Nation',tol=nation_tol)),
                ('Club_Left Binning', RareLabelEncoder(variables='Club_Left',tol=club_left_tol)),
                ##('League_Left Binning', RareLabelEncoder(variables='League_Left',tol=0.001)),
                ('One-Hot Encoder',OneHotEncoder(variables=['Age','Pos'],drop_last=True)),
                ('Position_Score_FW', PositionScore(pos='FW', balanced_classes=True)),
                ('Position_Score_MF', PositionScore(pos='MF', balanced_classes=True)),
                ('Position_Score_DF', PositionScore(pos='DF', balanced_classes=True)),
                ('Position_Score_GK', PositionScore(pos='GK', balanced_classes=True)),
                ('Class Target Encoding',CatBoostEncoder(cols=['Nation','Club_Left','League_Left'],random_state=101)),
                ('Feature Selector Classification',Feature_Selector_Classification(threshold=feature_select_thresh, random_state=101)),
            ]
            )

            pipe = clone(base_pipe_xgb)
            #using not the encoded y, pipelien will deal with it
            X_tr_t = pipe.fit_transform(X_tr, y_tr)   # fit on train
            X_va_t = pipe.transform(X_val)

            #xgboost is not able to process the raw string values
            y_tr_enc=le.transform(y_tr)
            y_val_enc=le.transform(y_val)

            # -------- NEW: combine balanced base weights with tuned boosts ---------------
            # compute base class weights on THIS FOLD's training labels
            base_w = compute_class_weight(
                class_weight='balanced',
                classes=np.arange(num_class),
                y=y_tr_enc
            )
            # final per-class weights = base * boost
            class_w = base_w * m
            # per-sample weights for DMatrix
            w_tr = class_w[y_tr_enc]
        # ---------------------------------------------------------------------------

            dtrain=xgboost.DMatrix(X_tr_t,label=y_tr_enc,weight=w_tr)
            dvalid=xgboost.DMatrix(X_va_t,label=y_val_enc)

            bst=xgboost.train(
                params=xgb_params,
                dtrain=dtrain,
                num_boost_round=8000,
                #just (dvalid,'valid') -> we care only about 
                evals=[(dtrain,'train'),(dvalid,'valid')],
                callbacks=[xgboost.callback.EarlyStopping(rounds=250,save_best=True),XGBoostPruningCallback(trial,'valid-mlogloss')],
                verbose_eval=0
            )

            # predict probabilities for valid fold, due to 
            proba = bst.predict(dvalid, iteration_range=(0, bst.best_iteration + 1))
            loss = log_loss(y_val_enc, proba, labels=np.arange(num_class))
            fold_losses.append(loss)
            fold_ns.append(len(y_val_enc))
            best_trees.append(bst.best_iteration)
        
        # average CV loss
        weighted_cv_logloss = float(np.average(fold_losses, weights=fold_ns))

        # -------- OPTIONAL: regularize the boosts so they stay near 1.0 ----------------
        # a small L2 penalty on log(m) is stable; tune 0.005–0.02 if needed
        penalty = 0.01 * float(np.mean(np.log(m) ** 2))
        objective = weighted_cv_logloss + penalty
        # -------------------------------------------------------------------------------

        #adding the number of iterations
        if best_trees:
            trial.set_user_attr("n_estimators", int(np.mean(best_trees)))
        #return float(np.average(fold_losses, weights=fold_ns))
        return objective

    study_first_run=optuna.create_study(direction='minimize',sampler=TPESampler(multivariate=True,group=True,constant_liar=True,n_startup_trials=15,seed=101),
                                        pruner=SuccessiveHalvingPruner(min_resource=100,reduction_factor=3))
    start_time=time.time()
    study_first_run.optimize(stage1,n_trials=40)
    end_time=time.time()

    # best normalized boosts
    best_boosts = np.array(study_first_run.best_trial.user_attrs["class_boosts"], dtype=float)

    # balanced weights on TRAIN_ALL
    y_train_all_enc = le.transform(y_train_all)
    base_w = compute_class_weight('balanced', classes=np.arange(num_class), y=y_train_all_enc).astype(float)

    # final per-class weights
    final_class_w = np.clip(base_w * best_boosts, 0.5, 2.0)

    study1_params=study_first_run.best_params
    #study1_params['grow_policy']='SymmetricTree'
    study1_best_iter = study_first_run.best_trial.user_attrs["n_estimators"]
    #study1_params['objective']='multi:softprob'
    study1_params["n_estimators"] = study1_best_iter
    study1_params["_final_class_weights"] = final_class_w.tolist()
    study1_params["_classes"] = le.classes_.tolist()
    study1_params['booster']='gbtree'

    study1_score=study_first_run.best_value

    logging.info(f'Best Params Run 1: {study1_params}')
    logging.info(f'Best Score Run 1: {study1_score}')
    logging.info(f'Run Time Run 1: {(end_time - start_time)/60:.2f} Minutes\n')

    logging.info('Starting the narrow hyperparameter tuning')
    def stage2(trial):
        nation_tol=trial.suggest_float('nation_tol',(study1_params['nation_tol']-study1_params['nation_tol']*0.3),(study1_params['nation_tol']+study1_params['nation_tol']*0.3))
        club_left_tol=trial.suggest_float('club_left_tol',(study1_params['club_left_tol']-study1_params['club_left_tol']*0.3),(study1_params['club_left_tol']+study1_params['club_left_tol']*0.3))
        #league_left_tol=trial.suggest_float('league_left_tol',(study1_params['league_left_tol']-study1_params['league_left_tol']*0.3),(study1_params['league_left_tol']+study1_params['league_left_tol']*0.3))
        feature_select_thresh=trial.suggest_float('feature_select_thresh',(study1_params['feature_select_thresh']-study1_params['feature_select_thresh']*0.3),(study1_params['feature_select_thresh']+study1_params['feature_select_thresh']*0.3))

        # --- CatBoost-only params (no num_leaves; no max_leaves unless Lossguide)
        xgb_params = {
            'booster': study1_params['booster'],
            #'multi:softmax' -> looks for accuracy; 'multi:softprob' looks for probability
            'objective': 'multi:softprob', #trial.suggest_categorical('objective',['multi:softmax','multi:softprob']),
            'num_class': num_class,
            #'n_estimators':8000,
            'tree_method':study1_params['tree_method'],
            'gamma': trial.suggest_float('gamma',(study1_params['gamma']-study1_params['gamma']*0.3),(study1_params['gamma']+study1_params['gamma']*0.3), log=True),
            'eta': trial.suggest_float('eta',(study1_params['eta']-study1_params['eta']*0.3),(study1_params['eta']+study1_params['eta']*0.3), log=True),
            'max_delta_step': trial.suggest_int('max_delta_step', int(study1_params['max_delta_step']-study1_params['max_delta_step']*0.3),int(study1_params['max_delta_step']+study1_params['max_delta_step']*0.3)),
            'min_child_weight': trial.suggest_float('min_child_weight',(study1_params['min_child_weight']-study1_params['min_child_weight']*0.3),(study1_params['min_child_weight']+study1_params['min_child_weight']*0.3), log=True),
            'grow_policy': study1_params['grow_policy'],
            'subsample': trial.suggest_float('subsample',(study1_params['subsample']-study1_params['subsample']*0.3),(study1_params['subsample']+study1_params['subsample']*0.3), log=True),
            'colsample_bytree': trial.suggest_float('colsample_bytree', (study1_params['colsample_bytree']-study1_params['colsample_bytree']*0.3),(study1_params['colsample_bytree']+study1_params['colsample_bytree']*0.3), log=True),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel',(study1_params['colsample_bylevel']-study1_params['colsample_bylevel']*0.3),(study1_params['colsample_bylevel']+study1_params['colsample_bylevel']*0.3), log=True),
            'colsample_bynode': trial.suggest_float('colsample_bynode',(study1_params['colsample_bynode']-study1_params['colsample_bynode']*0.3),(study1_params['colsample_bynode']+study1_params['colsample_bynode']*0.3), log=True),
            'alpha': trial.suggest_float('alpha',(study1_params['alpha']-study1_params['alpha']*0.3),(study1_params['alpha']+study1_params['alpha']*0.3), log=True),
            'lambda': trial.suggest_float('lambda',(study1_params['lambda']-study1_params['lambda']*0.3),(study1_params['lambda']+study1_params['lambda']*0.3), log=True),
            'eval_metric':'mlogloss',
            'nthread':1,
            'verbosity':0,
            'seed':101

        }
        if xgb_params['tree_method'] in ['hist','approx']:
            xgb_params['max_bin'] = trial.suggest_int('max_bin',int(study1_params['max_bin']-study1_params['max_bin']*0.3),int(study1_params['max_bin']+study1_params['max_bin']*0.3))
        if xgb_params['booster'] == 'dart':
            xgb_params['sample_type']= trial.suggest_categorical('sample_type',['uniform','weighted'])
            xgb_params['normalize_type']= trial.suggest_categorical('normalize_type',['tree','forest'])
            xgb_params['rate_drop']= trial.suggest_float('rate_drop',(study1_params['rate_drop']-study1_params['rate_drop']*0.3),(study1_params['rate_drop']+study1_params['rate_drop']*0.3))
            xgb_params['one_drop']= trial.suggest_int('one_drop',0,1)
            xgb_params['skip_drop']= trial.suggest_float('skip_drop',(study1_params['skip_drop']-study1_params['skip_drop']*0.3),(study1_params['skip_drop']+study1_params['skip_drop']*0.3))
        if xgb_params['grow_policy']=='depthwise':
            xgb_params['max_depth']=trial.suggest_int('max_depth',int(study1_params['max_depth']-study1_params['max_depth']*0.3),int(study1_params['max_depth']+study1_params['max_depth']*0.3))
        elif xgb_params['grow_policy']=='lossguide':
            xgb_params['max_leaves']=trial.suggest_int('max_leaves',int(study1_params['max_leaves']-study1_params['max_leaves']*0.3),int(study1_params['max_leaves']+study1_params['max_leaves']*0.3))
            xgb_params['max_depth']=0

        if xgb_params['subsample']>1.0:
            xgb_params['subsample']=1.0
        if xgb_params['colsample_bytree']>1.0:
            xgb_params['colsample_bytree']=1.0
        if xgb_params['colsample_bylevel']>1.0:
            xgb_params['colsample_bylevel']=1.0
        if xgb_params['colsample_bynode']>1.0:
            xgb_params['colsample_bynode']=1.0

        fold_losses,fold_ns,best_trees = [],[],[]
        #need to encode y features, otherwise XGBoost is not able to prrocess it

        # -------- NEW: per-class boost multipliers (around 1.0), tuned by Optuna --------
        # one multiplier per encoded class (use the global LabelEncoder `le` you already fit on y_train)
        m = np.array([trial.suggest_float(f"boost_{cls}", 0.8, 1.4) for cls in le.classes_], dtype=float)

        # optional: slightly tighter range for the dominant class in the TRAIN set
        dom = np.argmax(np.bincount(le.transform(y_train)))   # dominant encoded label in TRAIN
        m[dom] = trial.suggest_float(f"boost_{le.classes_[dom]}", 0.7, 1.2)

        # normalize so the average boost is ~1 (keeps overall scale stable)
        m = m / m.mean()

        # keep for later (so you can reuse best multipliers after the study)
        trial.set_user_attr("class_boosts", m.tolist())
        # ------------------------------------------- Splitting training set into train and val, so we can use early stopping in xgboost ------------------------------------

        for tr_idx,val_idx in cv.split(X_train,y_train,groups_fit):
            X_tr,X_val=X_train.iloc[tr_idx],X_train.iloc[val_idx]
            y_tr,y_val=y_train.iloc[tr_idx],y_train.iloc[val_idx]

            base_pipe_xgb=Pipeline(
            [
                ('NewFeatures',NewFeatureCreate()),
                ('Club_weight', ClubWeightadder(columns=['Club_Left'],top_club_weights={'Manchester City': 27,'Real Madrid': 26,'FC Bayern München': 25,'FC Paris Saint-Germain': 24,'Inter Mailand': 23,'FC Barcelona': 22,
                                                    'FC Arsenal': 21,'FC Liverpool': 20,'Atlético Madrid': 19,'Borussia Dortmund': 18,'Juventus Turin': 17,'AC Mailand': 16,'Manchester United': 15,'RasenBallsport Leipzig': 14,
                                                    'FC Chelsea': 13,'SSC Neapel': 12,'Newcastle United': 11,'Bayer 04 Leverkusen': 10,'AS Monaco': 9,'Olympique Marseille': 8,'Ajax Amsterdam': 7,'FC Porto': 6,'Benfica Lissabon': 5,
                                                    'Sporting Lissabon': 4,'Galatasaray': 3,'Fenerbahce': 2,'Tottenham Hotspur': 1},non_top_weight=0.05)),
                ('TopClubs',TopClubTransformer(columns=[('Club_Left','League_Left')])),
                ('League_Weight',LeagueWeightEncoder(top_league_weights={'Premier League': 5,'LaLiga': 4,'Bundesliga': 3,'Serie A': 2,'Ligue 1': 1 },columns=['League_Left'])),
                ('TopLeague',TopLeagueTransformer(['League_Left'])),
                ('Nation Binning',RareLabelEncoder(variables='Nation',tol=nation_tol)),
                ('Club_Left Binning', RareLabelEncoder(variables='Club_Left',tol=club_left_tol)),
                ##('League_Left Binning', RareLabelEncoder(variables='League_Left',tol=0.001)),
                ('One-Hot Encoder',OneHotEncoder(variables=['Age','Pos'],drop_last=True)),
                ('Position_Score_FW', PositionScore(pos='FW', balanced_classes=True)),
                ('Position_Score_MF', PositionScore(pos='MF', balanced_classes=True)),
                ('Position_Score_DF', PositionScore(pos='DF', balanced_classes=True)),
                ('Position_Score_GK', PositionScore(pos='GK', balanced_classes=True)),
                ('Class Target Encoding',CatBoostEncoder(cols=['Nation','Club_Left','League_Left'],random_state=101)),
                ('Feature Selector Classification',Feature_Selector_Classification(threshold=feature_select_thresh, random_state=101))
            ]
            )

            pipe = clone(base_pipe_xgb)
            #using not the encoded y, pipelien will deal with it
            X_tr_t = pipe.fit_transform(X_tr, y_tr)   # fit on train
            X_va_t = pipe.transform(X_val)

            #xgboost is not able to process the raw string values
            y_tr_enc=le.transform(y_tr)
            y_val_enc=le.transform(y_val)

            # -------- NEW: combine balanced base weights with tuned boosts ---------------
            # compute base class weights on THIS FOLD's training labels
            base_w = compute_class_weight(
                class_weight='balanced',
                classes=np.arange(num_class),
                y=y_tr_enc
            )
            # final per-class weights = base * boost
            class_w = base_w * m
            # per-sample weights for DMatrix
            w_tr = class_w[y_tr_enc]
        # ---------------------------------------------------------------------------

            dtrain=xgboost.DMatrix(X_tr_t,label=y_tr_enc,weight=w_tr)
            dvalid=xgboost.DMatrix(X_va_t,label=y_val_enc)

            bst=xgboost.train(
                params=xgb_params,
                dtrain=dtrain,
                num_boost_round=8000,
                #just (dvalid,'valid') -> we care only about 
                evals=[(dtrain,'train'),(dvalid,'valid')],
                callbacks=[xgboost.callback.EarlyStopping(rounds=250,save_best=True),XGBoostPruningCallback(trial,'valid-mlogloss')],
                verbose_eval=0
            )

            # predict probabilities for valid fold, due to 
            proba = bst.predict(dvalid, iteration_range=(0, bst.best_iteration + 1))
            loss = log_loss(y_val_enc, proba, labels=np.arange(num_class))
            fold_losses.append(loss)
            fold_ns.append(len(y_val_enc))
            best_trees.append(bst.best_iteration)
        
        # average CV loss
        weighted_cv_logloss = float(np.average(fold_losses, weights=fold_ns))

        # -------- OPTIONAL: regularize the boosts so they stay near 1.0 ----------------
        # a small L2 penalty on log(m) is stable; tune 0.005–0.02 if needed
        penalty = 0.01 * float(np.mean(np.log(m) ** 2))
        objective = weighted_cv_logloss + penalty
        # -------------------------------------------------------------------------------

        #adding the number of iterations
        if best_trees:
            trial.set_user_attr("n_estimators", int(np.mean(best_trees)))
        #return float(np.average(fold_losses, weights=fold_ns))
        return objective

    study_second_run=optuna.create_study(direction='minimize',sampler=TPESampler(multivariate=True,group=True,constant_liar=True,n_startup_trials=15,seed=101),
                                        pruner=SuccessiveHalvingPruner(min_resource=100,reduction_factor=3))
    start_time=time.time()
    study_second_run.optimize(stage2,n_trials=60,n_jobs=-1)
    end_time=time.time()

    # best normalized boosts
    best_boosts = np.array(study_first_run.best_trial.user_attrs["class_boosts"], dtype=float)

    # balanced weights on TRAIN_ALL
    y_train_all_enc = le.transform(y_train_all)
    base_w = compute_class_weight('balanced', classes=np.arange(num_class), y=y_train_all_enc).astype(float)

    # final per-class weights
    final_class_w = np.clip(base_w * best_boosts, 0.5, 2.0)

    study2_params=study_second_run.best_params
    study2_params['booster']=study1_params['booster']
    study2_params['objective']='multi:softprob'
    study2_params['grow_policy']=study1_params['grow_policy']
    study2_params['tree_method']=study1_params['tree_method']
    study2_params["_final_class_weights"] = final_class_w.tolist()
    study2_params["_classes"] = le.classes_.tolist()
    study2_best_iter = study_second_run.best_trial.user_attrs["n_estimators"]
    study2_params["n_estimators"] = study2_best_iter

    study2_score=study_second_run.best_value

    logging.info(f'Best Params of Run 2: {study2_params}')
    logging.info(f'Best Score of Run 2: {study2_score}')
    logging.info(f'Run Time of Run 2: {(end_time - start_time)/60:.2f} Minutes\n')

    final_params = study1_params if study_first_run.best_value < study_second_run.best_value else study2_params
    best_score=study_first_run.best_value if study_first_run.best_value < study_second_run.best_value else study_second_run.best_value
    xgboost_multi_loss=study_first_run.trials_dataframe()['value'] if study_first_run.best_value < study_second_run.best_value else study_second_run.trials_dataframe()['value']

    logging.info(f'Best Score of Run 1 {study1_score}')
    logging.info(f'Best Score of Run 2 {study2_score}')
    logging.info(f'Basline Log Loss Score that the model needs to beat: {-np.log(1/num_class)}\n')

    #---------------------------------------------------------------------------------------- learning curve for xgboost which addresses the different weights of the classes ----------------------------------------------------------------------------------------------------------------------------
    logging.info('Creating the learning curve')
    labels_fixed = np.arange(num_class)

    def neg_log_loss_fixed(est, X, y):
        proba = est.predict_proba(X)
        return -log_loss(y, proba, labels=labels_fixed)  # sklearn expects higher-is-better
    from sklearn.preprocessing import FunctionTransformer
    ToNumpy = FunctionTransformer(lambda X: X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X))

    xgb_params = {
            'booster': final_params['booster'],
            #'multi:softmax' -> looks for accuracy; 'multi:softprob' looks for probability
            'objective': 'multi:softprob', #trial.suggest_categorical('objective',['multi:softmax','multi:softprob']),
            'n_estimators':final_params['n_estimators'],
            #'num_class': num_class,
            'tree_method':final_params['tree_method'],
            'gamma': final_params['gamma'],
            'eta': final_params['eta'],
            'max_delta_step': final_params['max_delta_step'],
            'min_child_weight': final_params['min_child_weight'],
            'grow_policy': final_params['grow_policy'],
            'subsample': final_params['subsample'],
            'colsample_bytree': final_params['colsample_bytree'],
            'colsample_bylevel': final_params['colsample_bylevel'],
            'colsample_bynode': final_params['colsample_bynode'],
            'alpha': final_params['alpha'],
            'lambda': final_params['lambda'],
            'eval_metric':'mlogloss',
            #'nthread':-1,
            'verbosity':0,
            #'seed':101

        }
    if xgb_params['tree_method'] in ['hist','approx']:
        xgb_params['max_bin'] = final_params['max_bin']
    if xgb_params['booster'] == 'dart':
        xgb_params['sample_type']= final_params['sample_type']
        xgb_params['normalize_type']= final_params['normalize_type']
        xgb_params['rate_drop']= final_params['rate_drop']
        xgb_params['one_drop']= final_params['one_drop']
        xgb_params['skip_drop']= final_params['skip_drop']
    if xgb_params['grow_policy']=='depthwise':
        xgb_params['max_depth']=final_params['max_depth']
    elif xgb_params['grow_policy']=='lossguide':
        xgb_params['max_leaves']=final_params['max_leaves']
        xgb_params['max_depth']=0

    if xgb_params['subsample']>1.0:
        xgb_params['subsample']=1.0
    if xgb_params['colsample_bytree']>1.0:
        xgb_params['colsample_bytree']=1.0
    if xgb_params['colsample_bylevel']>1.0:
        xgb_params['colsample_bylevel']=1.0
    if xgb_params['colsample_bynode']>1.0:
        xgb_params['colsample_bynode']=1.0

    xgboost_final_pipeline=Pipeline(
            [
                ('NewFeatures',NewFeatureCreate()),
                ('Club_weight', ClubWeightadder(columns=['Club_Left'],top_club_weights={'Manchester City': 27,'Real Madrid': 26,'FC Bayern München': 25,'FC Paris Saint-Germain': 24,'Inter Mailand': 23,'FC Barcelona': 22,
                                                    'FC Arsenal': 21,'FC Liverpool': 20,'Atlético Madrid': 19,'Borussia Dortmund': 18,'Juventus Turin': 17,'AC Mailand': 16,'Manchester United': 15,'RasenBallsport Leipzig': 14,
                                                    'FC Chelsea': 13,'SSC Neapel': 12,'Newcastle United': 11,'Bayer 04 Leverkusen': 10,'AS Monaco': 9,'Olympique Marseille': 8,'Ajax Amsterdam': 7,'FC Porto': 6,'Benfica Lissabon': 5,
                                                    'Sporting Lissabon': 4,'Galatasaray': 3,'Fenerbahce': 2,'Tottenham Hotspur': 1},non_top_weight=0.05)),
                ('TopClubs',TopClubTransformer(columns=[('Club_Left','League_Left')])),
                ('League_Weight',LeagueWeightEncoder(top_league_weights={'Premier League': 5,'LaLiga': 4,'Bundesliga': 3,'Serie A': 2,'Ligue 1': 1 },columns=['League_Left'])),
                ('TopLeague',TopLeagueTransformer(['League_Left'])),
                ('Nation Binning',RareLabelEncoder(variables='Nation',tol=final_params['nation_tol'])),
                ('Club_Left Binning', RareLabelEncoder(variables='Club_Left',tol=final_params['club_left_tol'])),
                ##('League_Left Binning', RareLabelEncoder(variables='League_Left',tol=0.001)),
                ('One-Hot Encoder',OneHotEncoder(variables=['Age','Pos'],drop_last=True)),
                ('Position_Score_FW', PositionScore(pos='FW', balanced_classes=True)),
                ('Position_Score_MF', PositionScore(pos='MF', balanced_classes=True)),
                ('Position_Score_DF', PositionScore(pos='DF', balanced_classes=True)),
                ('Position_Score_GK', PositionScore(pos='GK', balanced_classes=True)),
                ('Class Target Encoding',CatBoostEncoder(cols=['Nation','Club_Left','League_Left'],random_state=101)),
                ('Feature Selector Classification',Feature_Selector_Classification(threshold=final_params['feature_select_thresh'], random_state=101)),
                #('ToNumpy', ToNumpy),  # <- ensure pure numeric array
                ('XGBoost Classifier',XGBClassifier(**xgb_params,n_jobs=-1,random_state=101))
            ]
    )

    ##We need to replace that, with a custom leanring_curve, we always need to use the custom/manual learning curve when passing weights no matter which model
    ##the other models where using parameters which tackeled these issues, with multiclass XGBoost we need to address weights via fit_params

    # from stage 2 (or final_params), already saved earlier:
    final_class_w = np.array(final_params["_final_class_weights"], dtype=float)
    classes_saved = final_params["_classes"]

    # sanity: ensure order matches the LabelEncoder used everywhere
    assert list(classes_saved) == list(le.classes_), "Class order mismatch!"

    # encode y for the data you're feeding to learning_curve (you already did this)
    # y_train_enc = le.transform(y_train)

    # per-sample weights for *the same rows you pass to learning_curve* (X_train, y_train_enc)
    w_train = final_class_w[y_train_enc]   # shape (len(y_train_enc),)

    fit_params = {'XGBoost Classifier__sample_weight': w_train}

    final_class_w = np.array(final_params["_final_class_weights"], dtype=float)
    w_train_full = final_class_w[y_train_enc]

    # 3) Run weighted learning curve
    ts_abs, tr_scores, va_scores = learning_curve_weighted(
        estimator=xgboost_final_pipeline,
        X=X_train,
        y=y_train_enc,
        groups=groups_fit,
        cv=cv,                                # your StratifiedGroupKFold
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring=neg_log_loss_fixed,                         # default: neg_log_loss
        labels=None,          # important: fixed label order
        sample_weight_full=w_train_full,      # <- weights for CURRENT X/y
        sample_weight_param='XGBoost Classifier__sample_weight',
        #extra_fit_params=extra_fit,
        use_early_stopping=False,              # if you want ES per fold
        random_state=101
    )

    for train_size, cv_train_scores, cv_val_scores in zip(ts_abs, tr_scores, va_scores):
        logging.info(f'Train Size: {np.mean(train_size)} | Train Score: {np.mean(cv_train_scores)*-1} | Val Score: {np.mean(cv_val_scores)*-1}')

    #------------------------------------------------------------------------- evaluating final model ------------------------------------------------------------------------------------

    year=datetime.date.today().year
    month=datetime.date.today().month
    day=datetime.date.today().day

    #saving the final dictionary, with all the important/needed values for the model pipeline
    joblib.dump(final_params, MODEL_PARAMS_MODELS_MULTI / f'multiclass_final_params_{day}_{month}_{year}.dict')

    #getting the, breaking down groups_train_all into group_val, groups_train_all has already been broken down into group_fit & finally into group_val
    ##why can't we just use groups_train_all? Due to the index splits for group_fit and group_val
    groups_val = groups_train_all.iloc[val_idx].reset_index(drop=True)

    X_train_val=pd.concat([X_train,X_val],axis=0).reset_index(drop=True)
    #y features is at that point an array, instead of pd.concat() we use np.concatenate()
    y_train_val_enc = np.concatenate((y_train_enc, y_val_enc), axis=0)
    groups_train_val = pd.concat([groups_fit, groups_val], axis=0).reset_index(drop=True)

    joblib.dump(X_train_val,MODEL_TRAIN_TEST_MULTI / 'X_Train_Data.CSV')
    joblib.dump(y_train_val_enc, MODEL_TRAIN_TEST_MULTI / 'y_Train_Data.CSV')
    joblib.dump(X_test,MODEL_TRAIN_TEST_MULTI / 'X_Test_Data.CSV')
    joblib.dump(y_test_enc,MODEL_TRAIN_TEST_MULTI / 'y_Test_Data.CSV')

    #create group-aware folds for calibration
    sgkf_cal = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=101)
    ##create an array that will store which fold each row belongs to => start by marking every sample as -1 (meaning "not assigned yet")
    ##after the loop, fold_ids[i] will contain a value 0–4 depending on which fold index row i belongs to
    fold_ids = -np.ones(len(X_train_val), dtype=int)   # -1 means "never validation"
    ##we only use val_index because for PredefinedSplit we label which rows will be validation in each fold
    ##for each fold, we assign the fold number to those validation rows => enumerate()
    for fold, (_, val_index) in enumerate(
            sgkf_cal.split(X_train_val, y_train_val_enc, groups_train_val)):
        fold_ids[val_index] = fold

    assert (fold_ids >= 0).all(), "Some samples not assigned to a fold."

    #PredefinedSplit will make sure CalibratedClassifierCV respects your group folds
    ps = PredefinedSplit(test_fold=fold_ids)

    # pulled from your saved final_params / stage 2 artifacts
    final_class_w = np.array(final_params["_final_class_weights"], dtype=float)
    classes_saved = final_params["_classes"]
    assert list(classes_saved) == list(le.classes_), "Class order mismatch!"

    # weights for the actual training and validation rows
    w_train_val = final_class_w[y_train_val_enc]

    xgb_params = {
            'booster': final_params['booster'],
            #'multi:softmax' -> looks for accuracy; 'multi:softprob' looks for probability
            'objective': 'multi:softprob', #trial.suggest_categorical('objective',['multi:softmax','multi:softprob']),
            'n_estimators':final_params['n_estimators'],
            #'num_class': num_class,
            'tree_method':final_params['tree_method'],
            'gamma': final_params['gamma'],
            'eta': final_params['eta'],
            'max_delta_step': final_params['max_delta_step'],
            'min_child_weight': final_params['min_child_weight'],
            'grow_policy': final_params['grow_policy'],
            'subsample': final_params['subsample'],
            'colsample_bytree': final_params['colsample_bytree'],
            'colsample_bylevel': final_params['colsample_bylevel'],
            'colsample_bynode': final_params['colsample_bynode'],
            'alpha': final_params['alpha'],
            'lambda': final_params['lambda'],
            #'eval_metric':'mlogloss',
            #'nthread':-1,
            'verbosity':0,
            #'seed':101

        }
    if xgb_params['tree_method'] in ['hist','approx']:
        xgb_params['max_bin'] = final_params['max_bin']
    if xgb_params['booster'] == 'dart':
        xgb_params['sample_type']= final_params['sample_type']
        xgb_params['normalize_type']= final_params['normalize_type']
        xgb_params['rate_drop']= final_params['rate_drop']
        xgb_params['one_drop']= final_params['one_drop']
        xgb_params['skip_drop']= final_params['skip_drop']
    if xgb_params['grow_policy']=='depthwise':
        xgb_params['max_depth']=final_params['max_depth']
    elif xgb_params['grow_policy']=='lossguide':
        xgb_params['max_leaves']=final_params['max_leaves']
        xgb_params['max_depth']=0

    if xgb_params['subsample']>1.0:
        xgb_params['subsample']=1.0
    if xgb_params['colsample_bytree']>1.0:
        xgb_params['colsample_bytree']=1.0
    if xgb_params['colsample_bylevel']>1.0:
        xgb_params['colsample_bylevel']=1.0
    if xgb_params['colsample_bynode']>1.0:
        xgb_params['colsample_bynode']=1.0

    xgboost_final_pipeline=Pipeline(
            [
                ('NewFeatures',NewFeatureCreate()),
                ('Club_weight', ClubWeightadder(columns=['Club_Left'],top_club_weights={'Manchester City': 27,'Real Madrid': 26,'FC Bayern München': 25,'FC Paris Saint-Germain': 24,'Inter Mailand': 23,'FC Barcelona': 22,
                                                    'FC Arsenal': 21,'FC Liverpool': 20,'Atlético Madrid': 19,'Borussia Dortmund': 18,'Juventus Turin': 17,'AC Mailand': 16,'Manchester United': 15,'RasenBallsport Leipzig': 14,
                                                    'FC Chelsea': 13,'SSC Neapel': 12,'Newcastle United': 11,'Bayer 04 Leverkusen': 10,'AS Monaco': 9,'Olympique Marseille': 8,'Ajax Amsterdam': 7,'FC Porto': 6,'Benfica Lissabon': 5,
                                                    'Sporting Lissabon': 4,'Galatasaray': 3,'Fenerbahce': 2,'Tottenham Hotspur': 1},non_top_weight=0.05)),
                ('TopClubs',TopClubTransformer(columns=[('Club_Left','League_Left')])),
                ('League_Weight',LeagueWeightEncoder(top_league_weights={'Premier League': 5,'LaLiga': 4,'Bundesliga': 3,'Serie A': 2,'Ligue 1': 1 },columns=['League_Left'])),
                ('TopLeague',TopLeagueTransformer(['League_Left'])),
                ('Nation Binning',RareLabelEncoder(variables='Nation',tol=final_params['nation_tol'])),
                ('Club_Left Binning', RareLabelEncoder(variables='Club_Left',tol=final_params['club_left_tol'])),
                ##('League_Left Binning', RareLabelEncoder(variables='League_Left',tol=0.001)),
                ('One-Hot Encoder',OneHotEncoder(variables=['Age','Pos'],drop_last=True)),
                ('Position_Score_FW', PositionScore(pos='FW', balanced_classes=True)),
                ('Position_Score_MF', PositionScore(pos='MF', balanced_classes=True)),
                ('Position_Score_DF', PositionScore(pos='DF', balanced_classes=True)),
                ('Position_Score_GK', PositionScore(pos='GK', balanced_classes=True)),
                ('Class Target Encoding',CatBoostEncoder(cols=['Nation','Club_Left','League_Left'],random_state=101)),
                ('Feature Selector Classification',Feature_Selector_Classification(threshold=final_params['feature_select_thresh'], random_state=101)),
                ('XGBoost Classifier',XGBClassifier(**xgb_params,n_jobs=-1,random_state=101))
            ]
    )

    ccv=CalibratedClassifierCV(estimator=xgboost_final_pipeline,method='sigmoid',cv=ps)
    ccv.fit(X_train_val,y_train_val_enc,**{"XGBoost Classifier__sample_weight": w_train_val})

    y_proba=ccv.predict_proba(X_test)
    y_pred=ccv.predict(X_test)

    #pos_idx = list(ccv.classes_).index(1)

    logging.info(f'Log Loss Score: {log_loss(y_test_enc,y_proba,labels=ccv.classes_)}')

    logging.info(f'Macro F1-Score: {f1_score(y_test_enc,y_pred,average="macro")}')

    logging.info(classification_report(y_test_enc,y_pred))

    #binarize the true labels for multiclass ROC
    y_test_bin = label_binarize(y_test_enc, classes=ccv.classes_)  # shape (n_samples, n_classes)
    n_classes = y_test_bin.shape[1]

    #compute per-class ROC curves and AUCs
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    #compute macro-average ROC
    #1. aggregate all FPR points
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    #2. interpolate mean TPR at each FPR
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes

    #3. compute macro AUC
    roc_auc["macro"] = auc(all_fpr, mean_tpr)
    logging.info(f'ROC-AUC Macro: {roc_auc["macro"]}')
    #ink_classes={v:k for k,v in ccv.classes_.items()}
    for i in range(n_classes):
        logging.info(f'{ccv.classes_[i]}: {roc_auc[i]:.3f}')

    #top-1 (normal accuracy)
    acc1 = top_k_accuracy_score(y_test_enc, y_proba, k=1,labels=ccv.classes_)

    #top-2
    acc3 = top_k_accuracy_score(y_test_enc, y_proba, k=2,labels=ccv.classes_)

    #top-3
    acc5 = top_k_accuracy_score(y_test_enc, y_proba, k=3,labels=ccv.classes_)

    logging.info(f"Top-1: {acc1}")
    logging.info(f"Top-2: {acc3}")
    logging.info(f"Top-3: {acc5}\n")

    joblib.dump(ccv, MODEL_PARAMS_MODELS_MULTI / f'multiclass_final_model_{day}_{month}_{year}.pkl')

    label_to_code = {lbl: i for i, lbl in enumerate(le.classes_)}
    joblib.dump(label_to_code, MODEL_PARAMS_MODELS_MULTI / 'multiclass_translation.dict')

    #----------------------------------------------------------------------------- setting up final model just fitting data, no test set remaining -------------------------------------------------------------------------
    final_params=joblib.load(MODEL_PARAMS_MODELS_MULTI / 'multiclass_final_params.dict')
    year=datetime.date.today().year
    month=datetime.date.today().month
    day=datetime.date.today().day

    X_multi=df_top.drop(['Player','Squad','League','League_Joined','Club_Joined','Transfer_Fee','Other_or_Top'],axis=1).copy()
    joblib.dump(list(X_multi.columns),MODEL_PARAMS_MODELS_MULTI/ 'multiclass_model_columns.joblib')
    y_multi=df_top['League_Joined']

    le = LabelEncoder()
    y_multi_enc=le.fit_transform(y_multi)

    joblib.dump(X_multi,MODEL_WHOLE_DATA_MULTI / 'X_Feat.CSV')
    joblib.dump(y_multi_enc,MODEL_WHOLE_DATA_MULTI / 'y_Feat.CSV')

    #pulled from your saved final_params / stage 2 artifacts
    final_class_w = np.array(final_params["_final_class_weights"], dtype=float)
    classes_saved = final_params["_classes"]
    assert list(classes_saved) == list(le.classes_), "Class order mismatch!"

    #weights for the actual training and validation rows
    w_multi = final_class_w[y_multi_enc]

    xgb_params = {
            'booster': final_params['booster'],
            #'multi:softmax' -> looks for accuracy; 'multi:softprob' looks for probability
            'objective': 'multi:softprob', #trial.suggest_categorical('objective',['multi:softmax','multi:softprob']),
            'n_estimators':final_params['n_estimators'],
            #'num_class': num_class,
            'tree_method':final_params['tree_method'],
            'gamma': final_params['gamma'],
            'eta': final_params['eta'],
            'max_delta_step': final_params['max_delta_step'],
            'min_child_weight': final_params['min_child_weight'],
            'grow_policy': final_params['grow_policy'],
            'subsample': final_params['subsample'],
            'colsample_bytree': final_params['colsample_bytree'],
            'colsample_bylevel': final_params['colsample_bylevel'],
            'colsample_bynode': final_params['colsample_bynode'],
            'alpha': final_params['alpha'],
            'lambda': final_params['lambda'],
            #'eval_metric':'mlogloss',
            #'nthread':-1,
            'verbosity':0,
            #'seed':101

        }
    if xgb_params['tree_method'] in ['hist','approx']:
        xgb_params['max_bin'] = final_params['max_bin']
    if xgb_params['booster'] == 'dart':
        xgb_params['sample_type']= final_params['sample_type']
        xgb_params['normalize_type']= final_params['normalize_type']
        xgb_params['rate_drop']= final_params['rate_drop']
        xgb_params['one_drop']= final_params['one_drop']
        xgb_params['skip_drop']= final_params['skip_drop']
    if xgb_params['grow_policy']=='depthwise':
        xgb_params['max_depth']=final_params['max_depth']
    elif xgb_params['grow_policy']=='lossguide':
        xgb_params['max_leaves']=final_params['max_leaves']
        xgb_params['max_depth']=0

    if xgb_params['subsample']>1.0:
        xgb_params['subsample']=1.0
    if xgb_params['colsample_bytree']>1.0:
        xgb_params['colsample_bytree']=1.0
    if xgb_params['colsample_bylevel']>1.0:
        xgb_params['colsample_bylevel']=1.0
    if xgb_params['colsample_bynode']>1.0:
        xgb_params['colsample_bynode']=1.0

    xgboost_final_pipeline=Pipeline(
            [
                ('NewFeatures',NewFeatureCreate()),
                ('Club_weight', ClubWeightadder(columns=['Club_Left'],top_club_weights={'Manchester City': 27,'Real Madrid': 26,'FC Bayern München': 25,'FC Paris Saint-Germain': 24,'Inter Mailand': 23,'FC Barcelona': 22,
                                                    'FC Arsenal': 21,'FC Liverpool': 20,'Atlético Madrid': 19,'Borussia Dortmund': 18,'Juventus Turin': 17,'AC Mailand': 16,'Manchester United': 15,'RasenBallsport Leipzig': 14,
                                                    'FC Chelsea': 13,'SSC Neapel': 12,'Newcastle United': 11,'Bayer 04 Leverkusen': 10,'AS Monaco': 9,'Olympique Marseille': 8,'Ajax Amsterdam': 7,'FC Porto': 6,'Benfica Lissabon': 5,
                                                    'Sporting Lissabon': 4,'Galatasaray': 3,'Fenerbahce': 2,'Tottenham Hotspur': 1},non_top_weight=0.05)),
                ('TopClubs',TopClubTransformer(columns=[('Club_Left','League_Left')])),
                ('League_Weight',LeagueWeightEncoder(top_league_weights={'Premier League': 5,'LaLiga': 4,'Bundesliga': 3,'Serie A': 2,'Ligue 1': 1 },columns=['League_Left'])),
                ('TopLeague',TopLeagueTransformer(['League_Left'])),
                ('Nation Binning',RareLabelEncoder(variables='Nation',tol=final_params['nation_tol'])),
                ('Club_Left Binning', RareLabelEncoder(variables='Club_Left',tol=final_params['club_left_tol'])),
                ##('League_Left Binning', RareLabelEncoder(variables='League_Left',tol=0.001)),
                ('One-Hot Encoder',OneHotEncoder(variables=['Age','Pos'],drop_last=True)),
                ('Position_Score_FW', PositionScore(pos='FW', balanced_classes=True)),
                ('Position_Score_MF', PositionScore(pos='MF', balanced_classes=True)),
                ('Position_Score_DF', PositionScore(pos='DF', balanced_classes=True)),
                ('Position_Score_GK', PositionScore(pos='GK', balanced_classes=True)),
                ('Class Target Encoding',CatBoostEncoder(cols=['Nation','Club_Left','League_Left'],random_state=101)),
                ('Feature Selector Classification',Feature_Selector_Classification(threshold=final_params['feature_select_thresh'], random_state=101)),
                ('XGBoost Classifier',XGBClassifier(**xgb_params,n_jobs=-1,random_state=101))
            ]
    )

    ccv=CalibratedClassifierCV(estimator=xgboost_final_pipeline,method='sigmoid',cv=5)
    ccv.fit(X_multi,y_multi_enc,**{"XGBoost Classifier__sample_weight": w_multi})

    joblib.dump(ccv, MODEL_PARAMS_MODELS_MULTI / f'multiclass_final_model_total_data.pkl')

if __name__ == '__main__':
    run_multiclass_training()