def run_binary_training():
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
    from optuna.samplers import TPESampler
    from optuna.pruners import SuccessiveHalvingPruner
    from optuna.integration import LightGBMPruningCallback,XGBoostPruningCallback
    import optuna.visualization.matplotlib as optuna_matplotlib
    import joblib
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
    MODEL_TRAIN_TEST_BINARY = ROOT / "DataSources" / "Model_Data" / "Binary" / "Train_Test"
    MODEL_WHOLE_DATA_BINARY = ROOT / "DataSources" / "Model_Data" / "Binary" / "Whole_Data"
    MODEL_PARAMS_MODELS_BINARY = ROOT / "Models" / "Final_Model" / "Binary"
    MODEL_PARAMS_MULTI = ROOT / "Models" / "Final_Model" / "Multiclass"

    

    #---------------------------------------------------------------------------------- loading the data ----------------------------------------------------------------------------------------------------------------------------
    logging.info('Loading the data set')
    try:
        df=pd.read_csv(DATA_STATS_RUMORS)
        top_5={'Premier League', 'Ligue 1', 'Serie A','Bundesliga', 'LaLiga'}
        df['Other_or_Top']=df['League_Joined'].isin(top_5).astype(int)
    except Exception as ex:
        logging.ERROR(f'The table was not able to be loaded {ex}')

    #---------------------------------------------------------------------------------- loading the data ----------------------------------------------------------------------------------------------------------------------------
    logging.info("Splitting the data into X and target features")
    X=df.drop(['Player','Squad','League','League_Joined','Club_Joined','Transfer_Fee','Other_or_Top'],axis=1).copy()
    y=df['Other_or_Top']
    logging.info("Setting up the reoccuring player names for the grouped-cv")
    players=df['Player'].str.strip().str.lower()

    sgf_train_test=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=101)

    logging.info("Creating the train, val & test set splits")
    #next, we will not use the standard split, we will do it ourselves
    ##Why? Due to the fact that we will predict probabilities, it is recommended to use CalibratedClassifierCV, therefore we will split the data into train, validation and test sets
    ###split into train & test, leave test as it is and use train for the next split
    ###80% train & 20% test
    train_idx,test_idx=next(sgf_train_test.split(X,y,groups=players))
    X_train_all,X_test=X.iloc[train_idx],X.iloc[test_idx]
    y_train_all,y_test=y.iloc[train_idx],y.iloc[test_idx]
    ####adding the player name, so the fold can make sure that not the same player name can show up, dataframe will be filtered by the train index
    groups_train_all=players.iloc[train_idx].reset_index(drop=True)

    ###creating the train and validation set
    sgkf_train_val = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=101)
    fit_idx, val_idx = next(sgkf_train_val.split(X_train_all, y_train_all, groups_train_all))
    X_train, X_val = X_train_all.iloc[fit_idx], X_train_all.iloc[val_idx]
    y_train, y_val = y_train_all.iloc[fit_idx], y_train_all.iloc[val_idx]
    groups_fit = groups_train_all.iloc[fit_idx].reset_index(drop=True)

    #--------------------------------- Note, not really needed for binary, when the y values are already numerical ---------------------------------
    logging.info("Creating the label encoder for the y/target value")
    le=LabelEncoder()
    y_train_enc=le.fit_transform(y_train)
    y_val_enc=le.transform(y_val)
    y_test_enc=le.transform(y_test)
    y_train_all_enc=le.transform(y_train_all)

    classes_sorted=np.array(sorted(y_train.unique()))
    #--------------------------------- Note, not really needed for binary, when the y values are already numerical ---------------------------------

    logging.info("Creating the class weights")
    #class weights for the base model logistic regression
    class_weights_models={0:1.3,1:0.8}

    #we will tune the different weight values in the Optuna hyperparameter tuning process, this code block is only needed when we want to use fixed weights
    weights=compute_class_weight(class_weight=class_weights_models,classes=classes_sorted,y=y_train)
    lgb_xgb_class_weight={cls:w for cls,w in zip(classes_sorted,weights)}
    num_class=len(y_train.unique())

    #inner CV for HPO (group-aware on FIT only)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=101)
    #-> use: for tr_idx, val_idx in cv.split(X_fit, y_fit, groups_fit): ...
    ##and keep all preprocessing inside your Pipeline per fold

    #creating brier score for the optuna run in cross_val_score
    brier_scorer = make_scorer(brier_score_loss, greater_is_better=False, needs_proba=True)

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
            'objective': 'binary:logistic', #trial.suggest_categorical('objective',['multi:softmax','multi:softprob']),
            #'n_estimators':8000,
            #'num_class': num_class,
            'tree_method':trial.suggest_categorical('tree_method',['hist','approx']),
            'gamma': trial.suggest_float('gamma', 0.9, 2.8, log=True),
            'eta': trial.suggest_float('eta', 0.008, 0.020, log=True),
            'max_delta_step': trial.suggest_int('max_delta_step', 1, 10),
            'min_child_weight': trial.suggest_float('min_child_weight',8.0, 16.0, log=True),
            #'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise','lossguide']),
            'grow_policy': 'lossguide',
            'subsample': trial.suggest_float('subsample',0.7, 0.8),
            'colsample_bytree': trial.suggest_float('colsample_bytree',0.65, 0.85,log=True),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel',0.65, 0.85, log=True),
            'colsample_bynode': trial.suggest_float('colsample_bynode',0.65, 0.85, log=True),
            'alpha': trial.suggest_float('alpha',1.5, 4.5, log=True),
            'lambda': trial.suggest_float('lambda',40.0, 120.0, log=True),
            'eval_metric':'logloss',
            #'custom_metric':brier_eval_xgb,
            'n_jobs':-1,
            'verbosity':0,
            'random_state':101

        }
        if xgb_params['tree_method'] in ['hist','approx']:
            xgb_params['max_bin'] = trial.suggest_int('max_bin',80,120)
        if xgb_params['booster'] == 'dart':
            xgb_params['sample_type']= trial.suggest_categorical('sample_type',['uniform','weighted'])
            xgb_params['normalize_type']= trial.suggest_categorical('normalize_type',['tree','forest'])
            xgb_params['rate_drop']= trial.suggest_float('rate_drop',0.1,0.8)
            xgb_params['one_drop']= trial.suggest_int('one_drop',0,1)
            xgb_params['skip_drop']= trial.suggest_float('skip_drop',0.1,0.8)
        if xgb_params['grow_policy']=='depthwise':
            xgb_params['max_depth']=trial.suggest_int('max_depth',5,7)
        elif xgb_params['grow_policy']=='lossguide':
            xgb_params['max_leaves']=trial.suggest_int('max_leaves',38, 65)
            xgb_params['max_depth']=0



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

        fold_losses,fold_ns,best_trees = [],[],[]

        num_class = 2
        classes_arr = np.array([0, 1])

        # Optional tuned boosts (kept around 1.0), one per class
        #boost_0 represents class 0
        #boost_1 represents class 1
        m = np.array([trial.suggest_float("boost_0", 1.1, 1.8),
                    trial.suggest_float("boost_1", 0.75, 1.0)])
        m = m / m.mean()
        trial.set_user_attr("class_boosts", m.tolist())
        # ---------------------------------- splitting training set into train & val; for early stopping ---------------------------------------------

        for tr_idx,val_idx in cv.split(X_train,y_train,groups_fit):
            X_tr,X_val=X_train.iloc[tr_idx],X_train.iloc[val_idx]
            y_tr,y_val=y_train.iloc[tr_idx],y_train.iloc[val_idx]

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
                classes=classes_arr,
                y=y_tr_enc
            )
            # final per-class weights = base * boost
            class_w = base_w * m
            # per-sample weights for DMatrix
            w_tr = class_w[y_tr_enc]
            w_tr = np.minimum(w_tr, np.percentile(w_tr, 95))
        # ---------------------------------------------------------------------------

            dtrain=xgboost.DMatrix(X_tr_t,label=y_tr_enc,weight=w_tr)
            dvalid=xgboost.DMatrix(X_va_t,label=y_val_enc)

            bst=xgboost.train(
                params=xgb_params,
                dtrain=dtrain,
                num_boost_round=10000,
                #just (dvalid,'valid') -> we care only about 
                evals=[(dtrain,'train'),(dvalid,'valid')],
                custom_metric=brier_eval_xgb,
                callbacks=[xgboost.callback.EarlyStopping(rounds=250,save_best=True),XGBoostPruningCallback(trial,'valid-brier')],
                verbose_eval=0
            )

            # predict probabilities for valid fold, due to 
            proba = bst.predict(dvalid, iteration_range=(0, bst.best_iteration + 1))
            loss = brier_score_loss(y_val_enc, proba, pos_label=1)
            fold_losses.append(loss)
            fold_ns.append(len(y_val_enc))
            best_trees.append(bst.best_iteration)
        
        # average CV loss
        weighted_cv_brier_score = float(np.average(fold_losses, weights=fold_ns))

        # -------- OPTIONAL: regularize the boosts so they stay near 1.0 ----------------
        # a small L2 penalty on log(m) is stable; tune 0.005–0.02 if needed
        penalty = 0.01 * float(np.mean(np.log(m) ** 2))
        objective = weighted_cv_brier_score + penalty
        # -------------------------------------------------------------------------------

        #adding the number of iterations
        if best_trees:
            trial.set_user_attr("n_estimators", int(np.mean(best_trees)))
        #return float(np.average(fold_losses, weights=fold_ns))
        return objective

    study_first_run=optuna.create_study(direction='minimize',sampler=TPESampler(multivariate=True,group=True,constant_liar=True,n_startup_trials=15,seed=101),
                                        pruner=SuccessiveHalvingPruner(min_resource=50,reduction_factor=3))
    start_time=time.time()
    study_first_run.optimize(stage1,n_trials=40)
    end_time=time.time()

    study1_params=study_first_run.best_params

    m_best = np.array(study_first_run.best_trial.user_attrs["class_boosts"])  # [m0, m1]

    # 2) compute base ratio on FULL train
    neg, pos = np.bincount(y_train_enc)
    base_spw = neg / max(pos, 1)

    # 3) fold the boosts into scale_pos_weight
    # class 0 gets m0, class 1 gets m1 -> ratio scales by (m1 / m0)
    spw_with_boosts = base_spw * (m_best[1] / m_best[0])

    # 4) put it in the param dict
    study1_params["scale_pos_weight"] = float(spw_with_boosts)

    #study1_params['grow_policy']='SymmetricTree'
    study1_best_iter = study_first_run.best_trial.user_attrs["n_estimators"]
    #study1_params['objective']='multi:softprob'
    study1_params["n_estimators"] = study1_best_iter
    study1_params['booster']='gbtree'
    study1_params['grow_policy']='lossguide'

    study1_score=study_first_run.best_value

    logging.info(f'Best Params: {study1_params}')
    logging.info(f'Best Score: {study1_score}')
    logging.info(f'Run Time: {(end_time - start_time)/60:.2f} Minutes\n')
    logging.info('Broad hyperparameter tuning is done, switchting to a more narrow hyperparameter tuning range')

    def stage2(trial):
        nation_tol=trial.suggest_float('nation_tol',(study1_params['nation_tol']-study1_params['nation_tol']*0.3),(study1_params['nation_tol']+study1_params['nation_tol']*0.3))
        club_left_tol=trial.suggest_float('club_left_tol',(study1_params['club_left_tol']-study1_params['club_left_tol']*0.3),(study1_params['club_left_tol']+study1_params['club_left_tol']*0.3))
        #league_left_tol=trial.suggest_float('league_left_tol',(study1_params['league_left_tol']-study1_params['league_left_tol']*0.3),(study1_params['league_left_tol']+study1_params['league_left_tol']*0.3))
        feature_select_thresh=trial.suggest_float('feature_select_thresh',(study1_params['feature_select_thresh']-study1_params['feature_select_thresh']*0.3),(study1_params['feature_select_thresh']+study1_params['feature_select_thresh']*0.3))

        # --- CatBoost-only params (no num_leaves; no max_leaves unless Lossguide)
        xgb_params = {
            'booster': study1_params['booster'],
            #'multi:softmax' -> looks for accuracy; 'multi:softprob' looks for probability
            'objective': 'binary:logistic', #trial.suggest_categorical('objective',['multi:softmax','multi:softprob']),
            #'num_class': num_class,
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
            'eval_metric':'logloss',
            #'custom_metric':brier_eval_xgb,
            'n_jobs':1,
            'verbosity':0,
            'random_state':101

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
            xgb_params['max_depth']=trial.suggest_int('max_depth',int(study1_params['max_bins']-study1_params['max_bins']*0.3),int(study1_params['max_bins']+study1_params['max_bins']*0.3))
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

        fold_losses,fold_ns,best_trees = [],[],[]

        num_class = 2
        classes_arr = np.array([0, 1])

        # Optional tuned boosts (kept around 1.0), one per class
        m = np.array([trial.suggest_float("boost_0",(study1_params['boost_0']-(study1_params['boost_0']*0.3)),(study1_params['boost_0']+(study1_params['boost_0']*0.3))),
                    trial.suggest_float("boost_1",(study1_params['boost_1']-(study1_params['boost_1']*0.3)),(study1_params['boost_1']+(study1_params['boost_1']*0.3)))])
        m = m / m.mean()
        trial.set_user_attr("class_boosts", m.tolist())
        # --------------------------------------  splitting training set into train & val for early stopping in xgboost -----------------------------------------

        for tr_idx,val_idx in cv.split(X_train,y_train,groups_fit):
            X_tr,X_val=X_train.iloc[tr_idx],X_train.iloc[val_idx]
            y_tr,y_val=y_train.iloc[tr_idx],y_train.iloc[val_idx]

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
                classes=classes_arr,
                y=y_tr_enc
            )
            # final per-class weights = base * boost
            class_w = base_w * m
            # per-sample weights for DMatrix
            w_tr = class_w[y_tr_enc]
            w_tr = np.minimum(w_tr, np.percentile(w_tr, 95))
        # ---------------------------------------------------------------------------

            dtrain=xgboost.DMatrix(X_tr_t,label=y_tr_enc,weight=w_tr)
            dvalid=xgboost.DMatrix(X_va_t,label=y_val_enc)

            bst=xgboost.train(
                params=xgb_params,
                dtrain=dtrain,
                num_boost_round=10000,
                #just (dvalid,'valid') -> we care only about 
                evals=[(dtrain,'train'),(dvalid,'valid')],
                custom_metric=brier_eval_xgb,
                callbacks=[xgboost.callback.EarlyStopping(rounds=250,save_best=True),XGBoostPruningCallback(trial,'valid-brier')],
                verbose_eval=0
            )

            # predict probabilities for valid fold, due to 
            proba = bst.predict(dvalid, iteration_range=(0, bst.best_iteration + 1))
            loss = brier_score_loss(y_val_enc, proba, pos_label=1)
            fold_losses.append(loss)
            fold_ns.append(len(y_val_enc))
            best_trees.append(bst.best_iteration)
        
        # average CV loss
        weighted_cv_brier_score = float(np.average(fold_losses, weights=fold_ns))

        # -------- OPTIONAL: regularize the boosts so they stay near 1.0 ----------------
        # a small L2 penalty on log(m) is stable; tune 0.005–0.02 if needed
        penalty = 0.01 * float(np.mean(np.log(m) ** 2))
        objective = weighted_cv_brier_score + penalty
        # -------------------------------------------------------------------------------

        #adding the number of iterations
        if best_trees:
            trial.set_user_attr("n_estimators", int(np.mean(best_trees)))
        #return float(np.average(fold_losses, weights=fold_ns))
        return objective

    study_second_run=optuna.create_study(direction='minimize',sampler=TPESampler(multivariate=True,group=True,constant_liar=True,n_startup_trials=15,seed=101),
                                        pruner=SuccessiveHalvingPruner(min_resource=50,reduction_factor=3))
    start_time=time.time()
    study_second_run.optimize(stage2,n_trials=60,n_jobs=-1)
    end_time=time.time()

    study2_params=study_second_run.best_params

    m_best = np.array(study_first_run.best_trial.user_attrs["class_boosts"])  # [m0, m1]

    # 2) compute base ratio on FULL train
    neg, pos = np.bincount(y_train_enc)
    base_spw = neg / max(pos, 1)

    # 3) fold the boosts into scale_pos_weight
    # class 0 gets m0, class 1 gets m1 -> ratio scales by (m1 / m0)
    spw_with_boosts = base_spw * (m_best[1] / m_best[0])

    # 4) put it in the param dict
    study2_params["scale_pos_weight"] = float(spw_with_boosts)

    study2_params['booster']=study1_params['booster']
    study2_params['objective']='binary:logistic'
    study2_params['grow_policy']=study1_params['grow_policy']
    #study2_params['objective']=study1_params['objective']
    study2_params['tree_method']=study1_params['tree_method']
    study2_best_iter = study_second_run.best_trial.user_attrs["n_estimators"]
    study2_params["n_estimators"] = study2_best_iter

    study2_score=study_second_run.best_value

    logging.info(f'Best Params: {study2_params}')
    logging.info(f'Best Score: {study2_score}')
    logging.info(f'Run Time: {(end_time - start_time)/60:.2f} Minutes')
    logging.info('Narrow hyperparameter tuning is done')

    final_params = study1_params if study_first_run.best_value < study_second_run.best_value else study2_params
    best_score=study_first_run.best_value if study_first_run.best_value < study_second_run.best_value else study_second_run.best_value

    #---------------------------------------------------------------------------------------- setting up the learning curve ----------------------------------------------------------------------------------------------------------------------------
    #in a binary problem we need to set up neg & pos, and place it as scale_pos_weight, in the param dict
    #neg, pos = np.bincount(y_train)
    logging.info('Starting learning curve')
    xgb_params = {
            'booster': final_params['booster'],
            #'multi:softmax' -> looks for accuracy; 'multi:softprob' looks for probability
            'objective': 'binary:logistic', #trial.suggest_categorical('objective',['multi:softmax','multi:softprob']),
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
            #when using scale_pos_weight, we do not need to use class_weights in XGBoost
            'scale_pos_weight':final_params['scale_pos_weight'],
            #'eval_metric':'logloss',
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

    train_sizes,train_scores,val_scores=learning_curve(
        estimator=xgboost_final_pipeline,
        X=X_train,
        y=y_train_enc,
        groups=groups_fit,
        train_sizes=np.linspace(0.1,1.0,10),
        cv=cv,
        shuffle=True,
        #we have tuned with the brier-score, for the learning-curve we will use log_loss, but why?
        ##brier-score => it measures calibration and sharpness of predicted probabilities, but it is less sensitive to overconfident mistakes
        ##log-loss => it heavily penalizes overconfidence in wrong predictions, but slightly more volatile when probabilities are extreme
        ##log-loss is better for a learning-curve, because it provides a monotonic, smooth, differentiable loss
        scoring='neg_log_loss',
        n_jobs=-1,
        random_state=101
        # we do not use fit_params, because we have tuned scale_pos_weight in the model
    )

    for train_size, cv_train_scores, cv_val_scores in zip(train_sizes, train_scores, val_scores):
        logging.info(f'Train Size: {np.mean(train_size)} | Train Score: {np.mean(cv_train_scores)*-1} | Val Score: {np.mean(cv_val_scores)*-1}')


    #---------------------------------------------------------------------------------------- eval of the model ----------------------------------------------------------------------------------------------------------------------------
    year=datetime.date.today().year
    month=datetime.date.today().month
    day=datetime.date.today().day

    #saving the final dictionary, with all the important/needed values for the model pipeline
    joblib.dump(final_params,MODEL_PARAMS_MODELS_BINARY / f'binary_final_params_{day}_{month}_{year}.dict')

    #getting the, breaking down groups_train_all into group_val, groups_train_all has already been broken down into group_fit & finally into group_val
    ##why can't we just use groups_train_all? Due to the index splits for group_fit and group_val
    groups_val = groups_train_all.iloc[val_idx].reset_index(drop=True)

    X_train_val=pd.concat([X_train,X_val],axis=0).reset_index(drop=True)
    #y features is at that point an array, instead of pd.concat() we use np.concatenate()
    y_train_val_enc = np.concatenate((y_train_enc, y_val_enc), axis=0)
    groups_train_val = pd.concat([groups_fit, groups_val], axis=0).reset_index(drop=True)

    joblib.dump(X_train_val,MODEL_TRAIN_TEST_BINARY / 'X_Train_Data.CSV')
    joblib.dump(y_train_val_enc,MODEL_TRAIN_TEST_BINARY / 'y_Train_Data.CSV')
    joblib.dump(X_test,MODEL_TRAIN_TEST_BINARY / 'X_Test_Data.CSV')
    joblib.dump(y_test_enc,MODEL_TRAIN_TEST_BINARY / 'y_Test_Data.CSV')

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
    #the code block above applied the the existing cross-validation folds on the combined val and training set, so we can use the same fold logic in the CalibratedClassifierCV
    ps = PredefinedSplit(test_fold=fold_ids)

    xgb_params = {
            'booster': final_params['booster'],
            #'multi:softmax' -> looks for accuracy; 'multi:softprob' looks for probability
            'objective': 'binary:logistic', #trial.suggest_categorical('objective',['multi:softmax','multi:softprob']),
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
            #when using scale_pos_weight, we do not need to use class_weights in XGBoost
            'scale_pos_weight':final_params['scale_pos_weight'],
            #we do not use early-stopping in the final model
            #'eval_metric':'logloss',
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
    ccv.fit(X_train_val,y_train_val_enc)

    y_proba=ccv.predict_proba(X_test)
    y_pred=ccv.predict(X_test)

    pos_idx = list(ccv.classes_).index(1)

    logging.info(f'Brier Score: {brier_score_loss(y_test_enc,y_proba[:,pos_idx],pos_label=1)}')

    logging.info(f'Macro F1-Score: {f1_score(y_test_enc,y_pred,average="macro")}')

    logging.info(classification_report(y_test_enc,y_pred))

    #ROC-AUC
    auc = roc_auc_score(y_test_enc, y_proba[:, pos_idx])
    logging.info(f'ROC-AUC: {auc:.3f}')

    joblib.dump(ccv, MODEL_PARAMS_MODELS_BINARY / f'binary_final_model.pkl')

    #---------------------------------------------------------------------------------------- creating the final model ----------------------------------------------------------------------------------------------------------------------------
    logging.info('Creating the final model')
    final_params=joblib.load(MODEL_PARAMS_MODELS_BINARY / 'binary_final_params.dict')

    year=datetime.date.today().year
    month=datetime.date.today().month
    day=datetime.date.today().day

    X_binary=df.drop(['Player','Squad','League','League_Joined','Club_Joined','Transfer_Fee','Other_or_Top'],axis=1).copy()
    joblib.dump(list(X_binary.columns),MODEL_PARAMS_MODELS_BINARY / 'binary_model_columns.joblib')
    y_binary=df['Other_or_Top']

    le_final=LabelEncoder()
    y_binary_enc=le_final.fit_transform(y_binary)

    joblib.dump(X_binary,MODEL_WHOLE_DATA_BINARY / 'X_Feat.CSV')
    joblib.dump(y_binary_enc,MODEL_WHOLE_DATA_BINARY / 'y_Feat.CSV')

    xgb_params = {
            'booster': final_params['booster'],
            #'multi:softmax' -> looks for accuracy; 'multi:softprob' looks for probability
            'objective': 'binary:logistic', #trial.suggest_categorical('objective',['multi:softmax','multi:softprob']),
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
            #when using scale_pos_weight, we do not need to use class_weights in XGBoost
            'scale_pos_weight':final_params['scale_pos_weight'],
            #we do not use early-stopping in the final model
            #'eval_metric':'logloss',
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
    ccv.fit(X_binary,y_binary_enc)

    joblib.dump(ccv,MODEL_PARAMS_MODELS_BINARY / f'binary_final_model_total_data.pkl')

if __name__ == "__main__":
    run_binary_training()