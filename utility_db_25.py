from sklearn.model_selection import StratifiedKFold,cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score,roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay, f1_score, precision_score, recall_score
import numpy as np
from sklearn.base import clone
from tqdm import tqdm
from IPython.display import clear_output
import matplotlib.pyplot as plt
import optuna
from sklearn.pipeline import Pipeline
from catboost import CatBoostClassifier
import nfl_data_py as nfl
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from catboost import  Pool, MetricVisualizer
from itertools import chain
    

def mark_columns(df,features, nc=None,cc = None):
        
    if nc is not None:
        cat_columns=[]
        for c in cc:
            if c in features:
                cat_columns.append(c)
        
        numeric_columns=[]
        for i in features:
            if df[i].dtype!='O' and i!='pass' and 'shiftSinceLineset' not in i and 'motionSinceLineset' not in i and  'offenseFormation' not in i and'receiverAlignment'not in i and 'Cover'not in i and 'roof' not in i and 'surface' not in i and 'is_no_huddle' not in i and 'is_no_motion' not in i:
                    numeric_columns.append(i)


    else:
        numeric_columns=[]
        for i in features:
            if df[i].dtype!='O' and i!='pass' and 'shiftSinceLineset' not in i and 'motionSinceLineset' not in i and  'offenseFormation' not in i and'receiverAlignment'not in i and 'Cover'not in i and 'roof' not in i and 'surface' not in i and 'is_no_huddle' not in i and 'is_no_motion' not in i:
                numeric_columns.append(i)
        
        cat_columns=[]
        for i in features:
            if i not in numeric_columns and i!='pass':
                cat_columns.append(i)
    
    return numeric_columns, cat_columns


def TrainML(model_class, X, y,n_splits,SEED):


    SKF = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    
    train_S = []
    test_S = []
    train_A = []
    test_A = []

    


    for fold, (train_idx, test_idx) in enumerate(tqdm(SKF.split(X, y), desc="Training Folds", total=n_splits)):
        X_train, X_val = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[test_idx]

        model = clone(model_class)
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
                  

        train_auc = roc_auc_score(y_train, y_train_pred)
        val_auc = roc_auc_score(y_val, y_val_pred)

        train_S.append(train_auc)
        test_S.append(val_auc)

        train_accuracy = accuracy_score(y_train, y_train_pred)
        val_accuracy = accuracy_score(y_val, y_val_pred)

        train_A.append(train_accuracy)
        test_A.append(val_accuracy)
       
        
        print(f"Fold {fold+1} - Train AUC: {train_auc:.4f}, Validation AUC: {val_auc:.4f}")
        clear_output(wait=True)

    print(f"Mean Train AUC --> {np.mean(train_S):.4f}")
    print(f"Mean Validation AUC ---> {np.mean(test_S):.4f}")
    print(f"Mean Train Accuracy --> {np.mean(train_A):.4f}")
    print(f"Mean Validation Accuracy ---> {np.mean(test_A):.4f}")
    
    cm = confusion_matrix(y_val, y_val_pred)

    cm_display = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = [0, 1])

    cm_display.plot()
    plt.show()

def calculate_and_plot_metrics(X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test, title, model, early_stopping):

    history=model.fit(X_train_scaled, y_train,  epochs=250,validation_data=(X_val_scaled, y_val),callbacks=[early_stopping])

    y_pred = model.predict(X_test_scaled)

    # Calculate metrics
    def calculate_metrics(y_true, y_pred):
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=1)
        recall = recall_score(y_true, y_pred, average='weighted')
        roc_auc=roc_auc_score(y_true, y_pred)
        return accuracy, f1, precision, recall, roc_auc

    y_preds=[]
    for i in y_pred:
        if i>.5:
            y_preds.append(1)
        else:
            y_preds.append(0)
 
    # Metrics for MLP
    accuracy_mlp, f1_mlp, precision_mlp, recall_mlp, roc = calculate_metrics(y_test, y_preds)


    fig, axes = plt.subplots(1, 2, figsize=(7, 4))

    # Confusion Matrix and Visualization 
    cm_mlp = confusion_matrix(y_test, y_preds)
    disp_mlp = ConfusionMatrixDisplay(confusion_matrix=cm_mlp)
    disp_mlp.plot(cmap='viridis', values_format='d', ax=axes[0])
    axes[0].set_title("MLP Confusion Matrix")

    # Loss Curve 
    loss_values = history.history['loss']
    epochs = range(1, len(loss_values)+1)

    axes[1].plot(epochs, loss_values, label='Training Loss')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Loss')

 

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

    return accuracy_mlp, f1_mlp, precision_mlp, recall_mlp, roc


################
# Optuna helper
################

def optuna_call(transformer,X,y,SKF,n_trials):

    def tune(objective):
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        params = study.best_params
        best_score = study.best_value
        print(f"Best score: {best_score}\n")
        print(f"Optimized parameters: {params}\n")
        return params

    def objective(trial):
        param = {
        "learning_rate": trial.suggest_float("learning_rate", 5e-2, 1e-1, log=True),
         "depth": trial.suggest_int("depth",5,8),
        "subsample": trial.suggest_float("subsample", 0.2, .95),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.2, .95),
        "iterations":trial.suggest_int("iterations",250, 500 )
        
        }   
        cat = CatBoostClassifier(**param,  logging_level='Silent')  
        pipeline=Pipeline([('tr', transformer), ('cat',  cat)], verbose = False)
        scores = cross_val_score( pipeline, X, y, cv=SKF, scoring="accuracy")
        return scores.mean()
    
    return tune(objective)

#############################################
#
#  function: proc_external
#  purpose: process external data sources
#
#############################################

def proc_external(xp_df, pr_df, cf_df, cu_df, df_games, df_plays, train_data,inj_df):

    # import ftn, nflverse play-by-play
    pbp = nfl.import_pbp_data([2022])
    ftn = nfl.import_ftn_data([2022])

    # make key types play nice
    pbp_ids = pbp[['play_id','game_id','old_game_id_x']]
    ftn['nflverse_play_id'] = ftn['nflverse_play_id'].astype(int)
    pbp_ids['play_id'] = pbp_ids['play_id'].astype(int)
    ftn['nflverse_game_id'] = ftn['nflverse_game_id'].astype(str)
    pbp_ids['game_id'] = pbp_ids['game_id'].astype(str)

    # merge nflverse keys s.t. ftn data can join into train_data
    ftn_merged = pbp_ids.merge(ftn,how='left',left_on=['play_id','game_id'],
                                right_on=['nflverse_play_id','nflverse_game_id'])

    ftn_merged = ftn_merged[['play_id','old_game_id_x','n_offense_backfield',
                            'n_defense_box','is_no_huddle','is_motion']].rename(columns={'old_game_id_x':'gameId',
                                                                                        'play_id':'playId'})

    # re-cast ID's back to int
    ftn_merged['gameId'] = ftn_merged['gameId'].astype(int)
    ftn_merged['playId'] = ftn_merged['playId'].astype(int)

    # merge in team pass-ratio, ftn data
    merged_id_df = train_data[['gameId','playId']]
    merged_base = merged_id_df.merge(ftn_merged,how='left',on=['gameId','playId'])
    merged_base = merged_base.merge(pr_df,how='left',on=['gameId','playId'])
    merged_base = merged_base.merge(xp_df,how='left',on=['gameId','playId'])

    # add in coverage data
    merged_base = merged_base.merge(df_games[['gameId','week']].drop_duplicates(),how='left',on=['gameId'])
    merged_base = merged_base.merge(df_plays[['gameId','playId',
                                          'possessionTeam','defensiveTeam']].drop_duplicates(),
                                how='left',on=['gameId','playId'])
    
    merged_base = merged_base.merge(cf_df,how='left',on=['possessionTeam','week'])
    merged_base = merged_base.merge(cu_df,how='left',on=['defensiveTeam','week'])

    # merge in snaps lost to injury
    merged_base = merged_base.merge(inj_df.drop(columns=['def_snaps_lost']),how='left',
                   left_on=['possessionTeam','week'], right_on=['club_code','week']).drop(columns=['club_code'])
    merged_base = merged_base.merge(inj_df.drop(columns=['off_snaps_lost']),how='left',
                   left_on=['defensiveTeam','week'], right_on=['club_code','week']).drop(columns=['club_code'])


    # pivot
    train_data = pd.concat([train_data,merged_base.iloc[:,2:]],axis=1)

    # re-cast types for train_data
    train_data[['is_no_huddle']] = train_data[['is_no_huddle']].astype(int)
    train_data[['is_motion']] = train_data[['is_motion']].astype(int)
    train_data['temp']=train_data['temp'].astype(float)
    train_data['humidity']=train_data['humidity'].astype(float)
    train_data['wind']=train_data['wind'].astype(float)

    return train_data

#############################################
#
#  function: build_transformer
#  purpose: avoid repeating transformer code
#
#############################################


# Create a transformer
def build_transformer(imputer,numeric_columns,kind):
    
    if kind == 'imputer':
        transformer = ColumnTransformer(
        transformers=[('imputer', imputer, numeric_columns)],
        remainder='passthrough'  # Pass through columns not specified
        )
    elif kind == 'scaler':
        transformer = ColumnTransformer(
        transformers=[
            ('scaler', MinMaxScaler(), numeric_columns)
        ],
        remainder='passthrough'  # Pass through columns not specified
        )
    else:
        raise ValueError('Kind must be imputer or scaler')
    
    return transformer


####################################################
#
#  function: get_final_features
#  purpose: use corr. data to get "final" features
#
#####################################################


def get_final_features(train_data,threshold,trim_rows):

    features=[]

    # remove ID, other identifying/non-input columns
    for col in train_data.columns:
        if 'Id' not in col and 'playNullifiedByPenalty' not in col and 'possessionTeam' not in col and 'defensiveTeam' not in col:
            features.append(col)

    #select final features based on correlation with target variable
    correlations=train_data[features].corr()[['pass']]
    final_features=list(correlations[((correlations['pass']>.09) | (correlations['pass']<-.09))].T.columns.values)

    # lose std dev feats. (redundant), low-record ct. feats.
    final_features = [x for x in final_features if 'std' not in x]
    final_features= train_data[final_features].sum().astype(int).sort_values().index[trim_rows:]

    #remove other redundant features
    threshold = threshold
    correlation_matrix = train_data[final_features].drop(columns='pass').corr()
    highly_correlated_features = set()
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > threshold:
                colname = correlation_matrix.columns[i]
                highly_correlated_features.add(colname)

    final_features=list(set(final_features)-highly_correlated_features)

    return final_features

##############################################
#
#  function: get columns to make momentum score
#  purpose: sum all shiftSinceLineset values 
#
##############################################


def get_momentum_cols(final_features):
    momentum_cols=[]
    rb_cols=[]
    for f in final_features:
        if 'shiftSinceLineset' in f and 'RB' not in f:
            momentum_cols.append(f)
        elif 'shiftSinceLineset' in f and 'RB' in f:
            rb_cols.append(f)
    return momentum_cols, rb_cols
def create_momentum_index(data, momentum_cols, rb_cols):
    data['presnap_momentum']=data[momentum_cols].sum(axis=1).fillna(0).astype(float)-data[rb_cols].sum(axis=1).fillna(0).astype(float)
    data.drop(columns=momentum_cols, inplace=True)
    data.drop(columns=rb_cols, inplace=True)
    return data


##############################################
#
#  function: get columns for motion complexity score
#  purpose: sum all motionSinceLineset columns
#
##############################################


def get_motion_cols(final_features):
    motion_cols=[]
    rb_motion_cols=[]

    for f in final_features:
        if 'motionSinceLineset' in f and 'RB' not in f:
            motion_cols.append(f)
        elif 'motionSinceLineset' in f and 'RB' in f:
            rb_motion_cols.append(f)
    return motion_cols, rb_motion_cols
def motion_complexity_score(data, motion_cols, rb_motion_cols):
    dir_cols=['dir_standard|mean|C_1',
    'dir_standard|mean|C_2',
    'dir_standard|mean|FB_1',
    'dir_standard|mean|G_1',
    'dir_standard|mean|G_2',
    'dir_standard|mean|G_3',
    'dir_standard|mean|ILB_1',
    'dir_standard|mean|RB_1',
    'dir_standard|mean|RB_2',
    'dir_standard|mean|TE_1',
    'dir_standard|mean|TE_2',
    'dir_standard|mean|TE_3',
    'dir_standard|mean|T_1',
    'dir_standard|mean|T_2',
    'dir_standard|mean|T_3',
    'dir_standard|mean|T_4',
    'dir_standard|mean|WR_1',
    'dir_standard|mean|WR_2',
    'dir_standard|mean|WR_3',
    'dir_standard|mean|WR_4',
    'dir_standard|mean|WR_5']
    x_cols=['x_standard|mean|C_1',
    'x_standard|mean|C_2',
    'x_standard|mean|FB_1',
    'x_standard|mean|G_1',
    'x_standard|mean|G_2',
    'x_standard|mean|G_3',
    'x_standard|mean|ILB_1',
    'x_standard|mean|RB_1',
    'x_standard|mean|RB_2',
    'x_standard|mean|TE_1',
    'x_standard|mean|TE_2',
    'x_standard|mean|TE_3',
    'x_standard|mean|T_1',
    'x_standard|mean|T_2',
    'x_standard|mean|T_3',
    'x_standard|mean|T_4',
    'x_standard|mean|WR_1',
    'x_standard|mean|WR_2',
    'x_standard|mean|WR_3',
    'x_standard|mean|WR_4',
    'x_standard|mean|WR_5']
    y_cols=['y_standard|mean|C_1',
    'y_standard|mean|C_2',
    'y_standard|mean|FB_1',
    'y_standard|mean|G_1',
    'y_standard|mean|G_2',
    'y_standard|mean|G_3',
    'y_standard|mean|ILB_1',
    'y_standard|mean|RB_1',
    'y_standard|mean|RB_2',
    'y_standard|mean|TE_1',
    'y_standard|mean|TE_2',
    'y_standard|mean|TE_3',
    'y_standard|mean|T_1',
    'y_standard|mean|T_2',
    'y_standard|mean|T_3',
    'y_standard|mean|T_4',
    'y_standard|mean|WR_1',
    'y_standard|mean|WR_2',
    'y_standard|mean|WR_3',
    'y_standard|mean|WR_4',
    'y_standard|mean|WR_5']
    for c in dir_cols:
        data[c+'_QBdiff']=abs(data['dir_standard|mean|QB_1']-data[c].astype(float))
    for c in y_cols:
        data[c+'_QBdiff']=abs(data['y_standard|mean|QB_1']-data[c].astype(float))
    for c in x_cols:
        data[c+'_QBdiff']=abs(data['x_standard|mean|QB_1']-data[c].astype(float))
    data['QBdff_TE']=data[['dir_standard|mean|TE_1_QBdiff','dir_standard|mean|TE_2_QBdiff','dir_standard|mean|TE_3_QBdiff']].mean(axis=1).astype(float)
    data['QBdff_RB']=data[['dir_standard|mean|RB_1_QBdiff','dir_standard|mean|RB_2_QBdiff']].mean(axis=1).astype(float)
    data['QBdff_G']=data[['dir_standard|mean|G_1_QBdiff','dir_standard|mean|G_2_QBdiff','dir_standard|mean|G_3_QBdiff']].mean(axis=1).astype(float)
    data['QBdff_T']=data[['dir_standard|mean|T_1_QBdiff','dir_standard|mean|T_2_QBdiff','dir_standard|mean|T_3_QBdiff', 'dir_standard|mean|T_4_QBdiff']].mean(axis=1).astype(float)
    data['QBdff_WR']=data[['dir_standard|mean|WR_1_QBdiff','dir_standard|mean|WR_2_QBdiff','dir_standard|mean|WR_3_QBdiff', 'dir_standard|mean|WR_4_QBdiff', 'dir_standard|mean|WR_5_QBdiff']].mean(axis=1).astype(float)
    data['QBdffy_TE']=data[['y_standard|mean|TE_1_QBdiff','y_standard|mean|TE_2_QBdiff','y_standard|mean|TE_3_QBdiff']].mean(axis=1).astype(float)
    data['QBdffy_RB']=data[['y_standard|mean|RB_1_QBdiff','y_standard|mean|RB_2_QBdiff']].mean(axis=1).astype(float)
    data['QBdffx_RB']=data[['x_standard|mean|RB_1_QBdiff','x_standard|mean|RB_2_QBdiff']].mean(axis=1).astype(float)
    data['QB_RB1_offset']=np.sqrt(np.square(data['x_standard|mean|RB_1_QBdiff']) + np.square(data['y_standard|mean|RB_1_QBdiff']**2))
    data['QBdffy_G']=data[['y_standard|mean|G_1_QBdiff','y_standard|mean|G_2_QBdiff','y_standard|mean|G_3_QBdiff']].mean(axis=1).astype(float)
    data['QBdffy_T']=data[['y_standard|mean|T_1_QBdiff','y_standard|mean|T_2_QBdiff','y_standard|mean|T_3_QBdiff', 'y_standard|mean|T_4_QBdiff']].mean(axis=1).astype(float)
    data['QBdffy_WR']=data[['y_standard|mean|WR_1_QBdiff','y_standard|mean|WR_2_QBdiff','y_standard|mean|WR_3_QBdiff', 'y_standard|mean|WR_4_QBdiff', 'y_standard|mean|WR_5_QBdiff']].mean(axis=1).astype(float)
    data['presnap_motion_complexity']=data[motion_cols].sum(axis=1).fillna(0).astype(float)-data[rb_motion_cols].sum(axis=1).fillna(0).astype(float)
    data['motion-momentum']= (data['presnap_motion_complexity']-data['presnap_momentum']).astype(float) #how many more people moved compared to how many shifted (over 2.5 yards)
    data['neg_Formations']=data[['offenseFormation_SINGLEBACK' ,'offenseFormation_I_FORM', 'offenseFormation_PISTOL']].sum(axis=1).astype(int)
    data['neg_alignment']=data[['receiverAlignment_1x1', 'receiverAlignment_2x1']].sum(axis=1).astype(int)
    data.drop(columns=['y_standard|mean|WR_1_QBdiff','y_standard|mean|WR_2_QBdiff','y_standard|mean|WR_3_QBdiff', 'y_standard|mean|WR_4_QBdiff', 'y_standard|mean|WR_5_QBdiff','y_standard|mean|RB_1_QBdiff','y_standard|mean|RB_2_QBdiff','y_standard|mean|T_1_QBdiff','y_standard|mean|T_2_QBdiff','y_standard|mean|T_3_QBdiff', 'y_standard|mean|T_4_QBdiff','y_standard|mean|G_1_QBdiff','y_standard|mean|G_2_QBdiff','y_standard|mean|G_3_QBdiff','y_standard|mean|TE_1_QBdiff','y_standard|mean|TE_2_QBdiff','y_standard|mean|TE_3_QBdiff', 'dir_standard|mean|WR_1_QBdiff','dir_standard|mean|WR_2_QBdiff','dir_standard|mean|WR_3_QBdiff', 'dir_standard|mean|WR_4_QBdiff', 'dir_standard|mean|WR_5_QBdiff','dir_standard|mean|RB_1_QBdiff','dir_standard|mean|RB_2_QBdiff','dir_standard|mean|T_1_QBdiff','dir_standard|mean|T_2_QBdiff','dir_standard|mean|T_3_QBdiff', 'dir_standard|mean|T_4_QBdiff','dir_standard|mean|G_1_QBdiff','dir_standard|mean|G_2_QBdiff','dir_standard|mean|G_3_QBdiff','dir_standard|mean|TE_1_QBdiff','dir_standard|mean|TE_2_QBdiff','dir_standard|mean|TE_3_QBdiff', 'offenseFormation_SINGLEBACK' ,'offenseFormation_I_FORM','offenseFormation_PISTOL','receiverAlignment_1x1', 'receiverAlignment_2x1'], inplace=True)
    data.drop(columns=motion_cols, inplace=True)
    data.drop(columns=dir_cols, inplace=True)
    data.drop(columns=y_cols, inplace=True)
    data.drop(columns=x_cols, inplace=True)
    return data


##############################################
#
#  function: build_catboost
#  purpose: construct catboost model, feats.
#
##############################################

def build_catboost(final_features, train_data, imputer, cat_params):


    

    X=train_data[final_features]
    y=train_data['pass']

    # delineate numeric, categorical columns
    numeric_columns=[]
    is_cat = (X.dtypes != float)

    for feature, feat_is_cat in is_cat.to_dict().items():
       if feat_is_cat:
            X[feature].fillna(0, inplace=True)
            X[feature].replace([np.inf, -np.inf], 0, inplace=True)
       else:
            numeric_columns.append(feature)

    transformer_impute = build_transformer(imputer,numeric_columns,kind='imputer')
    transformer_scale = build_transformer(imputer,numeric_columns,kind='scaler')
            
    # build transformer on purely numeric columns, transform X
    X_transform=transformer_impute.fit_transform(X)

    X_transform = pd.DataFrame(X_transform, columns=final_features)

    X_transform=transformer_scale.fit_transform(X_transform)

    X_transform = pd.DataFrame(X_transform, columns=final_features)

    for feature, feat_is_cat in is_cat.to_dict().items():
        if feat_is_cat:
            X_transform[feature].fillna(0, inplace=True)
            X_transform[feature].replace([np.inf, -np.inf], 0, inplace=True)
            X_transform[feature]=X_transform[feature].astype(int)

    # define pool, fit model
    cat_features_index = np.where(is_cat)[0]
    pool = Pool(X_transform, y, cat_features=cat_features_index, feature_names=list(X_transform.columns))

    model = CatBoostClassifier( **cat_params, verbose=False).fit(pool)

    return model, pool, cat_features_index, X_transform

####################
#
# function: test_ML
# purpose: get test statistics
#
####################

def test_ML(test_data, model,final_features,transformer_impute,transformer_scale):
    X=test_data[final_features]
    y_test=test_data['pass']
        
    #X_transform=transformer_impute.transform(X)

    #X_transform = pd.DataFrame(X_transform, columns=final_features)

    #X_transform=transformer_scale.transform(X_transform)

    #X_transform = pd.DataFrame(X_transform, columns=final_features)

    #is_cat = (X.dtypes != float)

    #X_transform = pd.DataFrame(X_transform, columns=final_features)
    #for feature, feat_is_cat in is_cat.to_dict().items():
     #   if feat_is_cat:
       #     X_transform[feature].fillna(0, inplace=True)
       #     X_transform[feature].replace([np.inf, -np.inf], 0, inplace=True)
        #    X_transform[feature]=X_transform[feature].astype(int)

    y_pred=model.predict(X)
    print(f"AUC --> {roc_auc_score(y_test, y_pred)}")
    print(f"Accuracy --> {accuracy_score(y_test, y_pred)}")
    
    
    cm = confusion_matrix(y_test, y_pred)

    cm_display = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = ["Run", "Pass"])
  
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    cm_display.plot(ax=axes[0])
    axes[0].set_title('Confusion Matrix')



    probs = model.predict_proba(X)
    preds = probs[:,1]
    fpr, tpr, threshold = roc_curve(y_test, preds)
    roc_auc = auc(fpr, tpr)

    axes[1].plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    axes[1].set_title('Receiver Operating Characteristic')
    axes[1].legend(loc = 'lower right')
    axes[1].plot([0, 1], [0, 1],'r--')
    
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_xlabel('False Positive Rate')
    plt.tight_layout()
    plt.show()


############################################
#
# function: calc_tempo
# purpose: incorporate tempo info 
#
############################################

def calc_tempo(df_plays):

    df_plays.sort_values(by=['gameId','playId'],inplace=True)
    df_plays.reset_index(inplace=True)
    last_team = df_plays['possessionTeam'][0] # monitor what the last team updated was, implies switch if different
    last_game = df_plays['gameId'][0] # if game switched, we're likely on a new drive
    pnum=0 # play number of drive
    pc = 0 # pass count
    curr_pr_ls = [] # pass rate for current drive
    pr_ls = [] # overall pass_rate ls
    curr_clock_ls = [] # play clock for run
    clock_ls = [] # play clock tracker
    curr_epa_ls = [] #list of current drive epa
    epa_ls = [] #overall epa list

    # loop over plays
    for index, row in df_plays.iterrows():

        curr_team = row['possessionTeam']
        curr_game = row['gameId']

        # if we've switched teams, reset drive tracking info/add last drive's info to running list
        if (last_team != curr_team) | (last_game != curr_game):

            # reset pass count, play number for drive

            last_team = curr_team # reset team to know we're on current drive now
            pc = 0 # reset pass count, etc.
            pnum = 0

            if last_game != curr_game:
                last_game = curr_game

            # append current clock, epa, pass rate stats to running lists
            clock_ls.append([10] + list(np.cumsum(curr_clock_ls)/np.arange(1,len(curr_clock_ls)+1))[:-1]) # assume 10 seconds left on play clock, can adjust later
            pr_ls.append([.6] + curr_pr_ls[:-1]) # lookback of one, use .6 for first play of drive (default pass rate)
            epa_ls.append([.0] + list(np.cumsum(curr_epa_ls)/np.arange(1,len(curr_epa_ls)+1))[:-1]) # inelegantly impute an EPA of zero for our first timestep
            
            # reset current drive stat lists
            curr_pr_ls = []
            curr_clock_ls = []
            curr_epa_ls = []
            

        # update current drive's pass rate
        if row['isDropback']:
            pc+=1
        pnum += 1
        pr = pc/pnum

        # get current mean clock used per drive
        clock = row['playClockAtSnap']
        epa = row['expectedPointsAdded']
        
        # update pass rate, play number, possession, etc. for current drive
        curr_pr_ls.append(pr)
        curr_clock_ls.append(clock)
        curr_epa_ls.append(epa)

    # if new drive not logged, append
    if len(list(chain(*pr_ls))) < len(df_plays):
        pr_ls.append([.6] + curr_pr_ls[:-1])
    if len(list(chain(*clock_ls))) < len(df_plays):
        clock_ls.append([10] + curr_clock_ls[:-1])
    if len(list(chain(*epa_ls))) < len(df_plays):
        epa_ls.append([0] + curr_epa_ls[:-1])

    # flatten running lists using using iter chain
    pr_flat = list(chain(*pr_ls))
    clock_flat = list(chain(*clock_ls))
    epa_flat = list(chain(*epa_ls))

    # estalish new features
    df_plays['drive_pass_rate'] = pr_flat
    df_plays['mean_clocksnap'] = clock_flat
    df_plays['mean_epa'] = epa_flat

    # define 'tempo'
    df_plays['tempo'] = (.1*df_plays['mean_clocksnap'] * df_plays['drive_pass_rate']) - df_plays['mean_epa']
    df_plays['tempo'] = df_plays['tempo']/df_plays['tempo'].max()

##########################################################
#
# function: get_game_pressure
# purpose: get pressure rate faced for game/team combos
#
######################################################

def get_game_pressure(df_play,df_player_play):

    df_pp_cp = df_player_play.groupby(['gameId','playId']).agg(pressure_play=('causedPressure','any')).reset_index()
    df_play = df_play.merge(df_pp_cp,how='left')
    df_play.sort_values(by=['gameId','possessionTeam','playId'],inplace=True)
    df_play['pressure_ewm_pre'] = df_play.groupby(['gameId','possessionTeam'])['pressure_play'].transform(lambda x: x.ewm(alpha=.1).mean())
    df_play['pressure_ewm'] = df_play.groupby(['gameId','possessionTeam']).pressure_ewm_pre.shift(1)
    df_play['pressure_ewm'] = df_play['pressure_ewm'].fillna(.19)
    return(df_play)