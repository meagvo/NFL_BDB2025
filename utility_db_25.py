from sklearn.model_selection import StratifiedKFold,cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score, precision_score, recall_score
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

    oof_non_rounded = np.zeros(len(y), dtype=float) 


    for fold, (train_idx, test_idx) in enumerate(tqdm(SKF.split(X, y), desc="Training Folds", total=n_splits)):
        X_train, X_val = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[test_idx]

        model = clone(model_class)
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        oof_non_rounded[test_idx] = y_val_pred
       
     

        train_auc = roc_auc_score(y_train, y_train_pred.round(0).astype(int))
        val_auc = roc_auc_score(y_val, y_val_pred.round(0))

        train_S.append(train_auc)
        test_S.append(val_auc)

        train_accuracy = accuracy_score(y_train, y_train_pred.round(0).astype(int))
        val_accuracy = accuracy_score(y_val, y_val_pred.round(0))

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
    final_features=list(correlations[((correlations['pass']>.03) | (correlations['pass']<-.03))].T.columns.values)

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
        
    X_transform=transformer_impute.transform(X)

    X_transform = pd.DataFrame(X_transform, columns=final_features)

    X_transform=transformer_scale.transform(X_transform)

    X_transform = pd.DataFrame(X_transform, columns=final_features)

    is_cat = (X.dtypes != float)

    X_transform = pd.DataFrame(X_transform, columns=final_features)
    for feature, feat_is_cat in is_cat.to_dict().items():
        if feat_is_cat:
            X_transform[feature].fillna(0, inplace=True)
            X_transform[feature].replace([np.inf, -np.inf], 0, inplace=True)
            X_transform[feature]=X_transform[feature].astype(int)

    y_pred=model.predict(X_transform)
    print(f"AUC --> {roc_auc_score(y_test, y_pred)}")
    print(f"Accuracy --> {accuracy_score(y_test, y_pred)}")
    
    
    cm = confusion_matrix(y_test, y_pred)

    cm_display = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = [0, 1])

    cm_display.plot()
    plt.show()