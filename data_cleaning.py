
import numpy as np 
import pandas as pd

def label_run_or_pass(df: pd.DataFrame) -> pd.DataFrame:
    print("INFO: Labeling plays as runs or passes")
    df["pass"] = np.where(df['isDropback']==True, 1, 0)
    return df
def reverse_deg(deg):
    if deg < 180:
        return deg + 180
    if deg >= 18:
        return deg - 180
def normalize_tracking(df_tracking):
    df_tracking['o_standard']=np.where(df_tracking["playDirection"] == "left",df_tracking['o'].apply(reverse_deg), df_tracking['o'])
    df_tracking['dir_standard']=np.where(df_tracking["playDirection"] == "left",df_tracking['dir'].apply(reverse_deg), df_tracking['dir'])
    df_tracking["x_standard"] = np.where(df_tracking["playDirection"] == "left", df_tracking["x"].apply(lambda x: 120 - x), df_tracking["x"])
    df_tracking["y_standard"] =np.where(df_tracking["playDirection"] == "left",  df_tracking["y"].apply(lambda y: 160/3 - y), df_tracking["y"])
    return df_tracking

def pivot_data(df):
    # pivot data and get dummy variables
    df_pivot=df.pivot(index=['gameId', 'playId','quarter', 'down', 'yardsToGo', 'score_differential', 'time_remaining', 'playNullifiedByPenalty','preSnapHomeTeamWinProbability', 'pass_rush_ratio', 'roof', 'surface', 'Rain', 'temp', 'humidity', 'wind',
           'preSnapVisitorTeamWinProbability',
           'absoluteYardlineNumber', 'offenseFormation', 'receiverAlignment',
           'pass'], columns=['position_pivot'], values=['s|max', 'a|max',
       'o_standard|mean', 'o_standard|std', 'dis|sum', 'dir_standard|mean',
       'dir_standard|std', 'x_standard|mean', 'x_standard|std',
       'y_standard|mean', 'y_standard|std', 'shiftSinceLineset', 'motionSinceLineset']).fillna(0).reset_index() 
    df_pivot.columns =df_pivot.columns.map('|'.join).str.strip('|')
    df_pivot.replace({False:0, True:1}, inplace=True)
    df_pivot = pd.get_dummies(df_pivot, columns=['offenseFormation','receiverAlignment', 'roof', 'surface'])
    return df_pivot
def feature_engineering(df_plays):
    #features for plays df 
    df_plays['time_remaining'] = df_plays['quarter'].map({1: 45, 2: 30, 3: 15, 4: 0, 5:0}) +                                df_plays['gameClock'].apply(lambda x: int(x.split(':')[0]) + int(x.split(':')[1])/60)
    df_plays['score_differential'] = df_plays['preSnapHomeScore'] - df_plays['preSnapVisitorScore']
    return df_plays
def get_position_count(df):
    df['pos_count']=(df[['gameId', 'playId', 'position', 'nflId']].drop_duplicates().sort_values(by=['gameId', 'playId', 'position']).groupby(by=['gameId', 'playId', 'position']).cumcount()+1)
    df['pos_count'].fillna(99, inplace=True)
    df['pos_count']=df['pos_count'].astype(int)
    #create a position_pivot column so that each player has a unique value during the play
    df['position_pivot']=df['position']+'_'+df['pos_count'].astype(str)
    return df
def change_columns_types(df):
    df[['is_no_huddle']] = df[['is_no_huddle']].astype(int)
    df[['is_motion']] = df[['is_motion']].astype(int)
    df['yardsToGo']=df['yardsToGo'].astype(float)
    df['down']=df['down'].astype(float)
    df['quarter']=df['quarter'].astype(float)
    return df
