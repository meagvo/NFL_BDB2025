import os 
import pandas as pd 
import numpy as np
from data_cleaning import normalize_tracking, feature_engineering, get_position_count, label_run_or_pass, pivot_data
import nfl_data_py as nfl
def load_weather_data():
#get game weather
    weather_df=nfl.import_pbp_data(years=[2022])[['old_game_id_x',  'weather' ]].drop_duplicates()
    weather_df['gameId']=weather_df['old_game_id_x'].astype(int)
    weather_df['Rain']=np.where(weather_df['weather'].str.contains('Rain'), 1, 0)
    weather_df[['temp', 'humidity', 'wind']]=weather_df['weather'].str.split(',', expand=True).iloc[:, :3]
    weather_df['temp'] = weather_df['temp'].str.split(':', expand=True).iloc[:, 1:2]
    weather_df['temp']=weather_df['temp'].str.extract('(\d+)').astype(float)
    weather_df['humidity'] = weather_df['humidity'].str.extract('(\d+)').astype(float)
    weather_df['wind'] = weather_df['wind'].str.extract('(\d+)').astype(float)
    return weather_df
#pull stadium data for current season games

def load_stadium_data():
    df_stadium=nfl.import_schedules([2022])
    df_stadium=df_stadium[df_stadium['week']<=9][[ 'old_game_id','roof', 'surface']]
    return df_stadium
def load_previous_year_data(year: int):
    #pull previous year pass rush data by team
    df_rush2021=nfl.import_ngs_data(stat_type='rushing', years=[year])[['team_abbr', 'rush_attempts']].groupby('team_abbr').sum().reset_index()
    df_pass2021=nfl.import_ngs_data(stat_type='passing', years=[year])[['team_abbr', 'attempts']].groupby('team_abbr').sum().reset_index()
    df_pass_rush2021=pd.merge(df_rush2021, df_pass2021, on='team_abbr', how='outer')
    df_pass_rush2021['pass_rush_ratio']=np.round((df_pass_rush2021['attempts']/df_pass_rush2021['rush_attempts']), 2)
    df_pass_rush2021.drop(columns=['rush_attempts', 'attempts'], inplace=True)
    return df_pass_rush2021
def load_ftn():
    pbp = nfl.import_pbp_data([2022])
    ftn = nfl.import_ftn_data([2022])
    pbp_ids = pbp[['play_id','game_id','old_game_id_x']]
    ftn['nflverse_play_id'] = ftn['nflverse_play_id'].astype(int)
    pbp_ids['play_id'] = pbp_ids['play_id'].astype(int)
    ftn['nflverse_game_id'] = ftn['nflverse_game_id'].astype(str)
    pbp_ids['game_id'] = pbp_ids['game_id'].astype(str)
    ftn_merged = pbp_ids.merge(ftn,how='left',left_on=['play_id','game_id'],
                right_on=['nflverse_play_id','nflverse_game_id'])
    ftn_merged = ftn_merged[['play_id','old_game_id_x','n_offense_backfield',
                            'n_defense_box','is_no_huddle','is_motion']].rename(columns={'old_game_id_x':'gameId',
                                                                                       'play_id':'playId'})
    ftn_merged['gameId'] = ftn_merged['gameId'].astype(int)
    ftn_merged['playId'] = ftn_merged['playId'].astype(int)
    return ftn_merged

def load_tracking_data(tracking_fname: str):
    df_tracking=pd.read_csv(tracking_fname)
    df_tracking=df_tracking[ (df_tracking['frameType']=='BEFORE_SNAP')& (df_tracking['event']!='huddle_break_offense')]
    df_tracking['gameplayId']=df_tracking['gameId'].astype(str)+'_'+df_tracking['playId'].astype(str)
    playstodrop=df_tracking[df_tracking['event'].isin(['huddle_start_offense', 'timeout_away'])][['gameplayId']].drop_duplicates() #plays with huddle start or timeout we should drop
    df_tracking = df_tracking[~df_tracking['gameplayId'].isin(playstodrop['gameplayId'])]
    
    return normalize_tracking(df_tracking)

def aggregate_data(  plays_fname, player_plays_fname, players_fname, tracking_fname_list, games_fname, xp_fname, pr_fname, cf_fname, cu_fname):
    """
    Create the aggregate dataframe by merging together the plays data and tracking data

    :param plays_fname: the filename of the plays data
    :param player_plays_fname: the filename of the playerplay data
    :param players_fname: the filename of the players data
    :param tracking_fname_list: a list of filenames of all tracking data

    :return df_final: the aggregate dataframe
    """

    # import files
    df_games=pd.read_csv(games_fname)
    
    df_games=pd.merge(df_games, load_stadium_data(),left_on='gameId', right_on='old_game_id', how='left')
    df_games=pd.merge(df_games, load_weather_data(),on='gameId', how='left')
    df_plays = feature_engineering(pd.read_csv(plays_fname))
    df_tracking = pd.concat(
        [load_tracking_data(tracking_fname) for tracking_fname in tracking_fname_list]
    )
    df_tracking=df_tracking[['gameId', 'playId', 'nflId','club' ,'o_standard', 'dir_standard', 'x_standard', 'y_standard', 's', 'a', 'dis']].groupby(['gameId', 'playId', 'nflId','club']).agg({'s':[ 'max',],'a':[ 'max'], 
    'o_standard':['mean', 'std'],'dis':['sum'],'dir_standard':['mean', 'std'], 'x_standard':['mean', 'std'], 'y_standard':['mean', 'std']}).reset_index()
    df_tracking.columns=df_tracking.columns.map('|'.join).str.strip('|')
    df_players = pd.read_csv(players_fname)
    df_player_plays=pd.read_csv(player_plays_fname)
    df_player_plays=pd.merge(df_player_plays, load_previous_year_data(2021), left_on='teamAbbr', right_on='team_abbr', how='outer')
    # aggregate plays, tracking, players tables
    df_agg1=pd.merge(pd.merge(df_tracking, df_plays, left_on=[ 'gameId', 'playId','club'], right_on=[ 'gameId', 'playId', 'possessionTeam'], how='inner'), df_player_plays, left_on=[ 'gameId', 'playId', 'nflId'], right_on=[ 'gameId', 'playId', 'nflId'], how='left' )
    df_agg2=pd.merge(df_agg1, df_games, on='gameId', how='inner')
    df_final=pivot_data(label_run_or_pass(get_position_count(pd.merge(df_agg2, df_players, on='nflId', how='inner'))))
    merged_id_df = df_final[['gameId','playId']].drop_duplicates()
    ftn_merged=load_ftn()
    xp_df = pd.read_csv(xp_fname).drop(columns='Unnamed: 0')
    pr_df = pd.read_csv(pr_fname).drop(columns='Unnamed: 0')
    merged_base = merged_id_df.merge(ftn_merged,how='left',on=['gameId','playId'])
    merged_base = merged_base.merge(pr_df,how='left',on=['gameId','playId'])
    merged_base = merged_base.merge(xp_df,how='left',on=['gameId','playId'])
    merged_base = merged_base.merge(df_games[['gameId','week']].drop_duplicates(),how='left',on=['gameId'])
    merged_base = merged_base.merge(df_plays[['gameId','playId',
                                          'possessionTeam','defensiveTeam']].drop_duplicates(),
                                how='left',on=['gameId','playId'])
    cf_df = pd.read_csv(cf_fname).drop(columns='Unnamed: 0')
    cu_df = pd.read_csv(cu_fname).drop(columns='Unnamed: 0')
    merged_base = merged_base.merge(cf_df,how='left',on=['possessionTeam','week'])
    merged_base = merged_base.merge(cu_df,how='left',on=['defensiveTeam','week'])
    
    return pd.concat([df_final,merged_base.iloc[:,2:]],axis=1)
