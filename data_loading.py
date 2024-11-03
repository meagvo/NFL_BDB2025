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

def aggregate_data(plays_fname, player_plays_fname, players_fname, tracking_fname_list, games_fname, xp_fname, pr_fname, cu_fname, inj_fname, c21_fname, pr21_fname, qbr_fname, agg_flag):
    # Route to whether we want test or train data
    if agg_flag == 'train':
        return aggregate_train(plays_fname, player_plays_fname, players_fname, tracking_fname_list, games_fname, xp_fname, pr_fname, cu_fname, inj_fname, c21_fname, pr21_fname, qbr_fname)
    elif agg_flag == 'test':
        return aggregate_test(plays_fname, player_plays_fname, players_fname, tracking_fname_list, games_fname, xp_fname, pr_fname, cu_fname, inj_fname, qbr_fname)
    
def aggregate_test(plays_fname, player_plays_fname, players_fname, tracking_fname_list, games_fname, xp_fname, pr_fname, cu_fname, inj_fname, qbr_fname):
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
    df_players.loc[df_players['nflId'] ==45244, 'position'] = 'TE' #update data for Taysom Hill
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
    
    cu_df = pd.read_csv(cu_fname)
    merged_base = merged_base.merge(cu_df,how='left',on=['defensiveTeam','week'])

    # add injury data
    inj_df = pd.read_csv(inj_fname)

    # merge in offensive snaps
    merged_base = merged_base.merge(inj_df.drop(columns=['def_snaps_lost']),how='left',
                    left_on=['possessionTeam','week'], right_on=['team','week']).drop(columns=['team'])

    # merge in defensive snaps
    merged_base = merged_base.merge(inj_df.drop(columns=['off_snaps_lost']),how='left',
                    left_on=['defensiveTeam','week'], right_on=['team','week']).drop(columns=['team'])

    # incorp qbr data, impute week 0 generic stats from years prior for direct snap plays
    qbr_df = pd.read_csv(qbr_fname)
    merged_base = merged_base.merge(qbr_df,how='left',on=['gameId','playId'])
    med_dict = {'qbr_total':57.500000,'pass_val':32.661255,'run_val':3.157565,'ybc_att':2.756089,'yac_att':0.700000,'qb_plays':0}
    merged_base.fillna(med_dict,inplace=True)

    # add situational xpass
    nfl_pbp = nfl.clean_nfl_data(nfl.import_pbp_data([2022]))
    xpass_df = nfl_pbp[nfl_pbp['season_type'] == 'REG'][['play_id','old_game_id_x','xpass']].rename(columns={'xpass':'xpass_situational'})
    xpass_df['play_id'] = xpass_df['play_id'].astype(int)
    xpass_df['old_game_id_x'] = xpass_df['old_game_id_x'].astype(int)

    merged_base = merged_base.merge(xpass_df,how='left',left_on=['gameId','playId'], right_on=['old_game_id_x','play_id'])
    merged_base['xpass_situational'] = merged_base['xpass_situational'].fillna(.5)
 
    return pd.concat([df_final,merged_base.iloc[:,2:]],axis=1)

def aggregate_train(  plays_fname, player_plays_fname, players_fname, tracking_fname_list, games_fname, xp_fname, pr_fname, cu_fname, inj_fname, c21_fname, pr21_fname, qbr_fname):
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
    df_players.loc[df_players['nflId'] ==45244, 'position'] = 'TE' #update data for Taysom Hill
    df_player_plays=pd.read_csv(player_plays_fname)
    df_player_plays=pd.merge(df_player_plays, load_previous_year_data(2021), left_on='teamAbbr', right_on='team_abbr', how='outer')
    # aggregate plays, tracking, players tables
    df_agg1=pd.merge(pd.merge(df_tracking, df_plays, left_on=[ 'gameId', 'playId','club'], right_on=[ 'gameId', 'playId', 'possessionTeam'], how='inner'), df_player_plays, left_on=[ 'gameId', 'playId', 'nflId'], right_on=[ 'gameId', 'playId', 'nflId'], how='left' )
    df_agg2=pd.merge(df_agg1, df_games, on='gameId', how='inner')
    df_final=pivot_data(label_run_or_pass(get_position_count(pd.merge(df_agg2, df_players, on='nflId', how='inner'))))
    merged_id_df = df_final[['gameId','playId']].drop_duplicates()
    
    # add week info
    merged_base = merged_id_df.merge(df_games[['gameId','week']].drop_duplicates(),how='left',on=['gameId'])

    # add in team info
    merged_base = merged_base.merge(df_plays[['gameId','playId',
                                          'possessionTeam','defensiveTeam']].drop_duplicates(),
                                how='left',on=['gameId','playId'])

    w1_ids = merged_base[merged_base['week'] == 1][['gameId','playId','week','possessionTeam','defensiveTeam']]
    w2_on_ids = merged_base[merged_base['week'] > 1][['gameId','playId','week','possessionTeam','defensiveTeam']]

    cov_21 = pd.read_csv(c21_fname)
    team_pr_21 = pd.read_csv(pr21_fname)
    xp_df = pd.read_csv(xp_fname).drop(columns='Unnamed: 0')
    pr_df = pd.read_csv(pr_fname).drop(columns='Unnamed: 0')


    # merge defensive pass rates for '21 into week 1
    w1_pr = w1_ids.merge(team_pr_21.drop(columns=['pass_rate_off']).rename(columns={'possessionTeam':'defensiveTeam'}),
                        on='defensiveTeam',how='left')

    # offensive
    w1_pr = w1_pr.merge(team_pr_21.drop(columns=['pass_rate_def']),on='possessionTeam',how='left')

    # set week 1 xpass to just be the team's default pass rate
    w1_pr['off_xpass'] = w1_pr['pass_rate_off'].copy()
    w1_pr['def_xpass'] = w1_pr['pass_rate_def'].copy()

    # subset to only defensive features, rename features
    cov_def = cov_21[[x for x in cov_21.columns if '_off' not in x]].rename(columns={'possessionTeam':'defensiveTeam'})
    cov_def = cov_def.rename(columns={'cover_2_def':'Cover-2_def','cover_0_def':'Cover-0_def'})

    #merge into running dataframe
    w1_merged = w1_pr.merge(cov_def,how='left',on='defensiveTeam')

    cu_df = pd.read_csv(cu_fname)
    cu_df['week'] = cu_df['week'].astype(int)
    cu_w2_on = w2_on_ids.merge(cu_df,how='left',left_on=['defensiveTeam','week'],right_on=['defensiveTeam','week'])

    # integrate pass rate df
    df_w2_on = cu_w2_on.merge(pr_df,how='left',on=['gameId','playId'])
    df_w2_on = df_w2_on.merge(xp_df,how='left',on=['gameId','playId'])

    #make columns ordered for week 2 onward
    w1_merged = w1_merged[df_w2_on.columns]

    # aggregate week 2-onward and week 1
    merged_base = pd.concat([w1_merged,df_w2_on],axis=0)

    # add injury data
    inj_df = pd.read_csv(inj_fname)

    # merge in offensive snaps
    merged_base = merged_base.merge(inj_df.drop(columns=['def_snaps_lost']),how='left',
                    left_on=['possessionTeam','week'], right_on=['team','week']).drop(columns=['team'])

    # merge in defensive snaps
    merged_base = merged_base.merge(inj_df.drop(columns=['off_snaps_lost']),how='left',
                    left_on=['defensiveTeam','week'], right_on=['team','week']).drop(columns=['team'])

    # add ftn
    ftn_merged = load_ftn()
    merged_base = merged_base.merge(ftn_merged,how='left',on=['gameId','playId'])

    # order df correctly
    cols_final = ['gameId', 'playId', 'n_offense_backfield', 'n_defense_box',
        'is_no_huddle', 'is_motion', 'pass_rate_off', 'pass_rate_def',
        'off_xpass', 'def_xpass', 'week', 'possessionTeam', 'defensiveTeam',
        'cover_3_def', 'cover_6_def', 'cover_1_def', 'Quarters_def',
        'Cover-2_def', 'Cover-0_def', 'Man_def', 'Other_def', 'Zone_def',
        'off_snaps_lost', 'def_snaps_lost']

    merged_base = merged_base[cols_final]

    # incorp qbr data, impute week 0 generic stats from years prior for direct snap plays
    qbr_df = pd.read_csv(qbr_fname)
    merged_base = merged_base.merge(qbr_df,how='left',on=['gameId','playId'])
    med_dict = {'qbr_total':57.500000,'pass_val':32.661255,'run_val':3.157565,'ybc_att':2.756089,'yac_att':0.700000,'qb_plays':0}
    merged_base.fillna(med_dict,inplace=True)

    # add situational xpass
    
    nfl_pbp = nfl.clean_nfl_data(nfl.import_pbp_data([2022]))
    xpass_df = nfl_pbp[nfl_pbp['season_type'] == 'REG'][['play_id','old_game_id_x','xpass']].rename(columns={'xpass':'xpass_situational'})
    xpass_df['play_id'] = xpass_df['play_id'].astype(int)
    xpass_df['old_game_id_x'] = xpass_df['old_game_id_x'].astype(int)

    merged_base = merged_base.merge(xpass_df,how='left',left_on=['gameId','playId'], right_on=['old_game_id_x','play_id'])
    merged_base['xpass_situational'] = merged_base['xpass_situational'].fillna(.5)
    

    return pd.concat([df_final,merged_base.iloc[:,2:]],axis=1)
