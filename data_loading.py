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
    pbp_ids = pbp[['play_id','game_id','old_game_id_x']].copy()
    ftn['nflverse_play_id'] = ftn['nflverse_play_id'].astype(int)
    pbp_ids['play_id'] = pbp_ids['play_id'].astype(int)
    ftn['nflverse_game_id'] = ftn['nflverse_game_id'].astype(str)
    pbp_ids['game_id'] = pbp_ids['game_id'].astype(str)
    ftn_merged_pre = pbp_ids.merge(ftn,how='left',left_on=['play_id','game_id'],
                right_on=['nflverse_play_id','nflverse_game_id'])
    ftn_merged = ftn_merged_pre[['play_id','old_game_id_x','n_offense_backfield',
                            'n_defense_box','is_no_huddle','is_motion']].copy().rename(columns={'old_game_id_x':'gameId',
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

def aggregate_data(plays_fname, player_plays_fname, players_fname, tracking_fname_list, games_fname, xp_fname, pr_fname, cu_fname, inj_fname, c21_fname, pr21_fname, qbr_fname, def_fname, agg_flag):
    # Route to whether we want test or train data

    ### TODO ### 
    # integrate this stuff one-by-one:
    df_players = pd.read_csv(players_fname)
    df_player_plays = pd.read_csv(player_plays_fname)
    df_plays = feature_engineering(pd.read_csv(plays_fname))
    df_games=pd.read_csv(games_fname)
    df_games=pd.merge(df_games, load_stadium_data(),left_on='gameId', right_on='old_game_id', how='left')
    df_games=pd.merge(df_games, load_weather_data(),on='gameId', how='left')
    df_plays = count_box_bmi(df_plays,df_players,df_player_plays)
    rate_df = get_qb_rates(df_plays,df_player_plays,df_players,df_games)

    # route between test/train
    if agg_flag == 'train':
        return aggregate_train(df_plays, df_player_plays, df_players, tracking_fname_list, df_games, xp_fname, pr_fname, cu_fname, inj_fname, c21_fname, pr21_fname, qbr_fname, def_fname,rate_df)
    elif agg_flag == 'test':
        return aggregate_test(df_plays, df_player_plays, df_players, tracking_fname_list, df_games, xp_fname, pr_fname, cu_fname, inj_fname, qbr_fname, def_fname,rate_df)
    
def aggregate_test(df_plays, df_player_plays, df_players, tracking_fname_list, df_games, xp_fname, pr_fname, cu_fname, inj_fname, qbr_fname, def_fname,rate_df):
    """
    Create the aggregate dataframe by merging together the plays data and tracking data

    :param plays_fname: the filename of the plays data
    :param player_plays_fname: the filename of the playerplay data
    :param players_fname: the filename of the players data
    :param tracking_fname_list: a list of filenames of all tracking data

    :return df_final: the aggregate dataframe
    """

    df_tracking = pd.concat(
        [load_tracking_data(tracking_fname) for tracking_fname in tracking_fname_list]
    )
    df_tracking=df_tracking[['gameId', 'playId', 'nflId','club' ,'o_standard', 'dir_standard', 'x_standard', 'y_standard', 's', 'a', 'dis']].groupby(['gameId', 'playId', 'nflId','club']).agg({'s':[ 'max',],'a':[ 'max'], 
    'o_standard':['mean', 'std'],'dis':['sum'],'dir_standard':['mean', 'std'], 'x_standard':['mean', 'std'], 'y_standard':['mean', 'std']}).reset_index()
    df_tracking.columns=df_tracking.columns.map('|'.join).str.strip('|')
    df_players.loc[df_players['nflId'] ==45244, 'position'] = 'TE' #update data for Taysom Hill
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

    
    # add new defensive metrics
    def_df = pd.read_csv(def_fname).drop(columns='Unnamed: 0')
    def_df['gameId'] = def_df['gameId'].astype(int)
    def_df['playId'] = def_df['playId'].astype(int)
    merged_base = merged_base.merge(def_df,how='left',left_on=['gameId','playId'], right_on=['gameId','playId'])

    # integrate tempo, bmi/box data
    merged_base = merged_base.merge(df_plays[['gameId','playId','tempo','box_ewm_dl_bmi','box_ewm','mean_DL_bmi']],how='left',left_on=['gameId','playId'], right_on=['gameId','playId'])
    merged_base = merged_base.merge(rate_df,how='left',on=['gameId','playId'])

    return pd.concat([df_final,merged_base.iloc[:,2:]],axis=1)

def aggregate_train( df_plays, df_player_plays, df_players, tracking_fname_list, df_games, xp_fname, pr_fname, cu_fname, inj_fname, c21_fname, pr21_fname, qbr_fname, def_fname,rate_df):
    """
    Create the aggregate dataframe by merging together the plays data and tracking data

    :param plays_fname: the filename of the plays data
    :param player_plays_fname: the filename of the playerplay data
    :param players_fname: the filename of the players data
    :param tracking_fname_list: a list of filenames of all tracking data

    :return df_final: the aggregate dataframe
    """
    
    df_tracking = pd.concat(
        [load_tracking_data(tracking_fname) for tracking_fname in tracking_fname_list]
    )
    df_tracking=df_tracking[['gameId', 'playId', 'nflId','club' ,'o_standard', 'dir_standard', 'x_standard', 'y_standard', 's', 'a', 'dis']].groupby(['gameId', 'playId', 'nflId','club']).agg({'s':[ 'max',],'a':[ 'max'], 
    'o_standard':['mean', 'std'],'dis':['sum'],'dir_standard':['mean', 'std'], 'x_standard':['mean', 'std'], 'y_standard':['mean', 'std']}).reset_index()
    df_tracking.columns=df_tracking.columns.map('|'.join).str.strip('|')
    df_players.loc[df_players['nflId'] ==45244, 'position'] = 'TE' #update data for Taysom Hill
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

    
    # add new defensive metrics
    def_df = pd.read_csv(def_fname).drop(columns='Unnamed: 0')
    def_df['gameId'] = def_df['gameId'].astype(int)
    def_df['playId'] = def_df['playId'].astype(int)
    merged_base = merged_base.merge(def_df,how='left',left_on=['gameId','playId'], right_on=['gameId','playId'])

    # add tempo, bmi/box data, pass rate info
    merged_base = merged_base.merge(df_plays[['gameId','playId','tempo','box_ewm_dl_bmi','box_ewm','mean_DL_bmi']],how='left',left_on=['gameId','playId'], right_on=['gameId','playId'])
    merged_base = merged_base.merge(rate_df,how='left',on=['gameId','playId'])

    return pd.concat([df_final,merged_base.iloc[:,2:]],axis=1)

############################################
#
# function: count_box_bmi
# purpose: incorporate bmi, box count features
#
############################################

def count_box_bmi(df_play, df_players, df_player_play):
    
    # add box count from ftn
    ftn_df = load_ftn()
    df_play = df_play.merge(ftn_df[['gameId','playId','n_defense_box']],how='left')
    
    # get box ct info
    df_play['box_ewm_pre'] = df_play.groupby(['gameId','possessionTeam'])['n_defense_box'].transform(lambda x: x.ewm(alpha=.1).mean())
    df_play['box_ewm'] = df_play.groupby(['gameId','possessionTeam']).box_ewm_pre.shift(1)
    df_play['box_ewm'] = df_play['box_ewm'].fillna(6)
    df_play[['gameId','playId','possessionTeam','n_defense_box','box_ewm']].head(6)

    # calc height, bmi
    df_players = pd.concat([df_players,df_players['height'].str.split('-',n=1,expand=True).rename(columns={0:'h_ft',1:'h_in_pre'})],axis=1)
    df_players['height_inches'] = df_players['h_ft'].astype(int)*12 + df_players['h_in_pre'].astype(int)
    df_players['bmi'] = df_players['weight'] /(df_players['height_inches']**2) # weight/height squared

    # incorporate data back into player-play
    df_bmi = df_player_play[['gameId','playId','nflId']].merge(df_players[['nflId','bmi','height_inches','weight','position']])

    # get DL BMI, reintegrate
    dl_df = df_bmi[df_bmi['position'].isin(['DT','NT','DE'])].groupby(['gameId','playId'])['bmi'].mean().reset_index().rename(columns={'bmi':'mean_DL_bmi'})
  
    # integrate into data
    df_play = df_play.merge(dl_df,how='left')

    # get final metric
    df_play['box_ewm_dl_bmi'] = df_play['box_ewm']*df_play['mean_DL_bmi']
    df_play.drop(columns=['n_defense_box'],inplace=True)
    
    return df_play

######################################################
#
# function: get_qb_rates
# purpose: get historical, season-long qb pass rates
#
######################################################

def get_qb_rates(df_play,df_player_play,df_players,df_games):

    # import nflverse data
    pfr_szn = nfl.import_seasonal_pfr('pass',[2021])
    isr = nfl.import_seasonal_rosters(years=[2021],columns=['player_id','pfr_id','gsis_it_id']).drop_duplicates()
    snap_df = nfl.import_snap_counts([2021])

    # get '21 nflverse snap data, merge into pfr pass info
    snap_szn = snap_df.groupby('pfr_player_id')['offense_snaps'].sum().reset_index()
    pfr_sub = pfr_szn[['pfr_id','pa_pass_att','pa_pass_yards','pass_attempts']].merge(snap_szn,how='left',left_on='pfr_id',right_on='pfr_player_id').drop(columns=['pfr_player_id'])

    # calculate PA, overall pass rates for '21
    pfr_sub['qb_pa_rate_pass'] = pfr_sub['pa_pass_att']/pfr_sub['pass_attempts']
    pfr_sub['qb_pa_rate_ovr'] = pfr_sub['pa_pass_att']/pfr_sub['offense_snaps']
    pfr_sub['qb_pass_rate'] = pfr_sub['pass_attempts']/pfr_sub['offense_snaps']
    
    # pa_pass att = arbitrary column, same amt of na's for all features
    pfr_id_rect = isr.merge(pfr_sub,how='left').dropna(subset='pa_pass_att').drop(columns=['player_id','pfr_id'])
    pfr_id_rect = pfr_id_rect[pfr_id_rect['pass_attempts'] >= 20]
    
    # bring in position, PA and pass flags
    df_pos = df_player_play[['gameId','playId','nflId']].merge(df_players[['nflId','position']],how='left')
    df_comp = df_pos.merge(df_play[['gameId','playId','playAction','isDropback']])
    df_comp = df_comp[~df_comp.nflId.isin([45244,54551])]

    # group QB passing data to game level, add explicit week info
    qb_grp = df_comp[df_comp.position=='QB'].groupby(['gameId','nflId']).agg(pa_ct_game=('playAction','sum'),snap_ct=('playAction','count'),pass_ct=('isDropback','sum')).reset_index()
    qb_trunc = qb_grp.merge(df_games[['gameId','week']].drop_duplicates(),how='left').sort_values(by=['nflId','week'])

    # get current-year (2022) pass ratios
    qb_trunc['qb_pass_rate'] = qb_trunc['pass_ct']/qb_trunc['snap_ct']
    qb_trunc['qb_pa_rate_ovr'] = qb_trunc['pa_ct_game']/qb_trunc['snap_ct']
    qb_trunc['qb_pa_rate_pass'] = qb_trunc['pa_ct_game']/qb_trunc['pass_ct']
    qb_trunc['week'] = qb_trunc['week'].astype(int)

    # adjust ID typing
    pfr_id_rect.rename(columns={'gsis_it_id':'nflId'},inplace=True)
    pfr_id_rect['nflId'] = pfr_id_rect['nflId'].astype(str)
    qb_trunc['nflId'] = qb_trunc['nflId'].astype(str)

    # set '21 data as week 1, uptick other weeks
    pfr_id_rect['week']=1
    qb_trunc['week']+=1

    # stack '21 (our proxy week 1) and '22 (week 2-9) data
    reduce_cols = ['nflId','week','qb_pass_rate','qb_pa_rate_ovr','qb_pa_rate_pass']
    qb_w_21 = pd.concat([pfr_id_rect[reduce_cols],qb_trunc[reduce_cols]],axis=0)

    # get cumulative sum for each feature
    mean_cols = ['qb_pass_rate','qb_pa_rate_ovr','qb_pa_rate_pass']
    qb_w_21.sort_values(by=['nflId','week'],inplace=True)
    qb_full = pd.concat([qb_w_21[['nflId','week']],qb_w_21.groupby(['nflId'])[mean_cols].cumsum()],axis=1)
    qb_full = pd.concat([qb_full,qb_full.groupby(['nflId']).agg(qb_week=('week','cumcount'))],axis=1)
    qb_full['qb_week']+=1 # num. start of qb's season; default start ind 0, so bump by 1

    # get average by dividing by week num.
    for col in mean_cols: 
        qb_full[col] = qb_full[col].values/qb_full['qb_week'].values

    qb_full.drop(columns=['qb_week'],inplace=True)

    # get EWM pass rate
    ewm_temp = qb_w_21.copy()
    for col in mean_cols:
        ewm_temp[col+'_ewm'] = ewm_temp.groupby(['nflId'])[col].transform(lambda x: x.ewm(alpha=.1).mean())
        ewm_temp.drop(columns=[col],inplace=True)

    # get all-week info with cartesian product
    qb_full = qb_full.merge(ewm_temp,how='left',on=['nflId','week'])
    ci = pd.merge(qb_trunc['nflId'].drop_duplicates(), pd.Series(list(range(1,10))).rename('week'), how='cross',copy=False).sort_values(by=['nflId','week'])
    qb_aw = ci.merge(qb_full,how='left')
    qb_ffill_pre = pd.concat([qb_aw['nflId'],qb_aw.groupby('nflId').ffill()],axis=1)

    # fix div. 0 issue
    qb_ffill_pre.loc[qb_ffill_pre['qb_pass_rate'] == 0,'qb_pa_rate_ovr'] = 0
    qb_ffill_pre.loc[qb_ffill_pre['qb_pass_rate'] == 0,'qb_pa_rate_pass'] = 0

    # fill rest w/base rates
    qb_ffill_pre['qb_pass_rate'] = qb_ffill_pre['qb_pass_rate'].fillna(.54)
    qb_ffill_pre['qb_pa_rate_ovr'] = qb_ffill_pre['qb_pa_rate_ovr'].fillna(.13)
    qb_ffill_pre['qb_pa_rate_pass'] = qb_ffill_pre['qb_pa_rate_pass'].fillna(.24)
    qb_ffill_pre['qb_pass_rate_ewm'] = qb_ffill_pre['qb_pass_rate_ewm'].fillna(.54)
    qb_ffill_pre['qb_pa_rate_ovr_ewm'] = qb_ffill_pre['qb_pa_rate_ovr_ewm'].fillna(.13)
    qb_ffill_pre['qb_pa_rate_pass_ewm'] = qb_ffill_pre['qb_pa_rate_pass_ewm'].fillna(.24)

    # reintegrate data
    qb_wk = df_comp[df_comp.position=='QB'].merge(df_games[['gameId','week']].drop_duplicates(),how='left')[['gameId','playId','nflId','week']]
    qb_ffill_pre['nflId'] = qb_ffill_pre['nflId'].astype(int)
    rate_df = qb_wk.merge(qb_ffill_pre,how='left')
    return rate_df.drop(columns=['nflId','week','qb_pa_rate_ovr','qb_pass_rate','qb_pa_rate_ovr_ewm','qb_pa_rate_pass_ewm'])
