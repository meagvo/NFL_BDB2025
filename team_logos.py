import nfl_data_py as nfl
import pandas as pd
import numpy as np
import os
import urllib.request
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image



def get_logo_df():
    
    # Pull the team description 
    logos = nfl.import_team_desc()
    logos = logos[['team_abbr', 'team_logo_espn']]
        # Initialize an empty list for the logo file paths
    logo_paths = []

        # Initialize an empty list for the team abbreviations
    team_abbr = []

        # Create a folder for the image files if it doesn't exist

    if not os.path.exists("logos"):
        os.makedirs("logos")
            # Pull the team logos from the URL and save them in the logos folder, save the file paths to
    for team in range(len(logos)):
        if logos['team_abbr'][team]=='NYJ':
            urllib.request.urlretrieve(logos['team_logo_espn'][team], f"logos/{logos['team_abbr'][team]}.tif")
            logo_paths.append(f"logos/{logos['team_abbr'][team]}.tif")
            team_abbr.append(logos['team_abbr'][team])
            image = Image.open(f"logos/{logos['team_abbr'][team]}.tif")
            new_image = image.resize((500, 500))
            new_image.save(f"logos/{logos['team_abbr'][team]}.tif")
        else:    
            urllib.request.urlretrieve(logos['team_logo_espn'][team], f"logos/{logos['team_abbr'][team]}.tif")
            logo_paths.append(f"logos/{logos['team_abbr'][team]}.tif")
            team_abbr.append(logos['team_abbr'][team])
            
                

    # Create a dictionary to put logo_paths and team_abbr in
    data = {'team_abbr' : team_abbr, 'Logo Path' : logo_paths}

    # Create a DataFrame from the dictionary
    return pd.DataFrame(data)
# Define a function for getting the image path and loading it into the visualization
def getImage(path):
    return OffsetImage(plt.imread(path), zoom=.1 ,alpha = 1)
def load_previous_year_passruns(year: int):
    #pull previous year pass rush data by team
    df_rush2021=nfl.import_ngs_data(stat_type='rushing', years=[year])[['team_abbr', 'rush_attempts']].groupby('team_abbr').sum().reset_index()
    df_pass2021=nfl.import_ngs_data(stat_type='passing', years=[year])[['team_abbr', 'attempts']].groupby('team_abbr').sum().reset_index()
    df_pass_rush2021=pd.merge(df_rush2021, df_pass2021, on='team_abbr', how='outer')
    df_pass_rush2021['pass_rush_ratio']=np.round((df_pass_rush2021['attempts']/df_pass_rush2021['rush_attempts']), 2)
    return df_pass_rush2021
def graph_run_pass():
    runpass2021=pd.merge(load_previous_year_passruns(2021), get_logo_df(), on='team_abbr', how='left')
    # Define plot size and autolayout
    fig, ax = plt.subplots(figsize=(20, 14), dpi=120)
    ax.scatter(runpass2021['attempts'], runpass2021['rush_attempts'], color='white')

    for index, row in runpass2021.iterrows():
        ab = AnnotationBbox(getImage(row['Logo Path']), (row['attempts'], row['rush_attempts']), frameon=False)
        ax.add_artist(ab)

    for i in range(len(runpass2021)):
        plt.annotate(runpass2021['pass_rush_ratio'][i], (runpass2021['attempts'][i], runpass2021['rush_attempts'][i]+10))
    plt.title("2021 NFL Season - Pass Attempts vs. Rush Attempts", fontdict={'fontsize':35});
    plt.xlabel("Pass Attempts", fontdict={'fontsize':21});
    plt.ylabel("Rush Attempts", fontdict={'fontsize':21});


#################################################################
#
#   function: get_partial_pass_rates
#   purpose: get pass rates for a year, windowed to max_week
#
################################################################

def get_partial_pass_rates(year: int, max_week: int):
    
    # pull year, week selection
    df_rush_pre = nfl.import_ngs_data(stat_type='rushing', years=[year])
    df_pass_pre = nfl.import_ngs_data(stat_type='passing', years=[year])
    rush_sub = df_rush_pre[(df_rush_pre['week'] > 0) & (df_rush_pre['week'] <= max_week)]
    pass_sub = df_pass_pre[(df_pass_pre['week'] > 0) & (df_pass_pre['week'] <= max_week)]

    # get aggregate stats
    df_rush = rush_sub[['team_abbr', 'rush_attempts']].groupby('team_abbr').sum().reset_index()
    df_pass = pass_sub[['team_abbr', 'attempts']].groupby('team_abbr').sum().reset_index()
    df_pass_ratio=df_rush.merge(df_pass, on='team_abbr', how='left')
    df_pass_ratio['pass_ratio']=np.round((df_pass_ratio['attempts']/(df_pass_ratio['rush_attempts']+df_pass_ratio['attempts'])), 2)
    
    return df_pass_ratio

#################################################################
#
#   function: get_median_feature
#   purpose: get team-based median for chosen feature
#
################################################################

def get_median_feature(data,feature):
    data_sub = data[['possessionTeam',feature]].copy().rename(columns={'LA':'LAC'})
    med_feat = data_sub.groupby('possessionTeam')[feature].median().reset_index()
    return med_feat

#################################################################
#
#   function: ggraph_feature_rate
#   purpose: plot chosen feature against pass rate for team so far
#
################################################################

def graph_feature_rate(year,max_week,feature,data):
    pr_cy = get_partial_pass_rates(year, max_week)
    med_feat = get_median_feature(data,feature)
    pre_logo_df = pr_cy.merge(med_feat,left_on='team_abbr',right_on='possessionTeam',how='left')
    df=pd.merge(pre_logo_df, get_logo_df(), on='team_abbr', how='left')
        # Define plot size and autolayout
    fig, ax = plt.subplots(figsize=(20, 14), dpi=120)
    ax.scatter(df[feature],df['pass_ratio'], color='white')

    for index, row in df.iterrows():
        ab = AnnotationBbox(getImage(row['Logo Path']), (row[feature],row['pass_ratio']), frameon=False)
        ax.add_artist(ab)

    for i in range(len(df)):
        plt.title(f"Pass Rate vs. {feature} ({year} through week {max_week})", fontdict={'fontsize':35});
        plt.xlabel(feature, fontdict={'fontsize':21});
        plt.ylabel("Pass Ratio", fontdict={'fontsize':21});