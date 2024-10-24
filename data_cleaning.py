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