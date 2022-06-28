import pandas as pd

dataframes_dir = "./DR3_pieces/"

starts = [8000000,16000000,24000000] #excluding the first one, which is loaded in separately

filename_base = "DR3_6D_kinematics_{}_to_{}.csv"


df = pd.read_csv(dataframes_dir + filename_base.format(0,8000000))

for start in starts:
    df_read = pd.read_csv(dataframes_dir + filename_base.format(start,start+8000000))
    print("Loading " + dataframes_dir + filename_base.format(start,start+8000000))

    df_bigger = pd.concat([df, df_read], verify_integrity=True, ignore_index=True)

    del(df_read)
    del(df)

    df = df_bigger
    del(df_bigger)

df.to_csv("DR3_6D_kinematics.csv")