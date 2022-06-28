import pandas as pd

dataframes_dir = "./pickle_pieces/"

filename_base = 'DR3_6D_kinematics_%s.pkl'

output_filename = "DR3_6D_kinematics.pkl"

df = pd.read_pickle(dataframes_dir + (filename_base % 0))

for i in range(1,30):
    df_read = pd.read_pickle(dataframes_dir + (filename_base % i))
    print(dataframes_dir + (filename_base % i))

    df_bigger = pd.concat([df, df_read], verify_integrity=True, ignore_index=True)

    del(df_read)
    del(df)

    df = df_bigger
    del(df_bigger)

print('Writing data to %s...' % output_filename)

df.to_pickle(output_filename)

print("Done!")