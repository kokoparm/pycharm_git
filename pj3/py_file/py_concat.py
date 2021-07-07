import pandas as pd
import pickle, os

csv_dir_path = '../read_file/'

print(os.listdir(csv_dir_path)[0])

df = pd.read_csv(csv_dir_path + os.listdir(csv_dir_path)[0], index_col=0)
df.dropna(inplace=True)
df.drop_duplicates()
df.to_csv('./crawling_csv/clean_data.csv')
for i in range(1, len(os.listdir(csv_dir_path))):
    raw_df = pd.read_csv(csv_dir_path + os.listdir(csv_dir_path)[i],
                         index_col=0)
    raw_df.dropna(inplace=True)
    raw_df.drop_duplicates()
    df = pd.concat([df, raw_df], ignore_index=True)

df.to_csv('./crawling_csv/claened_data(cate10).csv')