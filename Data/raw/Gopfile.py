import os
import glob
import pandas as pd

path = './data' 
all_files = glob.glob(os.path.join("C:/Users/ADMIN/Documents/Project/Data/raw/Kết quả các năm", "*.csv"))

li = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)

frame = pd.concat(li, axis=0, ignore_index=True)

print("Tổng số dòng:", len(frame))
print(frame.head())

frame.to_csv("Data/raw/premier_league_2017_2027.csv", index=False)