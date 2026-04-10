import glob
import pandas as pd

def create_main_df(file_list):
    
    for file in file_list:
        print(file)

tables_list = glob.glob('dataset/*.csv')

list_2500m = []
list_5000m = []

for filename in tables_list:
    if "5000km" in filename:
        temp_df = pd.read_csv(filename)
        list_5000m.append(temp_df)
    else:
        temp_df = pd.read_csv(filename)
        list_2500m.append(temp_df)

main_2500m_matrix = pd.concat(list_2500m, ignore_index=True)
main_5000m_matrix = pd.concat(list_5000m, ignore_index=True)

main_2500m_matrix.to_csv('dataset/FIRMS_2000-2025_2500m.csv', index=False)
main_5000m_matrix.to_csv('dataset/FIRMS_2000-2025_5000m.csv', index=False)