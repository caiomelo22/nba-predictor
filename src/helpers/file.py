import os
import pandas as pd
from joblib import dump


def save_file(directory, file, file_name):
    if not os.path.exists(directory):
        os.makedirs(directory)

    dump(file, f"{directory}/{file_name}.joblib")

def load_df_from_file(path, file_name):
    file_path = f"{path}/{file_name}.csv"
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        return None

def parse_df_to_csv(dataframe, columns, path, filename):
    if not os.path.exists(path):
        os.makedirs(path)
    final_df = pd.DataFrame(dataframe, columns=columns)
    final_df.to_csv("{}/{}".format(path, filename))
    return final_df
