from pyparsing import Word, alphas
import pandas as pd
import os
import shutil
from config import logger

def read_file(file_path, num_file, average_value):
    df = pd.read_csv(file_path, sep='\t', names=['x', 'y', 'z'])
    num, num_Bad, number_before_Bad = 0, 0, 0
    df = dict(df)
    for num in range(0, len(df['x'])):
        df, num_Bad, number_before_Bad = converting_bad_to_float(df, num, num_Bad, number_before_Bad)
    if num_Bad != 0:
        average_for_Bad = number_before_Bad / (num_Bad + 1)
        for i in range(num - num_Bad, num):
            df['z'][i + 1] = round(float(df['z'][i]) - average_for_Bad, 6)
    # if num_file != 1:
        # for key in df.keys():
            # average_value[key] = np.add(np.asarray(df[key], dtype=np.float64, order="C"), np.asarray(average_value[key], dtype=np.float64, order="C"))
    # else:
        # average_value = df
    return df

def converting_bad_to_float(df, num, num_Bad, number_before_Bad):
    if df['z'][num] == Word(alphas):
        num_Bad += 1
        if num_Bad == 1 and num != 0:
            number_before_Bad = float(df['z'][num-1])
    elif num_Bad > 0:
        average_for_Bad = (number_before_Bad - float(df['z'][num])) / (num_Bad + 1)
        for i in range(num - num_Bad, num):
            df['z'][i] = round(number_before_Bad - average_for_Bad, 6)
            number_before_Bad = df['z'][i]
        number_before_Bad = float(df['z'][num])
        num_Bad = 0
    return df, num_Bad, number_before_Bad


def convert_to_csv(folder_path, files_excluding_py):
    files_list_temp = []

    if os.path.isdir(folder_path + '/csv_dir/'):
        try:
            shutil.rmtree(folder_path+ '/csv_dir/')
        except:
            logger.warning("не получилось удалить csv_dir")
    os.mkdir(folder_path + '/csv_dir/')
    for file in files_excluding_py:
        path = folder_path + '/csv_dir/' + file
        files_list_temp.append(path + '.csv')
        shutil.copy(folder_path + '/' + file, path)
        with open(path, 'r') as file:
            lines = file.readlines()
        logger.info(path)    
        with open(path, 'w') as file:
            file.writelines(lines[13:])
        os.rename(path, path + '.csv')
    
    return files_list_temp