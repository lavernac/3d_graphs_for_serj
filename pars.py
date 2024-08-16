import tkinter as tk
from tkinter import filedialog
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import pandas as pd
import shutil
from pathlib import Path

def append_data(raw_data, rebuilded_data, dimension, i):
    for key in raw_data.keys():
        rebuilded_data[key][dimension].append(raw_data[key][i])

def append_df_list(raw_data, rebuilded_data, dimension, i):
    for key in rebuilded_data.keys():
        rebuilded_data[key].append([])
    append_data(raw_data, rebuilded_data, dimension, i)

def find_shortest_list(i, rebuilded_data, dimension, min_dimension_size):
    if i != 0 and len(rebuilded_data['x'][dimension-1]) < min_dimension_size:
        min_dimension_size = len(rebuilded_data['x'][dimension-1])
    return min_dimension_size

def rebuild_data(raw_data):
    dimension = 0
    rebuilded_data = {'x': [], 'y': [], 'z': []}
    min_dimension_size = 2**30
    for i in range(0, len(raw_data['x'])):
        if i == 0 or raw_data['x'][i-1] != raw_data['x'][i]:
            min_dimension_size = find_shortest_list(i, rebuilded_data, dimension, min_dimension_size)
            append_df_list(raw_data, rebuilded_data, dimension, i)
            dimension += 1
        else:
            append_data(raw_data, rebuilded_data, dimension-1, i)
    min_dimension_size = find_shortest_list(i, rebuilded_data, dimension, min_dimension_size)
    for key in rebuilded_data.keys():
        for i in range(0, dimension):
            rebuilded_data[key][i] = rebuilded_data[key][i][:min_dimension_size]
    return rebuilded_data

def draw_graph(df, file_name):
    for key in df.keys():
        df[key] = np.array(df[key])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(df['x'], df['y'], df['z'])
    # surf = ax.plot_surface(df['x'], df['y'], df['z'], cmap='jet')
    ax.set_xlabel('Ось X')
    ax.set_ylabel('Ось Y')
    ax.set_zlabel('Ось Z')
    # print("--- %s seconds final ---" % (time.time() - start_time))
    fig.colorbar(surf)
    
    fig.suptitle(Path(file_name).name[:len(Path(file_name).name)-4])
    plt.show()

def open_display():
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(title="Выбери папочку")
    if folder_path:
        files_excluding_py = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and not f.endswith(".py")]
        print("НАША ФОЛДЕР:", folder_path)
        print("ЧЁ ПО ФАЙЛАМ:", files_excluding_py)
        average_value = None
        start_time = time.time()
        files_excluding_py = convert_to_csv(folder_path, files_excluding_py, start_time)
        for kol_file, file_name in enumerate(files_excluding_py, start=1):
            file_path = os.path.join(folder_path, file_name)
            df, average_value = read_file(file_path, kol_file, average_value)
            print("--- %s seconds read ---" % (time.time() - start_time))
            df = dict(df)
            df = rebuild_data(df)
            print("--- %s seconds rebuild ---" % (time.time() - start_time))
            draw_graph(df, file_name)
        print("--- %s seconds full session ---" % (time.time() - start_time))
    else:
        print("ВЫ ЛОХ")

def read_file(file_path, kol_file, average_value):
    df = pd.read_csv(file_path, sep='\t', names=['x', 'y', 'z'])
    # cols = df.select_dtypes(exclude=['float']).columns
    # df[cols] = df[cols].apply(pd.to_numeric, downcast='float', errors='coerce')
    # df = df.astype(float, errors='coerce')
    for key in df.keys():
        df[key] = pd.to_numeric(df[key], errors='coerce')
    print(type(df['x'][0]))
    kol, kol_Bad, number_before_Bad = 0, 0, 0
    # # start_time = time.time()
    for kol in range(0, len(df['x'])):
        # if kol == 7:
            # print(type(df['z'][kol]), str(df['z'][kol]))
        df, kol_Bad, number_before_Bad = converting_bad_to_float(df, kol, kol_Bad, number_before_Bad)
        # if kol == 8:
            # print(number_before_Bad)
            # print(type(df['z'][kol]), df['z'][kol])
            # break
    # # print("%s seconds bad to bloat ---" % (time.time() - start_time))
    # if kol_Bad != 0:
    #     average_for_Bad = float(number_before_Bad) / (kol_Bad + 1)
    #     for i in range(kol - kol_Bad, kol):
    #         df['z'][i + 1] = round(df['z'][i] - average_for_Bad, 6)
    # if kol_file != 1:
    #     for key in df.keys():
    #         average_value[key] = np.add(df[key], average_value[key])
    #         average_value[key] /=2
    # else:
    #     average_value = df

    return df, average_value

def converting_bad_to_float(df, kol, kol_Bad, number_before_Bad):
    if kol_Bad == 0 and isinstance(df["z"][kol], float):
        number_before_Bad = df["z"][kol]
    elif str(df["z"][kol]) == 'nan':
        print(str(df["z"][kol]))
        kol_Bad += 1
    elif kol_Bad > 0 and isinstance(df["z"][kol], float):
        average_for_Bad = (number_before_Bad - df["z"][kol]) / (kol_Bad + 1)
        for i in range(kol - kol_Bad, kol):
            df["z"][i] = round(number_before_Bad - average_for_Bad, 6)
            number_before_Bad = df["z"][i]
        number_before_Bad = df["z"][kol]
        kol_Bad = 0
    return df, kol_Bad, number_before_Bad
# def converting_bad_to_float(df, kol, kol_Bad, number_before_Bad):
#     print(kol, kol_Bad)

def convert_to_csv(folder_path, files_excluding_py, start_time):
    files_list_temp = []

    if os.path.isdir(folder_path + '/csv_dir/'):
        shutil.rmtree(folder_path+ '/csv_dir/')
    os.mkdir(folder_path + '/csv_dir/')
    for file in files_excluding_py:
        path = folder_path + '/csv_dir/' + file
        files_list_temp.append(path + '.csv')
        shutil.copy(folder_path + '/' + file, path)
        with open(path, 'r') as file:
            lines = file.readlines()
        print(path)    
        with open(path, 'w') as file:
            file.writelines(lines[13:])
        os.rename(path, path + '.csv')
    
    print("--- %s seconds converting to csv ---" % (time.time() - start_time))
    return files_list_temp

open_display()