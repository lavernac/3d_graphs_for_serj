from pyparsing import Word, alphas
import tkinter as tk
from tkinter import filedialog
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import pandas as pd
import shutil

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
            df = read_file(file_path)
            print("--- %s seconds read ---" % (time.time() - start_time))
            df = dict(df)
            df = rebuild_data(df)
            print("--- %s seconds rebuild ---" % (time.time() - start_time))
            for key in df.keys():
                df[key] = np.array(df[key])
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            # Построение 3D поверхности
            surf = ax.plot_surface(df['x'], df['y'], df['z'], cmap='viridis')
            ax.set_xlabel('Ось X')
            ax.set_ylabel('Ось Y')
            ax.set_zlabel('Ось Z')
            print("--- %s seconds final ---" % (time.time() - start_time))
            plt.show()
            for i in range(0, len(df["x"])):
                print(df["x"][i], df["y"][i],df["z"][i])
            print("--- %s seconds full session ---" % (time.time() - start_time))
    else:
        print("ВЫ ЛОХ")

def convert_to_csv(folder_path, files_excluding_py, start_time):
    files_list_temp = []

    if os.path.isdir(folder_path + '/csv_dir/'):
        os.rmdir(folder_path + '/csv_dir/')
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
    
    print("--- %s seconds converting ---" % (time.time() - start_time))
    return files_list_temp

def read_file(file_path):
    print(file_path, "reading...")
    df = pd.read_csv(file_path, sep='\t', names=['x', 'y', 'z'])
    # print(df['x'][0], type(df['x'][0]))
    return df


open_display()