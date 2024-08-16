from pyparsing import Word, alphas
import tkinter as tk
from tkinter import filedialog
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import pandas as pd
import shutil
from pathlib import Path
from matplotlib.colors import LightSource
import math

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

def draw_graph(df, file_name, ratio):
    for key in df.keys():
        df[key] = np.array(df[key])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # light = LightSource(azdeg=315, altdeg=45)
    # rgb = light.shade(df['z'], cmap=plt.cm.viridis, vert_exag=0.1, blend_mode='soft')

    surf = ax.plot_surface(df['x'], df['y'], df['z'], cmap='jet')
    # ratio.append(ratio[1] + math.floor(ratio[0] / ratio[1]))
    ax.set_box_aspect(ratio)
    ax.set_xlabel('Ось X')
    ax.set_ylabel('Ось Y')
    ax.set_zlabel('Ось Z')
    # print("--- %s seconds final ---" % (time.time() - start_time))
    fig.colorbar(surf)
    fig.suptitle(Path(file_name).name[:len(Path(file_name).name)-4])
    fig.set_dpi(160)
    plt.show()

def get_ratios(folder_path, files_excluding_py):
    ratios = []
    for file in files_excluding_py:
        with open(folder_path + '/' + file, 'r') as cur_file:
            for index, line in enumerate(cur_file):
                if "X Size" in line:
                    ratios.append([])
                    ratios[len(ratios)-1].append(int(line.split('\t')[1]))
                if "Y Size" in line:
                    ratios[len(ratios)-1].append(int(line.split('\t')[1]))
                if index > 1:
                    if int(ratios[len(ratios)-1][0]) > int(ratios[len(ratios)-1][1]):
                        ratios[len(ratios)-1].append(ratios[len(ratios)-1][1] + math.floor(int(ratios[len(ratios)-1][0] / ratios[len(ratios)-1][1])) - 1)
                    else:
                        ratios[len(ratios)-1].append(ratios[len(ratios)-1][0] + math.floor(int(ratios[len(ratios)-1][1] / ratios[len(ratios)-1][0])) - 1)
                    break
    print(ratios)
    return ratios

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
        ratios = get_ratios(folder_path, files_excluding_py)
        print("--- %s seconds to get ratio ---" % (time.time() - start_time))
        files_csv = convert_to_csv(folder_path, files_excluding_py)
        print("--- %s seconds converting to csv ---" % (time.time() - start_time))
        for kol_file, file_name in enumerate(files_csv, start=1):
            file_path = os.path.join(folder_path, file_name)
            df, average_value = read_file(file_path, kol_file, average_value)
            print("--- %s seconds read ---" % (time.time() - start_time))
            df = rebuild_data(df)
            print("--- %s seconds rebuild ---" % (time.time() - start_time))
            draw_graph(df, file_name, ratios[kol_file-1])
        print("--- %s seconds full session ---" % (time.time() - start_time))
    else:
        print("ВЫ ЛОХ")

def read_file(file_path, kol_file, average_value):
    df = pd.read_csv(file_path, sep='\t', names=['x', 'y', 'z'])
    kol, kol_Bad, number_before_Bad = 0, 0, 0
    df = dict(df)
    for kol in range(0, len(df['x'])):
        df, kol_Bad, number_before_Bad = converting_bad_to_float(df, kol, kol_Bad, number_before_Bad)
    if kol_Bad != 0:
        average_for_Bad = number_before_Bad / (kol_Bad + 1)
        for i in range(kol - kol_Bad, kol):
            df['z'][i + 1] = round(float(df['z'][i]) - average_for_Bad, 6)
    if kol_file != 1:
        for key in df.keys():
            average_value[key] = np.add(np.asarray(df[key], dtype=np.float64, order="C"), np.asarray(average_value[key], dtype=np.float64, order="C"))
    else:
        average_value = df
    return df, average_value

def converting_bad_to_float(df, kol, kol_Bad, number_before_Bad):
    if df['z'][kol] == Word(alphas):
        kol_Bad += 1
        if kol_Bad == 1 and kol != 0:
            number_before_Bad = float(df['z'][kol-1])
    elif kol_Bad > 0:
        average_for_Bad = (number_before_Bad - float(df['z'][kol])) / (kol_Bad + 1)
        for i in range(kol - kol_Bad, kol):
            df['z'][i] = round(number_before_Bad - average_for_Bad, 6)
            number_before_Bad = df['z'][i]
        number_before_Bad = float(df['z'][kol])
        kol_Bad = 0
    return df, kol_Bad, number_before_Bad


def convert_to_csv(folder_path, files_excluding_py):
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
    
    return files_list_temp

open_display()