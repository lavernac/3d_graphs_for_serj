import numpy as np
from draw import draw_graph
from math import *
from config import logger

def append_data(raw_data, rebuilded_data, dimension, i):
    for key in raw_data.keys():
        rebuilded_data[key][dimension].append(float(raw_data[key][i]))

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

def get_lower_point(df):
    edge_points = [df['z'][1][1], df['z'][1][-2], df['z'][-2][1], df['z'][-2][-2]]
    lower_point = abs(min(edge_points, key=lambda i: float(i)))
    return lower_point

def graph_aligning(df, ratio):
    lower_point_pos = get_lower_point(df)
    for i in range(0, len(df['z'])):
        for j in range(0, len(df['z'][0])):
            df['z'][i][j] += lower_point_pos
            # df['z'][i][j] *= (ratio[0]/100 * i) * abs(cos(radians(sqrt(pow(df['z'][-2][1] - df['z'][i][j], 2) + pow(j - len(df['z'][i]), 2)))))
            df['z'][i][j] *= i / 1000 #x 0
            df['z'][i][j] *= j / 1000 #y 0
            # df['z'][i][j] *= (ratio[0]/100 * j)
            # if cos(radians(sqrt(pow(df['z'][-2][1] - df['z'][i][j], 2) + pow(j - len(df['z'][i]), 2)))) > 0:
                # df['z'][i][j] *= cos(radians(sqrt(pow(df['z'][-2][1] - df['z'][i][j], 2) + pow(j - len(df['z'][i]), 2))))
            # else:
                # df['z'][i][j] = 0
    for i in range(-1, -len(df['z']) - 1, -1):
        for j in range(-1, -len(df['z'][0]) - 1, -1):
            df['z'][i][j] *= i / 1000 #x max
            df['z'][i][j] *= j / 1000 #y max

def find_edges_of_spraying(df):
    edges_positions = [] 
    edge_height = max(df['z'][int(len(df['z'][0])/2)])/100*15 
    for i in range(0, len(df['z'])):
        flag = 0
        for j in range(0, len(df['z'][0])):
            if df['z'][i][j] > edge_height:
                edges_positions.append(abs(i))
                # logger.info("i:", i, "j:", j, "edges_positions:", edges_positions)
                flag = 1
                break
        if flag:
            break
    for i in range(0, len(df['z'])):
        flag = 0
        for j in range(0, len(df['z'][0])):
            if df['z'][j][i] > edge_height:
                edges_positions.append(abs(i))
                # logger.info("i:", i, "j:", j, "edges_positions:", edges_positions)
                flag = 1
                break
        if flag:
            break
    for i in range(-1, -len(df['z']) - 1, -1):
        flag = 0
        for j in range(-1, -len(df['z'][0]) - 1, -1):
            if df['z'][i][j] > edge_height:
                edges_positions.append(abs(i))
                # logger.info("i:", i, "j:", j, "edges_positions:", edges_positions)
                flag = 1
                break
        if flag:
            break
    for i in range(-1, -len(df['z']) - 1, -1):
        flag = 0
        for j in range(-1, -len(df['z'][0]) - 1, -1):
            if df['z'][j][i] > edge_height:
                edges_positions.append(abs(i))
                # logger.info("i:", i, "j:", j, "edges_positions:", edges_positions)
                flag = 1
                break
        if flag:
            break
    return edges_positions
    # df['z'][0][20] = 10 x = 0 y = 20!!!!!

def centering_by_x(df):
    edges_positions = find_edges_of_spraying(df)
    if edges_positions[0] - edges_positions[2] > edges_positions[2] - edges_positions[0]:
        x_shift_in_points = int((edges_positions[0] - edges_positions[2])/2)
        for key in df.keys():
            df[key] = df[key][x_shift_in_points:]
        x_step = df['x'][1][0] - df['x'][0][0]
        i = 0
        for j in range(0, x_shift_in_points):
            df['x'].append([-(x_step * i)] * len(df[key][0]))
            df['y'].append(list(np.linspace(0, df['y'][0][-1], len(df['y'][0]))))
            df['z'].append([0] * len(df[key][0]))
        logger.info("shift on x(cut from start):" + str(x_shift_in_points) + "len control:" + str(len(df['z'])) + str(len(df['z'][0])))  
    else:
        x_shift_in_points = int((edges_positions[2] - edges_positions[0])/2)
        for key in df.keys():
            df[key] = df[key][:len(df['z']) - x_shift_in_points]
        x_step = df['x'][1][0] - df['x'][0][0]
        i = 0
        while i < x_shift_in_points:
            df['x'].insert(0, [-(x_step * i)] * len(df[key][0]))
            df['y'].insert(0, list(np.linspace(0, df['y'][0][-1], len(df['y'][0]))))
            df['z'].insert(0, [0] * len(df[key][0]))
            i += 1
        logger.info("shift on x(cut from end):" + str(x_shift_in_points) + "len control:" + str(len(df['z'])) + str(len(df['z'][0])))   
    return edges_positions

def centering_by_y(df, edges_positions):
    if edges_positions[1] - edges_positions[3] > edges_positions[3] - edges_positions[1]:
        y_shift_in_points = int((edges_positions[1] - edges_positions[3])/2)
        for key in df.keys():
            for i in range(0, len(df[key])):
                df[key][i] = df[key][i][y_shift_in_points:]
        y_step = df['y'][0][2] - df['y'][0][1]
        for j in range(0, y_shift_in_points):
            for k in range(0, len(df['z'])):
                df['x'][k].append(df['x'][k][0])
                df['y'][k].append(df['y'][k][-1] + y_step)
                df['z'][k].append(0)
        logger.info("shift on y(cut from start):" + str(y_shift_in_points) + "len control:" + str(len(df['z'])) + str(len(df['z'][0])))
    else:
        y_shift_in_points = int((edges_positions[3] - edges_positions[1])/2)
        for key in df.keys():
            for i in range(0, len(df[key])):
                df[key][i] = df[key][i][:len(df['z']) - y_shift_in_points]
        y_step = df['y'][0][2] - df['y'][0][1]
        i = 0
        while i < y_shift_in_points:
            for k in range(0, len(df['z'])):
                df['x'][k].insert(0, df['x'][k][0])
                df['y'][k].insert(0, df['y'][k][0] - y_step)
                df['z'][k].insert(0, 0)
            i += 1
        logger.info("shift on y(cut from end):" + str(y_shift_in_points) + "len control:" + str(len(df['z'])) + str(len(df['z'][0]))) 

def centering_graph(df, ratio):
    edges_positions = centering_by_x(df)
    centering_by_y(df, edges_positions)
    
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
                        ratios[len(ratios)-1].append((ratios[len(ratios)-1][1] + floor(int(ratios[len(ratios)-1][0] / ratios[len(ratios)-1][1])) - 1) * 0.5) 
                    else:
                        ratios[len(ratios)-1].append((ratios[len(ratios)-1][0] + floor(int(ratios[len(ratios)-1][1] / ratios[len(ratios)-1][0])) - 1) * 0.5) 
                    break
    logger.info(str(ratios))
    return ratios

def calculate_average(df, number_of_files, ratio):
    average_data = {'x': [], 'y': [], 'z': []}
    average_data['x'] = df[0]['x']
    average_data['y'] = df[0]['y']
    for f_counter in range(0, number_of_files):
        if f_counter == 0:
            for i in range(0, len(df[0]['z'])):
                average_data['z'].append([])
                for j in range(0, len(df[0]['z'][0])):
                    average_data['z'][i].append(df[0]['z'][i][j])
        else:
            for i in range(0, len(df[0]['z'])):
                for j in range(0, len(df[0]['z'][0])):
                    average_data['z'][i][j] += df[f_counter]['z'][i][j]

    for i in range(0, len(df[0]['z'])):
        for j in range(0, len(df[0]['z'][0])):
            average_data['z'][i][j] /= number_of_files
    draw_graph(average_data, 'average ', ratio)  