import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def append_data(raw_data, rebuilded_data, dimension, i):
    for key in raw_data.keys():
        rebuilded_data[key][dimension].append(raw_data[key][i])

def append_new_list(raw_data, rebuilded_data, dimension, i):
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
            append_new_list(raw_data, rebuilded_data, dimension, i)
            dimension += 1
        else:
            append_data(raw_data, rebuilded_data, dimension-1, i)
    min_dimension_size = find_shortest_list(i, rebuilded_data, dimension, min_dimension_size)
    for key in rebuilded_data.keys():
        for i in range(0, dimension):
            rebuilded_data[key][i] = rebuilded_data[key][i][:min_dimension_size]
    return rebuilded_data

# d = {'x': [1., 1, 1, 2, 2, 2, 3, 3, 3],
#      'y': [0.5, 1, 1.5, 0.5, 1, 1.5, 0.5, 1, 1.5], 
#      'z': [1, 1.1, 1, 1.2, 1.15, 1,1 ,1, 1]}
d = {'x': [1, 1, 1, 2, 2, 2, 3, 3],
     'y': [0.5, 1, 1.5, 0.5, 1, 1.5, 0.5, 1], 
     'z': [1, 1.1, 1, 1.2, 1.15, 1,1 ,1]}
d = {'x': [1, 1, 1, 2, 2, 2],
     'y': [0.5, 1, 1.5, 0.5, 1, 1.5], 
     'z': [1, 1.1, 1, 1.2, 1.15, 1]}
# Создание сетки из данных X и Y
# Y, X = np.meshgrid(x, y)
d = rebuild_data(d)

# d['x'] = np.array(d['x'])
# d['y'] = np.array(d['y'])
# d['z'] = np.array(d['z'])
for key in d.keys():
        d[key] = np.array(d[key])
print(d['x'], d['y'], d['z'], sep='\n')
# print(x,y,z, sep='\n')
# Создание значений Z на основе X и Y
# Например, можно использовать функцию Z = X^2 + Y^2 для создания уникальных высот
# t, Z = np.meshgrid(x, y)  # Здесь мы создаем уникальные значения Z
# print(X, Y, sep='\n')
# print(Z)
# # Создание фигуры и 3D осей
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# # Построение 3D поверхности
surf = ax.plot_surface(d['x'], d['y'], d['z'], cmap='viridis')
# # ax.plot_wireframe(X, Y, Z, rstride=5, cstride=5)
# # ax.plot_trisurf(x, y, z)
# # Установка меток осей
ax.set_xlabel('Ось X')
ax.set_ylabel('Ось Y')
ax.set_zlabel('Ось Z')

# # Установка заголовка графика
ax.set_title('3D Поверхность из трех списков')

# # Добавление цветовой шкалы
# fig.colorbar(surf)

# # Отображение графика
plt.savefig("график.png", dpi=300)
plt.show()
