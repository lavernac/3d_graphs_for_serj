import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def append_new_list(x, y, z, x_temp, y_temp, z_temp, dimention):
    x_temp.append([])
    x_temp[dimention].append(x)
    y_temp.append([])
    y_temp[dimention].append(y)
    z_temp.append([])
    z_temp[dimention].append(z)

def rebuild_lists(x, y, z):
    x_temp = []
    y_temp = []
    z_temp = []
    dimention = 0
    for i in range(0, len(x)):
        if len(x_temp) == 0:
            append_new_list(x[0], y[0], z[0], x_temp, y_temp, z_temp, 0)
        elif x[i-1] != x[i]:
            dimention += 1
            append_new_list(x[i], y[i], z[i], x_temp, y_temp, z_temp, dimention)
        elif x[i-1] == x[i]:
            x_temp[dimention].append(x[i])
            y_temp[dimention].append(y[i])
            z_temp[dimention].append(z[i])
    return x_temp, y_temp, z_temp

x = [1, 1, 1, 2, 2, 2]  
y = [0.5, 1, 1.5, 0.5, 1, 1.5]  
z = [1, 1.1, 1, 1.2, 1.15, 1]
# Y, X = np.meshgrid(x, y)
x, y, z = rebuild_lists(x,y,z)
x = np.array(x)
y = np.array(y)
z = np.array(z)
print(x, y, z, sep='\n')
# t, Z = np.meshgrid(x, y)  # Здесь мы создаем уникальные значения Z
# print(X, Y, sep='\n')
# print(Z)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

surf = ax.plot_surface(x, y, z, cmap='magma')
# # ax.plot_wireframe(X, Y, Z, rstride=5, cstride=5)
# # ax.plot_trisurf(x, y, z)
ax.set_xlabel('Ось X')
ax.set_ylabel('Ось Y')
ax.set_zlabel('Ось Z')

# ax.set_title('3D Поверхность из трех списков')

# fig.colorbar(surf)

plt.show()