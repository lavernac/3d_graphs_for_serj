import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Создание данных
x = np.linspace(-50000, 50000, 1000)
y = np.linspace(-50000, 50000, 1000)
x, y = np.meshgrid(x, y)
z = np.sin(np.sqrt(x**2 + y**2))

# Создание фигуры и осей
fig = plt.figure(figsize=(10, 5))  # Установка размеров фигуры
ax = fig.add_subplot(111, projection='3d')

# Построение поверхности
ax.plot_surface(x, y, z, cmap='viridis', edgecolor='none')

# Установка соотношения сторон
# ratio = [x[int(len(x)/2)][len(x[0])-1], y[0][len(y[0])-1].max(), 500]
ratio = [500, 500, 500]
print(ratio)
ax.set_box_aspect(ratio)  # Установка соотношения сторон: X:Y:Z = 2:1:1
fig.set_dpi(200)
# Настройка осей
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')
ax.set_title('Rectangular 3D Plot')

plt.show()