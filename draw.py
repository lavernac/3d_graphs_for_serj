import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def draw_graph(df, file_name, ratio):
    for key in df.keys():
        df[key] = np.array(df[key])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # light = LightSource(azdeg=315, altdeg=45)
    # rgb = light.shade(df['z'], cmap=plt.cm.viridis, vert_exag=0.1, blend_mode='soft')
    
    surf = ax.plot_surface(df['x'], df['y'], df['z'], cmap='viridis')
    
    ax.set_box_aspect(ratio)
    ax.set_xlabel('Ось X')
    ax.set_ylabel('Ось Y')
    ax.set_zlabel('Ось Z')
    fig.colorbar(surf)
    fig.suptitle(Path(file_name).name[:len(Path(file_name).name)-4])
    fig.set_dpi(140)
    plt.show()