
import tkinter as tk
from tkinter import filedialog
import time
from converter import *
from processing import *
from config import logger


def open_display():
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(title="Выбери папочку")
    if folder_path:
        files_excluding_py = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and not f.endswith(".py")]
        logger.info("folder:" + str(folder_path))
        logger.info("files list:" + str(files_excluding_py))
        average_value = None
        df = []
        start_time = time.time()
        ratios = get_ratios(folder_path, files_excluding_py)
        logger.info("--- %s seconds to get ratios ---" % (time.time() - start_time))
        files_csv = convert_to_csv(folder_path, files_excluding_py)
        logger.info("--- %s seconds converting to csv ---" % (time.time() - start_time))
        for kol_file, file_name in enumerate(files_csv, start=1):
            file_path = os.path.join(folder_path, file_name)
            df.append(read_file(file_path, kol_file, average_value))
            logger.info("--- %s seconds read ---" % (time.time() - start_time))
            df[kol_file-1] = rebuild_data(df[kol_file-1])
            logger.info("--- %s seconds rebuild ---" % (time.time() - start_time))
            graph_aligning(df[kol_file-1], ratios[kol_file-1]) # чтобы вернуть график в исходный вид надо это закомментить
            logger.info("--- %s seconds aligning ---" % (time.time() - start_time))
            centering_graph(df[kol_file-1], ratios[kol_file-1]) # и это
            logger.info("--- %s seconds centering the graph ---" % (time.time() - start_time))
            # draw_graph(df[kol_file-1], file_name, ratios[kol_file-1])
        calculate_average(df, kol_file, ratios[0])
        # calculate_average(df, len(files_excluding_py) - , ratios[0])    
        logger.info("--- %s seconds full session ---" % (time.time() - start_time))
    else:
        ("Empty path")

if __name__ == "__main__":
    open_display()