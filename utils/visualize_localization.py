import os
import numpy as np
from typing import List, Dict
from collections import defaultdict
from dataclasses import dataclass
import matplotlib.pyplot as plt


@dataclass
class Sample:
    imf: str = None
    pose: np.ndarray = None
    r_err: float = None
    t_err: float = None


def average(data):
    return sum(data) / len(data)


@dataclass
class PlotTable:
    mini_x: float = None
    mini_y: float = None
    maxi_x: float = None
    maxi_y: float = None
    cols: int = None
    rows: int = None

    def update_range(self, x, y):
        if self.mini_x is None:
            self.mini_x = x
            self.mini_y = y
            self.maxi_x = x
            self.maxi_y = y
        else:
            self.mini_x = min(self.mini_x, x)
            self.mini_y = min(self.mini_y, y)
            self.maxi_x = max(self.maxi_x, x)
            self.maxi_y = max(self.maxi_y, y)

    def draw_figure(self, data: Dict, figure_name, sample_distance: float = 0.5):
        cols = self.maxi_x - self.mini_x + 1
        rows = self.maxi_y - self.mini_y + 1

        xv, yv = np.meshgrid(
            np.linspace(self.mini_x * sample_distance, self.maxi_x * sample_distance, cols),
            np.linspace(self.mini_y * sample_distance, self.maxi_y * sample_distance, rows),
        )
        zv = np.zeros(shape=(rows, cols), dtype=np.float32) - 1.0

        for (xi, yi), value in data.items():
            zv[yi - self.mini_y, xi - self.mini_x] = value

        z_min, z_max = zv.min(), zv.max()

        fig, ax = plt.subplots()
        cmap = plt.get_cmap('hot')
        c = ax.pcolormesh(xv, yv, zv, cmap=cmap, vmin=z_min, vmax=z_max)
        ax.set_title(figure_name)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        # set the limits of the plot to the limits of the data
        # ax.axis([xv.min(), xv.max(), yv.min(), yv.max()])
        ax.axis([xv.min(), xv.max(), yv.min(), yv.max()])
        fig.colorbar(c, ax=ax)
        fig.savefig(f"./output/{figure_name}.png")

    def plot_result(self, samples: List[Sample], sample_distance: float = 0.01):
        translation_statistic = defaultdict(list)
        rotation_statistic = defaultdict(list)
        for sample in samples:
            x, y, _ = sample.pose[0:3, 3]
            xi = int(x / sample_distance)
            yi = int(y / sample_distance)
            self.update_range(xi, yi)
            pos_key = (xi, yi)
            translation_statistic[pos_key].append(sample.t_err)
            rotation_statistic[pos_key].append(sample.r_err)
        for key in translation_statistic.keys():
            translation_statistic[key] = average(translation_statistic[key])
            rotation_statistic[key] = average(rotation_statistic[key])
        self.draw_figure(translation_statistic, "translation", sample_distance=sample_distance)
        self.draw_figure(rotation_statistic, "rotation", sample_distance=sample_distance)

    @staticmethod
    def save_result(samples: List[Sample], save_path: str):
        os.makedirs(save_path, exist_ok=True)
        meta_file = f"{save_path}/meta.txt"
        with open(meta_file, "w") as fh:

            for sample in samples:
                im_name = os.path.basename(sample.imf).replace(".color.png", "")

                save_file = f"{save_path}/{im_name}_pose.txt"
                np.savetxt(save_file, sample.pose)
                fh.write(f"{save_file} {sample.r_err} {sample.t_err}\n")
