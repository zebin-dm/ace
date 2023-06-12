from typing import List
from dataclasses import dataclass


@dataclass
class PoseError:
    t_err: float = 0
    r_err: float = 0
    counter: int = 0


class Metric():
    DEFAULT_CONFIG = {
        "pose_acc": [[0.01, 1], [0.02, 2], [0.05, 5], [0.1, 5]],  # [m, deg]
    }

    def __init__(self, config=None) -> None:
        if config is None:
            config = {}
        self.config = {**self.DEFAULT_CONFIG, **config}
        self.translation_error = []
        self.rotation_error = []
        self.pose_acc: List[PoseError] = []
        for t_err, r_err in self.config["pose_acc"]:
            self.pose_acc.append(PoseError(t_err=t_err, r_err=r_err))

    def update(self, t_err: float, r_err: float):
        self.translation_error.append(t_err)
        self.rotation_error.append(r_err)

        for i in range(len(self.pose_acc)):
            if t_err < self.pose_acc[i].t_err and r_err < self.pose_acc[
                    i].r_err:
                self.pose_acc[i].counter += 1

    def print(self):
        number = len(self.translation_error)
        accuracy = ""
        for pa in self.pose_acc:
            value = pa.counter / number * 100
            accuracy += f"{int(pa.t_err * 100)}cm/{int(pa.r_err)}deg: {value:.1f}, "

        self.translation_error.sort()
        self.rotation_error.sort()
        median_idx = number // 2
        median_rErr = self.rotation_error[median_idx]
        median_tErr = self.translation_error[median_idx] * 100
        average_rErr = sum(self.rotation_error) / number
        average_tErr = sum(self.translation_error) / number * 100
        print_text = f"""
Accuracy: {accuracy}
median_rErr: {median_rErr: .1f} deg, median_tErr: {median_tErr: .1f} cm,
average_rErr: {average_rErr: .1f} deg, average_tErr: {average_tErr:.1f} cm"""
        print(print_text)
