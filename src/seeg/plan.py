from dataclasses import dataclass
import os

from cycler import cycler
import mne
import nibabel as nib
import numpy as np
import numpy.typing as npt

from .plots.depths import create_contacts_plot, ROSA_COLOR_LIST
from .plots.draw import draw_cyl


@dataclass
class Trajectory():
    name: str
    base: npt.NDArray
    tip: npt.NDArray


def read_rosa_plan(file_name) -> list[Trajectory]:
    trajectories = list()
    rosa_to_RAS = np.asarray([[-1, 0, 0, 0],
                              [0, -1, 0, 0],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]]) #% we know the transformation from ROSA to RAS

    with open(file_name, "r") as file:
        line = file.readline().strip()
        while line != "[TRAJECTORY]":
            line = file.readline().strip()

        num_electrodes = int(file.readline().strip())
        for _ in range(num_electrodes):
            info = file.readline().strip().split(" ")
            name = info[0].split("^")[0]
            if name[-1] == "'":
                name = name.replace("'", "L")
            else:
                name += "R"
            base = nib.affines.apply_affine(rosa_to_RAS, np.asarray([float(x) for x in info[4:7]]))
            tip = nib.affines.apply_affine(rosa_to_RAS, np.asarray([float(x) for x in info[8:11]]))
            trajectories.append(Trajectory(name, base, tip))
            for _ in range(4):
                file.readline()

    return trajectories


def display_rosa_plan(SUBJECT_ID, SUBJECTS_DIR, LOCAL_ID, depth_list) -> mne.viz.Brain:
    file_name = os.path.join(SUBJECTS_DIR, SUBJECT_ID, LOCAL_ID+".ros")
    trajectories = read_rosa_plan(file_name)
    brain = create_contacts_plot(depth_list, SUBJECT_ID, SUBJECTS_DIR)
    for t, color in zip(trajectories, cycler(color=ROSA_COLOR_LIST)()):
        # print(t.name)
        base = t.base
        tip = t.tip
        draw_cyl(brain.plotter, tip, base, 1, color['color'], 1)

    return brain
