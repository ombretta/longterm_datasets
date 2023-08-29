#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 10:05:51 2020

@author: ombretta
"""

import os
import argparse

from .common_utils import load_data
from .video_segments_utils import create_action_video_segments

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "fill_credentials_here"

def video_path_formatter_breakfast(video_id):
    classes_ids = load_data("breakfast_classes_id.dat")
    video_id_fixed = video_id.replace("cereals", "cereal").replace("salat", "salad")
    label = [l for l in classes_ids if l in video_id_fixed][0]
    print("LABEL", label, "id", classes_ids[label])
    video_path = video_id_fixed.replace(label, classes_ids[label])
    video_id = "_".join(video_path.split("/")[-2:]).split(".")[0]
    return video_path, video_id


def video_path_formatter_CrossTask(video_id):
    root = "datasets/CrossTask/"
    video_path = os.path.join(root+video_id, video_id+".3gpp")
    print(video_path)
    return video_path, video_id


def main(timesteps_path, video_path_formatter, segments_path):

    timesteps = load_data(timesteps_path)
    print(timesteps.keys())

    for video_id in timesteps:

        print(video_id)

        video_path, id = video_path_formatter(video_id)

        create_action_video_segments(video_path,
                                     timesteps[video_id],
                                     segments_path+id)


if __name__ == '__main__':

    # Breakfast
    # timesteps_path = "annotations/breakfast/timesteps.dat"
    # video_path_formatter = video_path_formatter_breakfast
    # segments_path = "short_video_segments_TEST/breakfast/"
    #
    # main(timesteps_path, video_path_formatter, segments_path)

    # CrossTask
    timesteps_path = "annotations/CrossTask/timesteps.dat"
    video_path_formatter = video_path_formatter_CrossTask
    segments_path = "short_video_segments/CrossTask/"

    main(timesteps_path, video_path_formatter, segments_path)
