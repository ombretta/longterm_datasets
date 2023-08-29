#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 14:00:33 2022

@author: ombretta

This script generates Figure 2 from the paper.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import pickle as pkl


def save_data(data, filename):
    with open(filename, "wb") as f:
        pkl.dump(data, f)


def load_data(filename):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            data = pkl.load(f)
        return data
    else:
        print("File", filename, "does not exists.")


def get_label(video, dataset):
    # print(video)
    if dataset == "breakfast":
        return video.split("_")[-1]
    if dataset == "CrossTask":
        return video.split("_")[0]

def count_subactions(timesteps):
    subactions = []
    for video in timesteps:
        subactions += list(timesteps[video].keys())
    # print(subactions)
    number_subactions = len(list(set(subactions)))
    print("Number of short-term actions:", number_subactions)
    # print(list(set(subactions)))
    return number_subactions


def get_actions_frequency(timesteps, dataset):
    subaction_occurrences, subaction_frequency = {}, {}

    for video in timesteps.keys():
        video_class = get_label(video, dataset)
        # print(video_class)
        for subaction in timesteps[video]:
            if subaction not in subaction_occurrences:
                subaction_occurrences[subaction] = []
            subaction_occurrences[subaction] += [video_class]
            subaction_occurrences[subaction] = list(set(subaction_occurrences[subaction]))

    subaction_frequency = {i:[] for i in range(1,11)}
    # print(subaction_frequency)
    for subaction in subaction_occurrences:
        frequency = len(subaction_occurrences[subaction])
        subaction_frequency[frequency].append(subaction.replace("_", " "))

    # print(subaction_occurrences)
    print(subaction_frequency)
    return subaction_occurrences, subaction_frequency



def autolabel(rects, bar_label, ax, x_offset, y_poses):
    for idx, rect in enumerate(rects):
        height = rect.get_height()
        if bar_label[idx] != "":
            x_pos = rect.get_x() + rect.get_width() / 2
            y_pos = 1.01 * height

            ax.annotate(bar_label[idx], (x_pos, y_pos), xytext=(x_pos + x_offset[idx], y_poses[idx]),
                        arrowprops=dict(arrowstyle="->",
                                        connectionstyle="angle3,angleA=0,angleB=-90"), fontsize=50)


def count_videos_with_unique_actions(dataset, all_videos_actions, unique_actions):
    count_videos = 0
    for video in all_videos_actions:
        check_actions = [a for a in all_videos_actions[video] if a.replace("_", " ") in unique_actions]
        if any(check_actions): count_videos += 1
    print( len(all_videos_actions), "videos")
    print(dataset, ":", round(count_videos / len(all_videos_actions) * 100, 2), "% of videos have unique sub-actions.")


def create_timesteps_file_breakfast(dataset):
    timesteps = {}
    if dataset == "breakfast":
        annotations_path = "annotations/breakfast/segmentation_coarse/"
        classes = [f for f in os.listdir(annotations_path) if os.path.isdir(annotations_path+f)]
        for c in classes:
            videos = [v for v in os.listdir(annotations_path+c) if "txt" in v and " 2" not in v]
            for v in videos:
                timesteps[v.split("txt")[0]] = {}

                with open(annotations_path+c+"/"+v, "r") as f:
                    anno = f.readlines()
                for line in anno:
                    interval, subaction = line.split(" ")[:2]
                    start, end = int(interval.split("-")[0]), int(interval.split("-")[1])
                    if subaction.split("\n")[0] not in timesteps[v.split("txt")[0]]:
                        timesteps[v.split("txt")[0]][subaction.split("\n")[0]] = []
                    timesteps[v.split("txt")[0]][subaction.split("\n")[0]].append([start, end])
    # print(timesteps)
    # print(len(timesteps))


def calculate_subaction_uniqueness(dataset):
    print("Short-term action stats for", dataset)
    timesteps = load_data("annotations/" + dataset + "/all_timesteps_annotations.dat")
    n_subactions = count_subactions(timesteps)
    subaction_occurrences, subaction_frequency = get_actions_frequency(timesteps, dataset)
    uniqueness = [len(i) / n_subactions * 100 for i in list(subaction_frequency.values())]
    print("In how many long-term actions do short-term action appear:", uniqueness)
    print()
    return timesteps, uniqueness, subaction_frequency


def make_plot(data1, data2, subaction_frequency1, subaction_frequency2):

    # color_correct = "#FFB000" #"#97D8C4" #"#008000"
    color_br = "#FFB000" #"#DC267F"
    color_CT = "#648FFF"


    rc('font', **{'family': 'serif', 'serif': ['Times']})

    x_range = 6 #4
    X = np.arange(x_range)
    fig1 = plt.figure(figsize=(40, 10)) # 30, 20
    ax1 = fig1.add_axes([0, 0, 1, 1])
    plt.xticks(X, tuple([str(i) for i in range(1, x_range + 1)]), fontsize=50)
    width = 0.5

    bars_br = ax1.bar(X, data1[:x_range], color=color_br, width=-width / 2,
                      label="Breakfast dataset", align='edge')
    bars_CT = ax1.bar(X, data2[:x_range], color=color_CT, width=width / 2,
                      label="CrossTask dataset", align='edge')

    # bar_labels_br = ["\n".join(random.sample(actions, min(2, len(actions))))
    #                  for actions in subaction_frequency1.values()]
    # bar_labels_CT = ["\n".join(random.sample(actions, min(2, len(actions))))
    #                  for actions in subaction_frequency2.values()]

    bar_labels_br = ['fry pancake\npour coffee', 'pour sugar\nadd salt and pepper',
                     'crack egg\npour oil', 'pour milk\ntake bowl', '', 'take plate']
    bar_labels_CT = ['add kimchi\npeel banana', 'pour espresso\npour alcohol',
                     'add onion', 'pour egg\npour milk', 'stir mixture\npour water', 'add sugar']

    # y_poses = [30, 50, 40, 50]
    # x_offsets = [0.45, 0.05, 0.1, -0.3]
    y_poses = [30, 50, 40, 50] + [40 for i in range(6)]
    x_offsets = [0.45, 0.05, 0.1, -0.3, 0.1, -0.4]# + [0.1 for i in range(6)]
    autolabel(bars_br, bar_labels_br, ax1, x_offsets, y_poses)

    x_offsets = [0.2, 0.08, 0.2, 0.1, 0.1, -0.1] #+ [0.1 for i in range(6)]
    y_poses = [65, 20, 30, 20, 20, 30] #+ [30 for i in range(6)]
    autolabel(bars_CT, bar_labels_CT, ax1, x_offsets, y_poses)

    #plt.xlim([-0.5,3.5])
    plt.xlim([-0.5, 5.5])
    plt.ylim([0, 100])

    ax1.set_xticks(X, tuple([str(i) for i in range(1, x_range + 1)]))
    ax1.tick_params(axis='y', labelsize=50)
    # ax1.set_title("Breakfast dataset", fontsize="x-large")
    ax1.set_xlabel("Number of long-term action classes in which a short-term action appears", fontsize=55)
    ax1.set_ylabel("Amount of short-term actions (%)", fontsize=55)
    ax1.legend(loc='upper right', ncol=1, fontsize=55)
    fig1.savefig("Breakfast_CrossTask_subaction_unicity.png", bbox_inches="tight")
    fig1.savefig("Breakfast_CrossTask_subaction_unicity.pdf", bbox_inches="tight")



def main():
    timesteps_br, uniqueness_br, subaction_frequency_br = calculate_subaction_uniqueness("breakfast")
    timesteps_CT, uniqueness_CT, subaction_frequency_CT = calculate_subaction_uniqueness("CrossTask")

    make_plot(uniqueness_br, uniqueness_CT, subaction_frequency_br, subaction_frequency_CT)

    count_videos_with_unique_actions("breakfast", timesteps_br, subaction_frequency_br[1])
    count_videos_with_unique_actions("CrossTask", timesteps_CT, subaction_frequency_CT[1])

    return timesteps_br


if __name__ == '__main__':
    timesteps = main()

# %%



