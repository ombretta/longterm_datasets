#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 16:02:35 2021

@author: ombretta
"""

import csv

from matplotlib import pyplot as plt
from matplotlib import rc
from src.common_utils import load_data

from user_studies.analyze_results_segments import get_class_name



def create_bar_chart(ax, dataset_accuracies, categories, title, show_yticks=True, legend=True):

    color_correct = "#FFB000" #"#97D8C4" #"#008000"
    color_wrong = "#DC267F" #"#4059AD" #"#FF0000"
    color_dontknow = "#648FFF" #"#F4B942" #"#0000FF"

    rc('font', **{'family': 'serif', 'serif': ['Times']})

    pos = [1, 1.4, 1.8]
    for i, cat in enumerate(categories):
        labels = ["Correct", "Wrong", 'I am not sure'] if legend and i == 0 else ['', '', '']
        wrong = dataset_accuracies[cat][1]
        notsure = dataset_accuracies[cat][2]
        correct = dataset_accuracies[cat][0]
        ax.barh(cat, [correct], left=0, label=labels[0], color=color_correct, height=0.7)
        ax.barh(cat, [wrong], left=correct, label=labels[1], color=color_wrong, height=0.7)
        ax.barh(cat, [notsure], left=correct + wrong, label=labels[2], color=color_dontknow, height=0.7)

    # Labeling and formatting
    print(show_yticks)
    if not show_yticks:
        ax.set_yticklabels([])

    ax.set_xlabel('User votes (%)', fontsize=50)
    ax.tick_params(axis='x', labelsize=40)
    ax.tick_params(axis='y', labelsize=60)
    ax.set_title(title, fontsize=70)

    return ax


def create_bar_charts(dataset_accuracies, name, cat):
    # Plotting
    fig, (ax1) = plt.subplots(1, 1, figsize=(20, 13))

    ax1 = create_bar_chart(ax1, dataset_accuracies, cat[::-1], name)
    # plt.show()
    plt.subplots_adjust(wspace=0)

    legend = fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.04), ncol=3, fontsize=50)
    legend.get_frame().set_linewidth(0)  # Remove legend border

    plt.savefig(name+"_votes.png", bbox_inches='tight')
    plt.savefig(name+"_votes.pdf", bbox_inches='tight')

    return


def create_plot(accuracies, datasets, cat):
    # Plotting
    fig, axs = plt.subplots(1, len(datasets), figsize=(60, 10))

    for i in range(len(datasets)):
        legend = True if i == 0 else False
        show_yticks = True if i == 0 else False
        print(i, datasets[i], accuracies[i], show_yticks)
        axs[i] = create_bar_chart(axs[i], accuracies[i], cat[::-1], datasets[i], show_yticks=show_yticks, legend=legend)
    # plt.show()
    plt.subplots_adjust(wspace=0)

    legend = fig.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3, fontsize=60)
    legend.get_frame().set_linewidth(0)  # Remove legend border

    plt.savefig("all_votes.png", bbox_inches='tight')
    plt.savefig("all_votes.pdf", bbox_inches='tight')

    return


def main():

    # HARDCODED RESULTS
    datasets = ["(a) Breakfast", "(b) CrossTask", "(c) LVU - Relationship", "(d) LVU - Scene", "(e) LVU - Speaking"]
    cat = ["Full\nVideos", "Video\nSegments", "Selected\nSegments"]
    # cat = ["FV", "VS", "SS"]
    breakfast = {cat[0]: [86.78, 12.07, 1.15], cat[1]: [54.47, 25.49, 20.04], cat[2]: [76.36, 17.82, 5.82]}
    CrossTask = {cat[0]: [87.04, 8.56, 4.4], cat[1]: [68.7, 18.29, 13.01], cat[2]: [90.17, 8.15, 1.69]}
    LVU_relationship = {cat[0]: [74.44, 17.78, 7.78], cat[1]: [56.1, 30.89, 13.01], cat[2]: [70.21, 25.53, 4.26]}
    LVU_scene = {cat[0]: [90.28, 7.64, 2.08], cat[1]: [65.4, 22.26, 12.35], cat[2]: [84.4, 11.35, 4.26]}
    LVU_speaking = {cat[0]: [57.0, 38.0, 5.0], cat[1]: [47.64, 46.01, 6.34], cat[2]: [55.46, 42.86, 1.68]}

    # Plot results
    # create_bar_charts(breakfast, datasets[0], cat[1:]+[cat[0]])
    # create_bar_charts(CrossTask, datasets[1], cat[1:]+[cat[0]])
    # create_bar_charts(LVU_relationship, datasets[2], cat[1:]+[cat[0]])
    # create_bar_charts(LVU_scene, datasets[3], cat[1:]+[cat[0]])
    # create_bar_charts(LVU_speaking, datasets[4], cat[1:]+[cat[0]])

    create_plot([breakfast, CrossTask, LVU_relationship, LVU_scene, LVU_speaking], datasets, cat[1:]+[cat[0]])


if __name__ == "__main__":
    main()
