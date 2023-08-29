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



def get_video_names(dataset, video_type, video_accuracies):
    videos = list(video_accuracies.keys())
    if video_type == "segments":
        # labels = [get_class_name(dataset, video_accuracies[v]["0.mp4"]["gt"]) for v in videos]
        labels = [get_class_name(dataset, video_accuracies[v][list(video_accuracies[v].keys())[0]]["gt"]) for v in
                  videos]

        if dataset == "breakfast":
            video_names = [v.split("/")[-1].split("_")[1]+" "+v.split("/")[-1].split("_")[0]+": "+l
                           for v, l in zip(videos, labels)]
        elif dataset == "CrossTask":
            video_names = [v.split("/")[-1] + ": " + l for v, l in zip(videos, labels)]

        elif "LVU" in dataset:
            video_names = [v + ": " + l.capitalize().replace("_", "-") for v, l in zip(videos, labels)]
    else:
        labels = [get_class_name(dataset, video_accuracies[v]["gt"]) for v in videos]
        if dataset == "breakfast":
            video_names = [v.split("/")[-1].split("_")[0] + " " + v.split("/")[-2] + ": "+l
                           for v, l in zip(videos, labels)]
        elif dataset == "CrossTask":
            video_names = [v.split("/")[-1].split(".mp")[0] + ": " + l for v, l in zip(videos, labels)]

        elif "LVU" in dataset:
            video_names = [v + ": " + l.capitalize().replace("_", "-") for v, l in zip(videos, labels)]

    videos = [x for _, x in sorted(zip(labels, videos))]
    video_names = [x for _, x in sorted(zip(labels, video_names))]
    return videos, video_names


def get_LVU_sorted_segments(dataset, video):

    task = dataset.split("_")[1]
    segments_file = "datasets/LVU/" + task + "/" + task + "_segments.txt"
    ordered_segments = []

    with open(segments_file, "r", encoding="utf-16") as f:
        lines = f.readlines()

    mem = None
    for line in lines[1:]:
        if line.split("\t")[3] == "": continue

        video_url, video_len, segment_id, start, end, embed = line.replace("\"\"", "\"").split("\t")
        # print(video_url, video_len, segment_id, start, end, embed)

        if video_url != "":
            if len(ordered_segments) > 0: break
            mem = video_url

        if video in mem:

            embed_link = embed.split("src=\"")[1].split("\" title=")[0].replace("?clip=", "?controls=0&amp;clip=")
            segment_id = embed_link.split("clipt=")[1]
            ordered_segments.append(segment_id)

    return ordered_segments




def create_bar_chart(ax, video_accuracies, dataset, video_type, show_yticks=False, show_legend=False):

    color_correct = "#FFB000" #"#97D8C4" #"#008000"
    color_wrong = "#DC267F" #"#4059AD" #"#FF0000"
    color_dontknow = "#648FFF" #"#F4B942" #"#0000FF"

    rc('font', **{'family': 'serif', 'serif': ['Times']})

    categories = []

    videos, video_names = get_video_names(dataset, video_type, video_accuracies)

    all_dontknow = 0
    all_segments_votes = 0
    all_dontknow_seg1_2 = 0

    print(videos)

    for i, (video, video_name) in enumerate(zip(videos, video_names)):

        categories.append(video)
        if i == 0:
            label_correct, label_wrong, label_dontknow = 'Correct', 'Wrong', 'I am not sure'
        else:
            label_correct, label_wrong, label_dontknow = None, None, None

        if video_type == "segments":

            label_correct, label_wrong, label_dontknow = None, None, None

            segments_votes = 0

            if "LVU" not in dataset:
                segments = [str(i)+".mp4" for i in range(len(video_accuracies[video]))]
            else:
                segments = get_LVU_sorted_segments(dataset, video)

            print(video_name, ":", video_accuracies[video].keys(), end=" ")

            all_video_votes = sum([video_accuracies[video][s]['n_votes'] for s in segments])
            all_segments_votes += all_video_votes
            all_dontknow += sum([video_accuracies[video][s]['n_dontknow'] for s in segments])
            all_dontknow_seg1_2 += video_accuracies[video][segments[0]]['n_dontknow']
            if len(segments)>1: all_dontknow_seg1_2 += video_accuracies[video][segments[1]]['n_dontknow']

            print(all_segments_votes, all_dontknow, all_dontknow_seg1_2)

            for s in segments:
                # print("segment")
                correct = (video_accuracies[video][s]['n_correct']/all_video_votes)*100
                wrong = (video_accuracies[video][s]['n_wrong']/all_video_votes)*100
                dontknow = (video_accuracies[video][s]['n_dontknow']/all_video_votes)*100
                all = (video_accuracies[video][s]['n_votes']/all_video_votes)*100
                print(video_accuracies[video][s]['confidence'], video_accuracies[video][s]['most_chosen'],
                      "GT:", video_accuracies[video][s]['gt'])
                      # correct, wrong, dontknow, all, end="; ")


                # print(video_accuracies[video][s]['most_chosen'], ":", video_accuracies[video][s]['confidence'], "%", end="; ")

                ax.barh([video_name], [correct], left=segments_votes, label=label_correct, color=color_correct)
                ax.barh([video_name], [wrong], left=segments_votes+correct, label=label_wrong, color=color_wrong)
                ax.barh([video_name], [dontknow], left=segments_votes+correct+wrong, label=label_dontknow, color=color_dontknow)
                segments_votes += all
            print()

            ax.set_title("Video Segments", fontsize=24)




        if video_type == "full_videos":
            print(video, video_accuracies[video]['most_chosen'], ":", video_accuracies[video]['confidence'], "%")
            all = video_accuracies[video]['n_votes']
            correct = (video_accuracies[video]['n_correct']/all)*100
            wrong = (video_accuracies[video]['n_wrong']/all)*100
            dontknow = (video_accuracies[video]['n_dontknow']/all)*100

            ax.barh([video_name], [correct], left=0, label=label_correct, color=color_correct)
            ax.barh([video_name], [wrong], left=correct, label=label_wrong, color=color_wrong)
            ax.barh([video_name], [dontknow], left=correct + wrong, label=label_dontknow, color=color_dontknow)
            ax.set_title("Full Videos", fontsize=24)

    # Labeling and formatting
    ax.set_xlabel('Votes (%)', fontsize=24)
    ax.tick_params(axis='x', labelsize=24)
    ax.tick_params(axis='y', labelsize=24)

    if not show_yticks:
        plt.yticks([])
    else:
        ax.set_ylabel('Videos', fontsize=24)

    print("all_votes for segments", all_segments_votes)
    print("all I don't know votes", all_dontknow)
    print("I don't know votes in first segment", all_dontknow_seg1_2)
    # print("ratio", round(all_dontknow_seg1_2/all_dontknow*100,2))

    return ax


def create_two_bar_charts(accuracies_segments, accuracies_full_videos, dataset):


    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(28, 16)) # (28, 7)

    ax1 = create_bar_chart(ax1, accuracies_full_videos, dataset, "full_videos", True, True)
    ax2 = create_bar_chart(ax2, accuracies_segments, dataset, "segments")

    plt.subplots_adjust(wspace=0.02)

    legend = fig.legend(loc='upper center', bbox_to_anchor=(0.5, +0.05), ncol=3, fontsize=24)
    legend.get_frame().set_linewidth(0)  # Remove legend border

    plt.savefig(dataset+"_results.png", bbox_inches='tight')
    plt.savefig(dataset+"_results.pdf", bbox_inches='tight')

    return

def main():

    dataset = "breakfast"
    # dataset = "CrossTask"
    # dataset = "LVU_relationship"
    # dataset = "LVU_scene"
    # dataset = "LVU_speaking"

    # Load data
    segments_accuracies = load_data("user_studies/Accuracies/accuracies_"+dataset+"_segments.dat")
    full_video_accuracies = load_data("user_studies/Accuracies/accuracies_"+dataset+"_full_videos.dat")

    create_two_bar_charts(segments_accuracies, full_video_accuracies, dataset)


if __name__ == "__main__":
    main()
