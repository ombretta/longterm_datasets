#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 10:05:51 2020

@author: ombretta
"""
import math

from .common_utils import save_data
import os

import numpy as np

import subprocess

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "fill_credentials_here"


def read_videos_timesteps(root, timesteps_files, video_labels, classes_labels):
    '''Generates the timesteps file. Action 'NoHuman' and 'Stand' are' discarded.'''

    timesteps = {}

    for class_file in [f for f in os.listdir(timesteps_files) if "txt" in f]:
        print(class_file)
        with open(timesteps_files + class_file, "r") as f:
            file = f.readlines()

        cl = class_file[:-4]
        if cl != 'NoHuman' and cl != 'Stand':
            for line in file:
                video = line.split(" ")[0]

                if video not in timesteps:
                    timesteps[video] = {}

                if cl not in timesteps[video]:
                    timesteps[video][cl] = []

                start = float(line.split(" ")[1])
                end = float(line.split(" ")[2][:-1])
                timesteps[video][cl].append([start, end])

    if not os.path.exists(root + "annotations/timesteps_annotations.dat"):
        save_data(timesteps, root + "annotations/timesteps_annotations.dat")

    return timesteps


def check_video_labels(video_id, video_labels, classes_labels):
    '''Given a video id, returns the action labels of that video.
    The actions 'NoHuman' and 'Stand' are discarded.'''

    # print("Video", video_id)
    v_ex = video_labels[video_id]
    labels = []
    for lab in v_ex.split("_"):
        for cl in classes_labels:
            if int(lab) == classes_labels[cl]:
                # print(lab, cl)
                if cl != 'NoHuman' and cl != 'Stand':
                    labels.append(cl)
    return labels


def check_videos_per_class(class_name, video_labels, classes_labels):
    '''Given an action class, returns the videos that contain that action.'''

    print("Class", classes_labels[class_name], class_name)
    for video in video_labels:
        if str(classes_labels[class_name]) in video_labels[video]:
            print(video)  # , video_labels[video])


def check_video_timesteps(video_id, timesteps):
    '''Given a video id, returns actions and the action timesteps in the video.'''

    print("Video", video_id)
    for cl in timesteps[video_id]:
        print(cl)
        for interval in timesteps[video_id][cl]:
            print(interval)
    print(timesteps[video_id])


def get_video_duration(filename):
    '''Returns video duration in seconds.'''

    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                             "format=duration", "-of",
                             "default=noprint_wrappers=1:nokey=1", filename],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)

    tot_seconds = float(result.stdout)
    minutes = int(np.floor((tot_seconds) / 60))
    seconds = int(np.floor(tot_seconds - minutes * 60))

    print(filename, end=" ")
    print("%sm %ss" % (minutes, seconds))
    return tot_seconds


def get_mean_actions_duration(timesteps, video_id):
    '''Returns mean and std of actions duration in a video.'''

    actions_time = []
    for action in timesteps[video_id]:
        for t in timesteps[video_id][action]:
            actions_time.append(abs(t[1] - t[0]))
    mean_duration = np.mean(actions_time)
    std_duration = np.std(actions_time)
    return int(np.ceil(mean_duration)), int(np.ceil(std_duration))


def generate_no_sound_video(input_video_path, no_sound_video_path):
    '''Used to generate copies of the videos with no sound.
    We don't want to provide audio to the users.'''

    command = "ffmpeg -i " + input_video_path + " -c copy -an " + no_sound_video_path
    os.system(command)


def iterate_segments_cropping(segments_interval, segments_path, video_path):
    '''Loop over the segments interval and generate video segments.'''

    print("segments_interval", segments_interval)
    for i, segment in enumerate(segments_interval):
        print("Segment", i)
        start = int(round(segment[0]))
        duration = int(round(segment[1]))
        print("Start, duration", start, duration)
        segment_path = os.path.join(segments_path, str(i) + ".mp4")
        crop_segment(video_path, segment_path, start, duration)


def crop_segment(input_video_path, output_segment_path, start, duration):
    '''Cropping segment with ffmpeg. Example (crop video of 10 seconds):
    "ffmpeg -ss 30 -i input.wmv -c copy -t 10 output.wmv"
    The video path is given in input_video_path and the segments is saved
    in output_segment_path'''

    command = ["ffmpeg", "-i", input_video_path, "  -c:v libx264  "
               " -ss", str(start),
               "-an", "-to ", str(start + duration), output_segment_path]
    command = " ".join(command)
    print(command)
    os.system(command)


def get_action_uniform_intervals(timesteps, video_path, min_duration=1):
    ''' This function generates the timesteps for the action segments.
    Short segments (less than min_duration) are ignored.'''

    full_duration = get_video_duration(video_path)
    segments_interval = []

    for action in timesteps:
        for interval in timesteps[action]:
            start = math.floor(interval[0])
            end = math.ceil(interval[1])
            print("action, interval", action, interval)
            duration = end-start
            # print("duration", duration)
            if duration > min_duration:
                print("included", start, end, duration)

                segment_duration = (min(end,full_duration)-start)
                segments_interval.append([start, segment_duration])

    sorted(segments_interval)
    print(segments_interval)
    return segments_interval


def create_action_video_segments(video_path, timesteps, segments_dir="short_action_segments_TEST"):
    '''This function computes video segments according to the temporal location of the actions in the video.
    The actions timesteps are provided in the annotations of the dataset.'''

    if not os.path.exists(os.path.join(segments_dir)):
        os.mkdir(os.path.join(segments_dir))

    if True: #len(os.listdir(segments_dir)) == 0:
        segments_interval = get_action_uniform_intervals(timesteps, video_path)
        iterate_segments_cropping(segments_interval, segments_dir, video_path)
    return

