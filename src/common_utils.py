import os
import time

import csv

import numpy as np
import torch

import pickle as pkl

import pytube


def download_yt_video(url, output_path):
    try:
        # Create a YouTube object with the provided URL
        youtube = pytube.YouTube(url)

        # Get the first available video stream
        video = youtube.streams.first()

        # Download the video
        video.download(output_path)

        print("Video downloaded successfully!")
        return True

    except Exception as e:
        print("Error:", str(e))
        return None


def save_data(data, filename, with_torch=False):
    with open(filename, "wb") as f:
        if with_torch == True:
            torch.save(data, f)
        else:
            pkl.dump(data, f)


def load_data(filename, with_torch=False):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            data = torch.load(f, map_location='cpu') if with_torch == True else pkl.load(f)
        return data
    else:
        print("File", filename, "does not exists.")


def check_time(start_time):
    tot_seconds = time.time() - start_time
    hours = int(np.floor(tot_seconds/(60*60)))
    minutes = int(np.floor((tot_seconds-hours*60*60)/60))
    seconds = int(np.floor(tot_seconds-hours*60*60-minutes*60))
    milliseconds = int(np.floor((tot_seconds-hours*60*60-minutes*60-seconds)*1000))
    print("--- %sh %sm %ss %sms---" % (hours, minutes, seconds, milliseconds))


# Check memory usage
def check_memory(py):
    memoryUse = py.memory_info()[0]/2.**30  # memory use in GB...I think
    print('\nmemory use:', memoryUse, "GB")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def read_surveys_file(file_path):
    csv_rows, fields = [], []
    results = {}
    with open(file_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line = 0
        for row in csv_reader:
            if line == 0:
                fields = row
                print(len(row), fields)
            else:
                csv_rows.append(row)
                results['user' + str(line)] = {}
                for i in range(len(row)):
                    results['user' + str(line)][fields[i]] = row[i]
            line += 1

    print(line)
    # print(results)
    return results, fields