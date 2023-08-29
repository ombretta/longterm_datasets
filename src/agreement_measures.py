#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 11:25:52 2021

@author: ombretta
"""

import os
import json

import numpy as np
import krippendorff


'''Krippendorff's alpha from 
https://github.com/pln-fing-udelar/fast-krippendorff/blob/main/krippendorff/krippendorff.py

reliability_data : array_like, with shape (M, N)
Reliability data matrix which has the rate the i coder gave to the j unit, 
where M is the number of raters and N is the unit count.
Missing rates are represented with `np.nan`.
If it's provided then `value_counts` must not be provided.

alpha = 1 indicates perfect reliability
alpha = 0 indicates the absence of reliability. Units and the values assigned 
to them are statistically unrelated.
alpha < 0 when disagreements are systematic and exceed what can be expected by chance.
'''


def collect_annotations(summary_folder):
    print(summary_folder)
    summaries = {}
    files = [f for f in os.listdir(summary_folder) if "timesteps.txt" in f]
    for summary_file in files:
        with open(os.path.join(summary_folder, summary_file), "r") as f:
            summary = json.load(f)
            summaries[summary_file] = summary
            print(summary)
    return summaries
            

def create_annotation_matrix(summaries_dict):
    N = summaries_dict[list(summaries_dict.keys())[0]]["full_video_duration"]
    M = len(list(summaries_dict.keys()))
    
    reliability_data = np.zeros((M,N))
    
    for summary_name, i in zip(summaries_dict, range(M)):
        for chosen_interval in summaries_dict[summary_name]["chosen_timesteps"]:
            reliability_data[i,chosen_interval[0]:chosen_interval[1]] += 1
    
    return reliability_data

def krippendorff_alpha(reliability_data):
    if reliability_data.shape[0] <= 1:
        print("More than 1 worker is required to compute the Krippendorff's alpha.")
        return np.NaN
    alpha = krippendorff.alpha(reliability_data=reliability_data)
    return alpha


def compute_agreement(summary_folder):
    summaries_dict = collect_annotations(summary_folder)
    reliability_data = create_annotation_matrix(summaries_dict)
    alpha = krippendorff_alpha(reliability_data)
    return alpha
    

def simulate_gaussian_summaries(num_summaries, len_video, mean=100, 
                                std=50, summary_length=35):
    
    reliability_data_example = np.zeros((num_summaries, len_video))
    # Select video seconds from normal distribution centered in mean=meanwith std=std
    for i in range(num_summaries):
        choices = np.random.normal(loc=mean, scale=std, size=summary_length)
        choices = np.around(choices, decimals=0)
        for c in choices: 
            reliability_data_example[i,min(len_video-1,int(c))] += 1
    
    alpha_gaussian = krippendorff_alpha(reliability_data_example)
    return alpha_gaussian

