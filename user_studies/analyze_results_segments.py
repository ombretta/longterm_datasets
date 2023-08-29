#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 16:02:35 2021

@author: ombretta
"""

import os

import numpy as np
import collections

from matplotlib import pyplot as plt
from matplotlib import rc

from src.common_utils import read_surveys_file, load_data, save_data
from src.agreement_measures import krippendorff_alpha

def get_classes(dataset):
    if dataset == "breakfast":
        classes_id = load_data("breakfast_classes_id.dat")
        return list(classes_id.keys())
        # classes = [f for f in os.listdir(annotations_path) if os.path.isdir(annotations_path + f)]
        # classes = [c.replace("cereals", "cereal").replace("salat", "salad") for c in classes]
    elif dataset == "CrossTask":

        classes_id = load_data("CrossTask_classes_id.dat")
        return list(classes_id.values())
    elif "LVU" in dataset:
        task = dataset.split("_")[1]
        annotations_path = "annotations/LVU/"+task+"/test.csv"
        videos, annotations = read_annotations(annotations_path, {}, task)
        classes = list(set([annotations[c]['name'] for c in annotations]))
        if task == "relationship":
            classes = ["friends", "husband_wife", "boyfriend_girlfriend"]
        if task == "speaking":
            classes = ["confronts", "discusses", "explains", "teaches", "threatens"]
        return classes


def read_annotations(file, videos, annotations_type):
    annotations = {}

    with open(file, "r") as f:
        reader_obj = csv.reader(f, delimiter=' ')
        for i, row in enumerate(reader_obj):

            if i > 0:
                video = row[2]
                if video not in videos:
                    videos[video] = {}
                videos[video][annotations_type] = row[0]
                if row[0] not in annotations:
                    annotations[row[0]] = {'name': row[1], 'videos': []}
                annotations[row[0]]['videos'].append(video)
    return videos, annotations


def get_class_name(dataset, class_id):
    if dataset == "breakfast":
        return class_id.capitalize()
    elif dataset == "CrossTask":
        classes_id = load_data("CrossTask_classes_id.dat")
        return [c for c in classes_id if classes_id[c] == class_id][0]
    elif "LVU" in dataset:

        return class_id

def get_label(video, dataset):
    if dataset == "breakfast":
        classes_id = load_data("breakfast_classes_id.dat")
        for c in classes_id:
            if "_"+classes_id[c] in video:
                return c

    if dataset == "CrossTask":
        dir = "annotations/CrossTask/crosstask_release/annotations/"
        for file in os.listdir(dir):
            if video.split("/")[-2] in file or video.split("/")[-1] in file:
                return file.split("_")[0]

    if "LVU" in dataset:
        task = dataset.split("_")[1]
        annotations_path = "annotations/LVU/"+task+"/test.csv"
        _, annotations = read_annotations(annotations_path, {}, task)

        video_id = video

        if "embed/" in video:

            video_id = video.split("embed/")[1].split("?controls")[0]

        # print(annotations)
        # print(video, video_id)
        label = [annotations[c]['name'] for c in annotations if video_id in annotations[c]['videos']][0]
        if label == "friend": label = "friends"
        # classes = ["confronts", "discusses", "explains", "teaches", "threatens"]
        if label == "discuss": label = "discusses"
        return label
    return

def get_user_accuracies(fields, responses, dataset, low_accuracy=40):

    user_accuracies = {}
    assignments_to_reject = []

    for count, user_resp in enumerate(responses.values()):
        user_id = user_resp["WorkerId"]
        assignment_id = user_resp["AssignmentId"]

        # print("\n", count, user_id, end=": ")

        assignment_accuracy = {"correct": 0, "total": 0}

        if user_id not in user_accuracies:
            user_accuracies[user_id] = {"correct": 0, "total": 0}

        for video in range(len([f for f in fields if "Input.videoLink" in f])):

            video_link = user_resp["Input.videoLink" + str(video + 1)]
            label = get_label(video_link, dataset)
            user_accuracies[user_id]['HITId'] = user_resp['HITId']
            user_accuracies[user_id]['total'] += 1
            assignment_accuracy['total'] += 1

            # print(label, end=" ")

            if dataset == "breakfast" or "LVU" in dataset:
                dontknow_reply = user_resp["Answer.actionClass" + str(video + 1) + ".dontknow" + str(video + 1)]
            else:
                dontknow_reply = user_resp["Answer.actionClass" + str(video + 1) + ".dontknow_" + str(video + 1)]
            if 'true' in dontknow_reply:
                # print("dontknow", end="; ");
                continue

            for i, c in enumerate(get_classes(dataset)):

                if dataset == "breakfast" or "LVU" in dataset:
                    key = "Answer.actionClass" + str(video + 1) + "." + c + str(video + 1)
                else:
                    key = "Answer.actionClass" + str(video + 1) + "." + c + "_" + str(video + 1)

                if 'true' in user_resp[key]:
                    # print(c, label)
                    if label == c:
                        user_accuracies[user_id]['correct'] += 1
                        assignment_accuracy['correct'] += 1

        acc = (assignment_accuracy["correct"] / assignment_accuracy["total"]) * 100
        # print("Assignment accuracy", assignment_id, acc)
        if acc < low_accuracy:
            assignments_to_reject.append(assignment_id)
            print("Reject", user_id, user_resp['HITId'], user_resp['AssignmentStatus'])

    return user_accuracies, assignments_to_reject

def filter_out_bad_users(fields, responses, dataset, low_accuracy=40):
    bad_users = []
    user_accuracies, assignments_to_reject = get_user_accuracies(fields, responses, dataset, low_accuracy)
    mean_user_accuracy, std_user_accuracy = calculate_avg_user_accuracy(user_accuracies)

    for user in sorted(user_accuracies):
        acc = (user_accuracies[user]["correct"] / user_accuracies[user]["total"]) * 100
        if acc < low_accuracy:
            # print("Bad user!", user, acc)
            bad_users.append(user)
    print("Tot users:", len(user_accuracies),
          "Bad users:", len(bad_users),
          "Good users:", len(user_accuracies)-len(bad_users))
    return bad_users, assignments_to_reject

def get_responses(fields, responses, dataset, bad_users):
    results = {}
    reliability_data = {}

    classes = get_classes(dataset)
    n_videos = len([f for f in fields if "Input.videoLink" in f])

    for user_resp in responses.values():

        if user_resp["WorkerId"] in bad_users:
            continue

        if user_resp["WorkerId"] not in reliability_data:
            reliability_data[user_resp["WorkerId"]] = {}

        for video in range(n_videos):

            video_link = user_resp["Input.videoLink" + str(video + 1)]

            if video_link not in results:
                results[video_link] = {c: 0 for c in classes+["dontknow"]}

            for idx, c in enumerate(classes+["dontknow"]):
                if dataset == "breakfast" or "LVU" in dataset:
                    key = "Answer.actionClass"+str(video+1)+"."+c+str(video+1)
                else:
                    key = "Answer.actionClass" + str(video + 1) + "." + c + "_" +str(video + 1)

                if 'true' in user_resp[key]:
                    results[video_link][c] += 1
                    reliability_data[user_resp["WorkerId"]][video_link] = idx

    # print("RELIABILITY DATA", reliability_data)

    return results, reliability_data

def update_accuracy_per_video(video_type, dataset, accuracy_per_video, video, confidence,
                              correct, wrong, dontknow, all, label, prediction):

    if video_type == "segments":
        if "LVU" in dataset:
            root = video.split("embed/")[1].split("?controls")[0]
            segment_id = video.split("clipt=")[1]

        else:
            root = "/".join(video.split("/")[:-1])
            segment_id = video.split("/")[-1]
        if root not in accuracy_per_video:
            accuracy_per_video[root] = {}
        accuracy_per_video[root][segment_id] = {"confidence": confidence, "n_votes": all, "n_correct": correct,
                                                "n_wrong": wrong, "n_dontknow": dontknow,
                                                "gt": label, "most_chosen": prediction}
        # print(video, all, correct, confidence, label, prediction)

    elif video_type == "full_videos":
        if video not in accuracy_per_video:
            accuracy_per_video[video] = {}
        accuracy_per_video[video] = {"confidence": confidence, "n_votes": all, "n_correct": correct,
                                     "n_wrong": wrong, "n_dontknow": dontknow,
                                     "gt": label, "most_chosen": prediction}

    return accuracy_per_video

def get_responses_per_video(results, dataset, video_type):
    accuracy_per_video = {}
    for video in results:
        # print(video, results)
        label = get_label(video, dataset)

        # print(results[video])
        user_votes = np.array(list(results[video].values()))
        all = sum(user_votes)

        most_voted = np.argmax([results[video][l] for l in results[video] if l != "dontknow"])
        most_votes = np.max([results[video][l] for l in results[video] if l != "dontknow"])
        prediction = list(results[video].keys())[most_voted]

        confidence = round((most_votes / all) * 100, 2)
        correct = results[video][label]
        dontknow = results[video]["dontknow"]
        wrong = all - correct - dontknow

        accuracy_per_video = update_accuracy_per_video(video_type, dataset, accuracy_per_video, video, confidence,
                                                       correct, wrong, dontknow, all, label, prediction)

    return accuracy_per_video

def get_max_confident_segment(video_accuracies):
    ''' Assigning to the video the prediction of the segment classified with higher confidence.
    The "I don't know" answers are excluded.'''

    confidence_per_segment = np.array([video_accuracies[s]['confidence'] for s in video_accuracies])
    confidence = np.max(confidence_per_segment)
    most_confident_segment = list(video_accuracies.keys())[np.argmax(confidence_per_segment)]
    predicted_label_most_confident = video_accuracies[most_confident_segment]["most_chosen"]
    return predicted_label_most_confident, confidence

def get_segments_majority_vote(video_accuracies):
    ''' Assigning to the video the prediction obtained with majority vote.
    In case classes are voted equally often, pick the one with higher confidence. '''

    chosen_labels, confidences = [], {}
    segments = [s for s in video_accuracies]

    for s in sorted(segments):
        # print(video_accuracies[s]['confidence'], video_accuracies[s]['most_chosen'], end="; ")

        most_chosen = video_accuracies[s]['most_chosen']
        if most_chosen not in confidences:
            confidences[most_chosen] = []

        # Omit "I don't know" from majority vote.
        if most_chosen != "dontknow":
            confidences[most_chosen].append(video_accuracies[s]['confidence'])
            chosen_labels.append(most_chosen)

    if len(chosen_labels) == 0:
        return 'wrong'

    counter = collections.Counter(chosen_labels)
    len_counter = len(counter.most_common(5))

    if len_counter == 1: return counter.most_common(1)[0][0]

    i = 0
    while i < len_counter-1 and counter.most_common(len_counter)[i][1] == counter.most_common(len_counter)[i+1][1]:
        i += 1
    if i == 0:
        return counter.most_common(len_counter)[0][0]

    max_conf, pred = 0, None
    for j in range(i+1):
        if sum(confidences[counter.most_common(len_counter)[j][0]]) >= max_conf:
            max_conf = sum(confidences[counter.most_common(len_counter)[j][0]])
            pred = counter.most_common(len_counter)[j][0]
    return pred


def get_predicted_label(video_type, video_accuracies, video, dataset):
    video_label = get_label(video, dataset)

    if video_type == "segments":

        # print(video.split("/")[-1], video_label, ":", end=" ")

        # Most confident segment
        predicted_label_most_confident, confidence = get_max_confident_segment(video_accuracies[video])

        # Majority vote
        predicted_label_majority_vote = get_segments_majority_vote(video_accuracies[video])

        # print(video, video_label, predicted_label_most_confident, predicted_label_majority_vote)

        correct_votes = sum(np.array([video_accuracies[video][s]['n_correct'] for s in video_accuracies[video]]))
        wrong_votes = sum(np.array([video_accuracies[video][s]['n_wrong'] for s in video_accuracies[video]]))
        all_votes = sum(np.array([video_accuracies[video][s]['n_votes'] for s in video_accuracies[video]]))

    if video_type == "full_videos":
        predicted_label_most_confident = video_accuracies[video]["most_chosen"]
        confidence = video_accuracies[video]["confidence"]
        correct_votes = video_accuracies[video]["n_correct"]
        wrong_votes = video_accuracies[video]["n_wrong"]
        all_votes = video_accuracies[video]["n_votes"]
        predicted_label_majority_vote = predicted_label_most_confident


    return predicted_label_most_confident, predicted_label_majority_vote, confidence, correct_votes, wrong_votes, all_votes


def get_video_names(dataset, video_type, video_accuracies):
    videos = list(video_accuracies.keys())

    if video_type == "segments":

        labels = [get_class_name(dataset, video_accuracies[v][list(video_accuracies[v].keys())[0]]["gt"]) for v in videos]
        if dataset == "breakfast":
            video_names = [v.split("/")[-1].split("_")[1]+" "+v.split("/")[-1].split("_")[0]+": "+l
                           for v, l in zip(videos, labels)]
        elif dataset == "CrossTask":
            video_names = [v.split("/")[-1] + ": " + l for v, l in zip(videos, labels)]

        elif "LVU" in dataset:
            video_names = [v + ": " + l for v, l in zip(videos, labels)]
    else:
        labels = [get_class_name(dataset, video_accuracies[v]["gt"]) for v in videos]
        if dataset == "breakfast":
            video_names = [v.split("/")[-1].split("_")[0] + " " + v.split("/")[-2] + ": "+l
                           for v, l in zip(videos, labels)]
        elif dataset == "CrossTask":
            video_names = [v.split("/")[-1].split(".mp")[0] + ": " + l for v, l in zip(videos, labels)]

        elif "LVU" in dataset:
            video_names = [v + ": " + l for v, l in zip(videos, labels)]

    videos = [x for _, x in sorted(zip(labels, videos))]
    video_names = [x for _, x in sorted(zip(labels, video_names))]
    return videos, video_names


def create_bar_chart(video_accuracies, dataset, video_type):

    color_correct = "#FFB000" #"#008000"
    color_wrong = "#DC267F" #"#FF0000"
    color_dontknow = "#648FFF" #"#0000FF"

    rc('font', **{'family': 'serif', 'serif': ['Times']})

    categories = []

    # Plotting
    fig, ax = plt.subplots(figsize=(16, 12))

    videos, video_names = get_video_names(dataset, video_type, video_accuracies)

    for i, (video, video_name) in enumerate(zip(videos, video_names)):

        categories.append(video)
        if i == 0:
            label_correct, label_wrong, label_dontknow = 'Correct', 'Wrong', 'I am not sure'
        else:
            label_correct, label_wrong, label_dontknow = None, None, None

        if video_type == "segments":


            segments_votes = 0
            # segments = [str(i)+".mp4" for i in range(len(video_accuracies[video]))]
            segments = list(video_accuracies[video].keys())

            all_video_votes = sum([video_accuracies[video][s]['n_votes'] for s in segments])

            for s in segments:
                correct = (video_accuracies[video][s]['n_correct']/all_video_votes)*100
                wrong = (video_accuracies[video][s]['n_wrong']/all_video_votes)*100
                dontknow = (video_accuracies[video][s]['n_dontknow']/all_video_votes)*100
                all = (video_accuracies[video][s]['n_votes']/all_video_votes)*100
                # print(video_accuracies[video][s]['confidence'], video_accuracies[video][s]['most_chosen'],
                #       correct, wrong, dontknow, all, end="; ")
                ax.barh([video_name], [correct], left=segments_votes, label=label_correct, color=color_correct)
                ax.barh([video_name], [wrong], left=segments_votes+correct, label=label_wrong, color=color_wrong)
                ax.barh([video_name], [dontknow], left=segments_votes+correct+wrong, label=label_dontknow, color=color_dontknow)
                segments_votes += all
                label_correct, label_wrong, label_dontknow = None, None, None
            # print()

        if video_type == "full_videos":
            all = video_accuracies[video]['n_votes']
            correct = (video_accuracies[video]['n_correct']/all)*100
            wrong = (video_accuracies[video]['n_wrong']/all)*100
            dontknow = (video_accuracies[video]['n_dontknow']/all)*100

            ax.barh([video_name], [correct], left=0, label=label_correct, color=color_correct)
            ax.barh([video_name], [wrong], left=correct, label=label_wrong, color=color_wrong)
            ax.barh([video_name], [dontknow], left=correct + wrong, label=label_dontknow, color=color_dontknow)

    # Labeling and formatting
    ax.set_xlabel('Votes (%)', fontsize=20)
    ax.set_ylabel('Videos', fontsize=20)
    ax.set_title(dataset.replace("b", "B") + " - " + video_type.capitalize().replace("_", " "), fontsize=20)
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=16)

    legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=20)
    legend.get_frame().set_linewidth(0)  # Remove legend border

    plt.savefig(dataset+"_"+video_type+".png", bbox_inches='tight')

    return



def get_aggregated_accuracies(video_accuracies, dataset, video_type):

    confidence_per_video = {}
    classification_accuracy_most_confident = 0
    classification_accuracy_majority_vote = 0
    all_correct, all_wrong, all_votes = 0, 0, 0

    # print(video_accuracies)

    for video in video_accuracies:

        # print(video, video_accuracies[video])
        predicted_label_most_confident, predicted_label_majority_vote, confidence_per_video[video], \
            correct, wrong, tot = get_predicted_label(video_type, video_accuracies, video, dataset)

        all_correct += correct
        all_wrong += wrong
        all_votes += tot
        # print(video, dataset, predicted_label, get_label(video, dataset), n_correct, n_tot)

        if predicted_label_most_confident == get_label(video, dataset):
            classification_accuracy_most_confident += 1
        if predicted_label_majority_vote == get_label(video, dataset):
            classification_accuracy_majority_vote += 1

    tot_accuracy_most_confident = round(classification_accuracy_most_confident/len(video_accuracies)*100, 2)
    tot_accuracy_majority_vote = round(classification_accuracy_majority_vote/len(video_accuracies)*100, 2)
    tot_correct = round(all_correct / all_votes * 100, 2)
    tot_wrong = round(all_wrong / all_votes * 100, 2)

    print("% correct votes:", tot_correct)
    print("% wrong votes:", tot_wrong)
    print("Accuracy (most confident segment):", tot_accuracy_most_confident)
    print("Accuracy (majority vote):", tot_accuracy_majority_vote)

    return confidence_per_video


def average_accs(accs):
    mean = np.mean(accs)
    std = np.std(accs)
    return round(mean,2), round(std,2)


def calculate_avg_accuracy(accuracy_per_video):
    # Percentage of users who voted for the right class in the segment with highest agreement for the right class
    mean, std = average_accs(np.array(list(accuracy_per_video.values())))
    print("User confidence:", mean, "pm", std)
    return (mean, std)


def calculate_avg_user_accuracy(user_accuracies):
    print("User accuracy:")
    accuracy = []
    for user in user_accuracies:
        acc = (user_accuracies[user]["correct"] / user_accuracies[user]["total"]) * 100
        accuracy.append(acc)
    return average_accs(accuracy)

def count_users_per_HIT(results, dataset):
    responses = []
    for video in sorted(results):
        # print("count_users_per_HIT", video, results[video])
        label = get_label(video, dataset)
        n_responses = sum(list(results[video].values()))
        n_correct = results[video][label]
        # print(video, n_responses, round((n_correct/n_responses)*100,2), label)
        responses.append(n_responses)
    mean, std = average_accs(np.array(responses))
    print("Responses per clip:", mean, "pm", std)

def user_agreement(reliability_data, dataset):
    users = list(reliability_data.keys())
    print("USERS:", set(users))
    videos = sum([list(reliability_data[u].keys()) for u in users], [])
    videos = list(set(videos))

    matrix = np.empty((len(users), len(videos)))

    for i, user in enumerate(users):
        for j, video in enumerate(videos):
            if video in reliability_data[user]:
                matrix[i, j] = reliability_data[user][video]
            else:
                matrix[i, j] = np.NaN
    alpha = krippendorff_alpha(matrix)
    print("Krippendorff's alpha:", alpha)


def main():
    low_accuracy = 40
    dataset = "breakfast"
    # dataset = "CrossTask"
    # dataset = "LVU_relationship"
    # dataset = "LVU_scene"
    # dataset = "LVU_speaking"; low_accuracy = 40
    video_type = "segments"
    video_type = "full_videos"
    
    # Read MTurk responses from csv file
    results_file = "./user_studies/results_csv/"+dataset+"_"+video_type+"_results.csv"
    responses, fields = read_surveys_file(results_file)

    # Filter out bad users and get results
    bad_users, assignments_to_reject = filter_out_bad_users(fields, responses, dataset, low_accuracy)
    results, reliability_data = get_responses(fields, responses, dataset, bad_users)

    print(results)

    # Count responses per video
    count_users_per_HIT(results, dataset)

    # Check users agreement
    user_agreement(reliability_data, dataset)

    # Calculate the user accuracy per video
    accuracy_per_video = get_responses_per_video(results, dataset, video_type)

    # If the survey was done per segment, aggregate segment accuracy via maxpooling
    save_data(accuracy_per_video, "accuracies_"+dataset+"_"+video_type+".dat")
    print("accuracy_per_video", accuracy_per_video)

    create_bar_chart(accuracy_per_video, dataset, video_type)
    confidence_per_video = get_aggregated_accuracies(accuracy_per_video, dataset, video_type)
    print(len(confidence_per_video))

    mean, std = calculate_avg_accuracy(confidence_per_video)


if __name__ == "__main__":
    main()
