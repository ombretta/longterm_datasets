import math


def get_std(ns, stds):
    first_term = 0
    second_term = -len(ns)
    for i, (n, std) in enumerate(zip(ns, stds)):
        first_term += (n - 1) * math.pow(std, 2)
        second_term += n
    return round(math.sqrt((first_term) / (second_term)), 2)

def get_avg(ns, avgs):
    tot = sum(ns)
    average = 0
    for i, (n, avg) in enumerate(zip(ns, avgs)):
        average += avg*n
    return round(average/tot,2)

# FULL VIDEOS
breakfast_avg, breakfast_std, breakfast_n = 11.6, 0.8, 30
CrossTask_avg, CrossTask_std, CrossTask_n =  12.0, 0.0, 36
LVU_rel_avg, LVU_rel_std, LVU_rel_n =  10.0, 0.0, 9
LVU_scene_avg, LVU_scene_std, LVU_scene_n = 12.0, 0.0, 12
LVU_speak_avg, LVU_speak_std, LVU_speak_n = 10.0, 0.0, 10

N_FV = [breakfast_n, CrossTask_n, LVU_rel_n, LVU_scene_n, LVU_speak_n]
STD_FV = [breakfast_std, CrossTask_std, LVU_rel_std, LVU_scene_std, LVU_speak_std]
AVG_FV = [breakfast_avg, CrossTask_avg, LVU_rel_avg, LVU_scene_avg, LVU_speak_avg]
print("Full videos", get_avg(N_FV, AVG_FV), "pm", get_std(N_FV, STD_FV))

# VIDEO SEGMENTS
breakfast_s_avg, breakfast_s_std, breakfast_s_n = 17.5, 2.41, 154
CrossTask_s_avg, CrossTask_s_std, CrossTask_s_n = 10.2, 1.66, 348
LVU_rel_s_avg, LVU_rel_s_std, LVU_rel_s_n = 10.25, 0.83, 36
LVU_scene_s_avg, LVU_scene_s_std, LVU_scene_s_n = 11.71, 0.45, 56
LVU_speak_s_avg, LVU_speak_s_std, LVU_speak_s_n = 11.5, 0.5, 48

N_VS = [breakfast_s_n, CrossTask_s_n, LVU_rel_s_n, LVU_scene_s_n, LVU_speak_s_n]
STD_VS = [breakfast_s_std, CrossTask_s_std, LVU_rel_s_std, LVU_scene_s_std, LVU_speak_s_std]
AVG_VS = [breakfast_s_avg, CrossTask_s_avg, LVU_rel_s_avg, LVU_scene_s_avg, LVU_speak_s_avg]
print("Video segments", get_avg(N_VS, AVG_VS), "pm", get_std(N_VS, STD_VS))

# ALL
print("All", get_avg(N_FV+N_VS, AVG_FV+AVG_VS), "pm", get_std(N_FV+N_VS, STD_FV+STD_VS))