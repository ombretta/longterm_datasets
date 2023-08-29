# Are current long-term video understanding datasets long-term?
Official repository of the paper "Are current long-term video understanding datasets long-term?", published in CVEU 2023.

## Abstract

Many real-world applications, from sport analysis to surveillance, benefit from automatic long-term action recognition. In the current deep learning paradigm for automatic action recognition, it is imperative that models are trained and tested on datasets and tasks that evaluate if such models actually learn and reason over long-term information. In this work, we propose a method to evaluate how suitable a video dataset is to evaluate models for long-term action recognition. To this end, we define a long-term action as excluding all the videos that can be correctly recognized using solely short-term information. We test this definition on existing long-term classification tasks on three popular real-world datasets, namely Breakfast, CrossTask, and LVU, to determine if these datasets are truly evaluating long-term recognition. Our study reveals that these datasets can be effectively solved using shortcuts based on short-term information. Following this finding, we encourage long-term action recognition researchers to make use of datasets that need long-term information to be solved.

[Figure 1.pdf](https://github.com/ombretta/longterm_datasets/files/12464275/Figure.1.pdf)
Figure 1: Example of truly long-term actions. Top: Who is winning this soccer game?1, Bottom: Is this person shoplifting in the supermarket?2. In both cases, it is not possible to answer correctly without considering multiple short-term actions together, their order and relations over time.
