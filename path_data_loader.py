import os

import pandas as pd

from optimizations import *
import sys
from os.path import dirname

class PathDataLoader(object):
    def __init__(self, args):
        self.model_dir = "/PGPR"
        # Evaluate only the attributes that have been chosen and are avaiable in the chosen dataset
        self.sens_attribute_flags = {"Gender": args.show_gender_stats, "Age": args.show_age_stats, "Country": args.show_country_stats,
                 "Occupation": args.show_occupation_stats}
        self.dataset_name = args.dataset
        self.dataset_with_words = True if self.dataset_name in AMAZON_DATASETS else False
        self.agent_topk = args.agent_topk
        if args.eval_baseline or args.opt in ["soft", "softETD", "softSEP", "softLIR"]:
            self.load_uid_topk()
            self.load_uid_pid_path()
        else:
            self.uid_topk = {}
            self.uid_pid_explanation = {}
        self.load_pred_paths()
        if self.dataset_with_words:
            self.item_idxs, self.word_idxs, self.uid_timestamp = get_interaction2timestamp_with_words(self.dataset_name)
        else:
            self.item_idxs, self.uid_timestamp = get_interaction2timestamp_map(self.dataset_name)

        #Dependent by the model
        self.test_labels = load_labels(self.dataset_name, 'test')
        self.generate_LIR_matrix()
        self.generate_SEP_matrix()

    # Returns a dict that map the uid to the topk obtained by the models
    def load_uid_topk(self):
        self.uid_topk = {}
        topk_labels_file = open("paths/" + self.dataset_name + "/agent_topk=" + self.agent_topk + "/uid_topk.csv", 'r')
        reader = csv.reader(topk_labels_file, delimiter=",")
        next(reader, None)  # skip the headers
        for row in reader:
            uid = int(row[0])
            topk = row[1].split(" ")
            topk = [int(x) for x in topk if x != '']
            self.uid_topk[uid] = topk


    # Returns a dict of dict where every uid pid represent the list of paths starting from a user ending in a given product
    def load_uid_pid_path(self):
        self.uid_pid_explanation = {}
        uid_pid_path_topk_file = open("paths/" + self.dataset_name + "/agent_topk=" + self.agent_topk + "/uid_pid_explanation.csv")
        reader = csv.reader(uid_pid_path_topk_file, delimiter=",")
        next(reader, None)
        for row in reader:
            uid = int(row[0])
            pid = int(row[1])
            normalized_path = normalize_path(row[2])

            if uid not in self.uid_pid_explanation:
                self.uid_pid_explanation[uid] = {}
            if pid not in self.uid_pid_explanation[uid]:
                self.uid_pid_explanation[uid][pid] = []

            self.uid_pid_explanation[uid][pid] = normalized_path
        uid_pid_path_topk_file.close()

    # Returns a dict where every
    def load_pred_paths(self):
        self.pred_paths = {}
        uid_pid_path_topk_file = open("paths/" + self.dataset_name + "/agent_topk=" + self.agent_topk + "/pred_paths.csv")
        reader = csv.reader(uid_pid_path_topk_file, delimiter=",")
        next(reader, None)
        for row in reader:
            uid = int(row[0])
            pid = int(row[1])
            path_score = float(row[2])
            path_prob = float(row[3])
            normalized_path = normalize_path(row[4])

            if uid not in self.pred_paths:
                self.pred_paths[uid] = {}
            if pid not in self.pred_paths[uid]:
                self.pred_paths[uid][pid] = []
            self.pred_paths[uid][pid].append([path_score, path_prob, normalized_path])

        uid_pid_path_topk_file.close()
        #return self.pred_paths

    def load_best_pred_paths(self):
        self.best_pred_paths = {}
        uid_pid_path_topk_file = open("paths/" + self.dataset_name + "/agent_topk=" + self.agent_topk + "/best_pred_paths.csv")
        reader = csv.reader(uid_pid_path_topk_file, delimiter=",")
        next(reader, None)
        for row in reader:
            uid = int(row[0])
            pid = int(row[1])
            path_score = float(row[2])
            path_prob = float(row[3])
            normalized_path = normalize_path(row[4])

            if uid not in self.best_pred_paths:
                self.best_pred_paths[uid] = {}
            if pid not in self.best_pred_paths[uid]:
                self.best_pred_paths[uid][pid] = []
            self.best_pred_paths[uid][pid].append([path_score, path_prob, normalized_path])

        uid_pid_path_topk_file.close()
        #return self.best_pred_paths

    def generate_LIR_matrix(self):
        self.LIR_matrix = {}
        if self.dataset_name in AMAZON_DATASETS:
            self.LIR_matrix_words = {}
        for uid in self.uid_timestamp.keys():
            pid_lir_value = {}
            word_lir_value = {}
            if uid not in self.uid_timestamp:
                continue
            self.uid_timestamp[uid].sort()

            def normalized_ema(values):
                if max(values) == min(values):
                    values = np.array([i for i in range(len(values))])
                else:
                    values = np.array([i for i in values])
                values = pd.Series(values)
                ema_vals = values.ewm(span=len(values)).mean().tolist()
                min_res = min(ema_vals)
                max_res = max(ema_vals)
                return [(x - min_res) / (max_res - min_res) for x in ema_vals]

            if len(self.uid_timestamp[uid]) <= 1:  # Skips users with only one review in train (can happen with lastfm)
                continue
            ema_timestamps = normalized_ema(self.uid_timestamp[uid])
            assert len(self.item_idxs[uid]) == len(ema_timestamps)
            for idx, pid in enumerate(self.item_idxs[uid]):
                pid_lir_value[pid] = ema_timestamps[idx]

            if self.dataset_name in AMAZON_DATASETS:
                for idx, words in enumerate(self.word_idxs[uid]):
                    for word in words:
                        word_lir_value[word] = ema_timestamps[idx]
                self.LIR_matrix_words[uid] = word_lir_value
            self.LIR_matrix[uid] = pid_lir_value

    def generate_SEP_matrix(self):
        # Precompute entity distribution
        self.SEP_matrix = {}
        degrees = load_kg(self.dataset_name).degrees
        if "rproduct" in degrees:
            degrees["related_product"] = degrees["rproduct"] #tmp solution
            del degrees["rproduct"]
        for type, eid_indegree in degrees.items():
            pid_indegree_list = []
            biggest_indegree_value = float("-inf")
            smallest_indegree_value = float("inf")
            for pid, indegree in eid_indegree.items():
                biggest_indegree_value = max(indegree, biggest_indegree_value)
                smallest_indegree_value = min(indegree, smallest_indegree_value)
                pid_indegree_list.append([pid,indegree])  # idx = pid

            #Normalize indegree between 0 and 1
            normalized_indegree_list = [
                [x[0], (x[1] - smallest_indegree_value) / (biggest_indegree_value - smallest_indegree_value)] for x in
                pid_indegree_list]

            normalized_indegree_list.sort(key=lambda x: x[1])
            def normalized_ema(values):
                values = np.array([x for x in values])
                values = pd.Series(values)
                ema_vals = values.ewm(span=len(values)).mean().tolist()
                min_res = min(ema_vals)
                max_res = max(ema_vals)
                return [(x - min_res) / (max_res - min_res) for x in ema_vals]

            ema_es = normalized_ema([x[1] for x in normalized_indegree_list])
            pid_weigth = {}
            for idx in range(len(ema_es)):
                pid = normalized_indegree_list[idx][0]
                pid_weigth[pid] = ema_es[idx]

            self.SEP_matrix[type] = pid_weigth


