import pandas as pd

from utils import *

def get_interaction2timestamp_with_words(dataset_name):
    if dataset_name not in AMAZON_DATASETS:
        print("The dataset chosen doesn't use text reviews.")
        exit(-1)
    user2timestamp = {}
    item_idxs = {}
    word_idxs = {}
    file = open(DATASET_DIR[dataset_name] + "/train.txt", 'r')
    csv_reader = csv.reader(file, delimiter='\t')
    for row in csv_reader:
        uid = int(row[0])
        pid = int(row[1])
        timestamp = int(row[3])
        words = [int(w) for w in row[4].split(" ")]
        if uid not in user2timestamp:
            user2timestamp[uid] = []
        if uid not in item_idxs:
            item_idxs[uid] = []
        if uid not in word_idxs:
            word_idxs[uid] = []

        user2timestamp[uid].append(timestamp)
        item_idxs[uid].append(pid)
        word_idxs[uid].append(words)
    return item_idxs, word_idxs, user2timestamp

def get_dataset_pid2kg_pid_map(dataset_name):
    file = open(DATASET_DIR[dataset_name] + "/mappings/product_mappings.txt", "r")
    csv_reader = csv.reader(file, delimiter='\n')
    dataset_pid2kg_pid = {}
    next(csv_reader, None)
    for row in csv_reader:
        row = row[0].strip().split("\t")
        if dataset_name in AMAZON_DATASETS:
            dataset_pid2kg_pid[row[1]] = int(row[0])
        else:
            dataset_pid2kg_pid[int(row[1])] = int(row[0])
    file.close()
    return dataset_pid2kg_pid

def get_dataset_uid_to_kg_uid_map(dataset_name):
    review_uid_kg_uid = {}
    with open(DATASET_DIR[dataset_name] + "/mappings/user_mappings.txt", 'r') as file:
        reader = csv.reader(file, delimiter="\t")
        next(reader, None)
        for row in reader:
            uid_review = int(row[1]) if dataset_name not in AMAZON_DATASETS else row[1]
            uid_kg = int(row[0])
            review_uid_kg_uid[uid_review] = uid_kg
    return review_uid_kg_uid

def get_interaction2timestamp_map(dataset_name):
    user2timestamp = {}
    item_idxs = {}
    dataset2kg = get_dataset_pid2kg_pid_map(dataset_name) if dataset_name not in AMAZON_DATASETS else None
    file = open(DATASET_DIR[dataset_name] + "/train.txt", 'r')
    csv_reader = csv.reader(file, delimiter=' ')
    uid_mapping = get_dataset_uid_to_kg_uid_map(dataset_name)
    for row in csv_reader:
        uid = uid_mapping[int(row[0])] if dataset_name not in AMAZON_DATASETS else uid_mapping[row[0]]
        movie_id_ml = int(row[1]) if dataset_name not in AMAZON_DATASETS else row[1]
        if movie_id_ml not in dataset2kg: continue
        movie_id_kg = dataset2kg[movie_id_ml]
        timestamp = int(row[2]) if dataset_name == "lastfm" else int(row[3])
        if uid not in user2timestamp:
            user2timestamp[uid] = []
        if uid not in item_idxs:
            item_idxs[uid] = []
        user2timestamp[uid].append(timestamp)
        item_idxs[uid].append(movie_id_kg)
    return item_idxs, user2timestamp

def generate_LIR_matrix(dataset_name, uid_timestamp, item_idxs, word_idxs=None):
    LIR_matrix = {}
    LIR_matrix_words = {}
    for uid in uid_timestamp.keys():
        pid_lir_value = {}
        word_lir_value = {}
        if uid not in uid_timestamp:
            continue
        uid_timestamp[uid].sort()
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

        if len(uid_timestamp[uid]) <= 1: #Skips users with only one review in train (can happen with lastfm)
            continue
        ema_timestamps = normalized_ema(uid_timestamp[uid])
        assert len(item_idxs[uid]) == len(ema_timestamps)
        for idx, pid in enumerate(item_idxs[uid]):
            if type(pid) == list:
                print(pid)
            pid_lir_value[pid] = ema_timestamps[idx]

        if dataset_name in AMAZON_DATASETS:
            for idx, words in enumerate(word_idxs[uid]):
                for word in words:
                    word_lir_value[word] = ema_timestamps[idx]
            LIR_matrix_words[uid] = word_lir_value
        LIR_matrix[uid] = pid_lir_value
    return LIR_matrix, LIR_matrix_words

def generate_SEP_matrix(dataset_name):
    # Precompute entity distribution
    SEP_matrix = {}
    degrees = load_kg(dataset_name).degrees
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

        SEP_matrix[type] = pid_weigth
    return SEP_matrix