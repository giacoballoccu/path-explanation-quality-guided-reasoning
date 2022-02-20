import csv
# Dataset names.
import gzip
# from mapper import get_movie_mapping, get_mapping
import os
import pickle
import sys

from models.PGPR.utils import get_entities_without_user

ML1M = 'ml1m'
LASTFM = 'lastfm'
BEAUTY = 'beauty'
CD = 'cd'
CELL = 'cell'
CLOTH = 'cloth'
AMAZON_DATASETS = [CD, CELL, CLOTH, BEAUTY]
# Models

# Dataset directories.
KG_COMPLETATION_DATASET_DIR = {
    ML1M: './datasets/ml1m/joint-kg',
    LASTFM: './datasets/lastfm/kg-completion'
}

DATASET_DIR = {
    ML1M: './datasets/ml1m',
    LASTFM: './datasets/lastfm',
    BEAUTY: './datasets/Amazon_Beauty',
    CD: './datasets/Amazon_CDs',
    CELL: './datasets/Amazon_Cellphones',
    CLOTH: './datasets/Amazon_Clothing',
}

LABELS_DIR = {
    ML1M: {
        "kg": "models/PGPR/tmp/ml1m/kg.pkl",
        "train": "models/PGPR/tmp/ml1m/train_label.pkl",
        "test": "models/PGPR/tmp/ml1m/test_label.pkl",
    },
    LASTFM: {
        "kg": "models/PGPR/tmp/lastfm/kg.pkl",
        "train": "models/PGPR/tmp/lastfm/train_label.pkl",
        "test": "models/PGPR/tmp/lastfm/test_label.pkl",
    },
    CELL: {
        "kg": "models/PGPR/tmp/Amazon_Cellphones/kg.pkl",
        "train": "models/PGPR/tmp/Amazon_Cellphones/train_label.pkl",
        "test": "models/PGPR/tmp/Amazon_Cellphones/test_label.pkl",
    },
    BEAUTY: {
        "kg": "models/PGPR/tmp/Amazon_Beauty/kg.pkl",
        "train": "models/PGPR/tmp/Amazon_Beauty/train_label.pkl",
        "test": "models/PGPR/tmp/Amazon_Beauty/test_label.pkl",
    },
    CLOTH: {
        "kg": "models/PGPR/tmp/Amazon_Clothing/kg.pkl",
        "train": "models/PGPR/tmp/Amazon_Clothing/train_label.pkl",
        "test": "models/PGPR/tmp/Amazon_Clothing/test_label.pkl",
    }
}

PGPR_MODEL_DIR = "models/PGPR"

SENSIBLE_ATTRIBUTES = ["Gender", "Age", "Country", "Occupation"] #This order must be respected (dependency with metrics.py)
DATASET_SENSIBLE_ATTRIBUTE_MATRIX = {
    ML1M: {"Gender": True, "Age": True, "Country": False, "Occupation": True},
    LASTFM: {"Gender": True, "Age": True, "Country": True, "Occupation": False},
    CELL: {attribute: False for attribute in SENSIBLE_ATTRIBUTES},
    CD: {attribute: False for attribute in SENSIBLE_ATTRIBUTES},
    CLOTH: {attribute: False for attribute in SENSIBLE_ATTRIBUTES},
    BEAUTY: {attribute: False for attribute in SENSIBLE_ATTRIBUTES},
}
OVERALL_BETTER_ALPHA = {
    ML1M: {"LIR": 0.2, "SEP": 0.1, "ETD": 1},# "SEP-ETD": 0.15, "ETD-LIR": 0.25, "LIR_ETD": 0.5, "LIR_SEP_ETD": 0.25},
    LASTFM: {"LIR": 0.2, "SEP": 0.05, "ETD": 1},# "LIR_SEP": 0.2, "SEP_ETD": 0.15, "LIR_ETD": 0.35, "LIR_SEP_ETD": 0.2},
    CELL: {"LIR": 0.15, "SEP": 0.25, "ETD": 1},# "LIR_SEP": 0.2, "SEP_ETD": 0.1, "LIR_ETD": 0.2, "LIR_SEP_ETD": 0.35},
    BEAUTY: {"LIR": 0.2, "SEP": 0.25, "ETD": 1},# "LIR_SEP": 0.15, "SEP_ETD": 0.15, "LIR_ETD": 0.2 , "LIR_SEP_ETD": 0.25},
    CLOTH: {"LIR": 0.6, "SEP": 0.1, "ETD": 1},# "LIR_SEP": 0.45, "SEP_ETD": 0.3, "LIR_ETD": 0.4 , "LIR_SEP_ETD": 0.45},
}
# Selected relationships from the KG completion, used in the dataset_mapper part for dataset that have an external KG completion
SELECTED_RELATIONS = {
    ML1M: [0, 1, 2, 3, 8, 10, 14, 15, 16, 18],
    LASTFM: [0, 1, 2, 3, 4, 5, 6, 7, 8],
    CELL: [0, 1, 2, 3, 4, 5, 6, 7],
    CD: [0, 1, 2, 3, 4, 5, 6, 7],
    CLOTH: [0, 1, 2, 3, 4, 5, 6, 7],
    BEAUTY: [0, 1, 2, 3, 4, 5, 6, 7],
}

PATH_TYPES = {
    ML1M: ['watched', 'directed_by', 'belong_to', 'produced_by_company', 'produced_by_producer', 'starring','edited_by','wrote_by','cinematography','composed_by'],
    LASTFM: ['listened', 'belong_to', 'related_to', 'sang_by', 'mixed_by', 'produced_by_producer', 'original_version_of', 'related_to', 'alternative_version_of', 'featured_by'],
    BEAUTY: ['purchase', 'described_as', 'produced_by', 'belong_to', 'also_bought', 'also_viewed', 'bought_together', 'described_as', 'purchase'],
    CELL: ['purchase', 'described_as', 'produced_by', 'belongs_to', 'also_bought', 'also_viewed', 'bought_together', 'described_as', 'purchase'],
    CLOTH: ['purchase', 'described_as', 'produced_by', 'belongs_to', 'also_bought', 'also_viewed', 'bought_together', 'described_as', 'purchase'],
}

TOTAL_PATH_TYPES = {
    ML1M: len(set(PATH_TYPES[ML1M])),
    LASTFM: len(set(PATH_TYPES[LASTFM])),
    CELL: len(set(PATH_TYPES[CELL])),
    BEAUTY: len(set(PATH_TYPES[BEAUTY])),
    CLOTH: len(set(PATH_TYPES[CLOTH])),
}

# Model result directories.
TMP_DIR = {
    ML1M: './tmp/ml1m',
    LASTFM: './tmp/lastfm'
}


MAIN_PRODUCT_INTERACTION = {
    ML1M: ("movie", "watched"),
    LASTFM: ("song", "listened"),
    BEAUTY: ("product", "purchase"),
    CD: ("product", "purchase"),
    CELL: ("product", "purchase"),
    CLOTH: ("product", "purchase"),
}

def get_linked_interaction(path):
    return path[2]

def get_linked_interaction_id(path):
    return path[1][-1]

def get_shared_entity(path):
    return path[5]

def get_shared_entity_id(path):
    return path[2][-1]

def get_path_type(path):
    return path[-3] if path[-3] != "self_loop" else path[-6]

def get_kg_uid_to_gender_map(dataset_name):
    attribute = "Gender"
    if not DATASET_SENSIBLE_ATTRIBUTE_MATRIX[dataset_name][attribute]:
        print("The dataset selected doesn't possess the attribute {}".format(attribute))
        exit(-1)
    file = open(DATASET_DIR[dataset_name] + "/mappings/uid2gender.txt", 'r')
    csv_reader = csv.reader(file, delimiter='\n')
    uid_gender = {}
    gender2name = {0: "Male", 1: "Female"}
    uid_mapping = get_dataset_uid_to_kg_uid_map(dataset_name)  # 1->0
    for row in csv_reader:
        row = row[0].strip().split('\t')
        if dataset_name == "ml1m":
            uid_gender[uid_mapping[int(row[0])]] = 0 if row[1] == 'M' else 1
        else:
            uid_gender[uid_mapping[int(row[0])]] = 0 if row[1] == 'm' else 1
    return uid_gender, gender2name

def get_kg_uid_to_age_map(dataset_name):
    file = open(DATASET_DIR[dataset_name] + "/mappings/uid2age_map.txt", 'r')
    csv_reader = csv.reader(file, delimiter='\n')
    uid_age = {}
    age2name = {1: "Under 18", 18:  "18-24", 25:  "25-34", 35:  "35-44", 45:  "45-49", 50:  "50-55", 56:  "56+"}
    uid_mapping = get_dataset_uid_to_kg_uid_map(dataset_name)
    for row in csv_reader:
        row = row[0].strip().split('\t')
        uid_age[uid_mapping[int(row[0])]] = int(row[1])
    return uid_age, age2name

def get_kg_uid_to_occupation_map(dataset_name):
    file = open(DATASET_DIR[dataset_name] + "/mappings/uid2occupation.txt", 'r')
    csv_reader = csv.reader(file, delimiter='\n')
    uid_occ = {}
    occ2name = {0: "other", 1:  "academic/educator",  2:  "artist",  3:  "clerical/admin",  4:  "college/grad student",  5:  "customer service",  6:  "doctor/health care",  7:  "executive/managerial",  8:  "farmer",  9:  "homemaker", 10:  "K-12 student", 11:  "lawyer", 12:  "programmer", 13:  "retired", 14:  "sales/marketing", 15:  "scientist", 16:  "self-employed", 17:  "technician/engineer", 18:  "tradesman/craftsman", 19:  "unemployed", 20:  "writer"}
    uid_mapping = get_dataset_uid_to_kg_uid_map(dataset_name)  # 1->0
    for row in csv_reader:
        row = row[0].strip().split('\t')
        uid_occ[uid_mapping[int(row[0])]] = int(row[1])
    return uid_occ, occ2name

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
        user2timestamp[uid].append(timestamp)
        if uid not in item_idxs:
            item_idxs[uid] = []
            word_idxs[uid] = []
        item_idxs[uid].append(pid)
        word_idxs[uid].append(words)
    return item_idxs, word_idxs, user2timestamp

def get_interaction2timestamp_map(dataset_name):
    user2timestamp = {}
    item_idxs = {}
    dataset2kg = get_dataset_pid2kg_pid_map(dataset_name)
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


#Return the mapping between the id of the entity in the knowledge graph and his original entity id from the jointkg
def get_mapping(dataset_name, entity_name, old_id_as_key=False):
    mapping = {}
    file = open(DATASET_DIR[dataset_name] + "/mappings/" + entity_name + "id2dbid.txt", "r")
    csv_reader = csv.reader(file, delimiter='\t')
    next(csv_reader, None)
    for row in csv_reader:
        kg_id = int(row[0]) if dataset_name not in AMAZON_DATASETS else row[0]
        old_entity_id = int(row[1]) if dataset_name not in AMAZON_DATASETS else row[1]
        if old_id_as_key:
            mapping[old_entity_id] = kg_id
        else:
            mapping[kg_id] = old_entity_id
    return mapping


def get_all_entity_mappings(dataset_name):
    mappings = {}
    #main_entity, main_relation = MAIN_PRODUCT_INTERACTION[dataset_name]
    for entity in get_entities_without_user(dataset_name):
        if entity == 'movie':
            mappings[entity] = get_movie_mapping(dataset_name)
            continue
        if entity == 'song':
            mappings[entity] = get_song_mapping(dataset_name)
            continue
        mappings[entity] = get_mapping(dataset_name, entity, True)
    return mappings

def get_movie_mapping(dataset_name):
    valid = {}
    file = open(DATASET_DIR[dataset_name] + "/mappings/product_mappings.txt", "r")
    csv_reader = csv.reader(file, delimiter='\n')
    next(csv_reader, None)
    for row in csv_reader:
        row = row[0].strip().split("\t")
        valid[int(row[2])] = [int(row[0]), int(row[1])] #key: entityid, value: {kgid, movielandid}
    return valid

#CAN BE USE ONLY AFTER THE CREATION OF PRODUCT_MAPPINGS
def get_valid_products(dataset_name):
    valid = set()
    file = open(DATASET_DIR[dataset_name] + "/mappings/product_mappings.txt", "r")
    csv_reader = csv.reader(file, delimiter='\n')
    next(csv_reader, None)
    for row in csv_reader:
        row = row[0].strip().split("\t")
        if dataset_name not in AMAZON_DATASETS:
            valid.add(int(row[1]))
        else:
            valid.add(row[1])
    return valid

def get_invalid_users(dataset_name, reverse=False):
    invalid_users = {}
    file = open(DATASET_DIR[dataset_name] + "/invalid_users.txt", "r")
    reader = csv.reader(file, delimiter="\t")
    next(reader, None)
    for row in reader:
        amazon_id = row[0]
        previous_kg_id = row[1]
        if not reverse:
            invalid_users[amazon_id] = previous_kg_id
        else:
            invalid_users[previous_kg_id] = amazon_id
    return invalid_users

def get_invalid_products(dataset_name, reverse=False):
    invalid_products = {}
    file = open(DATASET_DIR[dataset_name] + "/invalid_products.txt", "r")
    reader = csv.reader(file)
    next(reader, None)
    for row in reader:
        amazon_pid = row[0]
        previous_kg_id = row[1]
        if not reverse:
            invalid_products[amazon_pid] = previous_kg_id
        else:
            invalid_products[previous_kg_id] = amazon_pid
    return invalid_products

def get_song_mapping(dataset_name):
    valid = {}
    file = open(DATASET_DIR[dataset_name] + "/mappings/product_mappings.txt", "r")
    csv_reader = csv.reader(file, delimiter='\n')
    next(csv_reader, None)
    for row in csv_reader:
        row = row[0].strip().split("\t")
        valid[int(row[2])] = [int(row[0]), int(row[1])] #key: entityid, value: {kgid, movielandid}
    return valid

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

def zip_file(filename):
    with open(filename, 'rb') as file:
        zipped = gzip.open(filename + '.gz', 'wb')
        zipped.writelines(file)
        zipped.close()
    file.close()

#Returns a string representing the path type
def get_path_type(path):
    return path[-1][0]

#Only for Amazon where you can have multiple interactions types
def get_interaction_type(path):
    return path[1][0]

def get_interaction_id(path):
    return path[1][-1]

def get_rec_pid(path):
    return int(path[-1][-1][-1])

def get_shared_entity(path):
    #In a path of size 3 the shared entity and the linked interaction are the same
    if len(path) == 3:
        shared_entity_name = path[1][-2]
        shared_entity_eid = int(path[1][-1])
    #On paths of size 4 the shared entity is separated from the linked interaction
    else:
        shared_entity_name = path[-2][1]
        shared_entity_eid = int(path[-2][-1])
    #Potentially there may be multiple shared entity if you allow path with length more than 4
    return shared_entity_name, shared_entity_eid

#Trasform a string separeted by space that rapresent the path in a list composed by triplets
def normalize_path(path_str):
    path = path_str.split(" ")
    normalized_path = []
    for i in range(0, len(path), 3):
        normalized_path.append((path[i], path[i + 1], path[i + 2]))
    return normalized_path

def load_labels(dataset, mode='train'):
    if mode == 'train':
        label_file = LABELS_DIR[dataset][mode]
        # CHANGED
    elif mode == 'test':
        label_file = LABELS_DIR[dataset][mode]
    else:
        raise Exception('mode should be one of {train, test}.')
    user_products = pickle.load(open(label_file, 'rb'))
    return user_products

def load_kg(dataset):
    kg_file = LABELS_DIR[dataset]["kg"]
    # CHANGED
    sys.path.append(r'models/PGPR')
    kg = pickle.load(open(kg_file, 'rb'))
    return kg

def ensure_result_folder(args):
    # Creation of results folders
    if not os.path.exists("./results"):
        os.makedirs("./results")
    if not os.path.exists("./results/" + args.dataset):
        os.makedirs("./results/" + args.dataset)
    if not os.path.exists("./results/" + args.dataset + "/agent_topk=" + args.agent_topk):
        os.makedirs("./results/" + args.dataset + "/agent_topk=" + args.agent_topk)
    result_base_path = "./results/" + args.dataset + "/agent_topk=" + args.agent_topk + "/"
    return result_base_path

def ensure_log_folder(args):
    if not os.path.exists("./logs"):
        os.makedirs("./logs")
    if not os.path.exists("./logs/" + args.dataset):
        os.makedirs("./logs/" + args.dataset)
    if not os.path.exists("./logs/" + args.dataset + "/agent_topk=" + args.agent_topk):
        os.makedirs("./logs/" + args.dataset + "/agent_topk=" + args.agent_topk)
    log_base_path = "./logs/" + args.dataset + "/agent_topk=" + args.agent_topk + "/"
    return log_base_path


def ensure_path_folder(args):
    path_dir = "./results/paths"
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)
    if not os.path.exists(path_dir + "/" + args.dataset):
        os.makedirs(path_dir + "/" + args.dataset)
    if not os.path.exists(path_dir + "/" + args.dataset + "/agent_topk=" + args.agent_topk):
        os.makedirs(path_dir + "/" + args.dataset + "/agent_topk=" + args.agent_topk)
    path_base_path = path_dir + "/" + args.dataset + "/agent_topk=" + args.agent_topk + "/"
    return path_base_path