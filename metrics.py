import numpy as np
from myutils import *
from easydict import EasyDict as edict

def dcg_at_k(r, k, method=1):
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k, method=0):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max

def measure_rec_quality(path_data):
    # Evaluate only the attributes that have been chosen and are avaiable in the chosen dataset
    flags = path_data.sens_attribute_flags
    attribute_list = get_attribute_list(path_data.dataset_name, flags)
    metrics_names = ["ndcg", "hr", "recall", "precision"]
    metrics = edict()
    for metric in metrics_names:
        metrics[metric] = {"Overall": []}
        for values in attribute_list.values():
            if len(attribute_list) == 1: break
            attribute_to_name = values[1]
            for _, name in attribute_to_name.items():
                metrics[metric][name] = []

    topk_matches = path_data.uid_topk
    test_labels = path_data.test_labels

    test_user_idxs = list(test_labels.keys())
    invalid_users = []
    for uid in test_user_idxs:
        if uid not in topk_matches: continue
        if len(topk_matches[uid]) < 10:
            invalid_users.append(uid)
            continue
        pred_list, rel_set = topk_matches[uid], test_labels[uid]
        if len(pred_list) == 0:
            continue

        k = 0
        hit_num = 0.0
        hit_list = []
        for pid in pred_list:
            k += 1
            if pid in rel_set:
                hit_num += 1
                hit_list.append(1)
            else:
                hit_list.append(0)

        ndcg = ndcg_at_k(hit_list, k)
        recall = hit_num / len(rel_set)
        precision = hit_num / len(pred_list)
        hit = 1.0 if hit_num > 0.0 else 0.0

        # Based on attribute
        for attribute in attribute_list.keys():
            if uid not in attribute_list[attribute][0]: continue
            attr_value = attribute_list[attribute][0][uid]
            if attr_value not in attribute_list[attribute][1]: continue #Few users may have the attribute missing (LASTFM)
            attr_name = attribute_list[attribute][1][attr_value]
            metrics["ndcg"][attr_name].append(ndcg)
            metrics["recall"][attr_name].append(recall)
            metrics["precision"][attr_name].append(precision)
            metrics["hr"][attr_name].append(hit)
        metrics["ndcg"]["Overall"].append(ndcg)
        metrics["recall"]["Overall"].append(recall)
        metrics["precision"]["Overall"].append(precision)
        metrics["hr"]["Overall"].append(hit)
    return metrics

def print_rec_metrics(dataset_name, flags, metrics):
    attribute_list = get_attribute_list(dataset_name, flags)

    print("\n---Recommandation Quality---")
    print("Average for the entire user base:", end=" ")
    for metric, values in metrics.items():
        print("{}: {:.3f}".format(metric, np.array(values["Overall"]).mean()), end=" | ")
    print("")

    for attribute_category, values in attribute_list.items():
        print("\n-Statistic with user grouped by {} attribute".format(attribute_category))
        for attribute in values[1].values():
            print("{} group".format(attribute), end=" ")
            for metric_name, groups_values in metrics.items():
                print("{}: {:.3f}".format(metric_name, np.array(groups_values[attribute]).mean()), end=" | ")
            print("")
    print("\n")

"""
Explanation metrics
"""

def topk_ETV(path_data):
    dataset_name = path_data.dataset_name

    def simpson_index(topk):
        n_path_for_patterns = {k: 0 for k in set(PATH_TYPES[dataset_name])}
        N = 0
        for path in topk:
            path = path
            path_type = get_path_type(path)
            if path_type == 'self_loop':
                path_type = 'described_as'
            n_path_for_patterns[path_type] += 1
            N += 1
        numerator = 0
        for path_type, n_path_type_ith in n_path_for_patterns.items():
            numerator += n_path_type_ith * (n_path_type_ith - 1)

        # N = 0
        # for item_path in pred_uv_paths.items():
        #    N += len(item_path[1])
        if N * (N - 1) == 0:
            return 0
        return 1 - (numerator / (N * (N - 1)))

    ETVs = {}
    for uid, topk in path_data.uid_topk.items():
        if uid not in path_data.test_labels: continue
        ETV = simpson_index([path_data.uid_pid_explanation[uid][pid] for pid in topk])
        ETVs[uid] = ETV
    return ETVs

def avg_ETV(path_data):
    uid_ETVs = topk_ETV(path_data)

    # Evaluate only the attributes that have been chosen and are avaiable in the chosen dataset
    flags = path_data.sens_attribute_flags
    attribute_list = get_attribute_list(path_data.dataset_name, flags)
    avg_groups_ETV = {}
    groups_ETV_scores = {}

    # Initialize group scores with empty list
    for attribute in attribute_list.keys():
        for _, attribute_label in attribute_list[attribute][1].items():
            groups_ETV_scores[attribute_label] = []

    if "Overall" not in groups_ETV_scores:
        groups_ETV_scores["Overall"] = []

    for uid, ETV in uid_ETVs.items():
        for attribute in attribute_list.keys():
            if uid not in attribute_list[attribute][0]: continue
            attr_value = attribute_list[attribute][0][uid]
            if attr_value not in attribute_list[attribute][1]: continue  # Few users may have the attribute missing (LASTFM)
            attr_name = attribute_list[attribute][1][attr_value]
            groups_ETV_scores[attr_name].append(ETV)
        groups_ETV_scores["Overall"].append(ETV)

    for attribute_label, group_scores in groups_ETV_scores.items():
        avg_groups_ETV[attribute_label] = np.array(group_scores).mean()

    explanation_type_variety = edict(
        avg_groups_ETV=avg_groups_ETV,
        groups_ETV_scores=groups_ETV_scores
    )
    return explanation_type_variety

def avg_LID(path_data):
    uid_LIDs = topk_LID(path_data)

    # Evaluate only the attributes that have been chosen and are avaiable in the chosen dataset
    flags = path_data.sens_attribute_flags
    attribute_list = get_attribute_list(path_data.dataset_name, flags)
    avg_groups_LID = {}
    groups_LID_scores = {}

    # Initialize group scores with empty list
    for attribute in attribute_list.keys():
        for _, attribute_label in attribute_list[attribute][1].items():
            groups_LID_scores[attribute_label] = []

    if "Overall" not in groups_LID_scores:
        groups_LID_scores["Overall"] = []

    for uid, LID in uid_LIDs.items():
        for attribute in attribute_list.keys():
            if uid not in attribute_list[attribute][0]: continue
            attr_value = attribute_list[attribute][0][uid]
            if attr_value not in attribute_list[attribute][1]: continue #Few users may have the attribute missing (LASTFM)
            attr_name = attribute_list[attribute][1][attr_value]
            groups_LID_scores[attr_name].append(LID)
        groups_LID_scores["Overall"].append(LID)


    for attribute_label, group_scores in groups_LID_scores.items():
        avg_groups_LID[attribute_label] = np.array(group_scores).mean()


    linked_interaction_diversity_results = edict(
        avg_groups_LID=avg_groups_LID,
        groups_LID_scores=groups_LID_scores
    )
    return linked_interaction_diversity_results

def topk_LID(path_data):
    LIDs = {}
    for uid, topk in path_data.uid_topk.items():
        if uid not in path_data.test_labels: continue
        unique_linked_interaction = set()
        count = 0
        for pid in topk:
            if pid not in path_data.uid_pid_explanation[uid]:
                continue
            current_path = path_data.uid_pid_explanation[uid][pid]
            li = get_linked_interaction_id(current_path)
            if current_path[1][0] == "mention":
                li += 10000 #pad in order to not make them overlap, this is a stupid workaround, fix it
            unique_linked_interaction.add(li)
        if len(topk) == 0 or len(unique_linked_interaction) == 0:
            count += 1
        LID = len(unique_linked_interaction) / len(topk)
        LIDs[uid] = LID
    print(count)
    return LIDs

def avg_SED(path_data):
    uid_SEDs = topk_SED(path_data)

    # Evaluate only the attributes that have been chosen and are avaiable in the chosen dataset
    flags = path_data.sens_attribute_flags
    attribute_list = get_attribute_list(path_data.dataset_name, flags)
    avg_groups_SED = {}
    groups_SED_scores = {}

    # Initialize group scores with empty list
    for attribute in attribute_list.keys():
        for _, attribute_label in attribute_list[attribute][1].items():
            groups_SED_scores[attribute_label] = []

    if "Overall" not in groups_SED_scores:
        groups_SED_scores["Overall"] = []

    for uid, SED in uid_SEDs.items():
        for attribute in attribute_list.keys():
            if uid not in attribute_list[attribute][0]: continue
            attr_value = attribute_list[attribute][0][uid]
            if attr_value not in attribute_list[attribute][1]: continue #Few users may have the attribute missing (LASTFM)
            attr_name = attribute_list[attribute][1][attr_value]
            groups_SED_scores[attr_name].append(SED)
        groups_SED_scores["Overall"].append(SED)


    for attribute_label, group_scores in groups_SED_scores.items():
        avg_groups_SED[attribute_label] = np.array(group_scores).mean()


    shared_entity_diversity_results = edict(
        avg_groups_SED=avg_groups_SED,
        groups_SED_scores=groups_SED_scores
    )
    return shared_entity_diversity_results

def topk_SED(path_data):
    SEDs = {}
    for uid, topk in path_data.uid_topk.items():
        if uid not in path_data.test_labels: continue
        unique_shared_entities = set()
        for pid in topk:
            if pid not in path_data.uid_pid_explanation[uid]:
                continue
            current_path = path_data.uid_pid_explanation[uid][pid]
            se = get_shared_entity_id(current_path)
            unique_shared_entities.add(se)
        if len(topk) > 0:
            SED = len(unique_shared_entities) / len(topk)
        else:
            SED = 1
        SEDs[uid] = SED
    return SEDs

def topk_ETD(path_data):
    ETDs = {}
    for uid, topk in path_data.uid_topk.items():
        if uid not in path_data.test_labels: continue
        unique_path_types = set()
        for pid in topk:
            if pid not in path_data.uid_pid_explanation[uid]:
                continue
            current_path = path_data.uid_pid_explanation[uid][pid]
            path_type = get_path_type(current_path)
            unique_path_types.add(path_type)
        ETD = len(unique_path_types) / TOTAL_PATH_TYPES[path_data.dataset_name]
        ETDs[uid] = ETD
    return ETDs

def get_attribute_list(dataset_name, flags):
    attribute_list = {}
    for attribute, flag in flags.items():
        if flag and DATASET_SENSIBLE_ATTRIBUTE_MATRIX[dataset_name][attribute]:
            attribute_list[attribute] = []

    for attribute in attribute_list.keys():
        if attribute == "Gender":
            user2attribute, attribute2name = get_kg_uid_to_gender_map(dataset_name)
        elif attribute == "Age":
            user2attribute, attribute2name = get_kg_uid_to_age_map(dataset_name)
        elif attribute == "Occupation":
            user2attribute, attribute2name = get_kg_uid_to_occupation_map(dataset_name)
        elif attribute == "Country":
            pass #implement country
        else:
            print("Unknown attribute")
        attribute_list[attribute] = [user2attribute, attribute2name]
    return attribute_list

def avg_ETD(path_data):
    uid_ETDs = topk_ETD(path_data)

    # Evaluate only the attributes that have been chosen and are avaiable in the chosen dataset
    flags = path_data.sens_attribute_flags
    attribute_list = get_attribute_list(path_data.dataset_name, flags)
    avg_groups_ETD = {}
    groups_ETD_scores = {}

    # Initialize group scores with empty list
    for attribute in attribute_list.keys():
        for _, attribute_label in attribute_list[attribute][1].items():
            groups_ETD_scores[attribute_label] = []

    if "Overall" not in groups_ETD_scores:
        groups_ETD_scores["Overall"] = []

    for uid, ETD in uid_ETDs.items():
        for attribute in attribute_list.keys():
            if uid not in attribute_list[attribute][0]: continue
            attr_value = attribute_list[attribute][0][uid]
            if attr_value not in attribute_list[attribute][1]: continue #Few users may have the attribute missing (LASTFM)
            attr_name = attribute_list[attribute][1][attr_value]
            groups_ETD_scores[attr_name].append(ETD)
        groups_ETD_scores["Overall"].append(ETD)


    for attribute_label, group_scores in groups_ETD_scores.items():
        avg_groups_ETD[attribute_label] = np.array(group_scores).mean()


    diversity_results = edict(
        avg_groups_ETD=avg_groups_ETD,
        groups_ETD_scores=groups_ETD_scores
    )
    return diversity_results




#Extract the value of LIR for the given user item path from the LIR_matrix
def LIR_single(path_data, path):
    uid = int(path[0][-1])
    if uid not in path_data.uid_timestamp or uid not in path_data.LIR_matrix or len(path_data.uid_timestamp[uid]) <= 1: return 0. #Should not enter there
    predicted_path = path
    linked_interaction = int(get_interaction_id(predicted_path))
    linked_interaction_type = get_interaction_type(predicted_path)

    #Handle the case of Amazon Dataset where a path may have different interaction types
    if linked_interaction_type == "mentions":
        LIR = path_data.LIR_matrix_words[uid][linked_interaction]
    elif linked_interaction_type == "watched" or linked_interaction_type == "listened" or linked_interaction_type == "purchase":
        LIR = path_data.LIR_matrix[uid][linked_interaction]
    else:
        LIR = 0.
    return LIR



# Returns a dict where to every uid is associated a value of LIR calculated based on his topk
def topk_LIR(path_data):
    LIR_topk = {}

    # Precompute user timestamps weigths
    LIR_matrix = path_data.LIR_matrix


    for uid in path_data.test_labels.keys(): #modified for pgpr labels
        LIR_single_topk = []
        if uid not in LIR_matrix or uid not in path_data.uid_topk:
            continue
        for pid in path_data.uid_topk[uid]:
            predicted_path = path_data.uid_pid_explanation[uid][pid]
            linked_interaction = int(get_interaction_id(predicted_path))
            linked_interaction_type = get_interaction_type(predicted_path)
            # Handle the case of Amazon Dataset where a path may have different interaction types
            if linked_interaction_type == "mentions":
                LIR = path_data.LIR_matrix_words[uid][linked_interaction]
            elif linked_interaction_type == "purchase" or linked_interaction_type == "watched" or linked_interaction_type == "listened":
                LIR = LIR_matrix[uid][linked_interaction]
            else:
                LIR = 0.
            LIR_single_topk.append(LIR)
        LIR_topk[uid] = np.array(LIR_single_topk).mean() if len(LIR_single_topk) != 0 else 0
    return LIR_topk


# Returns an avg value for the LIR of a given group
def avg_LIR(path_data):
    uid_LIR_score = topk_LIR(path_data)
    avg_groups_LIR = {}
    groups_LIR_scores = {}

    # Evaluate only the attributes that have been chosen and are avaiable in the chosen dataset
    flags = path_data.sens_attribute_flags
    attribute_list = get_attribute_list(path_data.dataset_name, flags)

    #Initialize group scores with empty list
    for attribute in attribute_list.keys():
        for _, attribute_label in attribute_list[attribute][1].items():
            groups_LIR_scores[attribute_label] = []

    if "Overall" not in groups_LIR_scores:
        groups_LIR_scores["Overall"] = []

    for uid, LIR_score in uid_LIR_score.items():
        for attribute in attribute_list.keys():
            if uid not in attribute_list[attribute][0]: continue
            attr_value = attribute_list[attribute][0][uid]
            if attr_value not in attribute_list[attribute][1]: continue #Few users may have the attribute missing (LASTFM)
            attr_name = attribute_list[attribute][1][attr_value]
            groups_LIR_scores[attr_name].append(LIR_score)
        groups_LIR_scores["Overall"].append(LIR_score)


    for attribute_label, group_scores in groups_LIR_scores.items():
         avg_groups_LIR[attribute_label] = np.array(group_scores).mean()

    LIR = edict(
        avg_groups_LIR=avg_groups_LIR,
        groups_LIR_scores=groups_LIR_scores,
    )

    return LIR

#Extract the value of SEP for the given user item path from the SEP_matrix
def SEP_single(path_data, path):
    related_entity_type, related_entity_id = get_shared_entity(path)
    SEP = path_data.SEP_matrix[related_entity_type][related_entity_id]
    return SEP


def topks_SEP(path_data):
    SEP_topk = {}

    # Precompute entity distribution
    exp_serendipity_matrix = path_data.SEP_matrix

    #Measure explanation serendipity for topk
    for uid in path_data.test_labels:
        SEP_single_topk = []
        if uid not in path_data.uid_topk: continue
        for pid in path_data.uid_topk[uid]:
            if pid not in path_data.uid_pid_explanation[uid]:
                #print("strano 2")
                continue
            path = path_data.uid_pid_explanation[uid][pid]
            related_entity_type, related_entity_id = get_shared_entity(path)
            SEP = exp_serendipity_matrix[related_entity_type][related_entity_id]
            SEP_single_topk.append(SEP)
        if len(SEP_single_topk) == 0: continue
        SEP_topk[uid] = np.array(SEP_single_topk).mean()
    return SEP_topk


def avg_SEP(path_data):
    uid_SEP = topks_SEP(path_data)
    avg_groups_SEP = {}
    groups_SEP_scores = {}

    # Evaluate only the attributes that have been chosen and are avaiable in the chosen dataset
    flags = path_data.sens_attribute_flags
    attribute_list = get_attribute_list(path_data.dataset_name, flags)

    # Initialize group scores with empty list
    for attribute in attribute_list.keys():
        for _, attribute_label in attribute_list[attribute][1].items():
            groups_SEP_scores[attribute_label] = []

    if "Overall" not in groups_SEP_scores:
        groups_SEP_scores["Overall"] = []

    for uid, SEP_score in uid_SEP.items():
        for attribute in attribute_list.keys():
            if uid not in attribute_list[attribute][0]: continue
            attr_value = attribute_list[attribute][0][uid]
            if attr_value not in attribute_list[attribute][1]: continue #Few users may have the attribute missing (LASTFM)
            attr_name = attribute_list[attribute][1][attr_value]
            groups_SEP_scores[attr_name].append(SEP_score)
        groups_SEP_scores["Overall"].append(SEP_score)

    for attribute_label, group_scores in groups_SEP_scores.items():
        avg_groups_SEP[attribute_label] = np.array(group_scores).mean()

    serendipity_results = edict(
        avg_groups_SEP=avg_groups_SEP,
        groups_SEP_scores=groups_SEP_scores,
    )
    return serendipity_results

def print_expquality_metrics(dataset_name, flags, metric_values):
    attribute_list = get_attribute_list(dataset_name, flags)
    print("\n---Explanation Quality---")
    print("Average for the entire user base:", end=" ")
    for metric, values in metric_values.items():
        print("{}: {:.3f}".format(metric, values["Overall"]), end= " | ")
    print("")

    for attribute_category, values in attribute_list.items():
        attributes = values[1].values()
        print("\n-Statistic with user grouped by {} attribute".format(attribute_category))
        for attribute in attributes:
            print("{} group".format(attribute), end=" ")
            for metric, values in metric_values.items():
                print("{}: {:.3f}".format(metric, values[attribute]), end=" | ")
            print("")

