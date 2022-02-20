from __future__ import absolute_import, division, print_function

import os
import argparse
from math import log

import torch as torch
from easydict import EasyDict as edict
from tqdm import tqdm
from functools import reduce
from kg_env import BatchKGEnvironment
from train_agent import ActorCritic
from utils import *
from extract_predicted_paths import *
from path_metrics import *
import pandas as pd

def evaluate(dataset_name, topk_matches, test_user_products):
    """Compute metrics for predicted recommendations.
    Args:
        topk_matches: a list or dict of product ids in ascending order.
    """
    invalid_users = []
    attribute_name = "Gender"
    if dataset_name not in AMAZON_DATASETS and dataset_name != BEAUTY: # ^^^
        if attribute_name == "Gender":
            user2attribute, attribute2name = get_user2gender(dataset_name)
        elif attribute_name == "Age":
            user2attribute, attribute2name = get_user2age()
        elif attribute_name == "Occupation":
            user2attribute, attribute2name = get_user2occupation()
        else:
            print("Not existing attribute selected attribute")
            return
    # Compute metrics
    metrics = edict(
        ndcg_other=edict(
            Male=[],
            Female=[],
            Overall=[]
        ),
        ndcg=edict(
            Male=[],
            Female=[],
            Overall=[]
        ),
        hr=edict(
            Male=[],
            Female=[],
            Overall=[]
        ),
        precision=edict(
            Male=[],
            Female=[],
            Overall=[]
        ),
        recall=edict(
            Male=[],
            Female=[],
            Overall=[]
        ),

    )
    ndcgs = []
    #uid2gender, gender2name = get_user2gender(dataset_name)
    test_user_idxs = list(test_user_products.keys())
    rel_size = []
    for uid in test_user_idxs:
        if uid not in topk_matches or len(topk_matches[uid]) < 10:
            invalid_users.append(uid)
            continue
        pred_list, rel_set = topk_matches[uid][::-1], test_user_products[uid]
        if len(pred_list) == 0:
            continue
        rel_size.append(len(rel_set))
        #if len(rel_set) < 10: continue
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

        #old
        dcg = 0.0
        hit_num = 0.0
        for i in range(len(pred_list)):
            if pred_list[i] in rel_set:
                dcg += 1. / (log(i + 2) / log(2))
                hit_num += 1
        # idcg
        idcg = 0.0
        for i in range(min(len(rel_set), len(pred_list))):
            idcg += 1. / (log(i + 2) / log(2))
        ndcg_other = dcg / idcg
        #recall = hit_num / len(rel_set)
        #precision = hit_num / len(pred_list)
        #hit = 1.0 if hit_num > 0.0 else 0.0

        ndcg = ndcg_at_k(hit_list, k)
        recall = hit_num / len(rel_set)
        precision = hit_num / len(pred_list)
        hit = 1.0 if hit_num > 0.0 else 0.0

        # Based on attribute
#        attribute_val = uid2gender[uid]
#        gender = gender2name[attribute_val]
        all = "Overall"

        # According to gender
 #       metrics.ndcg[gender].append(ndcg)
#        metrics.recall[gender].append(recall)
 #       metrics.precision[gender].append(precision)
 #       metrics.hr[gender].append(hit)

        # General
        metrics.ndcg_other[all].append(ndcg_other)
        metrics.ndcg[all].append(ndcg)
        metrics.hr[all].append(hit)
        metrics.recall[all].append(recall)
        metrics.precision[all].append(precision)

    print("Average test set size: ",np.array(rel_size).mean())
    for metric, groups_values in metrics.items():
        for group_id, values in groups_values.items():
            avg_metric_value = np.mean(values)*100 if metric == "ndcg_other" else np.mean(values)
            n_users = len(values)
            print("{} group  {}, noOfUser={}, PGPR {}={:.4f}".format(attribute_name, group_id, n_users, metric,
                                                                       avg_metric_value))
        print("\n")


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


def ndcg_at_k(r, k, method=1):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max



def batch_beam_search(env, model, uids, device, intrain=None, topk=[25, 5, 1]):
    def _batch_acts_to_masks(batch_acts):
        batch_masks = []
        for acts in batch_acts:
            num_acts = len(acts)
            act_mask = np.zeros(model.act_dim, dtype=np.uint8)
            act_mask[:num_acts] = 1
            batch_masks.append(act_mask)
        return np.vstack(batch_masks)

    def _batch_acts_to_lir(prev, batch_acts):
        batch_lirs = []
        for idx, acts in enumerate(batch_acts):
            lirs = [0] * model.act_dim
            for i, act in enumerate(acts):
                uid = prev[idx][0][-1]
                interaction = prev[idx][0][0]
                if act[0] != WATCHED and act[0] != MENTION:
                    lirs[i] = 0.
                    continue
                pid = act[-1]
                if interaction == WATCHED:
                    lir = env.LIR_matrix[uid][pid]
                elif interaction == MENTION:
                    lir = env.LIR_matrix_words[uid][pid]
                else:
                    lir = 0.
                lirs[i] = lir
            batch_lirs.append(np.array(lirs))
        return np.vstack(batch_lirs)

    def _batch_acts_to_sep(batch_acts):
        batch_seps = []
        for idx, acts in enumerate(batch_acts):
            seps = [0] * model.act_dim
            for i, act in enumerate(acts):
                r = act[0]
                if r == SELF_LOOP:
                    seps[i] = 0.
                    continue
                etype = RELATION_TO_ENTITY[env.dataset_name][r]
                eid = act[1]
                if eid not in env.SEP_matrix[etype]:
                    sep = 0.
                else:
                    sep = env.SEP_matrix[etype][eid]
                seps[i] = sep
            batch_seps.append(np.array(seps))
        return np.vstack(batch_seps)

    state_pool = env.reset(uids)  # numpy of [bs, dim]
    path_pool = env._batch_path  # list of list, size=bs
    probs_pool = [[] for _ in uids]
    model.eval()
    for hop in range(3):
        state_tensor = torch.FloatTensor(state_pool).to(device)
        acts_pool = env._batch_get_actions(path_pool, False)  # list of list, size=bs
        actmask_pool = _batch_acts_to_masks(acts_pool)  # numpy of [bs, dim]
        actmask_tensor = torch.ByteTensor(actmask_pool).to(device)
        probs, _ = model((state_tensor, actmask_tensor))  # Tensor of [bs, act_dim]

        probs_max, _ = torch.max(probs, 0)
        probs_min, _ = torch.min(probs, 0)

        if intrain == "LIR" and hop == 0:
            alpha = 0.65
            lir_scores = _batch_acts_to_lir(path_pool, acts_pool)
            lir_scores = torch.tensor(lir_scores).to(device)
            lir_max, _ = torch.max(lir_scores, 0)
            lir_min, _ = torch.min(lir_scores, 0)
            lir_max.to(device)
            lir_min.to(device)
            new_range = (probs_max - probs_min).to(device)
            lir_scores = ((lir_scores - lir_min) / (lir_max - lir_min)) * (new_range) + probs_min
            #torch.masked_select(lir_scores, actmask_tensor)
            probs = (alpha * (probs) + (1-alpha) * (lir_scores))
            probs = torch.nan_to_num(probs)
            probs += actmask_tensor.float() # In order to differ from masked actions
            #torch.masked_select(probs, actmask_tensor.bool())
        if intrain == "SEP" and hop == 1:
            alpha = 0.75
            sep_scores = _batch_acts_to_sep(acts_pool)
            sep_scores = torch.tensor(sep_scores).to(device)
            lir_max, _ = torch.max(sep_scores, 0)
            lir_min, _ = torch.min(sep_scores, 0)
            lir_max.to(device)
            lir_min.to(device)
            new_range = (probs_max - probs_min).to(device)
            sep_scores = ((sep_scores - lir_min) / (lir_max - lir_min)) * (new_range) + probs_min
            # torch.masked_select(lir_scores, actmask_tensor)
            probs = (alpha * (probs) + (1 - alpha) * (sep_scores))
            probs = torch.nan_to_num(probs)
            probs += actmask_tensor.float()  # In order to differ from masked actions
        topk_probs, topk_idxs = torch.topk(probs, topk[hop], dim=1)  # LongTensor of [bs, k]
        topk_idxs = topk_idxs.detach().cpu().numpy()
        topk_probs = topk_probs.detach().cpu().numpy()

        new_path_pool, new_probs_pool = [], []
        for row in range(topk_idxs.shape[0]):
            path = path_pool[row]
            probs = probs_pool[row]
            for idx, p in zip(topk_idxs[row], topk_probs[row]):
                if idx >= len(acts_pool[row]):  # act idx is invalid
                    continue
                relation, next_node_id = acts_pool[row][idx]  # (relation, next_node_id)

                if relation == SELF_LOOP:
                    next_node_type = path[-1][1]
                else:
                    next_node_type = KG_RELATION[env.dataset_name][path[-1][1]][relation] # Changing according to the dataset

                new_path = path + [(relation, next_node_type, next_node_id)]
                new_path_pool.append(new_path)
                new_probs_pool.append(probs + [p])
        path_pool = new_path_pool
        probs_pool = new_probs_pool
        if hop < 2:
            state_pool = env._batch_get_state(path_pool)

    return path_pool, probs_pool


def predict_paths(policy_file, path_file, args):
    print('Predicting paths...')
    env = BatchKGEnvironment(args.dataset, args.max_acts, max_path_len=args.max_path_len,
                             state_history=args.state_history)
    if args.dataset not in AMAZON_DATASETS:
        env.item_idxs, env.user2timestamp = get_interaction2timestamp_map(args.dataset)
    else:
        env.item_idxs, env.word2timestamp, env.user2timestamp = get_interaction2timestamp_with_words(
            args.dataset)
    pretrain_sd = torch.load(policy_file)
    model = ActorCritic(env.state_dim, env.act_dim, gamma=args.gamma, hidden_sizes=args.hidden).to(args.device)
    model_sd = model.state_dict()
    model_sd.update(pretrain_sd)
    model.load_state_dict(model_sd)

    test_labels = load_labels(args.dataset, 'test')
    test_uids = list(test_labels.keys())

    batch_size = 16
    start_idx = 0
    all_paths, all_probs = [], []
    pbar = tqdm(total=len(test_uids))
    while start_idx < len(test_uids):
        end_idx = min(start_idx + batch_size, len(test_uids))
        batch_uids = test_uids[start_idx:end_idx]
        paths, probs = batch_beam_search(env, model, batch_uids, args.device, args.metric, topk=args.topk)
        all_paths.extend(paths)
        all_probs.extend(probs)
        start_idx = end_idx
        pbar.update(batch_size)
    predicts = {'paths': all_paths, 'probs': all_probs}
    pickle.dump(predicts, open(path_file, 'wb'))


def evaluate_paths(dataset_name, path_file, train_labels, test_labels, exp_property=None):
    embeds = load_embed(args.dataset)
    user_embeds = embeds[USER]
    main_entity, main_relation = MAIN_PRODUCT_INTERACTION[dataset_name]
    product = main_entity
    watched_embeds = embeds[main_relation][0]
    movie_embeds = embeds[main_entity]
    scores = np.dot(user_embeds + watched_embeds, movie_embeds.T)
    if dataset_name not in AMAZON_DATASETS:
        uid2gender, _ = get_user2gender(dataset_name)
    # 1) Get all valid paths for each user, compute path score and path probability.
    results = pickle.load(open(path_file, 'rb'))
    pred_paths = {uid: {} for uid in test_labels}
    counter_total, counter_affected = 0, 0

    if exp_property == "LIR":
        if dataset_name not in AMAZON_DATASETS:
            item_idxs, user2timestamp = get_interaction2timestamp_map(dataset_name)
            LIR_matrix = generate_LIR_matrix(dataset_name, user2timestamp, item_idxs)
        else:
            item_idxs, word_idxs, user2timestamp = get_interaction2timestamp_with_words(dataset_name)
            LIR_matrix, LIR_matrix_words = generate_LIR_matrix(dataset_name, user2timestamp, item_idxs, word_idxs)
        del item_idxs, user2timestamp
    elif exp_property == "SEP":
        SEP_matrix = generate_SEP_matrix(dataset_name)
    elif exp_property == "ETD":
        uid_type = {}

    for path, probs in zip(results['paths'], results['probs']):
        if path[-1][1] != product:
            continue
        uid = path[0][2]
        if uid not in pred_paths:
            continue
        pid = path[-1][2]
        if pid not in pred_paths[uid]:
            pred_paths[uid][pid] = []

        path_score = scores[uid][pid] #path_score = (0.85 * scores[uid][pid]) + ((1-0.85) * SEP_matrix[path[2][1]][path[2][-1]])
        if exp_property == "LIR":
            li = path[1][2]
            li_type = path[1][-1]
            if path[1][0] == SELF_LOOP:
                pred_paths[uid][pid].append((0, 0, path))
                continue
            path_prob = reduce(lambda x, y: x * y, probs)
            if li_type == MENTION:
                path_prob = LIR_matrix_words[uid][li]
            elif li_type == WATCHED:
                path_prob = LIR_matrix[uid][li]
        elif exp_property == "SEP":
            se = path[2][-1]
            se_type = path[2][1]
            path_prob = SEP_matrix[se_type][se]
        elif exp_property == "ETD":
            if uid not in uid_type:
                uid_type[uid] = set()
            if path[-1][-1] in uid_type[uid]:
                path_prob += [0.5]
            path_prob = reduce(lambda x, y: x * y, probs)
        else:
            path_prob = reduce(lambda x, y: x * y, probs)
        pred_paths[uid][pid].append((path_score, path_prob, path))

    print(counter_affected, counter_total)
    if not os.path.isdir("../../paths/"):
        os.makedirs("../../paths/")

    extracted_path_dir = "../../paths/" + args.dataset
    if not os.path.isdir(extracted_path_dir):
        os.makedirs(extracted_path_dir)
    extracted_path_dir = "../../paths/" + args.dataset + "/agent_topk=" + '-'.join([str(x) for x in args.topk])
    if not os.path.isdir(extracted_path_dir):
        os.makedirs(extracted_path_dir)
    save_pred_paths(extracted_path_dir, pred_paths, train_labels)

    # 2) Pick best path for each user-product pair, also remove pid if it is in train set.
    best_pred_paths = {}
    for uid in pred_paths:
        if uid in train_labels:
            train_pids = set(train_labels[uid])
        else:
            print("Invalid train_pids")
        best_pred_paths[uid] = []
        for pid in pred_paths[uid]:
            if pid in train_pids:
                continue

            # Get the path with highest probability
            sorted_path = sorted(pred_paths[uid][pid], key=lambda x: x[1], reverse=True)
            best_pred_paths[uid].append(sorted_path[0])

    #save_best_pred_paths(extracted_path_dir, best_pred_paths)

    # 3) Compute top 10 recommended products for each user.
    sort_by = 'score'
    pred_labels = {}
    pred_paths_top10 = {}

    pred_paths_pattern_names = {}
    pred_pid_interaction_path_pattern = {}
    entity_rate_among_total = {}
    n_of_ptype = {}
    n_of_ptype_before = {}

    for uid in best_pred_paths:
        if sort_by == 'score':
            sorted_path = sorted(best_pred_paths[uid], key=lambda x: (x[0], x[1]), reverse=True)
        elif sort_by == 'prob':
            sorted_path = sorted(best_pred_paths[uid], key=lambda x: (x[1], x[0]), reverse=True)
        top10_pids = [p[-1][2] for _, _, p in sorted_path[:10]]  # from largest to smallest
        top10_paths = [p for _, _, p in sorted_path[:10]] #paths for the top10

        top10_path_pattern_names = [p[-1][0] for _, _, p in sorted_path[:10]] #Diversity
        top10_path_interaction_pid = [p[1][-1] for _,_, p in sorted_path[:10]] #Time relevance
        path_pattern_names = [p[-1][0] for _, _, p in sorted_path[:10]] #Diversity
        # add up to 10 pids if not enough
        if args.add_products and len(top10_pids) < 10:
            train_pids = set(train_labels[uid])
            cand_pids = np.argsort(scores[uid])
            for cand_pid in cand_pids[::-1]:
                if cand_pid in train_pids or cand_pid in top10_pids:
                    continue
                top10_pids.append(cand_pid)
                if len(top10_pids) >= 10:
                    break
        # end of add
        pred_labels[uid] = top10_pids[::-1]  # change order to from smallest to largest!
        pred_paths_top10[uid] = top10_paths[::-1]

        pred_paths_pattern_names[uid] = top10_path_pattern_names[::-1]
        pred_pid_interaction_path_pattern[uid] = top10_path_interaction_pid[::-1]
        for ptype in pred_paths_pattern_names[uid]:
            if ptype not in n_of_ptype:
                n_of_ptype[ptype] = 0
            n_of_ptype[ptype] += 1
        for ptype in path_pattern_names:
            if ptype not in n_of_ptype_before:
                n_of_ptype_before[ptype] = 0
            n_of_ptype_before[ptype] += 1


    #Save pred_labels and pred_explaination for assesment and reranking
    save_pred_labels(extracted_path_dir, pred_labels)
    save_pred_explainations(extracted_path_dir, pred_paths_top10, pred_labels)
    '''
    #Get informations about diversity of paths
    new_diversity_score_male = []
    new_diversity_score_female = []
    new_diversity_score_general = []
    max_different_path_for_topk_male = 0
    max_different_path_for_topk_female = 0
    for uid in test_labels:
        gender = uid2gender[uid]
        unique_pred_paths_pattern_names = set(pred_paths_pattern_names[uid])
        value = len(unique_pred_paths_pattern_names)
        if gender == 0:
            new_diversity_score_male.append(value)
            max_different_path_for_topk_male = max(value, max_different_path_for_topk_male)
        else:
            new_diversity_score_female.append(value)
            max_different_path_for_topk_female = max(value, max_different_path_for_topk_female)
        new_diversity_score_general.append(value)

    avg_diversity = np.mean(new_diversity_score_general)
    avg_male_diversity = np.mean(new_diversity_score_male)
    avg_female_diversity = np.mean(new_diversity_score_female)
    print("\n--- Explainability diversity ---")
    print("Average number of different explaination types in top10 for all {:.3f}".format(avg_diversity))
    print("Average number of different explaination types in top10 for male {:.3f}".format(avg_male_diversity))
    print("Average number of different explaination types in top10 for female {:.3f}".format(avg_female_diversity))
    print("\nExplaination produced in top10:")
    total = 0
    cf = 0
    cb = 0
    for ptype in n_of_ptype.keys():
        if ptype == 'watched':
            cf += n_of_ptype[ptype]
        else:
            cb += n_of_ptype[ptype]
        total += n_of_ptype[ptype]
        print("{}: {}".format(ptype, n_of_ptype[ptype]))
    print(
        "\nPercentage of content based explainations: {:.3f}%, Percentage of collaborative filtering explainations: {:.3f}%".format(
            cb / total * 100, cf / total * 100))

    print("\nExplaination produced by the agent:")
    total = 0
    cf = 0
    cb = 0
    for ptype in n_of_ptype_before.keys():
        if ptype == 'watched':
            cf += n_of_ptype_before[ptype]
        else:
            cb += n_of_ptype_before[ptype]
        total += n_of_ptype_before[ptype]
        print("{}: {}".format(ptype, n_of_ptype_before[ptype]))
    print(
        "\nPercentage of content based explainations: {:.3f}%, Percentage of collaborative filtering explainations: {:.3f}%".format(
            cb / total * 100, cf / total * 100))
    print("Max different path types for male: {}, female: {}\n".format(max_different_path_for_topk_male,
                                                                       max_different_path_for_topk_female))

    #Get informations about time relevance
    interaction2timestamp, user2timestamp = get_interaction2timestamp(dataset_name)
    #save_user_interactions(extracted_path_dir, user2timestamp)
    #save_interactions_timestamps(extracted_path_dir, interaction2timestamp)

    with open("./evaluation/" + args.dataset + "/PGPR/user2timestamp.csv", 'w+', newline='') as user2timestamp_file:
        header = ["uid", "timestamps"]
        writer = csv.writer(user2timestamp_file)
        writer.writerow(header)
        for uid, timestamps in user2timestamp.items():
            writer.writerow([uid, ' '.join([str(timestamp) for timestamp in timestamps])])
    user2timestamp_file.close()

    with open("./evaluation/" + args.dataset + "/PGPR/uid_pid_timestamp.csv", 'w+', newline='') as uid_pid_timestamp_file:
        header = ["uid", "pid", "interaction_timestamp"]
        writer = csv.writer(uid_pid_timestamp_file)
        writer.writerow(header)
        for uid, pid_list in interaction2timestamp.items():
            for pid, timestamp in pid_list.items():
                writer.writerow([uid, pid, str(timestamp)])
    uid_pid_timestamp_file.close()
    return
topk_time_relevance = []
    topk_time_relevance_male = []
    topk_time_relevance_female = []
    for uid in test_labels:
        time_relevance_topk = []
        gender = uid2gender[uid]
        user2timestamp[uid].sort()
        def normalized_ema(values):
            values = np.array([i for i in range(len(values))])
            values = pd.Series(values)
            ema_vals = values.ewm(span=10).mean().tolist()
            min_res = min(ema_vals)
            max_res = max(ema_vals)
            return [(x - min_res) / (max_res - min_res) for x in ema_vals]
        ema_timestamps = normalized_ema(user2timestamp[uid])
        for pid in pred_pid_interaction_path_pattern[uid]:
            interaction_timestamp = interaction2timestamp[uid][pid]
            timestamp_idx = user2timestamp[uid].index(interaction_timestamp)
            time_relevance = ema_timestamps[timestamp_idx]
            time_relevance_topk.append(time_relevance)

        avg_user_time_relevance_topk = np.array(time_relevance_topk).mean()
        if gender == 0:  # Male
            topk_time_relevance_male.append(avg_user_time_relevance_topk)
        else:  # Female
            topk_time_relevance_female.append(avg_user_time_relevance_topk)
        topk_time_relevance.append(avg_user_time_relevance_topk)

    avg_time_relevance_male = np.array(topk_time_relevance_male).mean()
    avg_time_relevance_female = np.array(topk_time_relevance_female).mean()
    avg_time_relevance = np.array(topk_time_relevance).mean()

    print("--- Time relevance ---")
    print("Average time relevance general: {:.3f}".format(avg_time_relevance))
    print("Average time relevance for male: {:.3f}".format(avg_time_relevance_male))
    print("Average time relevance for female: {:.3f}\n".format(avg_time_relevance_female))

    avg_entity_populary_male = []
    avg_entity_populary_female = []
    avg_entity_populary = []
    for uid in test_labels:
        gender = uid2gender[uid]
        for entity_popularity in entity_rate_among_total[uid]:
            avg_user = np.array(entity_popularity).mean()
        if gender == 0:  # Male
            avg_entity_populary_male.append(avg_user)
        else:  # Female
            avg_entity_populary_female.append(avg_user)
        avg_entity_populary.append(avg_user)

    avg_entity_populary_male = np.array(avg_entity_populary_male).mean()
    avg_entity_populary_female = np.array(avg_entity_populary_female).mean()
    avg_entity_populary = np.array(avg_entity_populary).mean()
    print("--- Entity that brought to rec popularity ---")
    print("Rate of items that brought to recommandation among all the item in their category(item popularity)")
    print("Item populary male: {:.3f}".format(avg_entity_populary_male))
    print("Item populary female: {:.3f}".format(avg_entity_populary_female))
    print("Item populary general: {:.3f}\n".format(avg_entity_populary))
    #4) Compute group explaination diversity unfairness (GEDU) TO MOVE
    f_male = []
    f_female = []
    personalization_score_male = []  # S_p(Qi)
    personalization_score_female = []  # S_p(Qi)
    diversity_score_male = []
    diversity_score_female = []
    n_paths_male = []
    n_paths_female = []
    for uid in test_labels:
        gender = uid2gender[uid]
        #Calculate for every user the personalization score and the diversity to obtain a f for every user
        personalization_score_user = 0.
        diversity_score_user = 0.
        nPaths = 0
        nItem = 0
        for rank, pid in enumerate(pred_labels[uid]):
            uv_path_pattern_name = pred_paths_pattern_names[uid][rank]
            w_pi = get_path_pattern_weigth(uv_path_pattern_name, pred_paths[uid][pid])
            lpath = sum(float(pred_path[0]) if get_path_pattern(pred_path) == uv_path_pattern_name else 0. for pred_path in
                        pred_paths[uid][pid])  # score of best paths

            user_item_path_distribution = get_user_item_path_distribution(pred_paths[uid], uv_path_pattern_name)
            s_p = (w_pi / user_item_path_distribution) * lpath
            personalization_score_user += s_p
            simpson_index = simpson_index_of_diversity(pid, pred_paths[uid])
            diversity_score_user += simpson_index
            nPaths += len(pred_paths[uid][pid])
            nItem += 1

        n_paths_male.append(nPaths/nItem) if gender == 0 else n_paths_female.append(nPaths/nItem)
        alpha = 0.75
        lambd = 10

        personalization_score_male.append(personalization_score_user) if gender == 0 else personalization_score_female.append(personalization_score_user)
        diversity_score_male.append(diversity_score_user) if gender == 0 else diversity_score_female.append(diversity_score_user)

        if gender == 0:
            f_male.append(alpha * personalization_score_user + (1 - alpha) * lambd * diversity_score_user)
        else:
            f_female.append(alpha * personalization_score_user + (1 - alpha) * lambd * diversity_score_user)

    avg_personalization_score_male = np.mean(personalization_score_male)
    avg_personalization_score_female = np.mean(personalization_score_female)
    avg_diversity_score_male = np.mean(diversity_score_male)
    avg_diversity_score_female = np.mean(diversity_score_female)
    avg_f_male = np.mean(f_male)
    avg_f_female = np.mean(f_female)
    gedu = abs(avg_f_male - avg_f_female)

    print(n_paths_male)
    print("Average n of explaination paths per item in top 10 for male group={:.3f}".format(np.mean(n_paths_male)))
    print(n_paths_female)
    print("Average n of explaination paths per item in top 10 for female group={:.3f}".format(np.mean(n_paths_female)))
    print("\nAverage personalization score male={:.3f}    | Average personalization score female={:.3f}".format(
        avg_personalization_score_male, avg_personalization_score_female))
    print("Average diversity score male={:.3f}  | Average diversity score female={:.3f}".format(
        avg_diversity_score_male,avg_diversity_score_female))
    print("Group Explanation Diversity Unfairness (GEDU)={:.3f}".format(gedu))
    '''
    evaluate(dataset_name, pred_labels, test_labels)


def get_user_item_path_distribution(pred_paths_per_user, path_pattern_name):  # redo
    n_item_with_same_path_pattern = 0
    total_item = 0
    for path_to_item in pred_paths_per_user.items():
        for path in path_to_item[1]:
            total_item += 1
            if path_pattern_name == get_path_pattern(path):
                n_item_with_same_path_pattern += 1
    return n_item_with_same_path_pattern / total_item


# Simpson index of diversity range 0-1 with 1 max diversity and 0 no diversity at all
def simpson_index_of_diversity(pid, pred_uv_paths):
    n_path_for_patterns = {}
    N = 0
    for path in pred_uv_paths[pid]:
        path_patter_value = get_path_pattern(path)
        if path_patter_value not in n_path_for_patterns:
            n_path_for_patterns[path_patter_value] = 0
        n_path_for_patterns[path_patter_value] += 1
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


# In formula w of pi log(2 + (number of patterns of same pattern type among uv paths / total number of paths among uv paths))
def get_path_pattern_weigth(path_pattern_name, pred_uv_paths):
    n_same_path_pattern = 0
    total_paths = len(pred_uv_paths)
    for path in pred_uv_paths:
        if path_pattern_name == get_path_pattern(path):
            n_same_path_pattern += 1
    return log(2 + (n_same_path_pattern / total_paths))


def test(args):
    policy_file = args.log_dir + '/policy_model_epoch_{}.ckpt'.format(args.epochs)
    path_file = args.log_dir + '/policy_paths_epoch{}.pkl'.format(args.epochs)

    train_labels = load_labels(args.dataset, 'train')
    test_labels = load_labels(args.dataset, 'test')
    kg = load_kg(args.dataset)
    if args.run_path:
        predict_paths(policy_file, path_file, args)
    if args.run_eval:
        evaluate_paths(args.dataset, path_file, train_labels, test_labels, args.metric)


if __name__ == '__main__':
    boolean = lambda x: (str(x).lower() == 'true')
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=ML1M, help='One of {cloth, beauty, cell, cd}')
    parser.add_argument('--name', type=str, default='train_agent', help='directory name.')
    parser.add_argument('--seed', type=int, default=123, help='random seed.')
    parser.add_argument('--gpu', type=str, default='0', help='gpu device.')
    parser.add_argument('--epochs', type=int, default=50, help='num of epochs.')
    parser.add_argument('--max_acts', type=int, default=250, help='Max number of actions.')
    parser.add_argument('--max_path_len', type=int, default=3, help='Max path length.')
    parser.add_argument('--gamma', type=float, default=0.99, help='reward discount factor.')
    parser.add_argument('--state_history', type=int, default=1, help='state history length')
    parser.add_argument('--hidden', type=int, nargs='*', default=[512, 256], help='number of samples')
    parser.add_argument('--add_products', type=boolean, default=False, help='Add predicted products up to 10')
    parser.add_argument('--topk', type=list, nargs='*', default=[20,50,1], help='number of samples')
    parser.add_argument('--metric', type=str, default="ETD", help="An explanation metric among (LIR, SEP, ETD) to optimize in-train")
    parser.add_argument('--alpha', type=str, default="None", help="How much importance to give to explanation property between 0 and 1")
    parser.add_argument('--run_path', type=boolean, default=False, help='Generate predicted path? (takes long time)')
    parser.add_argument('--run_eval', type=boolean, default=True, help='Run evaluation?')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'

    args.log_dir = TMP_DIR[args.dataset] + '/' + args.name
    test(args)

