import argparse
import sys

from path_data_loader import *
from myutils import *

def save_pred_explanations(chosen_optimization, folder_path, pred_paths_top10):
    print("Saving topks' explanations")
    # Save explanations to load the uid-pid selected explanation
    with open(folder_path + "/" + chosen_optimization + "_uid_pid_explanation.csv", 'w+', newline='') as uid_pid_explanation:
          header = ["uid", "pid", "path"]
          writer = csv.writer(uid_pid_explanation)
          writer.writerow(header)
          for uid, paths in pred_paths_top10.items():
              for pid, path in paths.items():
                  path_explanation = []
                  for tuple in path:
                      for x in tuple:
                          path_explanation.append(str(x))
                  writer.writerow([uid, path_explanation[-1], ' '.join(path_explanation)])
    uid_pid_explanation.close()

def eval_baseline(path_data):
    # Rec Quality
    rec_metrics_before = measure_rec_quality(path_data)
    print_rec_metrics(path_data.dataset_name, path_data.sens_attribute_flags, rec_metrics_before)
    # Exp Quality
    exp_metrics_before = {}
    distributions_exp_metrics_before = {}
    # Save average of values in topk for each metric
    lir_before = avg_LIR(path_data)
    sep_before = avg_SEP(path_data)
    etd_before = avg_ETD(path_data)
    lid_before = avg_LID(path_data)
    sed_before = avg_SED(path_data)
    etv_before = avg_ETV(path_data)

    exp_metrics_before["LIR"] = dict(lir_before.avg_groups_LIR)
    exp_metrics_before["SEP"] = dict(sep_before.avg_groups_SEP)
    exp_metrics_before["ETD"] = dict(etd_before.avg_groups_ETD)
    exp_metrics_before["LID"] = dict(lid_before.avg_groups_LID)
    exp_metrics_before["SED"] = dict(sed_before.avg_groups_SED)
    exp_metrics_before["ETV"] = dict(etv_before.avg_groups_ETV)

    # Save distributions of values in topk for each metric
    distributions_exp_metrics_before["LIR"] = dict(lir_before.groups_LIR_scores)
    distributions_exp_metrics_before["SEP"] = dict(sep_before.groups_SEP_scores)
    distributions_exp_metrics_before["ETD"] = dict(etd_before.groups_ETD_scores)
    distributions_exp_metrics_before["LID"] = dict(lid_before.groups_LID_scores)
    distributions_exp_metrics_before["SED"] = dict(sed_before.groups_SED_scores)
    distributions_exp_metrics_before["ETV"] = dict(etv_before.groups_ETV_scores)
    print_expquality_metrics(path_data.dataset_name, path_data.sens_attribute_flags, exp_metrics_before)

def performe_soft_optimization(chosen_optimization, path_data):
    if chosen_optimization == "softETD":
        soft_optimization_ETD(path_data)
    elif chosen_optimization == "softSEP":
        soft_optimization_SEP(path_data)
    elif chosen_optimization == "softLIR":
        soft_optimization_LIR(path_data)

    #Calculate exp quality metrics after optimization
    LIR_after = avg_LIR(path_data)
    SEP_after = avg_SEP(path_data)
    ETD_after = avg_ETD(path_data)
    rec_metrics_after = measure_rec_quality(path_data)
    print_rec_metrics(path_data.dataset_name, path_data.sens_attribute_flags, rec_metrics_after)

    avg_exp_metrics_after = {}
    distributions_exp_metrics_after = {}

    # Save average of values in topk for each metric
    avg_exp_metrics_after["LIR"] = dict(LIR_after.avg_groups_LIR)
    avg_exp_metrics_after["SEP"] = dict(SEP_after.avg_groups_SEP)
    avg_exp_metrics_after["ETD"] = dict(ETD_after.avg_groups_ETD)

    # Save distributions of values in topk for each metric
    distributions_exp_metrics_after["LIR"] = dict(LIR_after.groups_LIR_scores)
    distributions_exp_metrics_after["SEP"] = dict(SEP_after.groups_SEP_scores)
    distributions_exp_metrics_after["ETD"] = dict(ETD_after.groups_ETD_scores)
    print_expquality_metrics(path_data.dataset_name, path_data.sens_attribute_flags, avg_exp_metrics_after["LIR"],
                             avg_exp_metrics_after["SEP"],
                             avg_exp_metrics_after["ETD"])

def performe_weighted_optimization(alpha, chosen_optimization, path_data):
    # Apply the chosen optimization for chosen value of alpha
    print("--- Performing {} with alpha={}---".format(chosen_optimization, alpha))
    if chosen_optimization == "weightedETD":
        path_data = weighted_opt_ETD(path_data, alpha)
    elif chosen_optimization == "weightedSEP":
        path_data = weighted_opt_SEP(path_data, alpha)
    elif chosen_optimization == "weightedLIR":
        path_data = weighted_opt_LIR(path_data, alpha)
    elif chosen_optimization == "weightedSEP_ETD":
        path_data = weighted_opt_ETD_SEP(path_data, alpha)
    elif chosen_optimization == "weightedLIR_ETD":
        path_data = weighted_opt_ETD_LIR(path_data, alpha)
    elif chosen_optimization == "weightedLIR_SEP":
        path_data = weighted_opt_LIR_SEP(path_data, alpha)
    elif chosen_optimization == "weightedLIR_SEP_ETD":
        path_data = weighted_opt_ETD_SEP_LIR(path_data, alpha)
    else:
        print("The chosen optimization doesn't exist, ensure that the opt parameter is in: [weightedETD, weightedLIR, "
              "weightedSEP, weightedLIR_ETD, weigthedSEP_ETD, weightedLIR_SEP, weightedLIR_SEP_ETD]")
        exit(-1)
    rec_metrics_after = measure_rec_quality(path_data)
    print_rec_metrics(path_data.dataset_name, path_data.sens_attribute_flags, rec_metrics_after)

    exp_metrics_after = {}
    distributions_exp_metrics_after = {}

    # Save average of values in topk for each metric
    lir_after = avg_LIR(path_data)
    sep_after = avg_SEP(path_data)
    etd_after = avg_ETD(path_data)
    lid_after = avg_LID(path_data)
    sed_after = avg_SED(path_data)
    etv_after = avg_ETV(path_data)

    # Save average of values in topk for each metric
    exp_metrics_after["LIR"] = dict(lir_after.avg_groups_LIR)
    exp_metrics_after["SEP"] = dict(sep_after.avg_groups_SEP)
    exp_metrics_after["ETD"] = dict(etd_after.avg_groups_ETD)
    exp_metrics_after['LID'] = dict(lid_after.avg_groups_LID)
    exp_metrics_after['SED'] = dict(sed_after.avg_groups_SED)
    exp_metrics_after['ETV'] = dict(etv_after.avg_groups_ETV)

    # Save distributions of values in topk for each metric
    distributions_exp_metrics_after["LIR"] = dict(lir_after.groups_LIR_scores)
    distributions_exp_metrics_after["SEP"] = dict(sep_after.groups_SEP_scores)
    distributions_exp_metrics_after["ETD"] = dict(etd_after.groups_ETD_scores)
    print_expquality_metrics(path_data.dataset_name, path_data.sens_attribute_flags, exp_metrics_after)
    return path_data
if __name__ == '__main__':
    boolean = lambda x: (str(x).lower() == 'true')
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="ml1m", help='One of {ml1m, lastfm}')
    parser.add_argument('--agent_topk', type=str, default="25-50-1", help='One of {25-50-1, 10-12-1} or more if you compute the predpaths with PGPR')
    parser.add_argument('--opt', type=str, default="weigthed", help='weighted or soft')
    parser.add_argument('--alpha', type=float, default=-1, help="(Only for weighted opt) Determine the weigth of the optimized "
                                                                "explanation metric/s in reranking, -1 means test all alpha from 0. to 1. at step of 0.05")
    parser.add_argument('--metrics', type=list, default=["ETD"], help='One of {LIR, SEP, ETD, [ETD, LIR], [ETD, SEP], [SEP, LIR], [SEP, LIR, ETD]}')
    parser.add_argument('--eval_baseline', type=bool, default=True, help="Calculate stats for baseline")
    parser.add_argument('--log_enabled', type=bool, default=False, help='If true save log files instead of printing results')
    parser.add_argument('--show_gender_stats', type=bool, default=False, help="(Only LASTFM, ML1M) Show metrics obtained grouping for gender")
    parser.add_argument('--show_age_stats', type=bool, default=False, help="(Only LASTFM, ML1M) Show metrics obtained grouping for gender")
    parser.add_argument('--show_occupation_stats', type=bool, default=False, help="(Only ML1M) Show metrics obtained grouping for gender")
    parser.add_argument('--show_country_stats', type=bool, default=False, help='(Only LASTFM) Show metrics obtained grouping for country')
    parser.add_argument('--save_explanations', type=bool, default=False, help='Save post processed explanations and top-k')
    args = parser.parse_args()

    sys.path.append(r'models/PGPR')
    metric_order = {"LIR": 0, "SEP": 1, "ETD": 2} #Ordering label to allow input to be in random order
    metrics = args.metrics[0] if len(args.metrics) == 1 else '_'.join(args.metrics.sort(key=lambda x: metric_order[x]))
    chosen_optimization = args.opt + metrics

    #Best alphas

    #Paths and ensure folders
    result_base_path = ensure_result_folder(args)
    log_base_path = ensure_log_folder(args)
    #Load paths
    path_data = PathDataLoader(args)

    #Evaluate folders
    if args.eval_baseline:
        # Enable logging to file
        if args.log_enabled:
            log_path = log_base_path + "/" + chosen_optimization + ".txt"
            log_file = open(log_path, "w+")
            sys.stdout = log_file

        print("--- Baseline---")
        eval_baseline(path_data)
    #Performe soft optimization
    if args.opt == "soft":
        print("Performing Soft-Optimization...")
        performe_soft_optimization(chosen_optimization)

        #Save uid-pid-path recommandations if requested
        if args.save_explanations:
            path_base_path = ensure_path_folder(args)
            save_pred_explanations(path_base_path, path_data.uid_pid_explanation)

        if args.log_enabled:
            log_file.close()

    #Performe weighted optimization
    if args.opt == "weighted":
        alpha_optimizations = ["weightedETD", "weightedSEP", "weightedLIR", "weightedSEP_ETD",
                               "weightedLIR_ETD", "weightedLIR_SEP", "weightedLIR_SEP_ETD"]
        alpha = args.alpha
        if alpha < 0. or alpha > 1:
            print("Invalid alpha. Alpha values must be in range 0-1")
            exit(-1)
        if chosen_optimization not in alpha_optimizations:
            print("Invalid weighted optimization. The chosen optimization doesn't exist.")
            exit(-1)

        path_base_path = ensure_path_folder(args)
        metric = "ETD"
        chosen_optimization = "weighted" + metric
        save_pred_explanations("IN-" + metric, path_base_path, path_data.uid_pid_explanation)
        #path_data = performe_weighted_optimization(OVERALL_BETTER_ALPHA[args.dataset]["LIR"],  "weightedLIR", path_data)
        #path_data = performe_weighted_optimization(OVERALL_BETTER_ALPHA[args.dataset]["SEP"], "weightedSEP", path_data)
        #path_data = performe_weighted_optimization(0.35, "weightedETD", path_data)
        path_data = performe_weighted_optimization(OVERALL_BETTER_ALPHA[args.dataset]["ETD"], "weightedETD", path_data)
        save_pred_explanations("IN-" + chosen_optimization, path_base_path, path_data.uid_pid_explanation)
        #for metric, alpha in OVERALL_BETTER_ALPHA[args.dataset].items():
        #    chosen_optimization = "weighted" + metric
        #    path_data = performe_weighted_optimization(alpha, chosen_optimization, path_data)
        #Save uid-pid-path recommandations if requested
        if args.save_explanations:
            path_base_path = ensure_path_folder(args)
            save_pred_explanations("IN" + chosen_optimization, path_base_path, path_data.uid_pid_explanation)
        if args.log_enabled:
            log_file.close()
