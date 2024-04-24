import argparse

import numpy as np
import torch
import torch.optim as optim
from pathlib import Path
from models import Model
from data_loader import load_data, load_out_t, load_out_emb_t
from utils import (
    get_logger,
    get_evaluator,
    set_seed,
    get_training_config,
    check_writable,
    check_readable,
    compute_min_cut_loss,
    graph_split,
    get_pos_neg_pair
)
from train_and_eval import distill_run_transductive
import networkx as nx
import dgl
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

def get_args():
    parser = argparse.ArgumentParser(description="PyTorch DGL implementation")
    parser.add_argument("--device", type=int, default=3, help="CUDA device, -1 means CPU")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--log_level", type=int, default=20, help="Logger levels for run {10: DEBUG, 20: INFO, 30: WARNING}")
    parser.add_argument("--console_log",action="store_true",help="Set to True to display log info in console")
    parser.add_argument("--output_path", type=str, default="outputs", help="Path to save outputs")
    parser.add_argument("--num_exp", type=int, default=5, help="Repeat how many experiments")
    parser.add_argument("--exp_setting",type=str,default="tran",help="Experiment setting, one of [tran, ind]",)
    parser.add_argument("--eval_interval", type=int, default=1, help="Evaluate once per how many epochs")
    parser.add_argument("--save_results",action="store_true",help="Set to True to save the loss curves, trained model, and min-cut loss for the transductive setting")
    """Dataset"""
    parser.add_argument("--dataset", type=str, default="citeseer", help="Dataset")
    parser.add_argument("--data_path", type=str, default="./data", help="Path to data")
    parser.add_argument("--labelrate_train",type=int,default=20,help="How many labeled data per class as train set")
    parser.add_argument("--labelrate_val",type=int, default=30,help="How many labeled data per class in valid set")
    parser.add_argument("--split_idx",type=int,default=0,help="For Non-Homo datasets only, one of [0,1,2,3,4]")

    """Model"""
    parser.add_argument("--model_config_path",type=str,default=".conf.yaml",help="Path to model configeration")
    parser.add_argument("--teacher", type=str, default="GCN", help="Teacher model")
    parser.add_argument("--student", type=str, default="DMLP", help="Student model")
    parser.add_argument("--model_name", type=str, default="DMLP", help="model [DMLP]")
    parser.add_argument("--num_layers", type=int, default=2, help="Student model number of layers")
    parser.add_argument("--Kp", type=int, default=10, help="the numbers of positive")
    parser.add_argument("--Kn", type=int, default=10, help="the numbers of negative")
    parser.add_argument("--hidden_dim",type=int,default=1024,help="Student model hidden layer dimensions")
    parser.add_argument("--dropout_ratio_h", type=float, default=0.0)
    parser.add_argument("--dropout_ratio_a", type=float, default=0.0)
    parser.add_argument("--norm_type", type=str, default="none", help="One of [none, batch, layer]")

    parser.add_argument("--tau", type=float, default=1.0)
    """Optimization"""
    parser.add_argument("--learning_rate", type=float, default=0.005)
    parser.add_argument("--weight_decay", type=float, default=0.005)
    parser.add_argument("--max_epoch", type=int, default=500, help="Evaluate once per how many epochs")
    parser.add_argument("--patience",type=int,default=50,help="Early stop is the score on validation set does not improve for how many epochs")

    """Ablation"""

    parser.add_argument("--split_rate",type=float,default=0.2,help="Rate for graph split, see comment of graph_split for more details",)
    parser.add_argument("--compute_min_cut",action="store_true",help="Set to True to compute and store the min-cut loss",)

    """Distiall"""
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.8,
                        help="Parameter balances loss from hard labels and teacher outputs, take values in [0, 1]", )
    parser.add_argument("--gama", type=float, default=0.2)
    parser.add_argument( "--out_t_path", type=str, default="outputs", help="Path to load teacher outputs")

    # add-up
    parser.add_argument( "--feat_distill",action="store_true",help="Set to True to include feature distillation loss",)

    """parameter sensitivity"""

    args = parser.parse_args()
    return args


global_trans_dw_feature = None

def run(args, repeat):
    """
    Returns:
    score_lst: a list of evaluation results on test set.
    len(score_lst) = 1 for the transductive setting.
    len(score_lst) = 2 for the inductive/production setting.
    """

    """ Set seed, device, and logger """
    set_seed(args.seed)
    if torch.cuda.is_available() and args.device >= 0:
        device = torch.device("cuda:" + str(args.device))
    else:
        device = "cpu"

    if args.exp_setting == "tran":
        output_dir = Path.cwd().joinpath(
            args.output_path,
            "transductive",
            args.dataset,
            f"{args.teacher}_{args.student}",
            f"seed_{args.seed}",
        )
        out_t_dir = Path.cwd().joinpath(
            args.out_t_path,
            "transductive",
            args.dataset,
            args.teacher,
            f"seed_{args.seed}",
        )
    else:
        raise ValueError(f"Unknown experiment setting! {args.exp_setting}")
    args.output_dir = output_dir

    check_writable(output_dir, overwrite=False)
    check_readable(out_t_dir)

    logger = get_logger(output_dir.joinpath("log"), args.console_log, args.log_level)
    logger.info(f"output_dir: {output_dir}")
    logger.info(f"out_t_dir: {out_t_dir}")

    """ Load data and model config"""
    dense_adj_tensor, g, labels, idx_train, idx_val, idx_test = load_data(
        args.dataset,
        args,
        repeat
    )

    logger.info(f"Total {g.number_of_nodes()} nodes.")
    logger.info(f"Total {g.number_of_edges()} edges.")

    g = g.to(device)
    feats = g.ndata["feat"]
    feats = feats
    dense_adj_tensor = dense_adj_tensor
    #labels_one_hot = labels_one_hot
    args.feat_dim = g.ndata["feat"].shape[1]
    args.num_nodes = g.ndata["feat"].shape[0]
    args.label_dim = labels.int().max().item() + 1


    """ Model config """
    conf = {}
    #if args.model_config_path is not None:
    #    conf = get_training_config(
            # args.model_config_path, args.student, args.dataset
    #        args.exp_setting + args.model_config_path, args.student, args.dataset
   #     )  # Note: student config
    conf = dict(args.__dict__, **conf)
    conf["device"] = device
    logger.info(f"conf: {conf}")
    # print('conf: ', conf)

    #if args.exp_setting == "tran":
    idx_l = idx_train
    idx_t = torch.cat([idx_train, idx_val, idx_test])
    distill_indices = (idx_l, idx_t, idx_val, idx_test)


    """ Model init """
    model = Model(conf)

    optimizer = optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    criterion_l = torch.nn.NLLLoss()
    criterion_t = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
    evaluator = get_evaluator(conf["dataset"])

    """Load teacher model output"""
    out_t = load_out_t(out_t_dir)
    out_emb_t = load_out_emb_t(out_t_dir)
    out_emb_t = out_emb_t
    logger.info(
        f"teacher score on train data: {evaluator(out_t[idx_train], labels[idx_train])}"
    )
    logger.info(
        f"teacher score on val data: {evaluator(out_t[idx_val], labels[idx_val])}"
    )
    logger.info(
        f"teacher score on test data: {evaluator(out_t[idx_test], labels[idx_test])}"
    )

    """Data split and run"""
    loss_and_score = []
    pos_emb, Sim_neg = get_pos_neg_pair(dense_adj_tensor, feats, out_t, out_emb_t, args, device)
    out, score_val, score_test = distill_run_transductive(
            conf,
            model,
            dense_adj_tensor,
            feats,
            labels,
            out_t, #soft_label
            out_emb_t,#embbedings of the last layer
            distill_indices,
            criterion_l,
            criterion_t,
            evaluator,
            optimizer,
            logger,
            loss_and_score,
            g,
            pos_emb,
            Sim_neg,
            args
        )
    score_lst = [score_test]


    logger.info(
        f"num_layers: {conf['num_layers']}. hidden_dim: {conf['hidden_dim']}. dropout_ratio_a: {conf['dropout_ratio_a']}. dropout_ratio_h: {conf['dropout_ratio_h']}"
    )
    logger.info(f"# params {sum(p.numel() for p in model.parameters())}")

    """ Saving student outputs """
    out_np = out.detach().cpu().numpy()
    np.savez(output_dir.joinpath("out"), out_np)

    """ Saving loss curve and model """
    if args.save_results:
        # Loss curves
        loss_and_score = np.array(loss_and_score)
        np.savez(output_dir.joinpath("loss_and_score"), loss_and_score)

        # Model
        torch.save(model.state_dict(), output_dir.joinpath("model.pth"))

    """ Saving min-cut loss"""
    if args.exp_setting == "tran" and args.compute_min_cut:
        min_cut = compute_min_cut_loss(g, out)
        # with open(output_dir.parent.joinpath("min_cut_loss"), "a+") as f:
        #     f.write(f"{min_cut :.4f}\n")
        print('min_cut: ', min_cut, flush=True)

    return score_lst


def repeat_run(args):
    scores = []
    for seed in range(args.num_exp):
        args.seed = seed
        score_test = run(args,seed)
        scores.append(score_test)

    scores_np = np.array(scores)
    return scores,scores_np.mean(axis=0), scores_np.std(axis=0)


def main():
    args = get_args()
    if args.num_exp == 1:
        score = run(args)
        score_str = "".join([f"{s : .4f}\t" for s in score])
        if args.exp_setting == 'ind':
            score_prod = score[0] * 0.8 + score[1] * 0.2

    elif args.num_exp > 1:
        scores, score_mean, score_std = repeat_run(args)
        score_str = "".join(
            [f"{s : .4f}\t" for s in score_mean] + [f"{s : .4f}\t" for s in score_std]
        )

    with open(args.output_dir.parent.joinpath("exp_results"), "a+") as f:
        f.write(f"{score_str}\n")

    # for collecting aggregated results
    print(score_str, flush=True)
    if args.exp_setting == 'ind':
        print('prod: ', score_prod)


if __name__ == "__main__":
    args = get_args()
    main()
