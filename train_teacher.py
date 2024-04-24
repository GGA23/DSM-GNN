import argparse
import numpy as np
import torch
import torch.optim as optim
from pathlib import Path
from models import Model
from data_loader import load_data
from utils import (
    get_logger,
    get_evaluator,
    set_seed,
    get_training_config,
    check_writable,
    compute_min_cut_loss,
    graph_split,
)
from train_and_eval import run_transductive
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

def get_args():
    parser = argparse.ArgumentParser(description="PyTorch DGL implementation")
    parser.add_argument("--device", type=int, default=0, help="CUDA device, -1 means CPU")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--log_level",type=int,default=20,help="Logger levels for run {10: DEBUG, 20: INFO, 30: WARNING}")
    parser.add_argument("--console_log",action="store_true",help="Set to True to display log info in console")
    parser.add_argument("--output_path", type=str, default="outputs", help="Path to save outputs")
    parser.add_argument("--num_exp", type=int, default=5, help="Repeat how many experiments")
    parser.add_argument("--exp_setting",type=str,default="tran",help="Experiment setting, one of [tran, ind]")
    parser.add_argument("--eval_interval", type=int, default=1, help="Evaluate once per how many epochs")
    parser.add_argument("--save_results",action="store_true",help="Set to True to save the loss curves, trained model, and min-cut loss for the transductive setting",)

    """Dataset"""
    parser.add_argument("--dataset", type=str, default="cora", help="Dataset")
    #parser.add_argument("--data_path", type=str, default="./data", help="Path to data")
    parser.add_argument("--train_ratio", type=int, default=0.6,
                        help="How many labeled data per class as train set", )
    parser.add_argument("--labelrate_train",type=int,default=20,help="How many labeled data per class as train set",)
    parser.add_argument("--labelrate_val",type=int,default=30,help="How many labeled data per class in valid set",)
    parser.add_argument("--split_idx",type=int,default=0,help="For Non-Homo datasets only, one of [0,1,2,3,4]",)
    parser.add_argument('--train_rate', type=float, default=0.6, help='train set rate.')
    parser.add_argument('--val_rate', type=float, default=0.2, help='val set rate.')
    """Model"""
    parser.add_argument("--model_config_path",type=str,# default="./tran.conf.yaml",
                         default=".conf.yaml",help="Path to model configeration",)
    parser.add_argument("--teacher", type=str, default="MLP_A", help="Teacher model")
    parser.add_argument("--model_name", type=str, default="MLP_A", help="model [GCN, GAT, SAGE]")
    parser.add_argument("--num_layers", type=int, default=2, help="Model number of layers")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Model hidden layer dimensions")
    parser.add_argument("--dropout_ratio", type=float, default=0.6)
    parser.add_argument("--attn_dropout_ratio", type=float, default=0.3)
    parser.add_argument("--norm_type", type=str, default="none", help="One of [none, layer]")

    """Optimization"""
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=0.005)
    parser.add_argument("--max_epoch", type=int, default=500, help="Evaluate once per how many epochs")
    parser.add_argument("--patience",type=int,default=50,help="Early stop is the score on validation set does not improve for how many epochs",)

    """Ablation"""
    parser.add_argument("--split_rate",type=float,default=0.2, help="Rate for graph split, see comment of graph_split for more details",)
    parser.add_argument("--compute_min_cut",action="store_true",help="Set to True to compute and store the min-cut loss",)
    args = parser.parse_args()
    return args


def run(args,repeat=0):

    """ Set seed, device, and logger """
    # print('args.seed: ', args.seed)
    set_seed(args.seed)
    if torch.cuda.is_available() and args.device >= 0:
        device = torch.device("cuda:" + str(args.device))
    else:
        device = "cpu"



    output_dir = Path.cwd().joinpath(
            args.output_path,
            "transductive",
            args.dataset,
            args.teacher,
            f"seed_{args.seed}",
        )

    args.output_dir = output_dir

    check_writable(output_dir, overwrite=False)
    logger = get_logger(output_dir.joinpath("log"), args.console_log, args.log_level)
    logger.info(f"output_dir: {output_dir}")

    """ Load data """
    sp_adj_tensor,  g, labels, idx_train, idx_val, idx_test = load_data(
        args.dataset,
        args,
        repeat
    )
    logger.info(f"Total {g.number_of_nodes()} nodes.")
    logger.info(f"Total {g.number_of_edges()} edges.")

    feats = g.ndata["feat"]
    args.feat_dim = g.ndata["feat"].shape[1]
    args.label_dim = labels.int().max().item() + 1

    """ Model config """
    conf = {}
    #if args.model_config_path is not None:
    #    conf = get_training_config(args.exp_setting + args.model_config_path, args.teacher, args.dataset)
    conf = dict(args.__dict__, **conf)
    conf["device"] = device
    logger.info(f"conf: {conf}")
    conf['num_nodes'] = g.ndata["feat"].shape[0]
    #print(conf['model_name'])

    """ Model init """
    model = Model(conf)
    optimizer = optim.Adam(
        model.parameters(), lr=conf["learning_rate"], weight_decay=conf["weight_decay"]
    )
    criterion = torch.nn.NLLLoss()
    evaluator = get_evaluator(conf["dataset"])

    """ Data split and run """
    loss_and_score = []

    indices = (idx_train, idx_val, idx_test)

    out, score_val, score_test, emb_list = run_transductive(
            conf,
            model,
            g,
            sp_adj_tensor,
            feats,
            labels,
            indices,
            criterion,
            evaluator,
            optimizer,
            logger,
            loss_and_score,
        )
    score_lst = [score_test]
    #print(emb_list.size())
    #print(len(emb_list))
    #print(emb_list[-1].size())

    logger.info(
        f"num_layers: {conf['num_layers']}. hidden_dim: {conf['hidden_dim']}. dropout_ratio: {conf['dropout_ratio']}"
    )
    logger.info(f"# params {sum(p.numel() for p in model.parameters())}")

    """ Saving teacher outputs """
    if not args.compute_min_cut:
        if 'MLP' not in model.model_name:
            out_np = out.detach().cpu().numpy()
            np.savez(output_dir.joinpath("out"), out_np)
            out_emb_list = emb_list[-1].detach().cpu().numpy()  # last hidden layer
            np.savez(output_dir.joinpath("out_emb_list"), out_emb_list)
            #print(emb_list[1].size())
            #last_out_emb_list = emb_list[1].detach().cpu().numpy()  # last hidden layer
            #np.savez(output_dir.joinpath("last_out_emb_list"), last_out_emb_list)

        """ Saving loss curve and model """
        if args.save_results:
            # Loss curves
            loss_and_score = np.array(loss_and_score)
            np.savez(output_dir.joinpath("loss_and_score"), loss_and_score)

            # Model
            torch.save(model.state_dict(), output_dir.joinpath("model.pth"))

    """ Saving min-cut loss """
    if args.exp_setting == "tran" and args.compute_min_cut:
        min_cut = compute_min_cut_loss(g, out)
        # with open(output_dir.parent.joinpath("min_cut_loss"), "a+") as f:
        #     f.write(f"{min_cut :.4f}\n")
        print('min_cut: ', min_cut)

    return score_lst

#SEEDS=[1941488137,4198936517,983997847,4023022221,4019585660]
def repeat_run(args):
    scores = []
    for seed in range(args.num_exp):
        args.seed = seed
        temp_score = run(args,seed)
        scores.append(temp_score)
    scores_np = np.array(scores)
    return scores_np.mean(axis=0), scores_np.std(axis=0)


def main():
    args = get_args()
    if args.num_exp == 1:
        score = run(args)
        score_str = "".join([f"{s : .4f}\t" for s in score])

    elif args.num_exp > 1:
        score_mean, score_std = repeat_run(args)
        score_str = "".join(
            [f"{s : .4f}\t" for s in score_mean] + [f"{s : .4f}\t" for s in score_std]
        )

    with open(args.output_dir.parent.joinpath("exp_results"), "a+") as f:
        f.write(f"{score_str}\n")

    # for collecting aggregated results
    print(score_str)


if __name__ == "__main__":
    main()
