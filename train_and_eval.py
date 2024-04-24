import numpy as np
import copy
import torch
import dgl
import pandas as pd
from utils import set_seed,get_pos_neg_pair
import time
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def plotClusters(hidden_emb, true_labels,n,s,name='mykd'):
    #tqdm.write('Start plotting using TSNE...')
    # Doing dimensionality reduction for plotting
    tsne = TSNE(n_components=2)
    X_tsne = tsne.fit_transform(hidden_emb)
    # Plot figure
    fig = plt.figure(figsize=(5, 5), dpi=1000, )
    plt.subplot(1, 1, 1)
    plt.axis('off')
    col, size, true_labels = ['red', 'green', 'blue', 'brown', 'purple', 'yellow','pink'], 8, true_labels
    for i, point in enumerate(X_tsne):
        plt.scatter(point[0], point[1], s=size, c=col[true_labels[i]])
    fig.savefig('./img/{}/cora{}_{}.png'.format(name,s,n))
    #fig.savefig('./img/texas.svg', format='svg')
    fig.savefig('./img/{}/cora_{}_{}.eps'.format(name,s,n), format='eps', dpi=1000)
"""
1. Train and eval
"""
def train(model, data, feats, labels, criterion, optimizer, idx_train, lamb=1):
    """
    GNN full-batch training. Input the entire graph `g` as data.
    lamb: weight parameter lambda
    """
    model.train()

    # Compute loss and prediction
    _, logits = model(data, feats)
    out = logits.log_softmax(dim=1)
    loss = criterion(out[idx_train], labels[idx_train])
    loss_val = loss.item()

    loss *= lamb
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss_val

def evaluate(model, data, feats, labels, criterion, evaluator, idx_eval=None):
    """
    Returns:
    out: log probability of all input data
    loss & score (float): evaluated loss & score, if idx_eval is not None, only loss & score on those idx.
    """
    model.eval()

    emb_list, logits,_ = model(data, feats)

    out = logits.log_softmax(dim=1)
    if idx_eval is None:
        loss = criterion(out, labels)
        score = evaluator(out, labels)
    else:
        loss = criterion(out[idx_eval], labels[idx_eval])
        score = evaluator(out[idx_eval], labels[idx_eval])
    return out,  loss.item(), score, emb_list



"""
2. Run teacher
"""

def run_transductive(
        conf,
        model,
        g,
        dense_adj_tensor,
        feats,
        labels,
        indices,
        criterion,
        evaluator,
        optimizer,
        logger,
        loss_and_score,
):
    """
    Train and eval under the transductive setting.
    The train/valid/test split is specified by `indices`.
    The input graph is assumed to be large. Thus, SAGE is used for GNNs, mini-batch is used for MLPs.

    loss_and_score: Stores losses and scores.
    """
    set_seed(conf["seed"])
    device = conf["device"]

    idx_train, idx_val, idx_test = indices

    feats = feats.to(device)
    labels = labels.to(device)
    g = g.to(device)
    dense_adj_tensor = dense_adj_tensor.to(device)
    #data = g
    if conf["model_name"] == 'MLP_A':
        data = dense_adj_tensor
    else:
        data = g

    best_epoch, best_score_val, count = 0, 0, 0
    for epoch in range(1, conf["max_epoch"] + 1):

        loss = train(model, data, feats, labels, criterion, optimizer, idx_train)

        if epoch % conf["eval_interval"] == 0:

            out, loss_train, score_train, emb_list = evaluate(
                    model, data, feats, labels, criterion, evaluator, idx_train
            )
            # Use criterion & evaluator instead of evaluate to avoid redundant forward pass
            loss_val = criterion(out[idx_val], labels[idx_val]).item()
            score_val = evaluator(out[idx_val], labels[idx_val])
            loss_test = criterion(out[idx_test], labels[idx_test]).item()
            score_test = evaluator(out[idx_test], labels[idx_test])

        logger.debug(
                f"Ep {epoch:3d} | loss: {loss:.4f} | s_train: {score_train:.4f} | s_val: {score_val:.4f} | s_test: {score_test:.4f}"
            )
        loss_and_score += [
                [
                epoch,
                loss_train,
                loss_val,
                loss_test,
                score_train,
                score_val,
                score_test,
                ]
            ]

        if score_val >= best_score_val:
            best_epoch = epoch
            best_score_val = score_val
            state = copy.deepcopy(model.state_dict())
            count = 0
        else:
             count += 1

        if count == conf["patience"] or epoch == conf["max_epoch"]:
            break
        if epoch % 1 == 0:
            print(
                "\033[0;30;46m [{}] LT: {:.5f}  | Train: {:.4f}, Val: {:.4f}, Test: {:.4f} | Val Best: {:.4f}, Test Best: {:.4f}\033[0m".format(
                    epoch, loss_train,  score_train, score_val, score_test, best_score_val,  score_test))


    model.load_state_dict(state)

    out, _, score_val, emb_list = evaluate(
            model, data, feats, labels, criterion, evaluator, idx_val
        )

    score_test = evaluator(out[idx_test], labels[idx_test])
    logger.info(
        f"Best valid model at epoch: {best_epoch: 3d}, score_val: {score_val :.4f}, score_test: {score_test :.4f}"
    )
    print(f"Best valid model at epoch: {best_epoch: 3d}, score_val: {score_val :.4f}, score_test: {score_test :.4f}")
    return out, score_val, score_test, emb_list

def train_both_distillation(model, pos_emb, adj_neg, tau, adj, feats, labels, soft_out, out_emb, idx_l, idx_t, criterion_l, criterion_t, optimizer, conf):
    model.train()
    emb_list, logits, att = model(adj, feats)
    out = logits.log_softmax(dim=1)
    loss_l = criterion_l(out[idx_l], labels[idx_l])
    loss_t1 = criterion_t(emb_list[0].log_softmax(dim=1)[idx_t], soft_out[idx_t])
    loss_t2 = criterion_t(emb_list[1].log_softmax(dim=1)[idx_t], soft_out[idx_t])
    loss_dep = model.common_loss(emb_list[0],emb_list[1])
    cons_loss = model.cons_loss(logits, out_emb, pos_emb, adj_neg,tau).mean()
    loss = loss_l + conf['alpha']*(loss_t1 + loss_t2) + conf['beta']*loss_dep + conf['gama'] * cons_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return out, att, loss_l.item(), conf['alpha']*(loss_t1 + loss_t2) + conf['beta']*loss_dep + conf['gama']*cons_loss
    #return out, att, loss_l.item(), conf['alpha'] * (loss_t1 + loss_t2)  + conf['gama'] * cons_loss

"""
3. Distill
"""


def distill_run_transductive(
        conf,
        model,
        adj,
        feats,
        labels,
        out_t_all,
        out_emb_t_all,
        distill_indices,
        criterion_l,
        criterion_t,
        evaluator,
        optimizer,
        logger,
        loss_and_score,
        graph,
        pos_emb,
        Sim_neg,
        args
):
    """
    Distill training and eval under the transductive setting.
    The hard_label_train/soft_label_train/valid/test split is specified by `distill_indices`.
    The input graph is assumed to be large, and MLP is assumed to be the student model. Thus, node feature only and mini-batch is used.

    out_t: Soft labels produced by the teacher model.
    criterion_l & criterion_t: Loss used for hard labels (`labels`) and soft labels (`out_t`) respectively
    loss_and_score: Stores losses and scores.
    """
    set_seed(conf["seed"])
    device = conf["device"]
    idx_l, idx_t, idx_val, idx_test = distill_indices

    adj = adj.to(device)
    feats = feats.to(device)
    labels = labels.to(device)
    #labels_one_hot = labels_one_hot.to(device)
    soft_out = out_t_all.to(device)

    out_emb = out_emb_t_all.to(device)
    #print(out_emb.size())
    #adj_pos, adj_neg = get_pos_neg_pair(adj, feats, soft_out, args,device)
    #P_pos = torch.mm(adj_pos,out_emb)/adj_pos.sum(1).view(-1, 1)
    best_epoch, best_score_val, test_acc_val, count = 0, 0, 0, 0
    time_run = []
    for epoch in range(1, conf["max_epoch"] + 1):
        t_st = time.time()
        out, att, loss_train, loss_kd = train_both_distillation(
            model, pos_emb, Sim_neg,args.tau, adj, feats, labels, soft_out, out_emb, idx_l, idx_t, criterion_l, criterion_t, optimizer, conf,
        )
        time_epoch = time.time() - t_st  # each epoch train times
        time_run.append(time_epoch)

        loss = loss_train+loss_kd


        _, loss_l, score_l, _ = evaluate(
                model, adj, feats, labels, criterion_l, evaluator,idx_l
            )
        _, loss_val, score_val,_ = evaluate(
                model, adj, feats, labels, criterion_l, evaluator,idx_val
            )
        _, loss_test, score_test,_ = evaluate(
                model, adj, feats, labels, criterion_l, evaluator,idx_test
            )
        #list = [epoch, loss_l, loss_val, loss_test]
        #data = pd.DataFrame([list])
        #data.to_csv('./loss-data/chamloss_{}.csv'.format(args.seed), mode='a', header=False,
        #            index=False)  # mode设为a,就可以向csv文件追加数据了

        logger.debug(
                f"Ep {epoch:3d} | loss: {loss:.4f} | s_l: {score_l:.4f} | s_val: {score_val:.4f} | s_test: {score_test:.4f}"
            )
        loss_and_score += [
                [epoch, loss_l, loss_val, loss_test, score_l, score_val, score_test]
            ]

        if score_val >= best_score_val:
            best_epoch = epoch
            best_score_val = score_val
            test_acc_val = score_test
            state = copy.deepcopy(model.state_dict())
            #df = pd.DataFrame(att)
            #df.to_csv('./weight/{}/att_pre{}.csv'.format(args.dataset,args.seed), index=False, header=False)
            count = 0
        else:
            count += 1
        #if test_acc_val > 0.85:
        #    plotClusters(out.cpu().detach().numpy(), labels, epoch,args.seed)
        if count == conf["patience"] or epoch == conf["max_epoch"]:
            break
        if epoch % 1 == 0:
            print(
                "\033[0;30;46m [{}] LT: {:.5f}  | Train: {:.4f}, Val: {:.4f}, Test: {:.4f} | Val Best: {:.4f}, Test Best: {:.4f}\033[0m".format(
                    epoch, loss_train, score_l, score_val, score_test, best_score_val,  test_acc_val))

    model.load_state_dict(state)
    out, _, score_val,_ = evaluate(
        model, adj, feats, labels, criterion_l, evaluator, idx_val
    )
    # Use evaluator instead of evaluate to avoid redundant forward pass
    score_test = evaluator(out[idx_test], labels[idx_test])

    logger.info(
        f"Best valid model at epoch: {best_epoch: 3d}, score_val: {score_val :.4f}, score_test: {score_test :.4f}"
    )
    #print('average total running time:{} (s), average running time per epoch:{} (s)'.format(sum(time_run),sum(time_run)/int(len(time_run))))
    print('total running time:{} (s)'.format(sum(time_run)))
    return out, score_val, score_test
