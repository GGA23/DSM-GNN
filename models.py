import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv, SAGEConv, APPNPConv, GATConv
from torch.nn.modules.module import Module
import math
from torch.nn.parameter import Parameter

class Atten(Module):
    def __init__(self, in_features):
        super(Atten, self).__init__()
        self.in_features = in_features

        self.att_vec_k = Parameter(
                torch.FloatTensor(in_features, in_features))

        self.weight_h, self.weight_x= Parameter(
                torch.FloatTensor(in_features, in_features)), Parameter(
                torch.FloatTensor(in_features, in_features))

        self.att_vec_v = Parameter(torch.FloatTensor(2, 2))

        self.reset_parameters()

    def reset_parameters(self):

        std_att = 1. / math.sqrt(self.att_vec_k.size(1))
        stdv = 1. / math.sqrt(self.weight_h.size(1))
        std_att_vec = 1. / math.sqrt(self.att_vec_v.size(1))

        self.att_vec_k.data.uniform_(-std_att, std_att)

        self.weight_h.data.uniform_(-stdv, stdv)
        self.weight_x.data.uniform_(-stdv, stdv)

        self.att_vec_v.data.uniform_(-std_att_vec, std_att_vec)

    def Attention(self, output_x, output_h):  #
        tao = 2
        output_all = output_x + output_h
        K = torch.mean(torch.mm((output_all), self.att_vec_k), dim=0, keepdim=True)
        att = torch.softmax(torch.mm(torch.sigmoid(torch.cat(
            [torch.mm((output_x), K.T), torch.mm((output_h), K.T)], 1)), self.att_vec_v) / tao, 1)

        return att[:, 0][:, None], att[:, 1][:, None], att


    def forward(self, inputx,inputh):

        alpha_x, alpha_h, att= self.Attention(inputx, inputh)
        emb = alpha_x*inputx + alpha_h*inputh

        return emb, att

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class DMLP(nn.Module):
    def __init__(
            self,
            num_layers,
            node_num,
            input_dim,
            hidden_dim,
            output_dim,
            dropout_ratio_h,
            dropout_ratio_a,
            tau,
            norm_type="none",
    ):
        super(DMLP, self).__init__()
        self.num_layers = num_layers
        self.norm_type = norm_type
        self.dropout_h = nn.Dropout(dropout_ratio_h)
        self.dropout_a = nn.Dropout(dropout_ratio_a)
        self.layers_feat = nn.ModuleList()
        self.layers_adj = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.att = Atten(output_dim)
        self.tau = tau

        if num_layers == 1:
            self.layers_feat.append(nn.Linear(input_dim, output_dim))
            self.layers_adj.append(nn.Linear(node_num, output_dim))
        else:
            self.layers_feat.append(nn.Linear(input_dim, hidden_dim))
            self.layers_adj.append(nn.Linear(node_num, hidden_dim))


            for i in range(num_layers - 2):
                self.layers_feat.append(nn.Linear(hidden_dim, hidden_dim))
                self.layers_adj.append(nn.Linear(hidden_dim, hidden_dim))


            self.layers_feat.append(nn.Linear(hidden_dim, output_dim))
            self.layers_adj.append(nn.Linear(hidden_dim, output_dim))
            self.teacher_feature_encoder = nn.Linear(hidden_dim, hidden_dim)
            self.mlp_feature_encoder = nn.Linear(hidden_dim, hidden_dim)


    def forward(self, adj, feats):
        h = feats
        a = adj
        emb_list = []
        for l, layer in enumerate(self.layers_feat):

            a = self.layers_adj[l](a)
            h = layer(h)
            if l != self.num_layers - 1:
                a = F.relu(a)
                a = self.dropout_a(a)
                h = F.relu(h)
                h = self.dropout_h(h)

        emb_list.append(a)
        emb_list.append(h)
        z, att = self.att(a,h)
        emb_list.append(z)

        return emb_list, z, att

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1.t(), z2)

    def cons_loss(self, z1: torch.Tensor, z2: torch.Tensor, z3: torch.Tensor, z4:torch.Tensor):
        '''z1:student_emb; z2: teacher_emb; z3: pos_position; neg_position'''
        f = lambda x: torch.exp(x / self.tau)
        sim = f(self.sim(z1, z2))
        pos_sample = sim * z3
        neg_sample = sim * z4

        return -torch.log(
            pos_sample.sum(1)
            / (pos_sample.sum(1) + neg_sample.sum(1)))


class SAGE(nn.Module):
    def __init__(self,
                 num_layers,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 dropout_ratio,
                 activation,
                 norm_type="none"):
        super(SAGE, self).__init__()
        self.num_layers = num_layers
        self.norm_type = norm_type
        self.dropout = nn.Dropout(dropout_ratio)
        self.layers = nn.ModuleList()
        self.activation = activation

        if num_layers == 1:
            self.layers.append(SAGEConv(input_dim, output_dim, aggregator_type='gcn'))
        else:
            self.layers.append(SAGEConv(input_dim, hidden_dim, aggregator_type='gcn'))
            if self.norm_type == "batch":
                self.norms.append(nn.BatchNorm1d(hidden_dim))
            elif self.norm_type == "layer":
                self.norms.append(nn.LayerNorm(hidden_dim))
            for i in range(num_layers - 2):
                self.layers.append(SAGEConv(hidden_dim, hidden_dim, aggregator_type='gcn'))
                if self.norm_type == "batch":
                    self.norms.append(nn.BatchNorm1d(hidden_dim))
                elif self.norm_type == "layer":
                    self.norms.append(nn.LayerNorm(hidden_dim))
            self.layers.append(SAGEConv(hidden_dim, output_dim, aggregator_type='gcn'))

    def forward(self, g, feats):
        h = feats
        h_list = []

        for l, layer in enumerate(self.layers):
            h = layer(g, h)
            if l != self.num_layers - 1:
                h = self.activation(h)
                if self.norm_type != "none":
                    h = self.norms[l](h)
                h = self.dropout(h)
            h_list.append(h)

        return h_list, h


class GCN(nn.Module):
    def __init__(
            self,
            num_layers,
            input_dim,
            hidden_dim,
            output_dim,
            dropout_ratio,
            activation,
            norm_type="none",
    ):
        super().__init__()
        self.num_layers = num_layers
        self.norm_type = norm_type
        self.dropout = nn.Dropout(dropout_ratio)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        if num_layers == 1:
            self.layers.append(GraphConv(input_dim, output_dim, activation=activation))
        else:
            self.layers.append(GraphConv(input_dim, hidden_dim, activation=activation))
            if self.norm_type == "batch":
                self.norms.append(nn.BatchNorm1d(hidden_dim))
            elif self.norm_type == "layer":
                self.norms.append(nn.LayerNorm(hidden_dim))

            for i in range(num_layers - 2):
                self.layers.append(
                    GraphConv(hidden_dim, hidden_dim, activation=activation)
                )
                if self.norm_type == "batch":
                    self.norms.append(nn.BatchNorm1d(hidden_dim))
                elif self.norm_type == "layer":
                    self.norms.append(nn.LayerNorm(hidden_dim))

            self.layers.append(GraphConv(hidden_dim, output_dim))

    def forward(self, g, feats):
        h = feats
        h_list = []
        for l, layer in enumerate(self.layers):
            h = layer(g, h)
            if l != self.num_layers - 1:
                if self.norm_type != "none":
                    h = self.norms[l](h)
                h = self.dropout(h)
            h_list.append(h)
        return h_list, h

class GAT(nn.Module):
    def __init__(self,
                 num_layers,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 dropout_ratio,
                 activation,
                 num_heads=8,
                 attn_drop=0.3,
                 negative_slope=0.2,
                 residual=False):
        super(GAT, self).__init__()
        hidden_dim //= num_heads
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        heads = ([num_heads] * num_layers) + [1]

        self.layers.append(GATConv(input_dim, hidden_dim, heads[0], dropout_ratio, attn_drop, negative_slope, False, activation))
        for l in range(1, num_layers - 1):
            self.layers.append(GATConv(hidden_dim * heads[l-1], hidden_dim, heads[l], dropout_ratio, attn_drop, negative_slope, residual, activation))
        self.layers.append(GATConv(hidden_dim * heads[-2], output_dim, heads[-1], dropout_ratio, attn_drop, negative_slope, residual, None))

    def forward(self, g, feats):
        h = feats
        h_list = []

        for l, layer in enumerate(self.layers):
            h = layer(g, h)
            if l != self.num_layers - 1:
                h = h.flatten(1)
            else:
                h = h.mean(1)
            h_list.append(h)

        return h_list, h



class APPNP(nn.Module):
    def __init__(
            self,
            num_layers,
            input_dim,
            hidden_dim,
            output_dim,
            dropout_ratio,
            activation,
            norm_type="none",
            edge_drop=0.5,
            alpha=0.1,
            k=10,
    ):

        super(APPNP, self).__init__()
        self.num_layers = num_layers
        self.norm_type = norm_type
        self.activation = activation
        self.dropout = nn.Dropout(dropout_ratio)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        if num_layers == 1:
            self.layers.append(nn.Linear(input_dim, output_dim))
        else:
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            if self.norm_type == "batch":
                self.norms.append(nn.BatchNorm1d(hidden_dim))
            elif self.norm_type == "layer":
                self.norms.append(nn.LayerNorm(hidden_dim))

            for i in range(num_layers - 2):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
                if self.norm_type == "batch":
                    self.norms.append(nn.BatchNorm1d(hidden_dim))
                elif self.norm_type == "layer":
                    self.norms.append(nn.LayerNorm(hidden_dim))

            self.layers.append(nn.Linear(hidden_dim, output_dim))

        self.propagate = APPNPConv(k, alpha, edge_drop)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, g, feats):
        h = feats
        h_list = []
        for l, layer in enumerate(self.layers):
            h = layer(h)

            if l != self.num_layers - 1:
                h_list.append(h)
                if self.norm_type != "none":
                    h = self.norms[l](h)
                h = self.activation(h)
                h = self.dropout(h)

        h = self.propagate(g, h)
        return h_list, h

class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, dropout_ratio):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout_ratio)
        self.layers = nn.ModuleList()

        if num_layers == 1:
            self.layers.append(nn.Linear(input_dim, output_dim))
        else:
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            for i in range(num_layers - 2):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, feats):
        h = feats
        h_list = []

        for l, layer in enumerate(self.layers):
            h = layer(h)
            if l != self.num_layers - 1:
                h = F.relu(h)
                h = self.dropout(h)
            h_list.append(h)

        return h_list, h


class MLP_A(nn.Module):
    def __init__(self, num_layers, node_num, hidden_dim, output_dim, dropout_ratio):
        super(MLP_A, self).__init__()
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout_ratio)
        self.layers = nn.ModuleList()

        if num_layers == 1:
            self.layers.append(nn.Linear(node_num, output_dim))
        else:
            self.layers.append(nn.Linear(node_num, hidden_dim))
            for i in range(num_layers - 2):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, adj):
        s = adj
        s_list = []

        for l, layer in enumerate(self.layers):
            s = layer(s)
            if l != self.num_layers - 1:
                s = F.relu(s)
                s = self.dropout(s)
            s_list.append(s)

        return s_list, s
class Model(nn.Module):
    """
    Wrapper of different models
    """

    def __init__(self, conf):
        super(Model, self).__init__()
        self.model_name = conf["model_name"]
        if "DMLP" in conf["model_name"]:
            # origin
            self.encoder = DMLP(
                num_layers=conf["num_layers"],
                node_num= conf["num_nodes"],
                input_dim=conf["feat_dim"],
                hidden_dim=conf["hidden_dim"],
                output_dim=conf["label_dim"],
                dropout_ratio_h=conf["dropout_ratio_h"],
                dropout_ratio_a=conf["dropout_ratio_a"],
                tau=conf['tau'],
                norm_type=conf["norm_type"],
            ).to(conf["device"])
        elif "MLP" in self.model_name:
            self.encoder = MLP(num_layers=conf["num_layers"], input_dim=conf["feat_dim"],
                               hidden_dim=conf["hidden_dim"], output_dim=conf["label_dim"],
                               dropout_ratio=conf["dropout_ratio"]).to(conf["device"])
        elif "MLP_A" in self.model_name:
            self.encoder = MLP_A(num_layers=conf["num_layers"], node_num=conf["num_nodes"],
                               hidden_dim=conf["hidden_dim"], output_dim=conf["label_dim"],
                               dropout_ratio=conf["dropout_ratio"]).to(conf["device"])
        elif "SAGE" in conf["model_name"]:
            self.encoder = SAGE(
                num_layers=conf["num_layers"],
                input_dim=conf["feat_dim"],
                hidden_dim=conf["hidden_dim"],
                output_dim=conf["label_dim"],
                dropout_ratio=conf["dropout_ratio"],
                activation=F.relu,
                norm_type=conf["norm_type"],
            ).to(conf["device"])
        elif "GCN" in conf["model_name"]:
            self.encoder = GCN(
                num_layers=conf["num_layers"],
                input_dim=conf["feat_dim"],
                hidden_dim=conf["hidden_dim"],
                output_dim=conf["label_dim"],
                dropout_ratio=conf["dropout_ratio"],
                activation=F.relu,
                norm_type=conf["norm_type"],
            ).to(conf["device"])
        elif "GAT" in conf["model_name"]:
            self.encoder = GAT(
                num_layers=conf["num_layers"],
                input_dim=conf["feat_dim"],
                hidden_dim=conf["hidden_dim"],
                output_dim=conf["label_dim"],
                dropout_ratio=conf["dropout_ratio"],
                activation=F.relu,
                attn_drop=conf["attn_dropout_ratio"],
            ).to(conf["device"])
        elif "APPNP" in conf["model_name"]:
            self.encoder = APPNP(
                num_layers=conf["num_layers"],
                input_dim=conf["feat_dim"],
                hidden_dim=conf["hidden_dim"],
                output_dim=conf["label_dim"],
                dropout_ratio=conf["dropout_ratio"],
                activation=F.relu,
                norm_type=conf["norm_type"],
            ).to(conf["device"])

    def forward(self, data, feats):
        """
        data: a graph `g` or a `dataloader` of blocks
        """
        #print(self.model_name)
        #if "MLP_A" in self.model_name:
        #    return self.encoder(data)
        #if "MLP" in self.model_name:
        #    return self.encoder(feats)

        return self.encoder(data, feats)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def common_loss(self, emb1, emb2):
        emb1 = emb1 - torch.mean(emb1, dim=0, keepdim=True)
        emb2 = emb2 - torch.mean(emb2, dim=0, keepdim=True)
        emb1 = torch.nn.functional.normalize(emb1, p=2, dim=1)
        emb2 = torch.nn.functional.normalize(emb2, p=2, dim=1)
        cov1 = torch.matmul(emb1, emb1.t())
        cov2 = torch.matmul(emb2, emb2.t())
        cost = torch.mean((cov1 - cov2) ** 2)
        return cost

    def cons_loss(self, z1: torch.Tensor, z2: torch.Tensor, z3: torch.Tensor, z4: torch.Tensor, tau):
        '''z1:student_emb; z2: teacher_emb; z3: pos_emb; neg_position'''
        self.tau = tau
        f = lambda x: torch.exp(x / self.tau)
        pos_sim = f(F.cosine_similarity(z1, z3))
        sim = f(self.sim(z1, z2))
        #pos_sample = sim * z3
        neg_sample = sim * z4

        return -torch.log(
            pos_sim
            / (pos_sim + neg_sample.sum(1)))




