import torch
import torch.nn as nn
import copy
import torch.nn.functional as F


class MLP(nn.Module):


    def __init__(self, inp_size, outp_size, hidden_size):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(inp_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.PReLU(),
            nn.Linear(hidden_size, outp_size)
        )

    def forward(self, x):

        return self.net(x)


class GraphEncoder(nn.Module):


    def __init__(self,
                 gnn,
                 projection_hidden_size,
                 projection_size):

        super().__init__()

        self.gnn = gnn

        self.projector = MLP(512, projection_size, projection_hidden_size)

    def forward(self, adj, in_feats, sparse):
        representations = self.gnn(in_feats, adj, sparse)
        representations = representations.view(-1, representations.size(-1))
        projections = self.projector(representations)  # (batch, proj_dim)
        return projections


class EMA():

    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def update_moving_average(ema_updater, ma_model, current_model):
    'MOCO-like'
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)


def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val


def sim(h1, h2):

    z1 = F.normalize(h1, dim=-1, p=2)
    z2 = F.normalize(h2, dim=-1, p=2)
    return torch.mm(z1, z2.t())

####################################################################################


def agg_contrastive_loss(h, z, tau=0.8):
    def f(x, tau): return torch.exp(x/tau)
    cross_sim = f(sim(h, z), tau)
    return -torch.log(cross_sim.diag()/cross_sim.sum(dim=-1))


def interact_contrastive_loss(h1, h2, tau=0.8):
    def f(x, tau): return torch.exp(x/tau)
    intra_sim = f(sim(h1, h1), tau)
    inter_sim = f(sim(h1, h2), tau)
    return -torch.log(inter_sim.diag() /
                      (intra_sim.sum(dim=-1) + inter_sim.sum(dim=-1) - intra_sim.diag()))


class IMCSNet(nn.Module):

    def __init__(self,
                 gnn,
                 feat_size,
                 projection_size,
                 projection_hidden_size,
                 prediction_size,
                 prediction_hidden_size,
                 moving_average_decay,
                 beta,
                 alpha):

        super().__init__()


        self.online_encoder = GraphEncoder(
            gnn, projection_hidden_size, projection_size)
        self.target_encoder1 = copy.deepcopy(self.online_encoder)
        self.target_encoder2 = copy.deepcopy(self.online_encoder)

        set_requires_grad(self.target_encoder1, False)
        set_requires_grad(self.target_encoder2, False)

        self.target_ema_updater = EMA(moving_average_decay)
        self.online_predictor = MLP(
            projection_size, prediction_size, prediction_hidden_size)

        self.beta = beta
        self.alpha = alpha

    def reset_moving_average(self):
        del self.target_encoder1
        del self.target_encoder2
        self.target_encoder1 = None
        self.target_encoder2 = None

    def update_ma(self):
        assert self.target_encoder1 or self.target_encoder2 is not None, 'target encoder has not been created yet'
        update_moving_average(self.target_ema_updater,
                              self.target_encoder1, self.online_encoder)
        update_moving_average(self.target_ema_updater,
                              self.target_encoder2, self.online_encoder)

    def forward(self, adj, aug_adj_1, aug_adj_2, feat, aug_feat_1, aug_feat_2, sparse):
        online_proj = self.online_encoder(adj, feat, sparse)
        online_proj_1 = self.online_encoder(aug_adj_1, aug_feat_1, sparse)
        online_proj_2 = self.online_encoder(aug_adj_2, aug_feat_2, sparse)

        online_pred = self.online_predictor(online_proj)
        online_pred_1 = self.online_predictor(online_proj_1)
        online_pred_2 = self.online_predictor(online_proj_2)

        with torch.no_grad():
            target_proj_01 = self.target_encoder1(adj, feat, sparse)
            target_proj_11 = self.target_encoder1(aug_adj_1, aug_feat_1, sparse)
            target_proj_02 = self.target_encoder2(adj, feat, sparse)
            target_proj_22 = self.target_encoder2(aug_adj_2, aug_feat_2, sparse)

        structure_sim = (online_pred_1 @ online_pred_1.T) + (target_proj_01 @ target_proj_01.T) + (target_proj_22 @ target_proj_22.T)
        structure_sim = F.normalize(structure_sim)

        recon_structure_loss = torch.mean((adj.to_dense() - structure_sim) ** 2)

        l_cn_1 = self.alpha * agg_contrastive_loss(online_pred_1, target_proj_01.detach()) +\
            (1.0-self.alpha) * \
            agg_contrastive_loss(online_pred_1, target_proj_22.detach())

        l_cn_2 = self.alpha * agg_contrastive_loss(online_pred_2, target_proj_11.detach()) +\
            (1.0-self.alpha) * \
            agg_contrastive_loss(online_pred_2, target_proj_02.detach())

        l_cn = 0.5*(l_cn_1+l_cn_2)

        l_cv_0 = interact_contrastive_loss(online_pred, online_pred_1)

        l_cv_1 = interact_contrastive_loss(online_pred_1, online_pred_2)

        l_cv_2 = interact_contrastive_loss(online_pred_2, online_pred)

        l_cv = (l_cv_0+l_cv_1+l_cv_2)/3

        loss = l_cv * self.beta + l_cn * (1-self.beta) + recon_structure_loss

        return loss.mean()

##################################################################################