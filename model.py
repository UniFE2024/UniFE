import dgl
import dgl.function as fn
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class FeatureConv(nn.Module):
    def __init__(self, in_dim, concat_dim, out_dim, comp_fn, batchnorm, dropout):
        super(FeatureConv, self).__init__()
        self.in_dim = in_dim
        self.concat_dim = concat_dim
        self.out_dim = out_dim
        self.comp_fn = comp_fn
        self.actvation = th.tanh
        self.batchnorm = batchnorm
        self.dropout = nn.Dropout(dropout)
        real_ent_dim = self.in_dim
        real_rel_dim = self.in_dim
        self.W_O = nn.Linear(real_ent_dim, real_ent_dim, bias=False)
        self.W_I = nn.Linear(real_ent_dim, real_ent_dim, bias=False)
        self.W_S = nn.Linear(real_ent_dim, real_ent_dim, bias=False)
        self.out_dim = real_ent_dim
        self.loop_rel = nn.Parameter(th.Tensor(1, real_rel_dim))
        nn.init.xavier_normal_(self.loop_rel)
        self.W_R = nn.Linear(real_rel_dim, real_rel_dim, bias=False)
        if self.batchnorm:
            self.bn = nn.BatchNorm1d(self.out_dim)

    def forward(self, g, n_in_feats, r_feats):
        with g.local_scope():
            g.srcdata["h"] = n_in_feats
            r_feats = th.cat((r_feats, self.loop_rel), 0)
            g.edata["h"] = r_feats[g.edata["etype"]] * g.edata["norm"]

            if self.comp_fn == "sub":
                g.apply_edges(fn.u_sub_e("h", "h", out="comp_h"))
            elif self.comp_fn == "mul":
                g.apply_edges(fn.u_mul_e("h", "h", out="comp_h"))
            elif self.comp_fn == "ccorr":
                g.apply_edges(lambda edges: {"comp_h": ccorr(edges.src["h"], edges.data["h"])})
            else:
                raise Exception("Only supports sub, mul, and ccorr")

            comp_h = g.edata["comp_h"]
            in_edges_idx = th.nonzero(g.edata["in_edges_mask"], as_tuple=False).squeeze()
            out_edges_idx = th.nonzero(g.edata["out_edges_mask"], as_tuple=False).squeeze()
            comp_h_O = self.W_O(comp_h[out_edges_idx])
            comp_h_I = self.W_I(comp_h[in_edges_idx])
            new_comp_h = th.zeros(comp_h.shape[0], self.out_dim).to(comp_h.device)
            new_comp_h[out_edges_idx] = comp_h_O
            new_comp_h[in_edges_idx] = comp_h_I
            g.edata["new_comp_h"] = new_comp_h

            g.update_all(fn.copy_e("new_comp_h", "m"), fn.sum("m", "comp_edge"))

            if self.comp_fn == "sub":
                comp_h_s = n_in_feats - r_feats[-1]
            elif self.comp_fn == "mul":
                comp_h_s = n_in_feats * r_feats[-1]
            elif self.comp_fn == "ccorr":
                comp_h_s = ccorr(n_in_feats, r_feats[-1])

            n_out_feats = (self.W_S(comp_h_s) + self.dropout(g.ndata["comp_edge"])) * (1 / 3)
            r_out_feats = self.W_R(r_feats)
            if self.batchnorm:
                n_out_feats = self.bn(n_out_feats)
            if self.actvation is not None:
                n_out_feats = self.actvation(n_out_feats)

        return n_out_feats, r_out_feats[:-1]


class FeatureEncoder(nn.Module):
    def __init__(self, args, num_ent, num_rel, batchnorm=True):
        super(FeatureEncoder, self).__init__()

        self.num_bases = args.num_bases
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.in_dim = args.init_dim
        self.layer_size = args.layer_size
        self.ent_dim = args.ent_dim
        self.rel_dim = args.rel_dim
        self.comp_fn = args.opn
        self.batchnorm = batchnorm
        self.dropout = args.dropout
        self.layer_dropout = args.layer_dropout
        self.num_layer = len(args.layer_size)
        self.concat_dim = args.concat_dim
        self.score_func = args.score_func
        self.layers = nn.ModuleList()
        self.layers.append(
            FeatureConv(self.ent_dim, self.concat_dim, self.layer_size[0], comp_fn=self.comp_fn, batchnorm=self.batchnorm, dropout=self.dropout))
        for i in range(self.num_layer - 1):
            self.layers.append(
                FeatureConv(self.layer_size[i], self.concat_dim, self.layer_size[i + 1], comp_fn=self.comp_fn, batchnorm=self.batchnorm))
        if self.num_bases > 0:
            self.basis = nn.Parameter(th.Tensor(self.num_bases, self.in_dim))
            self.weights = nn.Parameter(th.Tensor(self.num_rel, self.num_bases))
            nn.init.xavier_normal_(self.basis)
            nn.init.xavier_normal_(self.weights)
        else:
            self.rel_embds = nn.Parameter(th.Tensor(self.num_rel, self.in_dim))
            nn.init.xavier_normal_(self.rel_embds)

        self.n_embds = nn.Parameter(th.Tensor(self.num_ent, self.in_dim))
        nn.init.xavier_normal_(self.n_embds)

        self.dropouts = nn.ModuleList()
        for i in range(self.num_layer):
            self.dropouts.append(nn.Dropout(self.layer_dropout[i]))

    def forward(self, graph):
        n_feats = self.n_embds
        if self.num_bases > 0:
            r_embds = th.mm(self.weights, self.basis)
            r_feats = r_embds
        else:
            r_feats = self.rel_embds

        for layer, dropout in zip(self.layers, self.dropouts):
            n_feats, r_feats = layer(graph, n_feats, r_feats)
            n_feats = dropout(n_feats)

        return n_feats, r_feats


class FeatureModule(nn.Module):
    def __init__(self, args, num_ent, num_rel, batchnorm=True):
        super(FeatureModule, self).__init__()

        self.embed_dim = args.layer_size[-1]
        self.hid_drop = args.hid_drop
        self.feat_drop = args.feat_drop
        self.ker_sz = args.ker_sz
        self.k_w = args.k_w
        self.k_h = args.k_h
        self.num_filt = args.num_filt
        self.comp_fn = args.opn
        self.score_func = args.score_func

        self.encoder = FeatureEncoder(args, num_ent, num_rel, batchnorm)

        self.bn0 = th.nn.BatchNorm2d(1)
        self.bn1 = th.nn.BatchNorm2d(self.num_filt)
        self.bn2 = th.nn.BatchNorm1d(self.embed_dim)

        self.hidden_drop = th.nn.Dropout(self.hid_drop)
        self.feature_drop = th.nn.Dropout(self.feat_drop)
        self.m_conv1 = th.nn.Conv2d(
            1,
            out_channels=self.num_filt,
            kernel_size=(self.ker_sz, self.ker_sz),
            stride=1,
            padding=0,
            bias=False,
        )

        flat_sz_h = int(2 * self.k_w) - self.ker_sz + 1
        flat_sz_w = self.k_h - self.ker_sz + 1
        self.flat_sz = flat_sz_h * flat_sz_w * self.num_filt
        self.fc = th.nn.Linear(self.flat_sz, self.embed_dim)

    def concat(self, e1_embed, rel_embed):
        e1_embed = e1_embed.view(-1, 1, self.embed_dim)
        rel_embed = rel_embed.view(-1, 1, self.embed_dim)
        stack_inp = th.cat([e1_embed, rel_embed], 1)
        stack_inp = th.transpose(stack_inp, 2, 1).reshape(
            (-1, 1, 2 * self.k_w, self.k_h)
        )
        return stack_inp

    def forward(self, graph, sub, rel):
        n_feats, r_feats = self.encoder(graph)
        sub_emb = n_feats[sub, :]
        rel_emb = r_feats[rel, :]

        if self.score_func == 'conve':
            stk_inp = self.concat(sub_emb, rel_emb)
            x = self.bn0(stk_inp)
            x = self.m_conv1(x)
            x = self.bn1(x)
            x = F.relu(x)
            x = self.feature_drop(x)
            x = x.view(-1, self.flat_sz)
            x = self.fc(x)
            x = self.hidden_drop(x)
            x = self.bn2(x)
            x = F.relu(x)
            x = th.mm(x, n_feats.transpose(1, 0))
        elif self.score_func == 'transe':
            obj_emb = sub_emb + rel_emb
            x = self.gamma - torch.norm(obj_emb.unsqueeze(1) - n_feats, p=1, dim=2)
        elif self.score_func == 'distmult':
            obj_emb = sub_emb * rel_emb
            x = torch.mm(obj_emb, n_feats.transpose(1, 0))
            x += self.bias.expand_as(x)
        elif self.score_func == 'complex':
            re_head, im_head = torch.chunk(sub_emb, 2, dim=1)
            re_relation, im_relation = torch.chunk(rel_emb, 2, dim=1)
            re_tail, im_tail = torch.chunk(n_feats, 2, dim=1)
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            x = re_score.mm(re_tail.transpose(1, 0)) + im_score.mm(im_tail.transpose(1, 0))
        else:
            raise ValueError("Not in [transe, distmult, complex, conve].")
        score = th.sigmoid(x)
        return score


class DualEncoder(nn.Module):
    def __init__(self, args, num_ent, num_rel, batchnorm=True):
        super(DualEncoder, self).__init__()

        self.num_bases = args.num_bases
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.in_dim = args.init_dim
        self.concat_dim = args.concat_dim
        self.layer_size = args.layer_size
        self.ent_dim = args.ent_dim
        self.rel_dim = args.rel_dim
        self.comp_fn = args.opn
        self.batchnorm = batchnorm
        self.dropout = args.dropout
        self.layer_dropout = args.layer_dropout
        self.num_layer = len(self.layer_size)
        self.n_basis = args.n_basis

        self.layers = nn.ModuleList()
        self.layers.append(
            FeatureConv(self.ent_dim, self.concat_dim, self.layer_size[0], comp_fn=self.comp_fn, batchnorm=self.batchnorm, dropout=self.dropout))
        self.struct_encoder_agg = nn.ModuleList()
        self.struct_encoder_agg.append(GCNConv(self.concat_dim, self.concat_dim))

        for i in range(self.num_layer - 1):
            self.layers.append(
                FeatureConv(self.layer_size[i], self.concat_dim, self.layer_size[i + 1], comp_fn=self.comp_fn, batchnorm=self.batchnorm,
                            dropout=self.dropout))
            self.struct_encoder_agg.append(GCNConv(self.concat_dim, self.concat_dim))

        if self.num_bases > 0:
            self.basis = nn.Parameter(th.Tensor(self.num_bases, self.init_dim))
            self.weights = nn.Parameter(th.Tensor(self.num_rel, self.num_bases))
            nn.init.xavier_normal_(self.basis)
            nn.init.xavier_normal_(self.weights)
        else:
            self.rel_embds = nn.Parameter(th.Tensor(self.num_rel, self.rel_dim))
            nn.init.xavier_normal_(self.rel_embds)

        self.n_embds = nn.Parameter(th.Tensor(self.num_ent, self.ent_dim))
        nn.init.xavier_normal_(self.n_embds)

        self.dropouts = nn.ModuleList()
        for i in range(self.num_layer):
            self.dropouts.append(nn.Dropout(self.layer_dropout[i]))
        self.embedding_sub_agg = nn.Linear(self.ent_dim + self.in_dim, self.ent_dim)
        self.Whp = torch.nn.Linear(self.ent_dim + self.in_dim, self.ent_dim)

        self.degree_basis = nn.Parameter(th.Tensor(args.basis_dim, self.in_dim))
        self.degree_weights_agg = nn.Parameter(th.Tensor(self.n_basis, args.basis_dim))
        nn.init.xavier_normal_(self.degree_basis)
        nn.init.xavier_normal_(self.degree_weights_agg)

    def forward(self, graph, degree_pattern_g):
        node_coef = self.init_pattern_g(degree_pattern_g)
        edge_index = torch.stack([graph.edges()[0], graph.edges()[1]], dim=0)
        n_feats = self.n_embds
        if self.num_bases > 0:
            r_embds = th.mm(self.weights, self.basis)
            r_feats = r_embds
        else:
            r_feats = self.rel_embds
        dn_feats = th.mm(node_coef, self.degree_basis)

        for index, (layer, dropout) in enumerate(zip(self.layers, self.dropouts)):
            n_feats = torch.cat([n_feats, dn_feats], dim=-1)
            n_feats = self.embedding_sub_agg(n_feats)
            n_feats, r_feats = layer(graph, n_feats, r_feats)
            n_feats = dropout(n_feats)
            dn_feats = self.struct_encoder_agg[index](dn_feats, edge_index)
            dn_feats = torch.tanh(dn_feats)

        n_feats = self.Whp(torch.cat([n_feats, dn_feats], dim=-1))

        return n_feats, r_feats

    def init_pattern_g(self, pattern_g):
        with pattern_g.local_scope():
            etypes = pattern_g.edata['etype']
            pattern_g.edata['edge_h'] = self.degree_weights_agg[etypes] / (pattern_g.edata['etype'] + 1).unsqueeze(1)
            message_func = dgl.function.copy_e('edge_h', 'msg')
            reduce_func = dgl.function.mean('msg', 'h')
            pattern_g.update_all(message_func, reduce_func)
            pattern_g.edata.pop('edge_h')
            rel_coef = pattern_g.ndata['h']

        return rel_coef


class UniFE(nn.Module):
    def __init__(self, args, num_ent, num_rel, batchnorm=True, gamma=10.0, epsilon=2.0):
        super(UniFE, self).__init__()

        self.embed_dim = args.layer_size[-1]
        self.gamma = gamma
        self.epsilon = epsilon
        self.embedding_range = torch.Tensor([(self.gamma + self.epsilon) / self.embed_dim])
        self.opn = args.opn
        self.score_func = args.score_func
        self.feat_and_struct_encoder = DualEncoder(args=args, num_ent=num_ent, num_rel=num_rel, batchnorm=batchnorm)

        if self.score_func == 'conve':
            self.hid_drop = args.hid_drop
            self.feat_drop = args.feat_drop
            self.ker_sz = args.ker_sz
            self.k_w = args.k_w
            self.k_h = args.k_h
            self.num_filt = args.num_filt
            self.bn0 = th.nn.BatchNorm2d(1)
            self.bn1 = th.nn.BatchNorm2d(self.num_filt)
            self.bn2 = th.nn.BatchNorm1d(self.embed_dim)

            self.hidden_drop = th.nn.Dropout(self.hid_drop)
            self.feature_drop = th.nn.Dropout(self.feat_drop)
            self.m_conv1 = th.nn.Conv2d(1, out_channels=self.num_filt, kernel_size=(self.ker_sz, self.ker_sz), stride=1, padding=0, bias=False, )

            flat_sz_h = int(2 * self.k_w) - self.ker_sz + 1
            flat_sz_w = self.k_h - self.ker_sz + 1
            self.flat_sz = flat_sz_h * flat_sz_w * self.num_filt
            self.fc = th.nn.Linear(self.flat_sz, self.embed_dim)

    def concat(self, e1_embed, rel_embed):
        e1_embed = e1_embed.view(-1, 1, self.embed_dim)
        rel_embed = rel_embed.view(-1, 1, self.embed_dim)
        stack_inp = th.cat([e1_embed, rel_embed], 1)
        stack_inp = th.transpose(stack_inp, 2, 1).reshape(
            (-1, 1, 2 * self.k_w, self.k_h)
        )
        return stack_inp

    def forward(self, graph, degree_pattern_g, sub, rel):
        n_feats, r_feats = self.feat_and_struct_encoder(graph, degree_pattern_g)
        sub_emb = n_feats[sub, :]
        rel_emb = r_feats[rel, :]

        if self.score_func == 'conve':
            stk_inp = self.concat(sub_emb, rel_emb)
            x = self.bn0(stk_inp)
            x = self.m_conv1(x)
            x = self.bn1(x)
            x = F.relu(x)
            x = self.feature_drop(x)
            x = x.view(-1, self.flat_sz)
            x = self.fc(x)
            x = self.hidden_drop(x)
            x = self.bn2(x)
            x = F.relu(x)
            x = th.mm(x, n_feats.transpose(1, 0))
        elif self.score_func == 'transe':
            obj_emb = sub_emb + rel_emb
            x = self.gamma - torch.norm(obj_emb.unsqueeze(1) - n_feats, p=1, dim=2)
        elif self.score_func == 'distmult':
            obj_emb = sub_emb * rel_emb
            x = torch.mm(obj_emb, n_feats.transpose(1, 0))
        elif self.score_func == 'complex':
            re_head, im_head = torch.chunk(sub_emb, 2, dim=1)
            re_relation, im_relation = torch.chunk(rel_emb, 2, dim=1)
            re_tail, im_tail = torch.chunk(n_feats, 2, dim=1)
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            x = re_score.mm(re_tail.transpose(1, 0)) + im_score.mm(im_tail.transpose(1, 0))
        else:
            raise ValueError("Not in [transe, distmult, complex, conve].")
        score = th.sigmoid(x)
        return score


def ccorr(a, b):
    return th.fft.irfftn(th.conj(th.fft.rfftn(a, (-1))) * th.fft.rfftn(b, (-1)), (-1))
