from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import dgl
import numpy as np
import torch
from ordered_set import OrderedSet
from collections import defaultdict as ddict
from torch.utils.data import Dataset, DataLoader


class UniFEData(object):
    def __init__(self, args, traindata, validdata, testdata, batch_size, data_idx, num_workers):
        self.data = {'train': traindata, 'valid': validdata, 'test': traindata}
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.args = args
        ent_set, rel_set = OrderedSet(), OrderedSet()
        for split in [traindata, testdata, validdata]:
            for triple in split:
                sub, rel, obj = triple
                ent_set.add(sub)
                rel_set.add(rel)
                ent_set.add(obj)

        self.ent2id = {ent: idx for idx, ent in enumerate(ent_set)}
        self.rel2id = {rel: idx for idx, rel in enumerate(rel_set)}
        self.rel2id.update({str(rel) + "_reverse": idx + len(self.rel2id)for idx, rel in enumerate(rel_set)})

        self.id2ent = {idx: ent for ent, idx in self.ent2id.items()}
        self.id2rel = {idx: rel for rel, idx in self.rel2id.items()}

        self.num_ent = len(self.ent2id)
        self.num_rel = len(self.rel2id) // 2

        self.data = ddict(list)
        sr2o = ddict(set)
        src,dst,rels = [],[],[]
        inver_src, inver_dst, inver_rels = [], [], []
        split_name = {0: 'train', 1: 'valid', 2: 'test'}

        for idx, split in enumerate([traindata, testdata, validdata]):
            for triple in split:
                sub, rel, obj = triple
                sub_id, rel_id, obj_id = (
                    self.ent2id[sub],
                    self.rel2id[rel],
                    self.ent2id[obj],
                )
                self.data[split_name[idx]].append((sub_id, rel_id, obj_id))

                if split_name[idx] == 'train':
                    sr2o[(sub_id, rel_id)].add(obj_id)
                    sr2o[(obj_id, rel_id + self.num_rel)].add(sub_id)
                    src.append(sub_id)
                    dst.append(obj_id)
                    rels.append(rel_id)
                    inver_src.append(obj_id)
                    inver_dst.append(sub_id)
                    inver_rels.append(rel_id + self.num_rel)

        src = src + inver_src
        dst = dst + inver_dst
        rels = rels + inver_rels
        self.g = dgl.graph((src, dst), num_nodes=self.num_ent)
        self.g.edata["etype"] = torch.Tensor(rels).long()

        in_edges_mask = [True] * (self.g.num_edges() // 2) + [False] * (self.g.num_edges() // 2 )
        out_edges_mask = [False] * (self.g.num_edges() // 2) + [True] * (self.g.num_edges() // 2)
        self.g.edata["in_edges_mask"] = torch.Tensor(in_edges_mask)
        self.g.edata["out_edges_mask"] = torch.Tensor(out_edges_mask)

        self.g = self.BSG_construction(args.n_basis, self.g, {k: list(v) for k, v in sr2o.items()})

        self.data = dict(self.data)
        self.sr2o = {k: list(v) for k, v in sr2o.items()}

        for split in [2, 1]:
            for sub, rel, obj in self.data[split_name[split]]:
                sr2o[(sub, rel)].add(obj)
                sr2o[(obj, rel + self.num_rel)].add(sub)

        self.sr2o_all = {k: list(v) for k, v in sr2o.items()}
        self.triples = ddict(list)
        for (sub, rel), obj in self.sr2o.items():
            self.triples["train"].append({"triple": (sub, rel, -1), "label": self.sr2o[(sub, rel)]})

        for split in [2, 1]:
            for sub, rel, obj in self.data[split_name[split]]:
                rel_inv = rel + self.num_rel
                self.triples["{}_{}".format(split_name[split], "tail")].append({    "triple": (sub, rel, obj),    "label": self.sr2o_all[(sub, rel)],})
                self.triples["{}_{}".format(split_name[split], "head")].append( {     "triple": (obj, rel_inv, sub),     "label": self.sr2o_all[(obj, rel_inv)], } )

        self.triples = dict(self.triples)

        def get_train_dataloader(split, batch_size, lbl_smooth, shuffle=True):
            return DataLoader(
                TrainDataset(self.triples[split], self.num_ent, lbl_smooth),
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=self.args.num_workers,
                collate_fn=TrainDataset.collate_fn,
            )

        def get_test_dataloader(split, batch_size, shuffle=True):
            return DataLoader(TestDataset(self.triples[split], self.num_ent),
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=self.args.num_workers,
                collate_fn=TestDataset.collate_fn,
            )

        self.sr_size = len(self.sr2o)

        self.dataloaders = {
            "train": get_train_dataloader("train", self.batch_size, args.lbl_smooth),
            "valid_head": get_test_dataloader("valid_head", self.batch_size),
            "valid_tail": get_test_dataloader("valid_tail", self.batch_size),
            "test_head": get_test_dataloader("test_head", self.batch_size),
            "test_tail": get_test_dataloader("test_tail", self.batch_size),
        }

    def BSG_construction(self, n_basis, ori_g, sr2o):
        basis_edges = torch.zeros([ori_g.num_edges(), n_basis])
        src_nodes = ori_g.edges()[0].numpy().tolist()
        rel = ori_g.edata["etype"].numpy().tolist()
        for i, (n, r) in enumerate(zip(src_nodes, rel)):
            basis_edges[i, max(1, min(len(sr2o[(n, r)]), n_basis)) - 1] = 1
        ori_g.edata['basis'] = basis_edges

        return ori_g


class TrainDataset(Dataset):
    def __init__(self, triples, num_ent, lbl_smooth):
        self.triples = triples
        self.num_ent = num_ent
        self.lbl_smooth = lbl_smooth
        self.entities = np.arange(self.num_ent, dtype=np.int32)

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        ele = self.triples[idx]
        triple, label = torch.LongTensor(ele["triple"]), np.int32(ele["label"])
        trp_label = self.get_label(label)
        if self.lbl_smooth != 0.0:
            trp_label = (1.0 - self.lbl_smooth) * trp_label + (
                    1.0 / self.num_ent
            )

        return triple, trp_label

    @staticmethod
    def collate_fn(data):
        triples = []
        labels = []
        for triple, label in data:
            triples.append(triple)
            labels.append(label)
        triple = torch.stack(triples, dim=0)
        trp_label = torch.stack(labels, dim=0)
        return triple, trp_label

    def get_label(self, label):
        y = np.zeros([self.num_ent], dtype=np.float32)
        for e2 in label:
            y[e2] = 1.0
        return torch.FloatTensor(y)


class TestDataset(Dataset):
    def __init__(self, triples, num_ent):
        self.triples = triples
        self.num_ent = num_ent

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        ele = self.triples[idx]
        triple, label = torch.LongTensor(ele["triple"]), np.int32(ele["label"])
        label = self.get_label(label)

        return triple, label

    @staticmethod
    def collate_fn(data):
        triples = []
        labels = []
        for triple, label in data:
            triples.append(triple)
            labels.append(label)
        triple = torch.stack(triples, dim=0)
        label = torch.stack(labels, dim=0)
        return triple, label

    def get_label(self, label):
        y = np.zeros([self.num_ent], dtype=np.float32)
        for e2 in label:
            y[e2] = 1.0
        return torch.FloatTensor(y)
