import os
import torch


def load_all_entities(datapath, datasets):
    client_entities_dict = dict()
    for client_dir in datasets:
        client_seq = int(client_dir)
        client_entities = []
        with open(os.path.join(datapath, client_dir, "entities.dict"), "r", encoding="utf-8") as fin:
            for line in fin.readlines():
                _, label = line.strip().split()
                client_entities.append(label)
        client_entities_dict[client_seq] = client_entities
    all_entities = []
    for client_seq in client_entities_dict.keys():
        all_entities.extend(client_entities_dict[client_seq])
    all_entities = list(set(all_entities))
    nentity = len(all_entities)
    client_entities_mapping = dict()
    for client_seq in client_entities_dict.keys():
        client_entities = client_entities_dict[client_seq]
        client_entities_mapping[client_seq] = [all_entities.index(client_entity) for client_entity in client_entities]
    return client_entities_dict, client_entities_mapping, nentity


def read_triples(file_path):
    triples = []
    with open(file_path, "r", encoding="utf-8") as fin:
        for line in fin.readlines():
            h, r, t = line.strip().split()
            triples.append((h, r, t))
    return triples


def calculate_weighted(*results_list):
    tensor = [torch.Tensor(li) for li in results_list]
    result = [round(((sum(li * tensor[-1])) / (sum(tensor[-1]))).item(), 5) for li in tensor]
    result[-1] = results_list[-1]
    return result


def calculate_and_round(*results_list):
    return [round(sum(li) / len(li), 5) for li in results_list]


def in_out_norm(graph):
    src, dst, EID = graph.edges(form="all")
    graph.edata["norm"] = torch.ones(EID.shape[0]).to(graph.device)
    in_edges_idx = torch.nonzero(graph.edata["in_edges_mask"], as_tuple=False).squeeze()
    out_edges_idx = torch.nonzero(graph.edata["out_edges_mask"], as_tuple=False).squeeze()

    for idx in [in_edges_idx, out_edges_idx]:
        u, v = src[idx], dst[idx]
        deg = torch.zeros(graph.num_nodes()).to(graph.device)
        n_idx, inverse_index, count = torch.unique(v, return_inverse=True, return_counts=True)
        deg[n_idx] = count.float()
        deg_inv = deg.pow(-0.5)
        deg_inv[deg_inv == float("inf")] = 0
        norm = deg_inv[u] * deg_inv[v]
        graph.edata["norm"][idx] = norm
    graph.edata["norm"] = graph.edata["norm"].unsqueeze(1)

    return graph
