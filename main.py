import argparse
import random
import time
import numpy as np
from client import Client
from server import Server
from utils import calculate_and_round, calculate_weighted
import os
import dgl
import torch
from model import UniFE, FeatureModule
from dataloader import UniFEData
from utils import load_all_entities, read_triples, in_out_norm


def init_data(args, datapath, group='C3FL'):
    datasets = [str(ds) for ds in args.use_ds]
    group_path = os.path.join(datapath, group)
    all_data = {}
    server_entities_dict, server_entities_mapping, all_nentity = load_all_entities(group_path, datasets)
    all_data['server_param'] = (server_entities_dict, server_entities_mapping, all_nentity)

    for data in datasets:
        server_data_path = os.path.join(group_path, data)
        train_path = os.path.join(server_data_path, "train.txt")
        traindata = read_triples(train_path)
        valid_path = os.path.join(server_data_path, "valid.txt")
        validdata = read_triples(valid_path)
        test_path = os.path.join(server_data_path, "test.txt")
        testdata = read_triples(test_path)

        client_data = UniFEData(args, traindata, validdata, testdata, args.local_batch_size, data, args.num_workers)
        dataloaders = client_data.dataloaders
        graph = client_data.g.to(args.device)
        num_rel = torch.max(graph.edata["etype"]).item() + 1
        graph = in_out_norm(graph)
        basis_edges = graph.edata['basis']
        rel_degree = torch.argmax(basis_edges, -1)
        bsg = dgl.graph((graph.edges()[0], graph.edges()[1]), num_nodes=graph.num_nodes())
        bsg.edata["etype"] = rel_degree
        bsg.edata["in_edges_mask"] = graph.edata["in_edges_mask"]
        bsg.edata["out_edges_mask"] = graph.edata["out_edges_mask"]
        bsg.edata["etype"] = rel_degree
        bsg = in_out_norm(bsg)
        print("train size: " + args.train_size_type)
        basis_size = torch.sum(basis_edges, dim=0)
        basis_size[basis_size <= 0] = 0
        if args.train_size_type == 'binary':
            basis_size[basis_size > 0] = 1
        all_data[data] = ({'dataloaders': dataloaders, 'kg': graph, 'bsg': bsg},
                          {'basis_size': basis_size, "sr_size": client_data.sr_size},
                          graph.num_nodes(), num_rel)
    return all_data


def init_federation(all_data, args):
    idx_clients = {}
    clients = []
    server_entities_dict, server_entities_mapping, all_nentity = all_data['server_param']
    all_data.pop('server_param', None)
    for ds in all_data.keys():
        idx = int(ds)
        client_data, train_size, nentity, nrelation = all_data[ds]
        if args.use_structure:
            cmodel_gc = UniFE(args, num_ent=nentity, num_rel=nrelation, batchnorm=True)
        else:
            cmodel_gc = FeatureModule(args, num_ent=nentity, num_rel=nrelation, batchnorm=True)

        optimizer = torch.optim.Adam(cmodel_gc.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        clients.append(Client(args, cmodel_gc, idx, ds, train_size, client_data['dataloaders'], optimizer, client_data['kg'], client_data['bsg']))
    if args.use_structure:
        smodel = UniFE(args, num_ent=nentity, num_rel=nrelation, batchnorm=True)
    else:
        smodel = FeatureModule(args, num_ent=nentity, num_rel=nrelation, batchnorm=True, )
    server = Server(smodel, all_nentity, server_entities_dict, server_entities_mapping, args.ent_dim, args.rel_dim, args.score_func, args.device)
    return clients, server, idx_clients


def train_pipeline(args, clients, server, rounds, local_epoch, frac=1.0):
    for client in clients:
        client.download_struct_params(server)
        if args.nembds_mode == 'agg' or args.nembds_mode == 'common':
            entity_embedding_dict = server.assign_embeddings()
            client.download_nembds(client.id, server, entity_embedding_dict)
    best_mrr = 0.
    best_round = 1
    early_stop = 0
    for c_round in range(1, rounds + 1):
        print(f"Round {c_round}")
        client_embedding_dict = dict()
        for client in clients:
            client.local_train(local_epoch)
            if args.nembds_mode == 'agg':
                client_embedding_dict[client.id] = client.get_entity_embeddings()
        if args.nembds_mode == 'agg':
            server.aggregate_feat_embeddings(client_embedding_dict)
            entity_embedding_dict = server.assign_embeddings()
        server.aggregate_struct_params(clients)

        for client in clients:
            client.download_struct_params(server)
            if args.nembds_mode == 'agg':
                client.download_nembds(client.id, server, entity_embedding_dict)

        if c_round % args.eval_rounds == 0 or c_round == 1:
            v_mrr_list = []
            v_count_list = []
            for idx in range(len(clients)):
                v_results, v_acc, = clients[idx].evaluate_unife('valid')
                v_mrr_list.append(v_results['mrr'])
                v_count_list.append(v_results['count'])
            if args.agg == 'weighted':
                v_mrr, v_count = calculate_weighted(v_mrr_list, v_count_list)
            else:
                v_mrr = calculate_and_round(v_mrr_list)
            if v_mrr < best_mrr:
                early_stop += 1
                print("[Bad Iter]: {},\tBest MRR: {},\t MRR: {}".format(early_stop, best_mrr, v_mrr))
            else:
                early_stop = 0
                best_round = c_round
                for client in clients:
                    client.save_model(best_round)
                print("Best MRR: {},\t MRR: {}".format(best_mrr, v_mrr))
                best_mrr = v_mrr
            if early_stop >= args.early_stop:
                break
    test_mrr_list = []
    test_h10_list = []
    test_h3_list = []
    test_h1_list = []
    test_count_list = []
    for client in clients:
        client.load_model(best_round)
        main, others = client.evaluate_unife('test')
        print('Test Epoch', best_round, main)
        test_mrr_list.append(main['mrr'])
        test_h10_list.append(main['hits@10'])
        test_h3_list.append(main['hits@3'])
        test_h1_list.append(main['hits@1'])
        test_count_list.append(main['count'])
    if args.agg == 'weighted':
        test_mrr, test_h10, test_h3, test_h1, t_count = calculate_weighted(test_mrr_list, test_h10_list, test_h3_list, test_h1_list, test_count_list)
    else:
        test_mrr, test_h10, test_h3, test_h1 = calculate_and_round(test_mrr_list, test_h10_list, test_h3_list, test_h1_list)
    print("Best Round:{},\tTest MRR: {},\tTest H@10: {},\tTest H@3: {},\tTest H@1: {}".format(best_round, test_mrr, test_h10, test_h3, test_h1))
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--setting', type=str, default='unife')
    parser.add_argument('--num_rounds', type=int, default=200)
    parser.add_argument('--local_epoch', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--datapath', type=str, default='./data')
    parser.add_argument('--datasets', type=str, default='C3FL')
    parser.add_argument('--n_basis', type=int, default=50)
    parser.add_argument('--basis_dim', type=int, default=50)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--ent_dim", type=int, default=128)
    parser.add_argument("--rel_dim", type=int, default=128)
    parser.add_argument("--num_rel_base", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--local_batch_size", default=1024, type=int)
    parser.add_argument("--use_structure", action='store_true')
    parser.add_argument("--concat_dim", default=64, type=int)
    parser.add_argument("--use_ds", nargs="?", default="[-1]")
    parser.add_argument("--eval_rounds", default=5, type=int)
    parser.add_argument("--early_stop", default=3, type=int)
    parser.add_argument("--nembds_mode", default='single', type=str, choices=['common', 'agg', 'single'])
    parser.add_argument("--train_size_type", default='sum', type=str, choices=['sum', 'binary'])
    parser.add_argument("--agg", default='weighted', type=str)
    parser.add_argument("--score_func", dest="score_func", default="conve")
    parser.add_argument("--opn", default="ccorr", choices=['sub', 'mul', 'ccorr'])
    parser.add_argument("--lbl_smooth", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_bases", default=-1, type=int)
    parser.add_argument("--init_dim", default=128, type=int)
    parser.add_argument("--layer_size", nargs="?", default="[64]")
    parser.add_argument("--gcn_drop", dest="dropout", default=0.1, type=float)
    parser.add_argument("--layer_dropout", nargs="?", default="[0.3]")
    parser.add_argument("--hid_drop", default=0.3, type=float, help="ConvE: Hidden dropout")
    parser.add_argument("--feat_drop", default=0.3, type=float, help="ConvE: Feature Dropout")
    parser.add_argument("--k_w", default=8, type=int, help="ConvE: k_w")
    parser.add_argument("--k_h", default=16, type=int, help="ConvE: k_h")
    parser.add_argument("--num_filt", default=64, type=int, help="ConvE: Number of filters in convolution")
    parser.add_argument("--ker_sz", default=7, type=int, help="ConvE: Kernel size")
    parser.add_argument('--pt_dir', default='./pt', type=str)

    try:
        args = parser.parse_args()
    except IOError as msg:
        parser.error(str(msg))

    if args.use_ds == '[-1]':
        if args.datasets == 'C3FL' or args.datasets == 'NELL3':
            args.use_ds = '[0,1,2]'
        elif args.datasets == 'C5FL':
            args.use_ds = '[0,1,2,3,4]'
        elif args.datasets == 'C10FL':
            args.use_ds = '[0,1,2,3,4,5,6,7,8,9]'

    args.layer_size, args.layer_dropout, args.use_ds = eval(args.layer_size), eval(args.layer_dropout), eval(args.use_ds)
    args.pt_dir = os.path.join(args.pt_dir, args.datasets + '_' + args.score_func)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    args.init_dim = args.dim
    args.ent_dim = args.dim
    args.rel_dim = args.dim
    args.concat_dim = args.dim
    args.num_filt = args.dim
    args.layer_size = [args.dim]

    if args.score_func == 'complex':
        args.ent_dim = args.ent_dim * 2
        args.rel_dim = args.rel_dim * 2
    if not os.path.exists(args.pt_dir):
        os.makedirs(args.pt_dir)

    print("Init data")
    all_data = init_data(args, args.datapath, args.datasets)
    print("Done")
    print("Init federation")
    init_clients, init_server, init_idx_clients = init_federation(all_data, args)
    print("Done.")

    start_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    print("Start Train Time: ", start_time)
    train_pipeline(args, init_clients, init_server, args.num_rounds, args.local_epoch)
    end_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    print("End Train Time: ", end_time)
