import torch
from parser import parameter_parser
from clustering import ClusteringMachine, ClusteringMachine4ea, AlignmentBatch
from clustergcn import ClusterGCNTrainer, ClusterGCN4ea
from utils import tab_printer, graph_reader, feature_reader, target_EAreader
from args_handler import check_args, load_args
import sys
import numpy as np
import pickle
import joblib
import pandas as pd
from load.kgs import read_kgs_from_folder
from largeEA.dataset import *
from largeEA.eval import *
from Dual_AMN.utils import *
import collections
import csv
import os

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def kg_dict_small(kgs):
    kgs_dict = {}
    for e in kgs.kg1.entities_list:
        kgs_dict[e] = 1
    for e in kgs.kg2.entities_list:
        kgs_dict[e] = 2
    return kgs_dict


def kg_dict_large(kgs):
    kgs_dict = {}
    for k, v in kgs.ent1.items():
        kgs_dict[v] = 1
    for k, v in kgs.ent2.items():
        kgs_dict[v] = 2
    return kgs_dict


def generate_relation_triple_dict(self):  # collect triplet, given h, its r and t; given t, its h and r.
    self.rt_dict, self.hr_dict = dict(), dict()
    for h, r, t in self.local_relation_triples_list:
        rt_set = self.rt_dict.get(h, set())
        rt_set.add((r, t))
        self.rt_dict[h] = rt_set
        hr_set = self.hr_dict.get(t, set())
        hr_set.add((h, r))
        self.hr_dict[t] = hr_set
    print("Number of rt_dict:", len(self.rt_dict))
    print("Number of hr_dict:", len(self.hr_dict))


def place_triplets(triplets, nodes_batch):
    batch = collections.defaultdict(list)
    node2batch = {}
    batch_triplets = []
    for i, cluster in enumerate(nodes_batch):
        for n in nodes_batch[cluster]:
            node2batch[n] = i
    removed = 0
    for h, r, t in triplets:
        h_batch, t_batch = node2batch.get(h, -1), node2batch.get(t, -1)
        if h_batch == t_batch and h_batch >= 0:
            batch[h_batch].append((h, r, t))
            batch_triplets.append((h, r, t))
        else:
            removed += 1
    print('split triplets complete, total {} triplets removed'.format(removed))

    return batch_triplets, batch, removed


def rearrange_ids(nodes, merge: bool, *to_map):
    ent_mappings = [{}, {}]
    rel_mappings = [{}, {}]
    ent_ids = [[], []]
    shift = 0
    for w, node_set in enumerate(nodes):
        for n in node_set:
            ent_mappings[w], nn, shift = add_cnt_for(ent_mappings[w], n, shift)
            ent_ids[w].append(nn)
        shift = len(ent_ids[w]) if merge else 0
    mapped = []
    shift = 0
    curr = 0
    for i, need in enumerate(to_map):
        now = []
        if len(need) == 0:
            mapped.append([])
            continue
        is_triple = len(need[0]) == 3
        for tu in need:
            if is_triple:
                h, t = ent_mappings[curr][tu[0]], ent_mappings[curr][tu[-1]]
                rel_mappings[curr], r, shift = add_cnt_for(rel_mappings[curr], tu[1], shift)
                now.append((h, r, t))
            else:
                now.append((ent_mappings[0][tu[0]], ent_mappings[1][tu[-1]]))
        mapped.append(now)
        curr += is_triple
        if not merge:
            shift = 0
    rel_ids = [list(rm.values()) for rm in rel_mappings]

    return ent_mappings, rel_mappings, ent_ids, rel_ids, mapped


def ill(pairs, device='cuda'):
    return torch.tensor(pairs, dtype=torch.long, device=device).t()


def main():
    """
    Parsing command line parameters, reading data, graph decomposition, fitting a ClusterGCN and scoring the model.
    """
    # args = parameter_parser()
    args = load_args(sys.argv[1])
    args.training_data = args.training_data + sys.argv[2] + '/'
    args.dataset_division = sys.argv[3]
    torch.manual_seed(args.seed)
    tab_printer(args)
    if args.data_size == 'small' or args.data_size == 'medium':
        d = OpenEAData('/home/uqkxin/OpenEA/run/datasets/EN_FR_15K_V1/', unsup=False)
        kgs = read_kgs_from_folder(args.training_data, args.dataset_division, args.alignment_module, args.ordered,
                                   remove_unlinked=False)
        # joblib.dump(kgs, 'kgs_fr15k')
        kgs.ent_dict = kg_dict_small(kgs)
        kgs.size = [kgs.kg1.entities_num, kgs.kg2.entities_num]
        kgs.ent1, kgs.ent2, kgs.ents = d.ent1, d.ent2, d.ents
        kgs.link = d.link
        kgs.rel1, kgs.rel2, kgs.rels = d.rel1, d.rel2, d.rels
        # kgs.test, kgs.train = d.test, d.train
        kgs.train_cnt, kgs.unsup = d.train_cnt, d.unsup
        kgs.triples1, kgs.triples2, kgs.triples = d.triple1, d.triple2, d.triples
        id_mapping1, id_mapping2 = {}, {}
        for e in kgs.ent1:
            id_mapping1[kgs.ent1[e]] = kgs.kg1.entities_id_dict[e]
        for e in kgs.ent2:
            id_mapping2[kgs.ent2[e]] = kgs.kg2.entities_id_dict[e]
        kgs.id_mapping = [id_mapping1, id_mapping2]
        kgs.id_mapping_inv = [{v: k for k, v in id_mapping1.items()}, {v: k for k, v in id_mapping2.items()}]
        kgs.train = np.array(
            [(kgs.id_mapping_inv[0][i[0]], kgs.id_mapping_inv[1][i[1]]) for i in kgs.train_links + kgs.valid_links])
        kgs.test = [(kgs.id_mapping_inv[0][i[0]], kgs.id_mapping_inv[1][i[1]]) for i in kgs.test_links]
    else:
        kgs = LargeScaleEAData('/home/uqkxin/LargeEA/mkdata/', 'fr', False, unsup=False)
        kgs.generate_relation_triple_dict()
        kgs.ent_dict = kg_dict_large(kgs)
        kgs.size = kgs.size()

    if not os.path.isfile(args.edge_path + 'edges.csv'):
        if args.data_size == 'small' or 'medium':
            target = {}
            for l in kgs.train_links + kgs.valid_links:
                target[l[1]] = l[0]
            with open(args.edge_path + 'edges.csv', 'a') as f:
                writer = csv.writer(f)
                writer.writerow(['id1', 'id2'])
                i = 0
                for t in kgs.kg1.relation_triples_list:
                    writer.writerow([t[0], t[2]])
                    i += 1
                for t in kgs.kg2.relation_triples_list:
                    tri = []
                    if target.get(t[0], None):
                        tri.append(target[t[0]])
                    else:
                        tri.append(t[0])
                    if target.get(t[2], None):
                        tri.append(target[t[2]])
                    else:
                        tri.append(t[2])
                    writer.writerow(tri)
                    i += 1
                print('kg edges number: ' + str(i))
        else:
            target = {}
            for l in d.train:
                target[l[1]] = l[0]
            with open(directory + 'edges.csv', 'a') as f:
                writer = csv.writer(f)
                writer.writerow(['id1', 'id2'])
                i = 0
                for t in d.triple1:
                    edge = []
                    if t[0] > len(d.ent2) - 1:
                        edge.append(t[0] + len(d.ent2))
                    else:
                        edge.append(2 * t[0])
                    if t[2] > len(d.ent2) - 1:
                        edge.append(t[2] + len(d.ent2))
                    else:
                        edge.append(2 * t[2])
                    writer.writerow(edge)
                    i += 1
                for t in d.triple2:
                    triple = []
                    if target.get(t[0]):
                        if target[t[0]] > len(d.ent2) - 1:
                            triple.append(len(d.ent2) + target[t[0]])
                        else:
                            triple.append(2 * target[t[0]])
                    else:
                        triple.append(2 * t[0] + 1)
                    if target.get(t[2]):
                        if target[t[2]] > len(d.ent2) - 1:
                            triple.append(len(d.ent2) + target[t[2]])
                        else:
                            triple.append(2 * target[t[2]])
                    else:
                        triple.append(2 * t[2] + 1)
                    writer.writerow(triple)
                    i += 1
                print('kg edges number: ' + str(i))
    # graph1 = graph_reader(args.edge_path + 'edges1.csv')
    # graph2 = graph_reader(args.edge_path + 'edges2.csv')
    graph = graph_reader(args.edge_path + 'edges.csv')
    # joblib.dump(graph, 'graph_15k')
    # node_size1 = len(graph1._node)
    # node_size2 = len(graph2._node)
    # features = feature_reader(args.features_path)
    # features = np.random.rand(node_size1 + node_size2, args.hidden_dim)
    # embedding_weight = torch.zeros((ent_num_sr + ent_num_tg, dim), dtype=torch.float)
    # nn.init.xavier_uniform_(embedding_weight)
    # self.feats_sr = nn.Parameter(embedding_weight[:ent_num_sr], requires_grad=True)
    # target = target_EAreader(args.target_path)
    target = kgs.train_links + kgs.valid_links
    clustering_machine = ClusteringMachine4ea(args, graph, target, kgs)
    clustering_machine.largeEA_partition(src_k=args.cluster_number, trg_k=args.cluster_number)
    # clustering_machine.decompose()
    # clustering_machine.split_clusters()
    clustering_machine.fill_cluster()
    clustering_machine.cluster2batch()
    # clustering_machine.duplicate_neighbours()
    # clustering_machine.align_clusters()
    model = ClusterGCN4ea(args, clustering_machine)
    model.run()
    # triple1_batch, batch1, removed1 = place_triplets(kgs.kg1.relation_triples_list,
    #                                                  clustering_machine.kg1_clusters)
    # triple2_batch, batch2, removed2 = place_triplets(kgs.kg2.relation_triples_list,
    #                                                  clustering_machine.kg2_clusters)
    # if args.data_size == 'small' or 'medium':
    #     curr_sim = None
    #     for i in clustering_machine.clusters:
    #         align_cluster = AlignmentBatch(batch1[i], batch2[i],
    #                        clustering_machine.kg1_clusters[i], clustering_machine.kg2_clusters[i], clustering_machine.sg_train_nodes[i],
    #                        clustering_machine.sg_test_nodes[i])
    #         model = ClusterGCN4ea(args, clustering_machine, align_cluster)
    #         model.run()
    #     sim = align_cluster.get_sim_mat(batch_emb, kgs.size)
    #     curr_sim = sim if curr_sim is None else curr_sim + sim
    # result = sparse_acc(curr_sim, ill(kgs.test_links, 'cpu'))
    # print('acc is', result)

    # gcn_trainer = ClusterGCN4ea(args, clustering_machine)
    # gcn_trainer.train()
    # gcn_trainer.test()


if __name__ == "__main__":
    main()
    # clustering_machine = joblib.load('cluster_machine_duplicate')

    # cluster analysis
    # target = np.array(pd.read_csv('/home/uqkxin/ClusterGCN/openea/enfr_15k/train_links.csv'))
    # target_dict = {i[0]:i[1] for i in list(target)}
    # for k in clustering_machine.sg_nodes:
    #     cluster = list(clustering_machine.sg_nodes[k].numpy())
    #     kg2 = 0
    #     for node in cluster:
    #         if node % 2 == 1 and node < 2*1365118:
    #             kg2 += 1
    #     ratio = kg2 / len(cluster)
    #     print('kg2 node ratio: ' + str(ratio))
    #
    # for k in clustering_machine.sg_nodes:
    #     cluster = list(clustering_machine.sg_nodes[k].numpy())
    #     seed = 0
    #     for node in cluster:
    #         if target_dict.get(node):
    #             seed += 1
    #     print('seed in this cluster: ' + str(seed))
