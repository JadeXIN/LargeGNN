import random
import networkx as nx
from sklearn.model_selection import train_test_split
from largeEA.utils import *
import largeEA.text_utils as text_utils
import collections
import nxmetis
import itertools
import heapq, operator
import numpy as np
from queue import Queue
import math
import ray
import os


def dfs(node, threshold, dic, kg, dist):
    score = 1 / (0.001 + dist)
    if score <= threshold:
        return
    else:
        if dic[node] > score:
            return
        else:
            dic[node] = max(dic[node], score)
    dist += 1
    if 1 / (0.001 + dist) <= threshold:
        return
    neighbors1 = [i[1] for i in kg.rt_dict.get(node, set())]
    neighbors2 = [i[0] for i in kg.hr_dict.get(node, set())]
    for neighbour in set(neighbors1 + neighbors2):
        dfs(neighbour, threshold, dic, kg, dist=dist)


@ray.remote
def node_importance_dfs_ray(node_set, seed_threshold, importance_dict, kg):
    for node in node_set:
        hop = 0
        dfs(node, seed_threshold, importance_dict, kg, hop)
    return importance_dict


def make_assoc(maps, src_len, trg_len, merge):
    assoc = np.empty(src_len + trg_len, dtype=np.int)
    shift = 0 if merge else 1
    shift = shift * src_len
    for idx, ent_mp in enumerate(maps):
        for k, v in ent_mp.items():
            assoc[v + idx * shift] = k
    return torch.tensor(assoc)


def rearrange_ids(nodes, merge: bool, *to_map):  # todo: remove rel map, keep original rel id
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

# def place_triplets(triplets, nodes_batch):
#     batch = collections.defaultdict(list)
#     node2batch = {}
#     batch_triplets = []
#     for i, cluster in enumerate(nodes_batch):
#         for n in nodes_batch[cluster]:
#             node2batch[n] = i
#     removed = 0
#     for h, r, t in triplets:
#         h_batch, t_batch = node2batch.get(h, -1), node2batch.get(t, -1)
#         if h_batch == t_batch and h_batch >= 0:
#             batch[h_batch].append((h, r, t))
#             batch_triplets.append((h, r, t))
#         else:
#             removed += 1
#     print('split triplets complete, total {} triplets removed'.format(removed))
#
#     return batch_triplets, batch, removed

def place_triplets(triplets, nodes_batch): # after divide the nodes, place the triples!!
    batch = collections.defaultdict(list)
    node2batch = {}
    for i, nodes in enumerate(nodes_batch):
        for n in nodes:
            node2batch[n] = i
    removed = 0
    for h, r, t in triplets:
        h_batch, t_batch = node2batch.get(h, -1), node2batch.get(t, -1)
        if h_batch == t_batch and h_batch >= 0:
            batch[h_batch].append((h, r, t))
        else:
            removed += 1
    print('split triplets complete, total {} triplets removed'.format(removed))

    return batch, removed

def share_triplets(src_triplet, trg_triplet, train_set, node_mapping, rel_mapping=None):
    if rel_mapping is None:
        rel_mapping = lambda x: x

    new_trg = []

    print('share triplet')  # parameter swapping
    for triplet in src_triplet:
        h, r, t = triplet
        if h in train_set and t in train_set:
            new_trg.append([node_mapping[h], rel_mapping(r), node_mapping[t]])

    return trg_triplet + new_trg

def overlaps(src: List[set], trg: List[set]):
    return np.array([[float(len(s.intersection(t))) / (float(len(s)) + 0.01) for t in trg] for s in src])

def filter_sim_mat(mat, threshold, greater=True, equal=False):
    if greater and equal:
        x, y = np.where(mat >= threshold)
    elif greater and not equal:
        x, y = np.where(mat > threshold)
    elif not greater and equal:
        x, y = np.where(mat <= threshold)
    else:
        x, y = np.where(mat < threshold)
    return set(zip(x, y))

def search_nearest_k(sim_mat, k):
    assert k > 0
    neighbors = set()
    num = sim_mat.shape[0]
    for i in range(num):
        rank = np.argpartition(-sim_mat[i, :], k)  # heap sort: top 50 is rank first, but not gurantee order.
        pairs = [j for j in itertools.product([i], rank[0:k])]
        neighbors |= set(pairs)
        # del rank
    assert len(neighbors) == num * k
    return neighbors

def exchange_xy(xy):
    return set([(y, x) for x, y in xy])

def galeshapley(suitor_pref_dict, reviewer_pref_dict, max_iteration):
    """ The Gale-Shapley algorithm. This is known to provide a unique, stable
    suitor-optimal matching. The algorithm is as follows:
    (1) Assign all suitors and reviewers to be unmatched.
    (2) Take any unmatched suitor, s, and their most preferred reviewer, r.
            - If r is unmatched, match s to r.
            - Else, if r is matched, consider their current partner, r_partner.
                - If r prefers s to r_partner, unmatch r_partner from r and
                  match s to r.
                - Else, leave s unmatched and remove r from their preference
                  list.
    (3) Go to (2) until all suitors are matched, then end.
    Parameters
    ----------
    suitor_pref_dict : dict
        A dictionary with suitors as keys and their respective preference lists
        as values
    reviewer_pref_dict : dict
        A dictionary with reviewers as keys and their respective preference
        lists as values
    max_iteration : int
        An integer as the maximum iterations
    Returns
    -------
    matching : dict
        The suitor-optimal (stable) matching with suitors as keys and the
        reviewer they are matched with as values
    """
    suitors = list(suitor_pref_dict.keys())
    matching = dict()
    rev_matching = dict()

    for i in range(max_iteration):
        if len(suitors) <= 0:
            break
        for s in suitors:
            if len(suitor_pref_dict[s]) == 0:
                continue
            r = suitor_pref_dict[s][0]
            if r not in matching.values():
                matching[s] = r
                rev_matching[r] = s
            else:
                r_partner = rev_matching.get(r)
                if reviewer_pref_dict[r].index(s) < reviewer_pref_dict[r].index(r_partner):
                    del matching[r_partner]
                    matching[s] = r
                    rev_matching[r] = s
                else:
                    suitor_pref_dict[s].remove(r)
        suitors = list(set(suitor_pref_dict.keys()) - set(matching.keys()))
    return matching

def make_pairs(src, trg, mp):
    return list(filter(lambda p: p[1] in trg, [(e, mp[e]) for e in set(filter(lambda x: x in mp, src))]))

def stable_matching(sim_mat, sim_th, k, cut=1000):
    t = time.time()

    kg1_candidates, kg2_candidates = dict(), dict()

    potential_aligned_pairs = filter_sim_mat(sim_mat, sim_th)
    if len(potential_aligned_pairs) == 0:
        return None
    if k <= 0:
        return potential_aligned_pairs
    nearest_k_neighbors1 = search_nearest_k(sim_mat, k)
    nearest_k_neighbors2 = search_nearest_k(sim_mat.T, k)
    nearest_k_neighbors = nearest_k_neighbors1 | exchange_xy(nearest_k_neighbors2)
    potential_aligned_pairs &= nearest_k_neighbors
    if len(potential_aligned_pairs) == 0:
        return None

    i_candidate = dict()
    i_candidate_sim = dict()
    j_candidate = dict()
    j_candidate_sim = dict()

    for i, j in potential_aligned_pairs:
        i_candidate_list = i_candidate.get(i, list())
        i_candidate_list.append(j)
        i_candidate[i] = i_candidate_list

        i_candidate_sim_list = i_candidate_sim.get(i, list())
        i_candidate_sim_list.append(sim_mat[i, j])
        i_candidate_sim[i] = i_candidate_sim_list

        j_candidate_list = j_candidate.get(j, list())
        j_candidate_list.append(i)
        j_candidate[j] = j_candidate_list

        j_candidate_sim_list = j_candidate_sim.get(j, list())
        j_candidate_sim_list.append(sim_mat[i, j])
        j_candidate_sim[j] = j_candidate_sim_list

    prefix1 = "x_"
    prefix2 = "y_"

    for i, i_candidate_list in i_candidate.items():
        i_candidate_sim_list = np.array(i_candidate_sim.get(i))
        sorts = np.argsort(-i_candidate_sim_list)
        i_sorted_candidate_list = np.array(i_candidate_list)[sorts].tolist()
        x_i = prefix1 + str(i)
        kg1_candidates[x_i] = [prefix2 + str(y) for y in i_sorted_candidate_list]
    for j, j_candidate_list in j_candidate.items():
        j_candidate_sim_list = np.array(j_candidate_sim.get(j))
        sorts = np.argsort(-j_candidate_sim_list)
        j_sorted_candidate_list = np.array(j_candidate_list)[sorts].tolist()
        y_j = prefix2 + str(j)
        kg2_candidates[y_j] = [prefix1 + str(x) for x in j_sorted_candidate_list]

    print("generating candidate lists costs time {:.1f} s ".format(time.time() - t),
          len(kg1_candidates),
          len(kg2_candidates))
    t = time.time()
    matching = galeshapley(kg1_candidates, kg2_candidates, cut)
    new_alignment = set()
    n = 0
    for i, j in matching.items():
        x = int(i.split('_')[-1])
        y = int(j.split('_')[-1])
        new_alignment.add((x, y))
        if x == y:
            n += 1
    cost = time.time() - t
    print("stable matching = {}, precision = {:.3f}%, time = {:.3f} s ".format(len(matching),
                                                                               n / len(matching) * 100, cost))
    return new_alignment

def construct_graph(edges, important_nodes=None, known_weight=1000, known_size=None,
                    cnt_as_weight=False, keep_inter_edges=False):
    g = nx.Graph()
    # edges = [(t[0], t[2]) for t in triples]
    if cnt_as_weight:
        edges = Partition.make_cnt_edges(Partition.make_cnt_edges(edges))
        g.add_weighted_edges_from(edges)
    else:
        g.add_edges_from(edges)
        nx.set_edge_attributes(g, 1, 'weight')
    # g.edges.data('weight', default=1)
    if important_nodes:
        subgraphs = []
        print('set important node weights:')
        for nodes in important_nodes:
            sn = nodes[0]
            g.add_edges_from([(sn, n) for n in nodes])
            subgraph = g.subgraph(nodes)
            if known_size:
                nx.set_node_attributes(subgraph, known_size, 'size')
            nx.set_edge_attributes(subgraph, known_weight, 'weight')
            subgraphs.append(subgraph)
            # subgraphs.append(g.subgraph(nodes))
            # subgraphs[-1].edges.data('weight', default=weight)
        print('compose subgraphs')
        for sg in subgraphs:
            g = nx.compose(g, sg)

        if keep_inter_edges:
            return g

        merged_important_nodes = set()
        for nodes in important_nodes:
            merged_important_nodes.update(nodes)
        # print('all important nodes merged')

        print('del inter edges:')
        for nodes in important_nodes:
            all_neighbors = [g.neighbors(n) for n in nodes]
            # neighbors = set()
            choices = []
            for n in all_neighbors:
                choices.append(merged_important_nodes.intersection(n) - set(nodes))
            edges = [(e1, e2) for idx, e1 in enumerate(nodes) for e2 in choices[idx]]
            g.remove_edges_from(edges)
    return g

def func_r(r_set, arg='second'):
    r_first = [i[0] for i in r_set]
    r_first = list(set(r_first))
    r_second = [i[1] for i in r_set]
    r_second = list(set(r_second))
    if arg == 'first':
        funcr = len(r_first)/len(r_set)
    else:
        funcr = len(r_second)/len(r_set)
    return funcr

def get_relation_dict(relation_triplet):
    relation_dict = {}
    for h, r, t in relation_triplet:
        r_set = relation_dict.get(r, set())
        r_set.add((h, t))
        relation_dict[r] = r_set
    return relation_dict

def gen_partition(corr_ind_1, src_nodes_1, trg_nodes_1, src_train_1, trg_train_1, corr_val_1, mapping_1, triple1_batch_1, triple2_batch_1):
    train_pair_cnt = 0
    test_pair_cnt = 0

    IDs_s_1 = []
    IDs_t_1 = []
    Trains_s_1 = []
    Trains_t_1 = []
    Triples_s_1 = []
    Triples_t_1 = []
    for src_id, src_corr in enumerate(corr_ind_1):
        ids1_1, train1_1 = src_nodes_1[src_id], src_train_1[src_id]
        train2_1, ids2_1, triple2_1 = [], [], []
        corr_rate = 0.
        for trg_rank, trg_id in enumerate(src_corr):
            train2_1 += trg_train_1[trg_id]
            ids2_1 += trg_nodes_1[trg_id]
            triple2_1 += triple2_batch_1[trg_id]
            corr_rate += corr_val_1[src_id][trg_rank]
        ids1_1, ids2_1, train1_1, train2_1 = map(set, [ids1_1, ids2_1, train1_1, train2_1])

        IDs_s_1.append(ids1_1)
        IDs_t_1.append(ids2_1)
        Trains_s_1.append(train1_1)
        Trains_t_1.append(train2_1)
        Triples_s_1.append(set(triple1_batch_1[src_id]))
        Triples_t_1.append(set(triple2_1))

        print('Train corr=', corr_rate)

        train_pairs = make_pairs(train1_1, train2_1, mapping_1)
        train_pair_cnt += len(train_pairs)
        test_pairs = make_pairs(ids1_1, ids2_1, mapping_1)
        test_pair_cnt += len(test_pairs)

    print("*************************************************************")
    print("Total trainig pairs: " + str(train_pair_cnt))
    print("Total testing pairs: " + str(test_pair_cnt - train_pair_cnt))
    print("Total links: " + str(test_pair_cnt))
    print("*************************************************************")

    return IDs_s_1, IDs_t_1, Trains_s_1, Trains_t_1, Triples_s_1, Triples_t_1

class ClusteringMachine4ea(object):
    """
    Clustering the graph, feature set and target.
    """
    def __init__(self, args, graph, target, kgs):
        """
        :param args: Arguments object with parameters.
        :param graph: Networkx Graph.
        :param features: Feature matrix (ndarray).
        :param target: Target vector (ndarray).
        """
        self.args = args
        self.graph = graph
        self.target = target
        self.kgs = kgs
        self.test_pair = kgs.test_links
        self.train_dict = [{i[0]: i[1] for i in self.target}, {i[1]: i[0] for i in self.target}]
        self.relation_dict1 = get_relation_dict(self.kgs.kg1.relation_triples_list)
        self.relation_dict2 = get_relation_dict(self.kgs.kg2.relation_triples_list)


    def decompose(self):
        """
        Decomposing the graph, partitioning the features and target, creating Torch arrays.
        """
        self.clusters = list(range(0, self.args.cluster_number))
        if self.args.clustering_method == "metis":
            print("\nMetis graph clustering started.\n")
            # self.metis_clustering()
            graph_edges = self.graph.edges
            big_g = construct_graph(list(graph_edges))
            min_cut, nodes = nxmetis.partition(big_g, self.args.cluster_number)
            self.sg_nodes = {}
            for cluster in range(0, self.args.cluster_number):
                self.sg_nodes[cluster] = nodes[cluster]
        else:
            print("\nRandom graph clustering started.\n")
            self.random_clustering()
        self.general_data_partitioning()

    def random_clustering(self):
        """
        Random clustering the nodes.
        """
        self.clusters = [cluster for cluster in range(self.args.cluster_number)]
        self.cluster_membership = {node: random.choice(self.clusters) for node in self.graph.nodes()}

    def general_data_partitioning(self):
        """
        Creating data partitions and train-test splits.
        """
        self.sg_edges = {}
        self.sg_train_nodes = {}
        self.sg_test_nodes = {}
        total_link = set()
        all_pairs = self.kgs.train_links + self.kgs.valid_links + self.kgs.test_links
        train_dict = [{i[0]: i[1] for i in self.target}, {i[1]: i[0] for i in self.target}]
        self.mapping = [{i[0]: i[1] for i in all_pairs}, {i[1]: i[0] for i in all_pairs}]
        for cluster in self.clusters:
            self.sg_train_nodes[cluster] = [[self.kgs.graph_id_inv[i][0], train_dict[0][self.kgs.graph_id_inv[i][0]]] for i in self.sg_nodes[cluster] if train_dict[0].get(self.kgs.graph_id_inv[i][0], -1) != -1]
            test_src = [self.kgs.graph_id_inv[i][0] for i in self.sg_nodes[cluster] if self.mapping[0].get(self.kgs.graph_id_inv[i][0], -1)!=-1]
            test_trg = [self.kgs.graph_id_inv[i][0] for i in self.sg_nodes[cluster] if self.mapping[1].get(self.kgs.graph_id_inv[i][0], -1)!=-1]
            self.sg_test_nodes[cluster] = make_pairs(test_src, test_trg, self.mapping[0])
            x = len(self.sg_test_nodes[cluster])/len(self.test_pair)
            total_link = total_link | set(self.sg_test_nodes[cluster])
            print('merge seed partition test pair ratio: '+ str(x))
        print('first total test pair ratio: ' + str(len(total_link)/len(self.test_pair)))

    def split_clusters(self):
        """
        split each cluster into kg1 and kg2
        """
        self.kg1_clusters = {}
        self.kg2_clusters = {}
        for i, cluster in self.sg_nodes.items():
            self.kg1_clusters[i] = []
            self.kg2_clusters[i] = []
            for node in cluster:
                node_id = self.kgs.graph_id_inv[node]
                if self.kgs.ent_dict.get(node_id[0]) == 1:
                    self.kg1_clusters[i].append(node_id[0])
                    if len(node_id) > 1:
                        self.kg2_clusters[i].append(node_id[-1])
                else:
                    self.kg2_clusters[i].append(node_id[0])
            self.kg1_clusters[i] = list(set(self.kg1_clusters[i]))
            self.kg2_clusters[i] = list(set(self.kg2_clusters[i]))

    def fill_cluster(self):
        """
        fill cluster to fixed size
        :return:
        """
        # todo: add nodes along path, not random
        if not self.args.dangling:
            size1 = max([len(self.kg1_clusters[i]) for i in self.kg1_clusters])
            for cluster in self.kg1_clusters:
                remain_nodes = list(set(self.kgs.kg1.entities_list) - set(self.kg1_clusters[cluster]))
                fill_nodes = random.sample(remain_nodes, size1 - len(self.kg1_clusters[cluster]))
                self.kg1_clusters[cluster] += fill_nodes
            size2 = max([len(self.kg2_clusters[i]) for i in self.kg2_clusters])
            for cluster in self.kg2_clusters:
                remain_nodes = list(set(self.kgs.kg2.entities_list) - set(self.kg2_clusters[cluster]))
                fill_nodes = random.sample(remain_nodes, size2 - len(self.kg2_clusters[cluster]))
                self.kg2_clusters[cluster] += fill_nodes
        else:
            size = max([len(self.kg1_clusters[i] + self.kg2_clusters[i]) for i in self.kg1_clusters])
            for cluster in self.clusters:
                size1, size2 = len(self.kg1_clusters[cluster]), len(self.kg2_clusters[cluster])
                if size1 < size2:
                    remain_nodes = list(set(self.kgs.kg1.entities_list) - set(self.kg1_clusters[cluster]))
                    fill_nodes = random.sample(remain_nodes, size - len(self.kg1_clusters[cluster]+self.kg2_clusters[cluster]))
                    self.kg1_clusters[cluster] += fill_nodes
                else:
                    remain_nodes = list(set(self.kgs.kg2.entities_list) - set(self.kg2_clusters[cluster]))
                    fill_nodes = random.sample(remain_nodes, size - len(self.kg1_clusters[cluster]+self.kg2_clusters[cluster]))
                    self.kg2_clusters[cluster] += fill_nodes
        total_link = set()
        for cluster in self.clusters:
            src_train = [i for i in self.kg1_clusters[cluster] if self.train_dict[0].get(i, -1)!=-1]
            trg_train = [i for i in self.kg2_clusters[cluster] if self.train_dict[1].get(i, -1)!=-1]
            self.sg_train_nodes[cluster] = make_pairs(src_train, trg_train, self.mapping[0])
            self.sg_test_nodes[cluster] = make_pairs(self.kg1_clusters[cluster], self.kg2_clusters[cluster], self.mapping[0])
            total_link = total_link | set(self.sg_test_nodes[cluster])
            print('merge seed partition train pair ratio: ' + str(len(self.sg_train_nodes[cluster]) / len(self.kgs.train_links + self.kgs.valid_links)))
            print('merge seed partition test pair ratio: '+ str(len(self.sg_test_nodes[cluster])/len(self.kgs.link)))
        print('final test pair ratio: ' + str(len(total_link) / len(self.kgs.link)))

    def place_triplets(self, triplets, nodes_batch):
        batch = collections.defaultdict(list)
        node2batch = {i: set() for i in self.kgs.kg1.entities_list + self.kgs.kg2.entities_list}
        batch_triplets = []
        for _, cluster in enumerate(nodes_batch):
            for n in nodes_batch[cluster]:
                node2batch[n].add(cluster)
        removed = 0
        removed_triples = []
        for h, r, t in triplets:
            h_batch, t_batch = node2batch.get(h, -1), node2batch.get(t, -1)
            join_batch = h_batch & t_batch
            if len(join_batch) > 0 and not join_batch == set([-1]):
                for b in list(join_batch):
                    batch[b].append((h, r, t))
                    batch_triplets.append((h, r, t))
            else:
                removed += 1
                removed_triples.append((h, r, t))
        print('split triplets complete, total {} triplets removed'.format(removed))
        batch_triplets = list(set(batch_triplets))

        return batch_triplets, batch, removed, removed_triples

    def cluster2batch(self):
        """
        formalize each cluster as a batch for training
        :return: AlignBatch class
        """
        self.triple1_batch, self.batch1, removed1, _ = self.place_triplets(self.kgs.kg1.relation_triples_list,
                                                         self.kg1_clusters)
        self.triple2_batch, self.batch2, removed2, _ = self.place_triplets(self.kgs.kg2.relation_triples_list,
                                                     self.kg2_clusters)
        self.align_clusters = {}
        for i in self.clusters:
            align_cluster = AlignmentBatch(self.batch1[i], self.batch2[i],
                           self.kg1_clusters[i], self.kg2_clusters[i], self.sg_train_nodes[i],
                           self.sg_test_nodes[i])
            self.align_clusters[i] = align_cluster

    def find_cutted(self):
        """
        find cutted entities
        :return:
        """
        self.triple1_batch, self.batch1, removed1, removed_triples1 = self.place_triplets(
            self.kgs.kg1.relation_triples_list,
            self.kg1_clusters)
        self.triple2_batch, self.batch2, removed2, removed_triples2 = self.place_triplets(
            self.kgs.kg2.relation_triples_list,
            self.kg2_clusters)

        cutted_e1, cutted_e2 = [], []
        for t in removed_triples1:
            cutted_e1.append(t[0])
            cutted_e1.append(t[-1])
        for t in removed_triples2:
            cutted_e2.append(t[0])
            cutted_e2.append(t[-1])
        return cutted_e1, cutted_e2

    def remove_isolated(self):
        """
        remove nodes without neighbors
        :return:
        """

    def fast_node_importance(self):
        t1 = time.time()
        print('start building node importance...')
        node_set1 = [i[0] for i in self.target]
        node_set2 = [i[1] for i in self.target]
        importance_dict1 = {node: 0 for node in self.kgs.kg1.entities_list}
        importance_dict2 = {node: 0 for node in self.kgs.kg2.entities_list}
        rests = list()
        for node_set, kg, importance_dict in [(node_set1, self.kgs.kg1, importance_dict1), (node_set2, self.kgs.kg2, importance_dict2)]:
            rest_dic = node_importance_dfs_ray.remote(node_set, self.args.seed_thre, importance_dict, kg)
            rests.append(rest_dic)
        importance_dict = dict()
        for dic in ray.get(rests):
            for k, v in dic.items():
                importance_dict[k] = v
        print('building node importance cost: ' + str(time.time() - t1))

        # assert len(importance_dict_slow) == len(importance_dict)
        # exit(0)
        return importance_dict

    def node_importance(self):
        """
        define importance of each node
        currently based on the distance to seed
        :return:
        """
        t1 = time.time()
        print('start building node importance...')
        importance_dict = {node: 0 for node in self.kgs.kg1.entities_list+self.kgs.kg2.entities_list}
        hop = 0
        score = 1/(0.001 + hop)
        node_set = [i[0] for i in self.target]
        while score > self.args.seed_thre:
            next_set = []
            t2 = time.time()
            for node in node_set:
                if importance_dict[node] == 0:
                    importance_dict[node] = score
                    neighbors1 = [i[1] for i in self.kgs.kg1.rt_dict.get(node, [])]
                    neighbors2 = [i[0] for i in self.kgs.kg1.hr_dict.get(node, [])]
                    next_set = next_set + neighbors1 + neighbors2
            hop += 1
            # score = 1 - 0.2 * hop
            score = 1/(0.001 + hop)
            node_set = next_set
            t3 = time.time()
            print(str(hop) +'-hop total cost: ' + str(t3-t2))
        hop = 0
        score = 1/(0.001 + hop)
        node_set = [i[1] for i in self.target]
        while score > self.args.seed_thre:
            next_set = []
            for node in node_set:
                if importance_dict[node] == 0:
                    importance_dict[node] = score
                    neighbors1 = [i[1] for i in self.kgs.kg2.rt_dict.get(node, set())]
                    neighbors2 = [i[0] for i in self.kgs.kg2.hr_dict.get(node, set())]
                    next_set = next_set + neighbors1 + neighbors2
            hop += 1
            # score = 1 - 0.2 * hop
            score = 1 / (0.001 + hop)
            node_set = next_set
        print('building node importance cost: '+ str(time.time()-t1))
        return importance_dict

    def influential_nodes(self, importance_dict):
        """
        define influential nodes.
        based on importance of node's neighbour
        :return:
        """
        t1 = time.time()
        print('start buidling influential nodes...')
        infl_dict = {}
        if self.args.r_func:
            r_func_dict1 = {}
            for r in self.relation_dict1:
                r_func_dict1[r] = [func_r(self.relation_dict1[r], arg='first'),
                                   func_r(self.relation_dict1[r], arg='second')]
            r_func_dict2 = {}
            for r in self.relation_dict2:
                r_func_dict2[r] = [func_r(self.relation_dict2[r], arg='first'),
                                   func_r(self.relation_dict2[r], arg='second')]
        for e in self.kgs.kg1.entities_list:
            neighbors1 = set([i[1] for i in self.kgs.kg1.rt_dict.get(e, set())])
            neighbors2 = set([i[0] for i in self.kgs.kg1.hr_dict.get(e, set())])
            if self.args.r_func:
                rt = self.kgs.kg1.rt_dict.get(e, set())
                score1 = sum([importance_dict[i[1]] * r_func_dict1[i[0]][0] for i in list(rt)])
                hr = self.kgs.kg1.hr_dict.get(e, set())
                score2 = sum([importance_dict[i[0]] * r_func_dict1[i[1]][1] for i in list(hr)])
                score = score1 + score2
            else:
                score = sum([importance_dict[node] for node in list(neighbors1 | neighbors2)])
            infl_dict[e] = score
        for e in self.kgs.kg2.entities_list:
            neighbors1 = set([i[1] for i in self.kgs.kg2.rt_dict.get(e, set())])
            neighbors2 = set([i[0] for i in self.kgs.kg2.hr_dict.get(e, set())])
            if self.args.r_func:
                rt = self.kgs.kg2.rt_dict.get(e, set())
                score1 = sum([importance_dict[i[1]] * r_func_dict2[i[0]][0] for i in list(rt)])
                hr = self.kgs.kg2.hr_dict.get(e, set())
                score2 = sum([importance_dict[i[0]] * r_func_dict2[i[1]][1] for i in list(hr)])
                score = score1 + score2
            else:
                score = sum([importance_dict[node] for node in list(neighbors1 | neighbors2)])
            infl_dict[e] = score
        t2=time.time()
        print('building influential node cost: ' + str(t2 - t1))
        return infl_dict

    def random_walk(self):
        """
        add neighbours via random walk
        :return:
        """
        self.important_dict = self.node_importance()
        self.infl_dict = self.influential_nodes(self.important_dict)
        cutted_e1, cutted_e2 = self.find_cutted()
        path_len = self.args.random_path
        for cluster in self.clusters:
            len1 = len(self.kg1_clusters[cluster])
            cutted_e = set(cutted_e1) & set(self.kg1_clusters[cluster])
            node_set_0 = set()
            for node in cutted_e:
                neighbors1 = set([i[1] for i in self.kgs.kg1.rt_dict.get(node, set())])
                neighbors2 = set([i[0] for i in self.kgs.kg1.hr_dict.get(node, set())])
                node_set_0 = node_set_0 | neighbors1 | neighbors2
            # node_set = cutted_e
            node_set = node_set_0
            curr_path = 0
            cluster_set = set(self.kg1_clusters[cluster])
            while len(cluster_set) < int(self.args.cluster_size/2):
                if curr_path <= path_len:
                    prob = [self.infl_dict[n] for n in node_set]
                    result = random.choices(list(node_set), weights=prob, k=1)[0]
                    # result = list(node_set)[prob.index(max(prob))]
                    cluster_set.add(result)
                    neighbors1 = set([i[1] for i in self.kgs.kg1.rt_dict.get(result, set())])
                    neighbors2 = set([i[0] for i in self.kgs.kg1.hr_dict.get(result, set())])
                    # node_set = neighbors1 | neighbors2 - cluster_set
                    node_set = neighbors1 | neighbors2
                    if len(node_set) == 0:
                        curr_path = 0
                        # node_set = cutted_e
                        node_set = node_set_0
                        continue
                    curr_path += 1
                else:
                    curr_path = 0
                    # node_set = cutted_e
                    node_set = node_set_0
            self.kg1_clusters[cluster] = list(set(cluster_set))
            len2 = len(self.kg1_clusters[cluster])
            print('after random walk, kg1 cluster size expand: ' + str(len2 / len1))

            len1 = len(self.kg2_clusters[cluster])
            cutted_e = set(cutted_e2) & set(self.kg2_clusters[cluster])
            node_set_0 = set()
            for node in cutted_e:
                neighbors1 = set([i[1] for i in self.kgs.kg2.rt_dict.get(node, set())])
                neighbors2 = set([i[0] for i in self.kgs.kg2.hr_dict.get(node, set())])
                node_set_0 = node_set_0 | neighbors1 | neighbors2
            # node_set = cutted_e
            node_set = node_set_0
            curr_path = 0
            cluster_set = set(self.kg2_clusters[cluster])
            while len(cluster_set) < int(self.args.cluster_size/2):
                if curr_path <= path_len:
                    prob = [self.infl_dict[n] for n in node_set]
                    result = random.choices(list(node_set), weights=prob, k=1)[0]
                    # result = list(node_set)[prob.index(max(prob))]
                    cluster_set.add(result)
                    neighbors1 = set([i[1] for i in self.kgs.kg2.rt_dict.get(result, set())])
                    neighbors2 = set([i[0] for i in self.kgs.kg2.hr_dict.get(result, set())])
                    # node_set = neighbors1 | neighbors2 - cluster_set
                    node_set = neighbors1 | neighbors2
                    if len(node_set) == 0:
                        curr_path = 0
                        # node_set = cutted_e
                        node_set = node_set_0
                        continue
                    curr_path += 1
                else:
                    curr_path = 0
                    # node_set = cutted_e
                    node_set = node_set_0
            self.kg2_clusters[cluster] = list(set(cluster_set))
            len2 = len(self.kg2_clusters[cluster])
            print('after random walk, kg2 cluster size expand: ' + str(len2 / len1))

    def core_rank(self):
        """
        pick nodes based on corerank score
        :return:
        """
        self.important_dict = self.node_importance()
        self.infl_dict = self.influential_nodes(self.important_dict)
        cutted_e1, cutted_e2 = self.find_cutted()
        path_len = self.args.random_path
        for cluster in self.clusters:
            len1 = len(self.kg1_clusters[cluster])
            cutted_e = set(cutted_e1) & set(self.kg1_clusters[cluster])
            node_set_0 = set()
            for node in cutted_e:
                neighbors1 = set([i[1] for i in self.kgs.kg1.rt_dict.get(node, set())])
                neighbors2 = set([i[0] for i in self.kgs.kg1.hr_dict.get(node, set())])
                node_set_0 = node_set_0 | neighbors1 | neighbors2
            # node_set = cutted_e
            node_set = node_set_0
            curr_path = 0
            cluster_set = set(self.kg1_clusters[cluster])
            while len(cluster_set) < int(self.args.cluster_size / 2) and len(node_set) > 0:
                if curr_path < path_len:
                    prob = [self.infl_dict[n] for n in node_set]
                    # result = random.choices(list(node_set), weights=prob, k=1)[0]
                    result = list(node_set)[prob.index(max(prob))]
                    cluster_set.add(result)
                    neighbors1 = set([i[1] for i in self.kgs.kg1.rt_dict.get(result, set())])
                    neighbors2 = set([i[0] for i in self.kgs.kg1.hr_dict.get(result, set())])
                    node_set = (neighbors1 | neighbors2) - cluster_set
                    if len(node_set) == 0:
                        curr_path = 0
                        # node_set = cutted_e
                        node_set = node_set_0 - cluster_set
                        continue
                    curr_path += 1
                else:
                    curr_path = 0
                    # node_set = cutted_e
                    node_set = node_set_0 - cluster_set
            self.kg1_clusters[cluster] = list(set(cluster_set))
            len2 = len(self.kg1_clusters[cluster])
            print('after core walk, kg1 cluster size expand: ' + str(len2 / len1))

            len1 = len(self.kg2_clusters[cluster])
            cutted_e = set(cutted_e2) & set(self.kg2_clusters[cluster])
            node_set_0 = set()
            for node in cutted_e:
                neighbors1 = set([i[1] for i in self.kgs.kg2.rt_dict.get(node, set())])
                neighbors2 = set([i[0] for i in self.kgs.kg2.hr_dict.get(node, set())])
                node_set_0 = node_set_0 | neighbors1 | neighbors2
            # node_set = cutted_e
            node_set = node_set_0
            curr_path = 0
            cluster_set = set(self.kg2_clusters[cluster])
            while len(cluster_set) < int(self.args.cluster_size / 2) and len(node_set) > 0:
                if curr_path < path_len:
                    prob = [self.infl_dict[n] for n in node_set]
                    # result = random.choices(list(node_set), weights=prob, k=1)[0]
                    result = list(node_set)[prob.index(max(prob))]
                    cluster_set.add(result)
                    neighbors1 = set([i[1] for i in self.kgs.kg2.rt_dict.get(result, set())])
                    neighbors2 = set([i[0] for i in self.kgs.kg2.hr_dict.get(result, set())])
                    # node_set = neighbors1 | neighbors2 - cluster_set
                    node_set = neighbors1 | neighbors2 - cluster_set
                    if len(node_set) == 0:
                        curr_path = 0
                        # node_set = cutted_e
                        node_set = node_set_0 - cluster_set
                        continue
                    curr_path += 1
                else:
                    curr_path = 0
                    # node_set = cutted_e
                    node_set = node_set_0 - cluster_set
            self.kg2_clusters[cluster] = list(set(cluster_set))
            len2 = len(self.kg2_clusters[cluster])
            print('after core walk, kg2 cluster size expand: ' + str(len2 / len1))

    def fast_core_path(self):
        def dfs(cluster_set, node, cand, node_hop, node_path, kg, dist):
            if dist <= self.args.random_path:
                if node in cand and node_hop[node] <= dist:
                    return
                if dist > 0:
                    cand.add(node)
                    node_hop[node] = dist
            else:
                return
            dist += 1
            if dist > self.args.random_path:
                return
            neighbors1 = [i[1] for i in kg.rt_dict.get(node, set())]
            neighbors2 = [i[0] for i in kg.hr_dict.get(node, set())]
            for neighbour in set(neighbors1 + neighbors2):
                if dist == 2 and node_hop.get(node, None) == 1:
                    if node_path.get(neighbour, None) is None:
                        node_path[neighbour] = node
                    else:
                        if self.infl_dict[node] > self.infl_dict[node_path[neighbour]]:
                            node_path[neighbour] = node
                dfs(cluster_set, neighbour, cand, node_hop, node_path, kg, dist=dist)
        dict_path = self.args.cm_path + self.args.training_data.split('/')[-2]
        dict_file = dict_path + '_important_dict'
        if os.path.exists(dict_file):
            with open(dict_file, 'rb') as f:
                self.important_dict = pickle.load(f)
        else:
            self.important_dict = self.fast_node_importance()
            # self.important_dict = self.node_importance()
            with open(dict_file, "wb") as f:
                pickle.dump(self.important_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        self.infl_dict = self.influential_nodes(self.important_dict)
        tt = time.time()
        cutted_e1, cutted_e2 = self.find_cutted()
        for cluster in self.clusters:
            cluster_set = set(self.kg1_clusters[cluster])
            len1 = len(self.kg1_clusters[cluster])
            cutted_e = set(cutted_e1) & cluster_set
            cand = set()
            hop = 0
            node_hop = {}
            node_path = {}
            t1 = time.time()
            for node in cutted_e:
                dfs(cluster_set, node, cand, node_hop, node_path, self.kgs.kg1, hop)
            print('generate recall candidates cost: ' + str(time.time() - t1))
            cand_dict = dict([(n, self.infl_dict[n] * math.pow(0.01, (node_hop[n]-1))) for n in cand])
            cand_dict = {k: v for k, v in sorted(cand_dict.items(), key=lambda item: -item[1])}
            print('2-hop candidate size: ' + str(len(cand_dict)))
            I = {}
            inter = 0
            for k, v in cand_dict.items():
                if len(cluster_set) >= int(self.args.cluster_size / 2):
                    break
                if node_hop[k] > 1:
                    k_path = node_path[k]
                    if k_path in cluster_set:
                        cluster_set.add(k)
                    else:
                        new = (cand_dict[k] + cand_dict[k_path])/2
                        cand_dict[k], cand_dict[k_path] = new, new
                        I[new] = (k, k_path)
                        if len(I) == 1:
                            inter = new
                else:
                    if cand_dict[k] >= inter:
                        cluster_set.add(k)
                    else:
                        m = max(I.keys())
                        pair = I[m]
                        cluster_set.add(pair[0])
                        cluster_set.add(pair[1])
                        del I[m]
                        if len(I)==0:
                            inter = 0
                        else:
                            inter = max(I.keys())
            self.kg1_clusters[cluster] = list(cluster_set)
            len2 = len(self.kg1_clusters[cluster])
            print('path rank of one cluster cost: ' + str(time.time() - t1))
            print('after path rank, kg1 cluster size is: ' + str(len2))

            cluster_set = set(self.kg2_clusters[cluster])
            len1 = len(self.kg2_clusters[cluster])
            cutted_e = set(cutted_e2) & cluster_set
            cand = set()
            hop = 0
            node_hop = {}
            node_path = {}
            t1 = time.time()
            for node in cutted_e:
                dfs(cluster_set, node, cand, node_hop, node_path, self.kgs.kg2, hop)
            print('generate recall candidates cost: ' + str(time.time() - t1))
            cand_dict = dict([(n, self.infl_dict[n] * math.pow(0.01, (node_hop[n]-1))) for n in cand])
            cand_dict = {k: v for k, v in sorted(cand_dict.items(), key=lambda item: -item[1])}
            print('2-hop candidate size: ' + str(len(cand_dict)))
            I = {}
            inter = 0
            for k, v in cand_dict.items():
                if len(cluster_set) >= int(self.args.cluster_size / 2):
                    break
                if node_hop[k] > 1:
                    k_path = node_path[k]
                    if k_path in cluster_set:
                        cluster_set.add(k)
                    else:
                        new = (cand_dict[k] + cand_dict[k_path])/2
                        cand_dict[k], cand_dict[k_path] = new, new
                        I[new] = (k, k_path)
                        if len(I) == 1:
                            inter = new
                else:
                    if cand_dict[k] >= inter:
                        cluster_set.add(k)
                    else:
                        m = max(I.keys())
                        pair = I[m]
                        cluster_set.add(pair[0])
                        cluster_set.add(pair[1])
                        del I[m]
                        if len(I) == 0:
                            inter = 0
                        else:
                            inter = max(I.keys())
            self.kg2_clusters[cluster] = list(cluster_set)
            len2 = len(self.kg2_clusters[cluster])
            print('path rank of one cluster cost: ' + str(time.time() - t1))
            print('after path rank, kg2 cluster size is: ' + str(len2))
        print('************** node recalling cost: ' + str(time.time()-tt) +' *******************')

    def core_path(self):
        """
        choose topk nodes forming path without isolated nodes
        :return:
        """
        t = time.time()
        self.important_dict = self.node_importance()
        self.infl_dict = self.influential_nodes(self.important_dict)
        cutted_e1, cutted_e2 = self.find_cutted()
        for cluster in self.clusters:
            cluster_set = set(self.kg1_clusters[cluster])
            len1 = len(self.kg1_clusters[cluster])
            cutted_e = set(cutted_e1) & set(self.kg1_clusters[cluster])
            cand = set()
            range = cutted_e
            hop = 0
            hop_dict = {}
            node_hop = {}
            t1 = time.time()
            while hop < self.args.random_path:
                hop_set = set()
                for node in range:
                    neighbors1 = set([i[1] for i in self.kgs.kg1.rt_dict.get(node, set())])
                    neighbors2 = set([i[0] for i in self.kgs.kg1.hr_dict.get(node, set())])
                    hop_set = hop_set | neighbors1 | neighbors2 - cluster_set
                range = hop_set - cluster_set - cand
                hop_dict[hop] = range
                hop += 1
                cand = cand | range
            t2 = time.time()
            print('generate recall candidates cost: ' + str(t2 - t1))
            for k, v in hop_dict.items():
                for i in list(v):
                    node_hop[i] = k
            cand_dict = dict([(n, self.infl_dict[n] * (1 - 0.99 * node_hop[n])) for n in cand])
            while len(cluster_set) < int(self.args.cluster_size / 2) and len(cand_dict) > 0:
                t3 = time.time()
                n = max(cand_dict, key=cand_dict.get)
                t4 = time.time()
                # print('picking max value cost: ' + str(t4 - t3))
                nei = set([i[1] for i in self.kgs.kg1.rt_dict.get(n, set())]) | set(
                    [i[0] for i in self.kgs.kg1.hr_dict.get(n, set())])
                if nei & cluster_set == set():
                    n_hop = node_hop[n]
                    paths = hop_dict[n_hop - 1] & nei
                    path_score = [self.infl_dict[x] for x in paths]
                    path = list(paths)[path_score.index(max(path_score))]
                    new_score = (cand_dict[path] + cand_dict[n])/2
                    cand_dict[n] = new_score
                    cand_dict[path] = new_score + 0.00000001
                else:
                    cluster_set.add(n)
                    del cand_dict[n]
                t5 = time.time()
                # print('checking isolated nodes: ' + str(t5 - t4))
            self.kg1_clusters[cluster] = list(set(cluster_set))
            len2 = len(self.kg1_clusters[cluster])
            t6 = time.time()
            print('path rank of one cluster cost: ' + str(t6 - t1))
            print('after path rank, kg1 cluster size expand: ' + str(len2 / len1))

            cluster_set = set(self.kg2_clusters[cluster])
            len1 = len(self.kg2_clusters[cluster])
            cutted_e = set(cutted_e2) & set(self.kg2_clusters[cluster])
            cand = set()
            range = cutted_e
            hop = 0
            hop_dict = {}
            node_hop = {}
            while hop < self.args.random_path:
                hop_set = set()
                for node in range:
                    neighbors1 = set([i[1] for i in self.kgs.kg2.rt_dict.get(node, set())])
                    neighbors2 = set([i[0] for i in self.kgs.kg2.hr_dict.get(node, set())])
                    hop_set = hop_set | neighbors1 | neighbors2 - cluster_set
                range = hop_set - cluster_set - cand
                hop_dict[hop] = range
                hop += 1
                cand = cand | range
            for k, v in hop_dict.items():
                for i in list(v):
                    node_hop[i] = k
            cand_dict = dict([(n, self.infl_dict[n] * (1 - 0.99 * node_hop[n])) for n in cand])
            while len(cluster_set) < int(self.args.cluster_size / 2) and len(cand_dict) > 0:
                n = max(cand_dict, key=cand_dict.get)
                nei = set([i[1] for i in self.kgs.kg2.rt_dict.get(n, set())]) | set(
                    [i[0] for i in self.kgs.kg2.hr_dict.get(n, set())])
                if nei & cluster_set == set():
                    n_hop = node_hop[n]
                    paths = hop_dict[n_hop - 1] & nei
                    path_score = [self.infl_dict[x] for x in paths]
                    path = list(paths)[path_score.index(max(path_score))]
                    new_score = (cand_dict[path] + cand_dict[n]) / 2
                    cand_dict[n] = new_score
                    cand_dict[path] = new_score + 0.00000001
                else:
                    cluster_set.add(n)
                    del cand_dict[n]
            self.kg2_clusters[cluster] = list(set(cluster_set))
            len2 = len(self.kg2_clusters[cluster])
            print('after path rank, kg2 cluster size expand: ' + str(len2 / len1))
        print('path rank cost total: ' + str(time.time()-t))

    def brute_force(self):
        """
        brute force solution of path with topk
        :return:
        """
        self.important_dict = self.node_importance()
        self.infl_dict = self.influential_nodes(self.important_dict)
        cutted_e1, cutted_e2 = self.find_cutted()
        for cluster in self.clusters:
            len1 = len(self.kg1_clusters[cluster])
            cutted_e = set(cutted_e1) & set(self.kg1_clusters[cluster])
            cand = set()
            range = cutted_e
            hop = 0
            while hop < self.args.random_path:
                hop_set = set()
                for node in range:
                    neighbors1 = set([i[1] for i in self.kgs.kg1.rt_dict.get(node, set())])
                    neighbors2 = set([i[0] for i in self.kgs.kg1.hr_dict.get(node, set())])
                    hop_set = hop_set | neighbors1 | neighbors2 - set(self.kg1_clusters[cluster])
                hop += 1
                range = hop_set - set(self.kg1_clusters[cluster]) - cand
                cand = cand | range
            budget = int(self.args.cluster_size / 2) - len1
            prob = np.array([self.infl_dict[n] for n in cand])
            topk_nodes = np.array(list(cand))[np.argpartition(prob, -budget)[-budget:]]
            for n in topk_nodes:
                nei = set([i[1] for i in self.kgs.kg1.rt_dict.get(n, set())]) | set([i[0] for i in self.kgs.kg1.hr_dict.get(n, set())])
                if nei & (set(self.kg1_clusters[cluster]) | (cand - set([n]))) == set():
                    cand.remove(n)

            all_paths = list(itertools.combinations(list(cand), budget))
            prob = [sum([self.infl_dict[n] for n in path]) for path in all_paths]
            result = all_paths[prob.index(max(prob))]
            self.kg1_clusters[cluster] = list(set(self.kg1_clusters[cluster]) | set(result))
            len2 = len(self.kg1_clusters[cluster])
            print('after brute force, kg1 cluster size expand: ' + str(len2 / len1))

            len1 = len(self.kg2_clusters[cluster])
            cutted_e = set(cutted_e2) & set(self.kg2_clusters[cluster])
            cand = set()
            range = cutted_e
            hop = 0
            while hop < self.args.random_path:
                hop_set = set()
                for node in range:
                    neighbors1 = set([i[1] for i in self.kgs.kg2.rt_dict.get(node, set())])
                    neighbors2 = set([i[0] for i in self.kgs.kg2.hr_dict.get(node, set())])
                    hop_set = hop_set | neighbors1 | neighbors2 - set(self.kg1_clusters[cluster])
                hop += 1
                range = hop_set - set(self.kg1_clusters[cluster]) - cand
                cand = cand | range
            budget = int(self.args.cluster_size / 2) - len1
            all_paths = list(itertools.combinations(list(cand), budget))
            prob = [sum([self.infl_dict[n] for n in path]) for path in all_paths]
            result = all_paths[prob.index(max(prob))]
            self.kg2_clusters[cluster] = list(set(self.kg2_clusters[cluster]) | set(result))
            len2 = len(self.kg2_clusters[cluster])
            print('after brute force, kg2 cluster size expand: ' + str(len2 / len1))


    def hop_corerank(self):
        """
        recall nodes consider hop, further, less weight
        :return:
        """
        self.important_dict = self.node_importance()
        self.infl_dict = self.influential_nodes(self.important_dict)
        cutted_e1, cutted_e2 = self.find_cutted()
        path_len = self.args.random_path
        for cluster in self.clusters:
            len1 = len(self.kg1_clusters[cluster])
            cutted_e = set(cutted_e1) & set(self.kg1_clusters[cluster])
            node_set_0 = set()
            for node in cutted_e:
                neighbors1 = set([i[1] for i in self.kgs.kg1.rt_dict.get(node, set())])
                neighbors2 = set([i[0] for i in self.kgs.kg1.hr_dict.get(node, set())])
                node_set_0 = node_set_0 | neighbors1 | neighbors2
            # node_set = cutted_e
            node_set = node_set_0
            curr_path = 0
            cluster_set = set(self.kg1_clusters[cluster])
            while len(cluster_set) < int(self.args.cluster_size / 2):
                while len(node_set) > 0:
                    prob = [self.infl_dict[n] for n in node_set]
                    result = list(node_set)[prob.index(max(prob))]
                    cluster_set.add(result)
                    node_set = node_set - result

                if curr_path <= path_len:
                    prob = [self.infl_dict[n] for n in node_set]
                    # result = random.choices(list(node_set), weights=prob, k=1)[0]
                    result = list(node_set)[prob.index(max(prob))]
                    cluster_set.add(result)
                    neighbors1 = set([i[1] for i in self.kgs.kg1.rt_dict.get(result, set())])
                    neighbors2 = set([i[0] for i in self.kgs.kg1.hr_dict.get(result, set())])
                    node_set = (neighbors1 | neighbors2) - cluster_set
                    if len(node_set) == 0:
                        curr_path = 0
                        # node_set = cutted_e
                        node_set = node_set_0 - cluster_set
                        continue
                    curr_path += 1
                else:
                    curr_path = 0
                    # node_set = cutted_e
                    node_set = node_set_0 - cluster_set
            self.kg1_clusters[cluster] = list(set(cluster_set))
            len2 = len(self.kg1_clusters[cluster])
            print('after random walk, kg1 cluster size expand: ' + str(len2 / len1))

    def fill_seed_neighbour(self):
        """
        fill neighbour of cutted seeds
        :return:
        """
        cutted_e1, cutted_e2 = self.find_cutted()
        for cluster in self.clusters:
            len1 = len(self.kg1_clusters[cluster])
            cutted_e = set(cutted_e1) & set(self.kg1_clusters[cluster])
            cutted_seed = []
            for e in cutted_e:
                if self.train_dict[0].get(e, -1) != -1:
                    cutted_seed.append(e)
            add_nodes = set()
            for s in cutted_seed:
                neighbors1 = set([i[1] for i in self.kgs.kg1.rt_dict.get(s, set())])
                neighbors2 = set([i[0] for i in self.kgs.kg1.hr_dict.get(s, set())])
                node_set1 = neighbors1 | neighbors2
                add_nodes = add_nodes | node_set1
            if self.args.fill_seed:
                add_nodes = set([i for i in list(add_nodes) if i in self.train_dict[0]])
            self.kg1_clusters[cluster] = list(set(self.kg1_clusters[cluster]) | add_nodes)
            len2 = len(self.kg1_clusters[cluster])
            print('after fill seed neighbours, kg1 cluster size expand: ' + str(len2 / len1))

            len1 = len(self.kg2_clusters[cluster])
            cutted_e = set(cutted_e2) & set(self.kg2_clusters[cluster])
            cutted_seed = []
            for e in cutted_e:
                if self.train_dict[1].get(e, -1) != -1:
                    cutted_seed.append(e)
            add_nodes = set()
            for s in cutted_seed:
                neighbors1 = set([i[1] for i in self.kgs.kg2.rt_dict.get(s, set())])
                neighbors2 = set([i[0] for i in self.kgs.kg2.hr_dict.get(s, set())])
                node_set1 = neighbors1 | neighbors2
                add_nodes = add_nodes | node_set1
            if self.args.fill_seed:
                add_nodes = set([i for i in list(add_nodes) if i in self.train_dict[1]])
            self.kg2_clusters[cluster] = list(set(self.kg2_clusters[cluster]) | add_nodes)
            len2 = len(self.kg2_clusters[cluster])
            print('after fill seed neighbours, kg2 cluster size expand: ' + str(len2 / len1))

    def fill_seed(self):
        """
        fill seeds
        :return:
        """
        cutted_e1, cutted_e2 = self.find_cutted()
        for cluster in self.clusters:
            len1 = len(self.kg1_clusters[cluster])
            cutted_e = set(cutted_e1) & set(self.kg1_clusters[cluster])
            added_seeds = set()
            for e in cutted_e:
                neighbors1 = set([i[1] for i in self.kgs.kg1.rt_dict.get(e, set())])
                neighbors2 = set([i[0] for i in self.kgs.kg1.hr_dict.get(e, set())])
                node_set1 = neighbors1 | neighbors2
                seed_onehop = []
                for node in node_set1:
                    if node in self.kgs.train_entities1 + self.kgs.valid_entities1:
                        seed_onehop.append(node)
                added_seeds = added_seeds | set(seed_onehop)
                if self.args.two_hop:
                    seed_twohop = []
                    for seed1 in seed_onehop:
                        neighbors1 = set([i[1] for i in self.kgs.kg1.rt_dict.get(seed1, set())])
                        neighbors2 = set([i[0] for i in self.kgs.kg1.hr_dict.get(seed1, set())])
                        seed_set1 = neighbors1 | neighbors2
                        for s in list(seed_set1):
                            if s in self.kgs.train_entities1 + self.kgs.valid_entities1:
                                seed_twohop.append(s)
                    added_seeds = added_seeds | set(seed_twohop)
            self.kg1_clusters[cluster] = list(set(self.kg1_clusters[cluster]) | added_seeds)
            len2 = len(self.kg1_clusters[cluster])
            print('after add 2-hop seeds, kg1 cluster size expand: ' + str(len2 / len1))

            len1 = len(self.kg2_clusters[cluster])
            cutted_e = set(cutted_e2) & set(self.kg2_clusters[cluster])
            added_seeds = set()
            for e in cutted_e:
                neighbors1 = set([i[1] for i in self.kgs.kg2.rt_dict.get(e, set())])
                neighbors2 = set([i[0] for i in self.kgs.kg2.hr_dict.get(e, set())])
                node_set1 = neighbors1 | neighbors2
                seed_onehop = []
                for node in node_set1:
                    if node in self.kgs.train_entities2 + self.kgs.valid_entities2:
                        seed_onehop.append(node)
                added_seeds = added_seeds | set(seed_onehop)
                if self.args.two_hop:
                    seed_twohop = []
                    for seed1 in seed_onehop:
                        neighbors1 = set([i[1] for i in self.kgs.kg2.rt_dict.get(seed1, set())])
                        neighbors2 = set([i[0] for i in self.kgs.kg2.hr_dict.get(seed1, set())])
                        seed_set1 = neighbors1 | neighbors2
                        for s in list(seed_set1):
                            if s in self.kgs.train_entities2 + self.kgs.valid_entities2:
                                seed_twohop.append(s)
                    added_seeds = added_seeds | set(seed_twohop)
            self.kg2_clusters[cluster] = list(set(self.kg2_clusters[cluster]) | added_seeds)
            len2 = len(self.kg2_clusters[cluster])
            print('after add 2-hop seeds, kg2 cluster size expand: ' + str(len2 / len1))

    def duplicate_neighbours(self):
        """
        duplicate cutted nodes.
        """
        cutted_e1, cutted_e2 = self.find_cutted()
        for cluster in self.clusters:
            len1 = len(self.kg1_clusters[cluster])
            cutted_e = set(cutted_e1) & set(self.kg1_clusters[cluster])
            node_set1 = set()
            for node in cutted_e:
                neighbors1 = set([i[1] for i in self.kgs.kg1.rt_dict.get(node, set())])
                neighbors2 = set([i[0] for i in self.kgs.kg1.hr_dict.get(node, set())])
                node_set1 = node_set1 | neighbors1 | neighbors2
            self.kg1_clusters[cluster] = list(set(self.kg1_clusters[cluster]) | node_set1)
            len2 = len(self.kg1_clusters[cluster])
            print('after duplicate, cluster size expand: ' + str(len2/len1))

            len1 = len(self.kg2_clusters[cluster])
            cutted_e = set(cutted_e2) & set(self.kg2_clusters[cluster])
            node_set2 = set()
            for node in cutted_e:
                neighbors1 = set([i[1] for i in self.kgs.kg2.rt_dict.get(node, set())])
                neighbors2 = set([i[0] for i in self.kgs.kg2.hr_dict.get(node, set())])
                node_set2 = node_set2 | neighbors1 | neighbors2
            self.kg2_clusters[cluster] = list(set(self.kg2_clusters[cluster]) | node_set2)
            len2 = len(node_set2)
            print('after duplicate, cluster size expand: ' + str(len2 / len1))


class AlignmentBatch:
    def __init__(self, triple1, triple2, src_nodes, trg_nodes, train_pairs, test_pairs, *args,
                 **kwargs):
        self.merge = True
        print("Batch info: ", '\n\t'.join(map(lambda x, y: '='.join(map(str, [x, y])),
                                              ['triple1', 'triple2', 'srcNodes', 'trgNodes',
                                               'trainPairs', 'testPairs'],
                                              map(len,
                                                  [triple1, triple2, src_nodes, trg_nodes, train_pairs, test_pairs]))))
        self.ent_maps, self.rel_maps, self.ent_ids, self.rel_ids, \
        [t1, t2, self.train_ill, self.test_ill] = rearrange_ids([src_nodes, trg_nodes], self.merge,
                                                      triple1, triple2, train_pairs, test_pairs)
        self.triples = t1 + t2
        self.test_pairs = test_pairs
        self.train_pairs = train_pairs
        self.len_src, self.len_trg = len(src_nodes), len(trg_nodes)
        self.shift = len(src_nodes)
        self.assoc = make_assoc(self.ent_maps, self.len_src, self.len_trg, self.merge)

    @staticmethod
    def get_ei(triple):
        return torch.tensor([[t[0], t[-1]] for t in triple]).t()

    @staticmethod
    def get_et(triple):
        return torch.tensor([t[1] for t in triple])

    @property
    def test_set(self):
        return torch.tensor(self.test_pairs).t()

    @torch.no_grad()
    def get_sim_mat(self, all_embeds, size):
        if isinstance(all_embeds, tuple):
            embeds = all_embeds
        else:
            embeds = [all_embeds[:self.shift], all_embeds[self.shift:]]
        ind, val = text_utils.get_batch_sim(embeds)
        ind = torch.stack(
            [self.assoc[ind[0]],
             self.assoc[ind[1] + self.shift]]
        )
        return ind2sparse(ind, size, values=val)


class Partition:
    def __init__(self, args, data):
        self.args = args
        self.data = data
        self.train_set, self.train_map = self.get_train_node_sets(data)

    @staticmethod
    def get_train_node_sets(data):
        train_pairs = data.train
        all_pairs = data.link
        node_sets = [set(tp[0] for tp in train_pairs),
                     set(tp[1] for tp in train_pairs)]
        train_map = [{tp[0]: tp[1] for tp in all_pairs},
                     {tp[1]: tp[0] for tp in all_pairs}]
        return node_sets, train_map

    @staticmethod
    def construct_edge_graph(triples, important_nodes=None, weight=1000):
        edges_map = collections.defaultdict(list)
        for n, tr in enumerate(triples):
            h, _, t = tr
            edges_map[h].append(n)
            edges_map[t].append(n)

        print("total nodes with triple:", len(edges_map))

        g = nx.Graph()
        now, total = 0, len(edges_map)
        merged_important_nodes = set()
        if important_nodes is not None:
            for nodes in important_nodes:
                merged_important_nodes.update(nodes)

        for node, edges in edges_map.items():

            if important_nodes is None:
                node_weight = 1
            else:
                node_weight = weight if node in merged_important_nodes else 1
            curr_node_graph: nx.DiGraph = nx.complete_graph(edges)
            for u, v in curr_node_graph.edges:
                g.add_edge(u, v, weight=g.get_edge_data(u, v, {}).get('weight', 0) + node_weight)

            now += 1
            if now % 50000 == 0:
                print('create graph', now, 'complete')

        return g

    def partition_by_edge(self, src=0, k=30):
        g0 = self.construct_edge_graph(self.data.triples[src])
        trg = 1 - src
        print('construct src graph complete, total nodes={0}, total edges={1}'
              .format(len(g0.nodes), len(g0.edges)))
        mincut, src_edges = nxmetis.partition(g0, k)
        print('src graph partition complete, mincut=', mincut)

        pass

    @staticmethod
    def make_cnt_edges(lst):
        mp = collections.defaultdict(int)
        for item in lst:
            mp[item] += 1

        return [(k[0], k[1], v) for k, v in mp.items()]

    @staticmethod
    def construct_graph(triples, important_nodes=None, known_weight=1000, known_size=None,
                        cnt_as_weight=False, keep_inter_edges=False):
        g = nx.Graph()
        edges = [(t[0], t[2]) for t in triples]
        if cnt_as_weight:
            edges = Partition.make_cnt_edges(Partition.make_cnt_edges(edges))
            g.add_weighted_edges_from(edges)
        else:
            g.add_edges_from(edges)
            nx.set_edge_attributes(g, 1, 'weight')
        # g.edges.data('weight', default=1)
        if important_nodes:
            subgraphs = []
            print('set important node weights:')
            for nodes in important_nodes:
                sn = nodes[0]
                g.add_edges_from([(sn, n) for n in nodes])
                subgraph = g.subgraph(nodes)
                if known_size:
                    nx.set_node_attributes(subgraph, known_size, 'size')
                nx.set_edge_attributes(subgraph, known_weight, 'weight')
                subgraphs.append(subgraph)
                # subgraphs.append(g.subgraph(nodes))
                # subgraphs[-1].edges.data('weight', default=weight)
            print('compose subgraphs')
            for sg in subgraphs:
                g = nx.compose(g, sg)

            if keep_inter_edges:
                return g

            merged_important_nodes = set()
            for nodes in important_nodes:
                merged_important_nodes.update(nodes)
            # print('all important nodes merged')

            print('del inter edges:')
            for nodes in important_nodes:
                all_neighbors = [g.neighbors(n) for n in nodes]
                # neighbors = set()
                choices = []
                for n in all_neighbors:
                    choices.append(merged_important_nodes.intersection(n) - set(nodes))
                edges = [(e1, e2) for idx, e1 in enumerate(nodes) for e2 in choices[idx]]
                g.remove_edges_from(edges)
        return g

    def subgraph_trainset(self, node_lists, src=0, no_trg=False):
        src_train = []
        train = self.train_set[src]
        mp = self.train_map[src]
        for i, nodes in enumerate(node_lists):
            curr = []
            for n in nodes:
                if n in train:
                    curr.append(n)
            src_train.append(curr)
        if no_trg:
            return src_train
        trg_train = [[mp[e] for e in curr] for curr in src_train]
        return src_train, trg_train

    @staticmethod
    def share_triplets(src_triplet, trg_triplet, train_set, node_mapping, rel_mapping=None):
        if rel_mapping is None:
            rel_mapping = lambda x: x

        new_trg = []

        print('share triplet')  # parameter swapping
        for triplet in tqdm(src_triplet):
            h, r, t = triplet
            if h in train_set and t in train_set:
                new_trg.append([node_mapping[h], rel_mapping(r), node_mapping[t]])

        return trg_triplet + new_trg

    def random_partition(self, src=0, src_k=20, trg_k=20, *args, **kwargs):
        trg = 1 - src
        assert trg_k == src_k
        src_node_len, trg_node_len = len(self.data.ents[src]), len(self.data.ents[trg])
        src_nodes = set(range(src_node_len))
        trg_nodes = set(range(trg_node_len))
        src_train, trg_train = self.train_set[src], self.train_set[trg]
        src_test, trg_test = map(lambda x, y: x - y, [src_nodes, trg_nodes], [src_train, trg_train])

        def split_k_parts(nodes: set, k) -> List[set]:
            nodes_list = list(nodes)
            random.shuffle(nodes_list)
            print('total {} of nodes to split'.format(len(nodes_list)))

            batch_size = int(len(nodes_list) / k)
            ret = []
            for i_batch in trange(0, len(nodes_list), batch_size):
                i_end = min(i_batch + batch_size, len(nodes_list))
                ret.append(set(nodes_list[i_batch:i_end]))
            return ret

        src_test, src_train, trg_test = map(split_k_parts, [src_test, src_train, trg_test], [src_k] * 3)
        src_nodes = [x for x in map(lambda x, y: x.union(y), src_test, src_train)]
        src_train, trg_train = self.subgraph_trainset(src_train, src)
        trg_nodes = [x for x in map(lambda x, y: x.union(y), trg_test, trg_train)]
        trg_train = self.subgraph_trainset(trg_train, trg, True)
        return src_nodes, trg_nodes, src_train, trg_train

    def partition(self, src=0, src_k=30, trg_k=125, share_triplets=True):
        trg = 1 - src
        if share_triplets:
            trg_triplets = self.share_triplets(self.data.triples[src], self.data.triples[trg],
                                               self.train_set[src], self.train_map[src])
            src_triplets = self.share_triplets(self.data.triples[trg], self.data.triples[src],
                                               self.train_set[trg], self.train_map[trg])
        else:
            src_triplets, trg_triplets = reversed(self.data.triples)
        g0 = self.construct_graph(src_triplets, cnt_as_weight=True)
        print('construct src graph complete, total nodes={0}, total edges={1}'
              .format(len(g0.nodes), len(g0.edges)))
        mincut, src_nodes = nxmetis.partition(g0, src_k)
        print('src graph partition complete, mincut=', mincut)
        src_train, trg_train = self.subgraph_trainset(src_nodes, src)
        print('filter trainset complete')
        # g1 = self.construct_graph(trg_triplets, None, keep_inter_edges=True)
        g1 = self.construct_graph(trg_triplets, trg_train, keep_inter_edges=False)
        print('construct trg graph complete')
        mincut, trg_nodes = nxmetis.partition(g1, trg_k)
        print('trg graph partition complete, mincut=', mincut)
        return src_nodes, trg_nodes, src_train, self.subgraph_trainset(trg_nodes, trg, True)

    def eval_align(self, src_sets, trg_sets, which=0, opname='minus', rhs=None):
        src_sets = [set(l) for l in src_sets]
        trg_sets = [set(l) for l in trg_sets]
        if opname == 'minus':
            op = lambda x, y: x - y
        elif opname == 'cross':
            op = lambda x, y: x.intersection(y)
        else:
            raise NotImplementedError
        if rhs:
            src_sets = [op(s, set(rhs[0])) for i, s in enumerate(src_sets)]
            trg_sets = [op(t, set(rhs[1])) for i, t in enumerate(trg_sets)]

        src_sets = [set(self.train_map[which][i] for i in s) for s in src_sets]

        s2t, t2s = overlaps(src_sets, trg_sets), overlaps(trg_sets, src_sets)
        s2t, t2s = torch.from_numpy(s2t), torch.from_numpy(t2s)
        s2t, t2s = torch.topk(s2t, k=5)[0], torch.topk(t2s, k=5)[0]
        print(s2t.sum(dim=1).mean().item(), '\n', s2t.sum(dim=1).numpy())
        print(t2s.sum(dim=1).mean().item(), '\n', t2s.sum(dim=1).numpy())
        stat(np.array([len(s) for s in src_sets], dtype=float), 'src align')
        stat(np.array([len(s) for s in trg_sets], dtype=float), 'trg align')
        # stat(s2t, 's2t', True)s2t
        # stat(t2s, 't2s', True)

    def eval_partition(self, src_nodes, trg_nodes, src_train, trg_train, *args, **kwargs):
        srclen = np.array([len(s) for s in src_nodes], dtype=float)
        trglen = np.array([len(s) for s in trg_nodes], dtype=float)
        srctlen = np.array([len(s) for s in src_train], dtype=float)
        trgtlen = np.array([len(s) for s in trg_train], dtype=float)
        stat(srclen, 'src', False)
        stat(trglen, 'trg', False)
        stat(srctlen, 'src train', False)
        stat(trgtlen, 'trg train', False)
        stat(srctlen / srclen, 'src ratio', False)
        stat(trgtlen / trglen, 'trg ratio', False)

        print("--Nodes")
        self.eval_align(src_nodes, trg_nodes)

        print("--Train Nodes")
        self.eval_align(src_nodes, trg_nodes, opname='cross', rhs=self.train_set)

        print('--Eval Nodes')
        self.eval_align(src_nodes, trg_nodes, rhs=self.train_set)