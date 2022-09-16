from clustering import ClusteringMachine4ea
from clustergcn import ClusterGCN4ea
from utils import tab_printer
from args_handler import load_args
import sys
from load.kgs import read_kgs_from_folder
from largeEA.eval import *
from Dual_AMN.utils import *
import os
import networkx as nx
import warnings
import joblib
import pickle

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
warnings.filterwarnings('ignore')
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'


def kg_dict_small(kgs):
    kgs_dict = {}
    for e in kgs.kg1.entities_list:
        kgs_dict[e] = 1
    for e in kgs.kg2.entities_list:
        kgs_dict[e] = 2
    return kgs_dict


def kg_dict_large(kgs):
    kgs_dict = {}
    # for k, v in kgs.ent1.items():
    #     kgs_dict[v] = 1
    # for k, v in kgs.ent2.items():
    #     kgs_dict[v] = 2
    for e, v in kgs.entities1_id_dict.items():
        kgs_dict[v] = 1
    for e, v in kgs.entities2_id_dict.items():
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


def main():
    print(sys.argv)
    t1 = time.time()
    args = load_args(sys.argv[1])
    args.training_data = args.training_data + sys.argv[2] + '/'
    args.dataset_division = sys.argv[3]
    torch.manual_seed(args.seed)
    tab_printer(args)

    cm_path = args.cm_path + args.training_data.split('/')[-2] + '_subgraphs_' + \
              str(args.cluster_number) + "_" + str(args.cluster_size) + "/"
    cm_path_file = cm_path + "subgraphs"
    if os.path.exists(cm_path_file + 'not'):
        print('****** loading clustering machine ******')
        with open(cm_path_file, 'rb') as f:
            clustering_machine = pickle.load(f)
    else:
        if args.data_size == 'small' or args.data_size == 'medium':
            kgs = read_kgs_from_folder(args.training_data, args.dataset_division, args.alignment_module, args.ordered,
                                       remove_unlinked=False)
            kgs.ent_dict = kg_dict_small(kgs)
            kgs.size = [kgs.kg1.entities_num, kgs.kg2.entities_num]
            kgs.link = kgs.train_links + kgs.valid_links + kgs.test_links
        else:
            kgs = read_kgs_from_folder(args.training_data, args.dataset_division, args.alignment_module, args.ordered,
                                       remove_unlinked=False, large=True)
            kgs.ent_dict = kg_dict_small(kgs)
            kgs.size = [kgs.kg1.entities_num, kgs.kg2.entities_num]
            kgs.link = kgs.train_links + kgs.valid_links + kgs.test_links

        target = kgs.train_links + kgs.valid_links
        tt = time.time()

        train_dict = {i[1]: i[0] for i in target}
        if args.data_size == 'large':
            train_dict = {i[1]: i[0] for i in list(target)}
            graph_id = {}
            i = 0
            for k, v in kgs.kg1.entities_id_dict.items():
                if kgs.kg1.hr_dict.get(v, -1) != -1 or kgs.kg1.rt_dict.get(v, -1) != -1:
                    graph_id[v] = i
                    i += 1
            for k, v in kgs.kg2.entities_id_dict.items():
                if train_dict.get(v, -1) != -1:
                    graph_id[v] = graph_id[train_dict[v]]
                else:
                    if kgs.kg2.hr_dict.get(v, -1) != -1 or kgs.kg2.rt_dict.get(v, -1) != -1:
                        graph_id[v] = i
                        i += 1
        else:
            graph_id = {}
            i = 0
            for k, v in kgs.kg1.entities_id_dict.items():
                graph_id[v] = i
                i += 1
            for k, v in kgs.kg2.entities_id_dict.items():
                if train_dict.get(v, -1) != -1:
                    graph_id[v] = graph_id[train_dict[v]]
                else:
                    graph_id[v] = i
                    i += 1
        edges = []
        for t in kgs.kg1.relation_triples_list + kgs.kg2.relation_triples_list:
            edges.append((graph_id[t[0]], graph_id[t[-1]]))
        graph = nx.from_edgelist(edges)
        kgs.graph_id = graph_id
        graph_id_inv = {i: [] for i in list(graph_id.values())}
        for k, v in graph_id.items():
            graph_id_inv[v].append(k)
        kgs.graph_id_inv = graph_id_inv

        clustering_machine = ClusteringMachine4ea(args, graph, target, kgs)

        clustering_machine.decompose()
        clustering_machine.split_clusters()

        clustering_machine.fast_core_path()
        print('********* centrality-based subgraph generation cost: ' + str(time.time()-tt) + ' **********************')

        clustering_machine.fill_cluster()
        clustering_machine.cluster2batch()
        # if not os.path.exists(cm_path):
        #     os.mkdir(cm_path)
        # with open(cm_path_file, "wb") as f:
        #     pickle.dump(clustering_machine, f, protocol=pickle.HIGHEST_PROTOCOL)
    print('************* total partition time: ' + str(time.time()-t1) + ' *********************')

    model = ClusterGCN4ea(args, clustering_machine)
    model.run()


if __name__ == "__main__":
    main()

    
