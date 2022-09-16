import pickle
import random
import time
import os

import numpy as np
import tensorflow.keras as keras
from Dual_AMN.evaluate import *
from Dual_AMN.fast_evaluate import test_by_faiss_batch, test_by_faiss_batch_one_by_one, \
    test_by_faiss_batch_hits1_one_by_one, test_by_faiss_batch_prf_one_by_one, compute_prf
from Dual_AMN.layer import NR_GraphAttention
from largeEA.utils import apply
from Dual_AMN.utils import get_matrix
import torch


class ClusterGCN4ea(object):
    """
    Training a ClusterGCN.
    """

    def __init__(self, args, clustering_machine, **kwargs):
        """
        :param ags: Arguments object.
        :param clustering_machine:
        """
        self.args = args
        self.clustering_machine = clustering_machine
        self.align_clusters = self.clustering_machine.align_clusters
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.whole_size = sum(self.clustering_machine.kgs.size)
        self.whole_size_r = 2 * self.clustering_machine.kgs.relations_num  # todo: large dataset need modify

        gpus = tf.config.experimental.list_physical_devices("GPU")
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

        # gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
        # self.sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
        # config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        # self.sess = tf.Session(config=config)
        self.make_batch()
        self.evaluater = evaluate(self.clustering_machine.kgs.test_links)

    def construct_adj(self, ent_sizes, rel_sizes, triples):
        entsz = ent_sizes[0] + ent_sizes[1]
        relsz = rel_sizes[0] + rel_sizes[1]
        adj_matrix, r_index, r_val, adj_features, rel_features = get_matrix(triples, entsz, relsz)
        adj_matrix, r_index, r_val, adj_features, rel_features = \
            adj_matrix, np.array(r_index), np.array(r_val), adj_features, rel_features
        adj_matrix = np.stack(adj_matrix.nonzero(), axis=1)
        rel_matrix, rel_val = np.stack(rel_features.nonzero(), axis=1), rel_features.data
        ent_matrix, ent_val = np.stack(adj_features.nonzero(), axis=1), adj_features.data
        return adj_matrix, adj_features, r_index, r_val, rel_features, rel_matrix, ent_matrix, ent_val

    def get_embedding(self, cluster, index_a, index_b, ent_ind, rel_ind, vec=None):
        if vec is None:
            inputs = [self.cluster_adj_matrix[cluster], self.cluster_r_index[cluster], self.cluster_r_val[cluster],
                      self.cluster_rel_matrix[cluster], self.cluster_ent_matrix[cluster]]
            inputs = [np.expand_dims(item, axis=0) for item in inputs] + [ent_ind, rel_ind]
            vec = self.get_emb.predict_on_batch(inputs)
        Lvec = np.array([vec[e] for e in index_a])
        Rvec = np.array([vec[e] for e in index_b])
        Lvec = Lvec / (np.linalg.norm(Lvec, axis=-1, keepdims=True) + 1e-5)
        Rvec = Rvec / (np.linalg.norm(Rvec, axis=-1, keepdims=True) + 1e-5)
        return Lvec, Rvec

    def get_ent_embedding(self, cluster, ent_ind, rel_ind, dense_shape):
        neg_ent_index = np.array(random.sample(range(self.whole_size), self.node_size)).reshape((1, -1))
        intra_neg_ent_index = np.array(random.sample(range(self.node_size), self.batch_size)).reshape((1, -1))
        inputs = [self.cluster_adj_matrix[cluster], self.cluster_r_index[cluster], self.cluster_r_val[cluster],
                  self.cluster_rel_matrix[cluster], self.cluster_ent_matrix[cluster]]
        inputs = [np.expand_dims(item, axis=0) for item in inputs] + \
                 [ent_ind, neg_ent_index, intra_neg_ent_index, rel_ind, dense_shape]
        vec = self.get_emb.predict_on_batch(inputs)
        vec = vec / (np.linalg.norm(vec, axis=-1, keepdims=True) + 1e-5)
        return vec

    def get_curr_embeddings(self, device=None):
        vec = self.get_embedding()
        vec = np.array(vec)
        sep = self.ent_sizes[0]
        vecs = vec[:sep], vec[sep:]
        vecs = apply(torch.from_numpy, *vecs)
        tf.compat.v1.reset_default_graph()
        return vecs if device is None else apply(lambda x: x.to(device), *vecs)

    def get_trgat(self, dropout_rate=0, gamma=3, lr=0.005, depth=2, **default_params):
        node_hidden = default_params['node_hidden']
        adj_input = Input(shape=(None, 2))
        index_input = Input(shape=(None, 2), dtype='int64')
        val_input = Input(shape=(None,))
        rel_adj = Input(shape=(None, 2))
        ent_adj = Input(shape=(None, 2))

        ent_ind = Input(batch_shape=(1, self.node_size), dtype='int64')
        neg_ent_ind = Input(batch_shape=(1, self.node_size), dtype='int64')
        intra_neg_ent_ind = Input(batch_shape=(1, self.batch_size), dtype='int64')
        rel_ind = Input(batch_shape=(1, self.rel_size), dtype='int64')
        dense_shape = Input(shape=(None, None), dtype='int64', name="dense_shape")

        embeds = TokenEmbedding(self.whole_size + 1, node_hidden, trainable=True, name="embed")(
            [ent_ind, neg_ent_ind, intra_neg_ent_ind])
        ent_emb, neg_ent_emb, intra_neg_ent_emb = embeds[0], embeds[1], embeds[2]
        embeds = TokenEmbedding(self.whole_size_r + 1, node_hidden, trainable=True)([rel_ind, rel_ind, rel_ind])
        rel_emb = embeds[0]

        def avg(tensor, size, node_size):
            adj = K.cast(K.squeeze(tensor[0], axis=0), dtype="int64")
            adj = tf.SparseTensor(indices=adj, values=tf.ones_like(adj[:, 0], dtype='float32'),
                                  dense_shape=(node_size, size))
            adj = tf.sparse.softmax(adj)
            return tf.sparse.sparse_dense_matmul(adj, tensor[1])

        opt = [rel_emb, adj_input, index_input, val_input, dense_shape]
        ent_feature = Lambda(avg, arguments={'size': self.node_size, 'node_size': self.node_size})([ent_adj, ent_emb])
        rel_feature = Lambda(avg, arguments={'size': self.rel_size, 'node_size': self.node_size})([rel_adj, rel_emb])

        e_encoder = NR_GraphAttention(self.node_size, activation="tanh",
                                      rel_size=self.rel_size,
                                      use_bias=True,
                                      depth=depth,
                                      triple_size=self.triple_size)

        r_encoder = NR_GraphAttention(self.node_size, activation="tanh",
                                      rel_size=self.rel_size,
                                      use_bias=True,
                                      depth=depth,
                                      triple_size=self.triple_size)

        out_feature = Concatenate(-1)([e_encoder([ent_feature] + opt), r_encoder([rel_feature] + opt)])
        # out_feature = Dropout(dropout_rate)(out_feature)
        print('out_feature:', out_feature.shape)

        alignment_input = Input(shape=(None, 2))

        def align_loss(tensor):
            def reconstruction_loss(embed1, embed2, drop=0.3):
                embed1 = Dropout(drop)(embed1)
                embed2 = Dropout(drop)(embed2)
                return K.sum(tf.reduce_mean(tf.square(embed1 - embed2), axis=1))

            def squared_dist(x, norm=False):
                source_vec, tgt_vec = x
                if norm:
                    source_vec = tf.nn.l2_normalize(source_vec, axis=-1)
                    tgt_vec = tf.nn.l2_normalize(tgt_vec, axis=-1)
                row_norms_source = tf.reduce_sum(tf.square(source_vec), axis=1)
                row_norms_source = tf.reshape(row_norms_source, [-1, 1])  # Column vector.
                row_norms_tgt = tf.reduce_sum(tf.square(tgt_vec), axis=1)
                row_norms_tgt = tf.reshape(row_norms_tgt, [1, -1])  # Row vector.
                return row_norms_source + row_norms_tgt - 2 * tf.matmul(source_vec, tgt_vec, transpose_b=True)

            def neg_pair_align_loss(embed1, embed2, margin_):
                neg_dis = squared_dist([embed1, K.stop_gradient(embed2)], norm=False)
                cro_clus_loss = (neg_dis - K.stop_gradient(K.mean(neg_dis, axis=-1, keepdims=True) + margin_)) / \
                                K.stop_gradient(K.std(neg_dis, axis=-1, keepdims=True))
                lamb, tau = 10, 10
                return K.mean(tf.reduce_logsumexp(lamb * cro_clus_loss + tau, axis=-1))

            def neg_align_loss(positive_dis, embed1, embed2, margin, ll=None, rr=None, stop_gradient=False):
                if stop_gradient:
                    embed2 = K.stop_gradient(embed2)
                neg_dis = squared_dist([embed1, embed2])
                cro_clus_loss = positive_dis - neg_dis + margin

                if ll is not None:
                    cro_clus_loss = cro_clus_loss * (1 - K.one_hot(indices=ll, num_classes=self.node_size) -
                                                     K.one_hot(indices=rr, num_classes=self.node_size))

                cro_clus_loss = (cro_clus_loss - K.stop_gradient(K.mean(cro_clus_loss, axis=-1, keepdims=True))) / \
                                K.stop_gradient(K.std(cro_clus_loss, axis=-1, keepdims=True))

                lamb, tau = 10, 10
                return tf.reduce_logsumexp(lamb * cro_clus_loss + tau, axis=-1)

            emb = tensor[1]
            emb = Dropout(dropout_rate)(emb)
            neighbor_emb = tensor[-1]
            input_emb = tensor[2]
            l, r = K.cast(tensor[0][0, :, 0], 'int32'), K.cast(tensor[0][0, :, 1], 'int32')
            l_emb, r_emb = K.gather(reference=emb, indices=l), K.gather(reference=emb, indices=r)

            pos_dis = K.sum(K.square(l_emb - r_emb), axis=-1, keepdims=True)

            l_loss = neg_align_loss(pos_dis, l_emb, emb, gamma, l, r)
            r_loss = neg_align_loss(pos_dis, r_emb, emb, gamma, l, r)

            # input_l_emb = K.gather(reference=tensor[2], indices=l)
            # input_r_emb = K.gather(reference=tensor[2], indices=r)
            # other_subgraph_emb = tensor[3]
            # intra_clu_emb = tensor[4]

            # neg_align_loss = neg_align_loss(pos_dis, input_l_emb, other_clu_emb, gamma/5., stop_gradient=True) + \
            #                  neg_align_loss(pos_dis, input_r_emb, other_clu_emb, gamma/5., stop_gradient=True)

            margin = 0.9
            # cross_subgraph_neg_align_loss = neg_pair_align_loss(input_l_emb, other_subgraph_emb, margin) + \
            #                                 neg_pair_align_loss(input_r_emb, other_subgraph_emb, margin)
            # neg_pair_align_loss(intra_clu_emb, other_clu_emb, margin)

            # ent_reconstruction_loss = reconstruction_loss(input_emb, neighbor_emb)

            # return K.mean(l_loss + r_loss) + \
            #        self.args.cross_neg_loss_w * cross_subgraph_neg_align_loss + \
            #        self.args.reconstruction_loss_w * ent_reconstruction_loss
            return K.mean(l_loss + r_loss)

        loss = Lambda(align_loss)([alignment_input, out_feature, ent_emb, neg_ent_emb, intra_neg_ent_emb, ent_feature])

        inputs = [adj_input, index_input, val_input, rel_adj, ent_adj]
        train_model = keras.Model(inputs=inputs + [alignment_input] + [ent_ind] + [neg_ent_ind] + [intra_neg_ent_ind] +
                                         [rel_ind] + [dense_shape],
                                  outputs=loss)
        # train_model.compile(loss=lambda y_true, y_pred: y_pred, optimizer=keras.optimizers.RMSprop(lr))
        train_model.compile(loss=lambda y_true, y_pred: y_pred, optimizer=tf.optimizers.RMSprop(lr))

        feature_model = keras.Model(inputs=inputs + [ent_ind] + [neg_ent_ind] + [intra_neg_ent_ind] +
                                           [rel_ind] + [dense_shape],
                                    outputs=out_feature)

        return train_model, feature_model

    def load_pair(self, link):
        dev_pair = self.update_devset(link.cpu().numpy())
        return dev_pair

    def update_devset(self, pairs):
        # pairs = [pairs[0], pairs[1]]
        devset = [[p, i] for i, p in enumerate(pairs)]
        dev_pair = np.array(devset).T
        return dev_pair

    def make_batch(self):
        """
        formalize each cluster as a batch for training
        :return: all batches
        """
        self.cluster_triples = {}
        self.cluster_sup = {}
        self.cluster_link = {}
        self.ent_sizes, rel_sizes = {}, {}
        self.cluster_adj_matrix, self.cluster_adj_features, self.cluster_r_index, self.cluster_r_val, \
        self.cluster_rel_features, self.cluster_rel_matrix, self.cluster_ent_matrix, self.cluster_ent_val \
            = {}, {}, {}, {}, {}, {}, {}, {}
        for cluster in self.clustering_machine.clusters:
            cm_path = self.args.cm_path + self.args.training_data.split('/')[-2] + '_subgraphs_' + \
                      str(self.args.cluster_number) + "_" + str(self.args.cluster_size) + "/"
            saved_data_path = cm_path + 'subgraph_' + str(cluster) + '_data'

            align_cluster = self.clustering_machine.align_clusters[cluster]
            self.cluster_triples[cluster] = align_cluster.triples
            self.cluster_sup[cluster] = align_cluster.train_ill
            self.cluster_link[cluster] = torch.tensor(align_cluster.test_ill)
            ent_sizes = [len(i) for i in align_cluster.ent_ids]
            rel_sizes = [len(i) for i in align_cluster.rel_ids]

            if os.path.exists(saved_data_path):
                print('loading subgraph info from', saved_data_path)
                (adj_matrix, adj_features, r_index, r_val, rel_features, rel_matrix, ent_matrix, ent_val) = \
                    pickle.load(open(saved_data_path, 'rb'))
            else:
                adj_matrix, adj_features, r_index, r_val, rel_features, rel_matrix, ent_matrix, ent_val = \
                    self.construct_adj(ent_sizes, rel_sizes, align_cluster.triples)
                saved = (adj_matrix, adj_features, r_index, r_val, rel_features, rel_matrix, ent_matrix, ent_val)
                # if not os.path.exists(saved_data_path):
                #     os.mkdir(saved_data_path)
                # pickle.dump(saved, open(saved_data_path, 'wb'))
                # print('subgraph info saved to', saved_data_path)

            self.cluster_adj_matrix[cluster] = adj_matrix
            self.cluster_adj_features[cluster] = adj_features
            self.cluster_r_index[cluster] = r_index
            self.cluster_r_val[cluster] = r_val
            self.cluster_rel_features[cluster] = rel_features
            self.cluster_rel_matrix[cluster] = rel_matrix
            self.cluster_ent_matrix[cluster] = ent_matrix
            self.cluster_ent_val[cluster] = ent_val
        self.node_size = max([self.cluster_adj_features[i].shape[0] for i in self.cluster_adj_features])
        self.rel_size = max([self.cluster_rel_features[i].shape[1] for i in self.cluster_rel_features])
        self.triple_size = max([len(self.cluster_adj_matrix[i]) for i in self.cluster_adj_matrix])  # todo:
        self.batch_size = self.args.batch_size
        default_params = dict(
            dropout_rate=self.args.dropout,
            node_size=self.node_size,
            rel_size=self.rel_size,
            depth=2,
            gamma=1,
            # todo
            node_hidden=128,
            rel_hidden=128,
            # node_hidden=12,
            # rel_hidden=12,
            triple_size=self.triple_size,
            batch_size=self.batch_size,
            lr=self.args.learning_rate
        )
        self.model, self.get_emb = self.get_trgat(**default_params)
        self.model.summary()

    def generate_input_from_cluster(self, cluster):
        sup = self.cluster_sup[cluster]
        np.random.shuffle(sup)

        adj_matrix, adj_features, r_index, r_val, rel_features, rel_matrix, ent_matrix, ent_val = \
            self.cluster_adj_matrix[cluster], self.cluster_adj_features[cluster], \
            self.cluster_r_index[cluster], self.cluster_r_val[cluster], self.cluster_rel_features[cluster], \
            self.cluster_rel_matrix[cluster], self.cluster_ent_matrix[cluster], self.cluster_ent_val[
                cluster]

        ent_index = np.array(list(self.clustering_machine.align_clusters[cluster].ent_maps[0].keys()) +
                             list(self.clustering_machine.align_clusters[cluster].ent_maps[
                                      1].keys())).reshape((1, -1))
        ent_index = np.concatenate(
            (ent_index, self.whole_size * np.ones(shape=(1, self.node_size - ent_index.shape[-1]),
                                                  dtype='int64')), axis=-1)
        rel_list1 = list(self.clustering_machine.align_clusters[cluster].rel_maps[0].keys()) + \
                    list(self.clustering_machine.align_clusters[cluster].rel_maps[1].keys())
        rel_list2 = [i + self.clustering_machine.kgs.relations_num for i in rel_list1]
        rel_index = np.array(rel_list1 + rel_list2).reshape((1, -1))
        rel_index = np.concatenate((rel_index, self.whole_size_r *
                                    np.ones(shape=(1, self.rel_size - rel_index.shape[-1]), dtype='int64')), axis=-1)

        dense_shape = np.expand_dims(np.array([len(adj_matrix), self.rel_size]), axis=0)
        return sup, adj_matrix, r_index, r_val, rel_matrix, ent_matrix, ent_index, rel_index, dense_shape

    def run(self):
        epoch = 150
        max_hits1 = 0
        for i in range(1, epoch):
            print()
            print("epoch:", i)
            epoch_t = time.time()
            self.node_count_seen = 0
            self.accumulated_training_loss = 0
            clusters = self.clustering_machine.clusters
            print("clusters:", len(clusters))
            np.random.shuffle(clusters)
            print('++++++++++++++++++++++ cluster order ++++++++++++++++: ' + str(clusters))

            for cluster in clusters:
                t = time.time()
                sup, adj_matrix, r_index, r_val, rel_matrix, ent_matrix, ent_index, rel_index, dense_shape = \
                    self.generate_input_from_cluster(cluster)
                # if len(sup) == 0:
                #     sup = [(0, 0)]
                for pairs in [sup[i * self.batch_size:(i + 1) * self.batch_size] for i in
                              range(len(sup) // self.batch_size + 1)]:
                    if len(pairs) == 0:
                        continue
                    neg_ent_index = np.array(random.sample(range(self.whole_size), self.node_size)).reshape((1, -1))
                    intra_neg_ent_index = np.array(random.sample(ent_index[0].tolist(), self.batch_size)).reshape(
                        (1, -1))
                    inputs = [adj_matrix, r_index, r_val, rel_matrix, ent_matrix, pairs]
                    inputs = [np.expand_dims(item, axis=0) for item in inputs] + \
                             [ent_index, neg_ent_index, intra_neg_ent_index, rel_index, dense_shape]
                    self.model.train_on_batch(inputs, np.zeros((1, 1)))
                print("****** successfully run cluster {} ******, time = {:.3f} s ".format(cluster, time.time() - t))

            print("epoch training time = {:.3f} s ".format(time.time() - epoch_t))
            # get embeddings for test
            ent_id_dict = {i: [] for i in range(self.clustering_machine.kgs.entities_num)}
            for cluster in self.clustering_machine.clusters:
                _, _, _, _, _, _, ent_index, rel_index, dense_shape = self.generate_input_from_cluster(cluster)
                ent_vec = self.get_ent_embedding(cluster, ent_index, rel_index, dense_shape)
                for ii, idx in enumerate(ent_index.tolist()[0]):
                    ent_id_dict[idx].append(ent_vec[ii])
            for idx in ent_id_dict:
                ent_id_dict[idx] = np.mean(np.array(ent_id_dict[idx]), axis=0)

            rest_set_1, rest_set_2 = [i[0] for i in self.clustering_machine.kgs.test_links], \
                                     [i[1] for i in self.clustering_machine.kgs.test_links]
            ent_emb = np.array([ent_id_dict[i] for i in ent_id_dict])
            source_vec = np.array([ent_emb[e] for e in rest_set_1])
            tgt_vec = np.array([ent_emb[e] for e in rest_set_2])

            self.evaluater.test(source_vec, tgt_vec)

            # test_by_faiss_batch(source_vec, tgt_vec)
            # test_by_faiss_batch_one_by_one(source_vec, tgt_vec, top_k=[1])

            # hits1 = test_by_faiss_batch_hits1_one_by_one(source_vec, tgt_vec)
            # if hits1 >= max_hits1:
            #     max_hits1 = hits1
            # else:
            #     print()
            #     print("early stop and max hits1:", max_hits1)

            #     # for p r f1 evaluation
            #     start = time.time()
            #     source_entities1 = self.clustering_machine.kgs.test_entities1 + \
            #                        list(set(self.clustering_machine.kgs.kg1.entities_list) -
            #                             set(self.clustering_machine.kgs.train_entities1) -
            #                             set(self.clustering_machine.kgs.valid_entities1) -
            #                             set(self.clustering_machine.kgs.test_entities1))
            #     candidate_entities2 = self.clustering_machine.kgs.test_entities2 + \
            #                           list(set(self.clustering_machine.kgs.kg2.entities_list) -
            #                                set(self.clustering_machine.kgs.train_entities2) -
            #                                set(self.clustering_machine.kgs.valid_entities2) -
            #                                set(self.clustering_machine.kgs.test_entities2))

            #     source_vec = np.array([ent_emb[e] for e in source_entities1])
            #     tgt_vec = np.array([ent_emb[e] for e in candidate_entities2])
            #     s_t_pairs = test_by_faiss_batch_prf_one_by_one(tgt_vec, source_vec)
            #     t_s_pairs = test_by_faiss_batch_prf_one_by_one(source_vec, tgt_vec)
            #     compute_prf(len(self.clustering_machine.kgs.test_entities1), s_t_pairs, t_s_pairs)
            #     print("evaluation time = {:.3f} s ".format(time.time() - start))

                # test_by_faiss_batch(source_vec, tgt_vec)
                # test_by_faiss_batch_one_by_one(source_vec, tgt_vec, top_k=[1])
                # exit(0)


class TokenEmbedding(keras.layers.Embedding):
    """Embedding layer with weights returned."""

    def compute_output_shape(self, input_shape):
        # return self.input_dim, self.output_dim
        return input_shape[1], self.output_dim

    def compute_mask(self, inputs, mask=None):
        return None

    def call(self, inputs):
        # return self.embeddings
        return [tf.squeeze(tf.nn.embedding_lookup(self.embeddings, inputs[0])),
                tf.squeeze(tf.nn.embedding_lookup(self.embeddings, inputs[1])),
                tf.squeeze(tf.nn.embedding_lookup(self.embeddings, inputs[2]))]