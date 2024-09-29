'''
Tensorflow implementation of SR-HyperFM
'''

import os
import numpy as np
import tensorflow as tf
from time import time
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score, log_loss
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
from tensorflow.contrib.rnn import GRUCell

def normalize(inputs, epsilon=1e-8):
    '''
    Applies layer normalization
    Args:
        inputs: A tensor with 2 or more dimensions
        epsilon: A floating number to prevent Zero Division
    Returns:
        A tensor with the same shape and data dtype
    '''
    inputs_shape = inputs.get_shape()
    params_shape = inputs_shape[-1:]

    mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
    beta = tf.Variable(tf.zeros(params_shape))
    gamma = tf.Variable(tf.ones(params_shape))
    normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
    outputs = gamma * normalized + beta

    return outputs

class SRFM():
    def __init__(self, args, feature_size, run_cnt):

        self.feature_size = feature_size  # denote as n, dimension of concatenated features
        self.field_size = args.field_size  # denote as M, number of total feature fields
        self.embedding_size = args.embedding_size  # denote as d, size of the feature embedding
        self.blocks = args.blocks  # number of the blocks
        self.heads = args.heads  # number of the heads
        self.block_shape = args.block_shape
        self.output_size = args.block_shape[-1]
        self.has_residual = args.has_residual
        self.deep_layers = args.deep_layers  # whether to joint train with deep networks as described in paper

        # 新增参数
        self.hypergraph_sample = args.hypergraph_sample
        self.hypergraph_feature = args.hypergraph_feature
        self.hyperedge_num = args.hyperedge_num
        
        self.k = args.k
        self.ks = args.ks

        self.batch_norm = args.batch_norm
        self.batch_norm_decay = args.batch_norm_decay
        self.drop_keep_prob = args.dropout_keep_prob
        self.l2_reg = args.l2_reg
        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.optimizer_type = args.optimizer_type

        self.data_path = args.data_path

        self.save_path = args.save_path + str(run_cnt) + '/'
        self.is_save = args.is_save
        if (args.is_save == True and os.path.exists(self.save_path) == False):
            os.makedirs(self.save_path)

        self.verbose = args.verbose
        self.random_seed = args.random_seed
        self.loss_type = args.loss_type
        self.eval_metric = roc_auc_score
        self.best_loss = 1.0
        self.greater_is_better = args.greater_is_better
        self.train_result, self.valid_result = [], []
        self.train_loss, self.valid_loss = [], []

        self._init_graph()

    def _init_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():

            tf.set_random_seed(self.random_seed)

            self.feat_index = tf.placeholder(tf.int32, shape=[None, None],
                                             name="feat_index")  # None * M
            self.feat_value = tf.placeholder(tf.float32, shape=[None, None],
                                             name="feat_value")  # None * M
            self.label = tf.placeholder(tf.float32, shape=[None, 1], name="label")  # None * 1

            # In our implementation, the shape of dropout_keep_prob is [3], used in 3 different places.
            self.dropout_keep_prob = tf.placeholder(tf.float32, shape=[None], name="dropout_keep_prob")
            self.train_phase = tf.placeholder(tf.bool, name="train_phase")


            self.weights = self._initialize_weights()
    
            self.p_out = tf.Variable(tf.random_normal([1, self.embedding_size], stddev=0.1), name='p_out')
            self.W_out = tf.Variable(tf.random_normal([self.embedding_size, self.embedding_size], stddev=0.1), name='W_out')
            self.b_out = tf.Variable(tf.random_normal([self.embedding_size], stddev=0.1), name='b_out')
    
            self.H_fea = tf.Variable(tf.random_uniform([self.field_size, self.hyperedge_num], minval=0, maxval=1), trainable=True, name="H_fea")
    
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
    
            self.embeddings = tf.nn.embedding_lookup(self.weights["feature_embeddings"], self.feat_index)  # None * M * d
            self.embeddings = tf.multiply(self.embeddings, tf.expand_dims(self.feat_value, -1))  # None * M * d
            self.embeddings = tf.nn.dropout(self.embeddings, self.dropout_keep_prob[1])  # None * M * d
    
            with tf.variable_scope("sample_hypergraph_mlp"):
                self.sample_hypergraph_dense = tf.layers.Dense(units=self.embedding_size, activation=tf.nn.relu)
            with tf.variable_scope("feature_hypergraph_mlp"):
                self.feature_hypergraph_dense = tf.layers.Dense(units=self.embedding_size, activation=tf.nn.relu)
    
            self.saver = tf.train.Saver(max_to_keep=5)
            init = tf.global_variables_initializer()
            self.sess = self._init_session()
            self.sess.run(init)
            self.sess.run(tf.variables_initializer(tf.global_variables()))
            self.count_param()

    def _build_sample_hypergraph(self, feat_values):
        with self.graph.as_default():
            self.sample_hypergraph_edges = []
        
            for feature_idx in range(self.field_size):
                feature_values = feat_values[:, feature_idx]
                unique_values = np.unique(feature_values)
        
                for value in unique_values:
                    samples_with_value = np.where(feature_values == value)[0]
                    if len(samples_with_value) > 1:
                        self.sample_hypergraph_edges.append(samples_with_value)
        
            if not self.sample_hypergraph_edges:
                for i in range(self.batch_size):
                    self.sample_hypergraph_edges.append(np.array([i], dtype=np.int32))
        
            max_len = max(len(edge) for edge in self.sample_hypergraph_edges)
            padded_edges = [np.pad(edge, (0, max_len - len(edge)), 'constant', constant_values=-1) for edge in self.sample_hypergraph_edges]
            padded_edges = np.array(padded_edges).astype(np.int32)
            self.sample_hypergraph_edges = tf.constant(padded_edges, dtype=tf.int32)

    
    def sample_hypergraph_aggregation(self, embeddings):
        sample_hypergraph_node_embeddings = tf.reshape(embeddings, shape=[-1, self.field_size * self.embedding_size])
        aggregated_embeddings = []
    
        for edge in tf.unstack(self.sample_hypergraph_edges):
            valid_nodes = tf.boolean_mask(edge, edge >= 0)
            edge_node_embeddings = tf.nn.embedding_lookup(sample_hypergraph_node_embeddings, valid_nodes)
            edge_agg = tf.reduce_sum(edge_node_embeddings, axis=0)
            aggregated_embeddings.append(edge_agg)
    
        self.sample_hypergraph_hyperedge_embeddings = tf.stack(aggregated_embeddings)
        self.sample_hypergraph_hyperedge_embeddings = tf.expand_dims(self.sample_hypergraph_hyperedge_embeddings, axis=0)
        with tf.variable_scope("sample_hypergraph_mlp", reuse=tf.AUTO_REUSE):
            self.sample_hypergraph_hyperedge_embeddings = self.sample_hypergraph_dense(self.sample_hypergraph_hyperedge_embeddings)

 
    def _build_feature_hypergraph(self):
        with self.graph.as_default():        
            self.H_fea = tf.Variable(tf.random_uniform([self.field_size, self.hyperedge_num], minval=0, maxval=1), trainable=True, name="H_fea")
    

    def feature_hypergraph_aggregation(self, embeddings):
        feature_hypergraph_node_embeddings = embeddings
        edge_weights = tf.nn.softmax(self.H_fea, dim=0)
    
        mask = edge_weights >= 0.5
        edge_weights = tf.where(mask, edge_weights, tf.zeros_like(edge_weights))
    
        current_batch_size = tf.shape(feature_hypergraph_node_embeddings)[0]
    
        edge_weights = tf.expand_dims(edge_weights, axis=0)
        edge_weights = tf.tile(edge_weights, [current_batch_size, 1, 1])
    
        node2edge_embeddings = tf.transpose(feature_hypergraph_node_embeddings, perm=[0, 2, 1])
        node2edge_embeddings = tf.matmul(node2edge_embeddings, edge_weights)
        node2edge_embeddings = tf.transpose(node2edge_embeddings, perm=[0, 2, 1])
    
        with tf.variable_scope("feature_hypergraph_mlp", reuse=tf.AUTO_REUSE):
            self.feature_hypergraph_hyperedge_embeddings = self.feature_hypergraph_dense(node2edge_embeddings)
    
        print("Shape of self.feature_hypergraph_hyperedge_embeddings:", self.feature_hypergraph_hyperedge_embeddings.shape)

    
    def dual_hypergraph_message_passing(self, feat_values, embeddings):
        if self.hypergraph_sample:
            self._build_sample_hypergraph(feat_values)
            self.sample_hypergraph_aggregation(embeddings)
    
        if self.hypergraph_feature:
            #self._build_feature_hypergraph()
            self.feature_hypergraph_aggregation(embeddings)
    
        print("Shape of sample_hypergraph_embeddings:", self.sample_hypergraph_hyperedge_embeddings.shape)
        print("Shape of feature_hypergraph_hyperedge_embeddings:", self.feature_hypergraph_hyperedge_embeddings.shape)


    def count_param(self):
        k = (np.sum([np.prod(v.get_shape().as_list())
                     for v in tf.trainable_variables()]))

        print("total parameters :%d" % k)
        print("extra parameters : %d" % (k - self.feature_size * self.embedding_size))

    def _init_session(self):
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        return tf.Session(config=config)

    def _initialize_weights(self):
        weights = dict()

        # embeddings
        weights["feature_embeddings"] = tf.Variable(
            tf.random_normal([self.feature_size, self.embedding_size], 0.0, 0.01),
            name="feature_embeddings")  # feature_size(n) * d

        input_size = self.embedding_size

        # dense layers
        if self.deep_layers != None:
            num_layer = len(self.deep_layers)
            layer0_size = self.field_size * self.embedding_size
            glorot = np.sqrt(2.0 / (layer0_size + self.deep_layers[0]))
            weights["layer_0"] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(layer0_size, self.deep_layers[0])), dtype=np.float32)
            weights["bias_0"] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers[0])),
                                            dtype=np.float32)  # 1 * layers[0]
            for i in range(1, num_layer):
                glorot = np.sqrt(2.0 / (self.deep_layers[i - 1] + self.deep_layers[i]))
                weights["layer_%d" % i] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(self.deep_layers[i - 1], self.deep_layers[i])),
                    dtype=np.float32)  # layers[i-1] * layers[i]
                weights["bias_%d" % i] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers[i])),
                    dtype=np.float32)  # 1 * layer[i]
            glorot = np.sqrt(2.0 / (self.deep_layers[-1] + 1))
            weights["prediction_dense"] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(self.deep_layers[-1], 1)),
                dtype=np.float32, name="prediction_dense")
            weights["prediction_bias_dense"] = tf.Variable(
                np.random.normal(), dtype=np.float32, name="prediction_bias_dense")

        # ---------- prediciton weight ------------------#
        glorot = np.sqrt(2.0 / (input_size + 1)) # Glorot 初始化
        weights["prediction"] = tf.Variable(
            np.random.normal(loc=0, scale=glorot, size=(input_size, 1)),
            dtype=np.float32, name="prediction")
        weights["prediction_bias"] = tf.Variable(
            np.random.normal(), dtype=np.float32, name="prediction_bias")

        return weights

    def batch_norm_layer(self, x, train_phase, scope_bn):
        bn_train = batch_norm(x, decay=self.batch_norm_decay, center=True, scale=True, updates_collections=None,
                              is_training=True, reuse=None, trainable=True, scope=scope_bn)
        bn_inference = batch_norm(x, decay=self.batch_norm_decay, center=True, scale=True, updates_collections=None,
                                  is_training=False, reuse=True, trainable=True, scope=scope_bn)
        z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
        return z

    def get_batch(self, Xi, Xv, y, batch_size, index):
        start = index * batch_size
        end = (index + 1) * batch_size
        end = end if end < len(y) else len(y)
        return Xi[start:end], Xv[start:end], [[y_] for y_ in y[start:end]]

    # shuffle three lists simutaneously
    def shuffle_in_unison_scary(self, a, b, c):
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)
        np.random.set_state(rng_state)
        np.random.shuffle(c)

    def fit_on_batch(self, Xi, Xv, y):
        with self.graph.as_default():
            feat_values = np.reshape(Xv, [-1, self.field_size])
    
            feed_dict = {
                self.feat_index: Xi,
                self.feat_value: Xv,
                self.label: y,
                self.dropout_keep_prob: [1.0, 1.0, 1.0],
                self.train_phase: True
            }
    
            self.dual_hypergraph_message_passing(feat_values, self.embeddings)
    
            e_i = tf.reshape(self.feature_hypergraph_hyperedge_embeddings, [-1, self.embedding_size])  # [batch_size * hyperedge_num, embedding_size]
            a_prime_i = tf.reduce_sum(self.p_out * tf.nn.relu(tf.matmul(e_i, self.W_out) + self.b_out), axis=1)  # [batch_size * hyperedge_num]
    
            current_batch_size = tf.shape(self.feature_hypergraph_hyperedge_embeddings)[0]
            a_prime_i = tf.reshape(a_prime_i, [current_batch_size, self.hyperedge_num])  # [batch_size, hyperedge_num]
    
            a_out_i = tf.nn.softmax(a_prime_i, dim=1)  # [batch_size, hyperedge_num]
            a_out_i = tf.expand_dims(a_out_i, axis=2)  # [batch_size, hyperedge_num, 1]
            weighted_hyperedge_embeddings = a_out_i * self.feature_hypergraph_hyperedge_embeddings  # [batch_size, hyperedge_num, embedding_size]
            aggregated_hyperedge_embeddings = tf.reduce_sum(weighted_hyperedge_embeddings, axis=1)  # [batch_size, embedding_size]
    
            out = tf.add(tf.matmul(aggregated_hyperedge_embeddings, self.weights["prediction"]),
                              self.weights["prediction_bias"], name='logits')  # None * 1
    
            if self.deep_layers is not None:
                y_dense = tf.reshape(self.embeddings, shape=[-1, self.field_size * self.embedding_size])
                for i in range(0, len(self.deep_layers)):
                    y_dense = tf.add(tf.matmul(y_dense, self.weights["layer_%d" % i]),
                                          self.weights["bias_%d" % i])  # None * layer[i]
                    if self.batch_norm:
                        y_dense = self.batch_norm_layer(y_dense, train_phase=self.train_phase,
                                                             scope_bn="bn_%d" % i)
                    y_dense = tf.nn.relu(y_dense)
                    y_dense = tf.nn.dropout(y_dense, self.dropout_keep_prob[2])
                y_dense = tf.add(tf.matmul(y_dense, self.weights["prediction_dense"]),
                                      self.weights["prediction_bias_dense"], name='logits_dense')  # None * 1
                out += y_dense
    
            if self.loss_type == "logloss":
                out = tf.nn.sigmoid(out, name='pred')
                loss = tf.losses.log_loss(self.label, out)
            elif self.loss_type == "mse":
                loss = tf.nn.l2_loss(tf.subtract(self.label, out))
    
            if self.l2_reg > 0:
                if self.deep_layers is not None:
                    for i in range(len(self.deep_layers)):
                        loss += tf.contrib.layers.l2_regularizer(self.l2_reg)(self.weights["layer_%d" % i])
    
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                               beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(loss, global_step=self.global_step)
    
            step, loss, _ = self.sess.run([self.global_step, loss, optimizer], feed_dict=feed_dict)
            return step, loss
    
    # Since the train data is very large, they can not be fit into the memory at the same time.
    # We separate the whole train data into several files and call "fit_once" for each file.
    def fit_once(self, Xi_train, Xv_train, y_train,
                 epoch, file_count, Xi_valid=None,
                 Xv_valid=None, y_valid=None,
                 early_stopping=False):

        has_valid = Xv_valid is not None
        last_step = 0
        t1 = time()
        self.shuffle_in_unison_scary(Xi_train, Xv_train, y_train)
        total_batch = int(len(y_train) / self.batch_size)
        for i in range(total_batch):
            Xi_batch, Xv_batch, y_batch = self.get_batch(Xi_train, Xv_train, y_train, self.batch_size, i)
            step, loss = self.fit_on_batch(Xi_batch, Xv_batch, y_batch)
            last_step = step

        # evaluate training and validation datasets
        train_result, train_loss = self.evaluate(Xi_train, Xv_train, y_train)
        self.train_result.append(train_result)
        self.train_loss.append(train_loss)
        if has_valid:
            valid_result, valid_loss = self.evaluate(Xi_valid, Xv_valid, y_valid)
            self.valid_result.append(valid_result)
            self.valid_loss.append(valid_loss)
            if valid_loss < self.best_loss and self.is_save == True:
                old_loss = self.best_loss
                self.best_loss = valid_loss
                self.saver.save(self.sess, self.save_path + 'model.ckpt', global_step=last_step)
                print("[%d-%d] model saved!. Valid loss is improved from %.4f to %.4f"
                      % (epoch, file_count, old_loss, self.best_loss))

        if self.verbose > 0 and ((epoch - 1) * 9 + file_count) % self.verbose == 0:
            if has_valid:
                print(
                    "[%d-%d] train-result=%.4f, train-logloss=%.4f, valid-result=%.4f, valid-logloss=%.4f [%.1f s]" % (
                    epoch, file_count, train_result, train_loss, valid_result, valid_loss, time() - t1))
            else:
                print("[%d-%d] train-result=%.4f [%.1f s]" \
                      % (epoch, file_count, train_result, time() - t1))
        if has_valid and early_stopping and self.training_termination(self.valid_loss):
            return False
        else:
            return True

    def training_termination(self, valid_result):
        if len(valid_result) > 5:
            if self.greater_is_better:
                if valid_result[-1] < valid_result[-2] and \
                        valid_result[-2] < valid_result[-3] and \
                        valid_result[-3] < valid_result[-4] and \
                        valid_result[-4] < valid_result[-5]:
                    return True
            else:
                if valid_result[-1] > valid_result[-2] and \
                        valid_result[-2] > valid_result[-3] and \
                        valid_result[-3] > valid_result[-4] and \
                        valid_result[-4] > valid_result[-5]:
                    return True
        return False

    def predict(self, Xi, Xv):
        """
        :param Xi: list of list of feature indices of each sample in the dataset
        :param Xv: list of list of feature values of each sample in the dataset
        :return: predicted probability of each sample
        """
        # dummy y
        dummy_y = [1] * len(Xi)
        batch_index = 0
        Xi_batch, Xv_batch, y_batch = self.get_batch(Xi, Xv, dummy_y, self.batch_size, batch_index)
        y_pred = None
        while len(Xi_batch) > 0:
            num_batch = len(y_batch)
            feed_dict = {self.feat_index: Xi_batch,
                         self.feat_value: Xv_batch,
                         self.label: y_batch,
                         self.dropout_keep_prob: [1.0] * len(self.drop_keep_prob),
                         self.train_phase: False}
            batch_out = self.sess.run(self.out, feed_dict=feed_dict)

            if batch_index == 0:
                y_pred = np.reshape(batch_out, (num_batch,))
            else:
                y_pred = np.concatenate((y_pred, np.reshape(batch_out, (num_batch,))))

            batch_index += 1
            Xi_batch, Xv_batch, y_batch = self.get_batch(Xi, Xv, dummy_y, self.batch_size, batch_index)

        return y_pred

    def visualize(self, Xi, Xv, y):
        """
        :param Xi: list of list of feature indices of each sample in the dataset
        :param Xv: list of list of feature values of each sample in the dataset
        :return: predicted probability of each sample
        """

        # for visualization
        visualization = []

        # dummy y
        dummy_y = [1] * len(Xi)
        batch_index = 0
        Xi_batch, Xv_batch, y_batch = self.get_batch(Xi, Xv, dummy_y, self.batch_size, batch_index)
        y_pred = None
        while len(Xi_batch) > 0:
            num_batch = len(y_batch)
            feed_dict = {self.feat_index: Xi_batch,
                         self.feat_value: Xv_batch,
                         self.label: y_batch,
                         self.dropout_keep_prob: [1.0] * len(self.drop_keep_prob),
                         self.train_phase: False}
            batch_out, batch_vis = self.sess.run([self.out, self.v_list], feed_dict=feed_dict)


            if batch_index == 0:
                y_pred = np.reshape(batch_out, (num_batch,))
                vis = np.reshape(batch_vis, (num_batch, self.blocks, self.field_size, self.field_size))
            else:
                y_pred = np.concatenate((y_pred, np.reshape(batch_out, (num_batch,))))
                vis = np.concatenate(
                    (vis, np.reshape(batch_vis, (num_batch, self.blocks, self.field_size, self.field_size))))

            # record the sample along with the visualization
            for i in range(num_batch):
                l = []
                l.append(Xi_batch[i])
                l.append(Xv_batch[i])
                l.append(y_batch[i])
                l.append(batch_out[i])
                l.append(batch_vis[i])
                visualization.append(l)

            batch_index += 1
            Xi_batch, Xv_batch, y_batch = self.get_batch(Xi, Xv, dummy_y, self.batch_size, batch_index)

        np.save(self.data_path+"/visualization-" + str(self.ks) + ".npy", visualization)

        return visualization

    def evaluate(self, Xi, Xv, y):
        """
        :param Xi: list of list of feature indices of each sample in the dataset
        :param Xv: list of list of feature values of each sample in the dataset
        :param y: label of each sample in the dataset
        :return: metric of the evaluation
        """
        y_pred = self.predict(Xi, Xv)
        y_pred = np.clip(y_pred, 1e-6, 1 - 1e-6)
        return self.eval_metric(y, y_pred), log_loss(y, y_pred)

    def restore(self, save_path=None):
        if (save_path == None):
            save_path = self.save_path
        ckpt = tf.train.get_checkpoint_state(save_path)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            if self.verbose > 0:
                print("restored from %s" % (save_path))
