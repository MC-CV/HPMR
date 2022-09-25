
import tensorflow as tf
import os
import sys
import pandas as pd
import copy
import random
from utility.helper import *
from utility.batch_test import *

def load_data(csv_file):
    tp = pd.read_csv(csv_file, sep='\t')
    return tp

class HPMR(object):
    def __init__(self, max_item_view, max_item_cart, max_item_buy, max_item_pwc, max_item_pwb, max_item_cwb, data_config):
        # argument settings
        self.model_type = 'HPMR'
        self.adj_type = args.adj_type
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.n_fold = 100
        self.wid=eval(args.wid)    # 0.1 for beibei, 0.01 for taobao
        self.buy_adj = data_config['buy_adj']
        self.pv_adj = data_config['pv_adj']
        self.cart_adj = data_config['cart_adj']
        self.pwc_adj = data_config['pwc_adj']
        self.pwb_adj = data_config['pwb_adj']
        self.cwb_adj = data_config['cwb_adj']
        self.lr = args.lr
        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size
        self.n_layers = args.gnn_layer
        self.decay = args.decay # 10 for beibei,1e-1 for taobao
        self.verbose = args.verbose
        self.max_item_view = max_item_view
        self.max_item_cart = max_item_cart
        self.max_item_buy = max_item_buy
        
        self.max_item_pwc = max_item_pwc
        self.max_item_pwb = max_item_pwb
        self.max_item_cwb = max_item_cwb      
        self.coefficient = eval(args.coefficient) # 0.0/6, 5.0/6,1.0/6 for beibei and 1.0/6, 4.0/6, 1.0/6 for taobao
        self.alpha = eval(args.alpha)
        self.n_relations=3

        '''
        *********************************************************
        Create Placeholder for Input Data & Dropout.
        '''
        # placeholder definition
        self.input_u = tf.placeholder(tf.int32, [None, 1], name="input_uid")
        self.lable_view = tf.placeholder(tf.int32, [None, self.max_item_view], name="lable_view")
        self.lable_cart = tf.placeholder(tf.int32, [None, self.max_item_cart], name="lable_cart")
        self.lable_buy = tf.placeholder(tf.int32, [None, self.max_item_buy], name="lable_buy")
        self.lable_pwc = tf.placeholder(tf.int32, [None, self.max_item_pwc], name="lable_pwc")
        self.lable_pwb = tf.placeholder(tf.int32, [None, self.max_item_pwb], name="lable_pwb")
        self.lable_cwb = tf.placeholder(tf.int32, [None, self.max_item_cwb], name="lable_cwb")


        self.users = tf.placeholder(tf.int32, shape=(None,))
        self.pos_items = tf.placeholder(tf.int32, shape=(None,))

        self.node_dropout = tf.placeholder(tf.float32, shape=[None])
        self.mess_dropout = tf.placeholder(tf.float32, shape=[None])

        """
        *********************************************************
        Create Model Parameters (i.e., Initialize Weights).
        """
        # initialization of model parameters
        self.weights = self._init_weights()

        """
        *********************************************************
        Compute Graph-based Representations of all users & items via Message-Passing Mechanism of Graph Neural Networks.
        """

        self.ua_embeddings, self.ia_embeddings, self.r0, self.r1, self.r2, self.r_pwc, self.r_pwb, self.r_cwb = self._create_gcn_embed()

        """
        *********************************************************
        The module of the denoise.
        """
        
        # first_denoise
        self.ua_vc_clear, self.ua_vc_noise = self.denoise(self.ua_embeddings[0], self.ua_embeddings[1])
        self.ia_vc_clear, self.ia_vc_noise = self.denoise(self.ia_embeddings[0], self.ia_embeddings[1])
    
        self.ua_vb_clear, self.ua_vb_noise = self.denoise(self.ua_embeddings[0], self.ua_embeddings[2])
        self.ia_vb_clear, self.ia_vb_noise = self.denoise(self.ia_embeddings[0], self.ia_embeddings[2])
        
        self.ua_cb_clear, self.ua_cb_noise = self.denoise(self.ua_embeddings[1], self.ua_embeddings[2])
        self.ia_cb_clear, self.ia_cb_noise = self.denoise(self.ia_embeddings[1], self.ia_embeddings[2])

        
        # re-enhance_denoise
        self.ua_vc_clear_re, _ = self.denoise(self.ua_vc_noise, self.ua_embeddings[0])
        self.ia_vc_clear_re, _ = self.denoise(self.ia_vc_noise, self.ia_embeddings[0])
    
        self.ua_vb_clear_re, _ = self.denoise(self.ua_vb_noise, self.ua_embeddings[0])
        self.ia_vb_clear_re, _ = self.denoise(self.ia_vb_noise, self.ia_embeddings[0])
        
        self.ua_cb_clear_re, _ = self.denoise(self.ua_cb_noise, self.ua_embeddings[1])
        self.ia_cb_clear_re, _ = self.denoise(self.ia_cb_noise, self.ia_embeddings[1])

        self.ua_vc_clear_re *= args.re_mult
        self.ia_vc_clear_re *= args.re_mult
        self.ua_vb_clear_re *= args.re_mult
        self.ia_vb_clear_re *= args.re_mult
        self.ua_cb_clear_re *= args.re_mult
        self.ia_cb_clear_re *= args.re_mult
        
        """for training"""
        """embeddings for unique loss"""           

        self.ua_unique_embs = [self.ua_vc_noise, self.ua_vb_noise, self.ua_cb_noise]
        self.ia_unique_embs = [self.ia_vc_noise, self.ia_vb_noise, self.ia_cb_noise]
        
        
        """embeddings for re-enhance"""
        all_vc_noise_embs = tf.reduce_mean(self.gnns_transfer(tf.concat([self.ua_vc_clear_re, self.ia_vc_clear_re], axis=0), self.A_fold_hat_pv, args.transfer_gnn_layer, self.r0, 'clear'), 0)
        vc_noise_embs_user, vc_noise_embs_item = tf.split(all_vc_noise_embs, [self.n_users, self.n_items], 0)

        all_vb_noise_embs = tf.reduce_mean(self.gnns_transfer(tf.concat([self.ua_vb_clear_re, self.ia_vb_clear_re], axis=0), self.A_fold_hat_pv, args.transfer_gnn_layer, self.r0, 'clear'), 0)
        vb_noise_embs_user, vb_noise_embs_item = tf.split(all_vb_noise_embs, [self.n_users, self.n_items], 0)        
        
        all_cb_noise_embs = tf.reduce_mean(self.gnns_transfer(tf.concat([self.ua_cb_clear_re, self.ia_cb_clear_re], axis=0), self.A_fold_hat_cart, args.transfer_gnn_layer, self.r1, 'clear'), 0)
        cb_noise_embs_user, cb_noise_embs_item = tf.split(all_cb_noise_embs, [self.n_users, self.n_items], 0)        
        
        
        """embeddings for transfer"""
        all_vc_clear_embs = tf.reduce_mean(self.gnns_transfer(tf.concat([self.ua_vc_clear, self.ia_vc_clear], axis=0), self.A_fold_hat_cart, args.transfer_gnn_layer, self.r1, 'clear'), 0)
        vc_clear_embs_user, vc_clear_embs_item = tf.split(all_vc_clear_embs, [self.n_users, self.n_items], 0)

        all_vb_clear_embs = tf.reduce_mean(self.gnns_transfer(tf.concat([self.ua_vb_clear, self.ia_vb_clear], axis=0), self.A_fold_hat_buy, args.transfer_gnn_layer, self.r2, 'clear'), 0)
        vb_clear_embs_user, vb_clear_embs_item = tf.split(all_vb_clear_embs, [self.n_users, self.n_items], 0)  
        
        all_cb_clear_embs = tf.reduce_mean(self.gnns_transfer(tf.concat([self.ua_cb_clear, self.ia_cb_clear], axis=0), self.A_fold_hat_buy, args.transfer_gnn_layer, self.r2, 'clear'), 0)
        cb_clear_embs_user, cb_clear_embs_item = tf.split(all_cb_clear_embs, [self.n_users, self.n_items], 0)           

        """
        *********************************************************
        Establish the final representations for user-item pairs in batch.
        for test
        """
        self.u_g_embeddings = tf.nn.embedding_lookup(self.ua_embeddings[-1], self.users)
        self.pos_i_g_embeddings = tf.nn.embedding_lookup(self.ia_embeddings[-1], self.pos_items)
        
        self.dot = tf.einsum('ac,bc->abc', self.u_g_embeddings, self.pos_i_g_embeddings)
        self.batch_ratings = tf.einsum('ajk,lk->aj', self.dot, self.r2)    
           
        self.uid = []       
        for idx, ua_embedding in enumerate(self.ua_embeddings):
            # View
            if idx == 0:
                uid_tmp = tf.nn.embedding_lookup((ua_embedding+vc_noise_embs_user+vb_noise_embs_user)/3, self.input_u)
                uid_tmp = tf.reshape(uid_tmp, [-1, self.emb_dim])
                self.uid.append(uid_tmp)
            # Cart
            if idx == 1:
                uid_tmp = tf.nn.embedding_lookup((ua_embedding+vc_clear_embs_user+cb_noise_embs_user)/3, self.input_u)
                uid_tmp = tf.reshape(uid_tmp, [-1, self.emb_dim])
                self.uid.append(uid_tmp)
            # Buy
            if idx == 2:
                uid_tmp = tf.nn.embedding_lookup((ua_embedding+vb_clear_embs_user+cb_clear_embs_user)/3, self.input_u)
                uid_tmp = tf.reshape(uid_tmp, [-1, self.emb_dim])
                self.uid.append(uid_tmp) 
                    
        self.uid_main = []
        for idx, ua_embedding in enumerate(self.ua_embeddings):
                uid_tmp = tf.nn.embedding_lookup(ua_embedding, self.input_u)
                uid_tmp = tf.reshape(uid_tmp, [-1, self.emb_dim])
                self.uid_main.append(uid_tmp)            
 
        self.noise_uid = []
        for idx, ua_embedding in enumerate(self.ua_unique_embs):
            uid_tmp = tf.nn.embedding_lookup(ua_embedding, self.input_u)
            uid_tmp = tf.reshape(uid_tmp, [-1, self.emb_dim])
            self.noise_uid.append(uid_tmp)        
        
        # transfer predict
        self.pos_rv = self._get_pos_emb(self.ia_embeddings[0], self.lable_view, self.uid[0], self.r0)
        self.pos_rc = self._get_pos_emb(self.ia_embeddings[1], self.lable_cart, self.uid[1], self.r1)
        self.pos_rb = self._get_pos_emb(self.ia_embeddings[2], self.lable_buy, self.uid[2], self.r2)
        
        # main process predict
        self.pos_rv_main = self._get_pos_emb(self.ia_embeddings[0], self.lable_view, self.uid_main[0], self.r0)
        self.pos_rc_main = self._get_pos_emb(self.ia_embeddings[1], self.lable_cart, self.uid_main[1], self.r1)
        self.pos_rb_main = self._get_pos_emb(self.ia_embeddings[2], self.lable_buy, self.uid_main[2], self.r2)
        
        # unique predict
        self.pos_pwc_noise = self._get_pos_emb(self.ia_unique_embs[0], self.lable_pwc, self.noise_uid[0], self.r_pwc)
        self.pos_pwb_noise = self._get_pos_emb(self.ia_unique_embs[1], self.lable_pwb, self.noise_uid[1], self.r_pwb)
        self.pos_cwb_noise = self._get_pos_emb(self.ia_unique_embs[2], self.lable_cwb, self.noise_uid[2], self.r_cwb)

        self.mf_loss, self.emb_loss = self.create_non_sampling_loss()
                      
        self.loss = self.mf_loss + self.emb_loss
        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def _get_pos_emb(self, ia_embeddings, lable_beh, uid_beh, r):
        pos_beh = tf.nn.embedding_lookup(ia_embeddings, lable_beh)
        pos_num_beh = tf.cast(tf.not_equal(lable_beh, self.n_items), 'float32')
        pos_beh = tf.einsum('ab,abc->abc', pos_num_beh, pos_beh)
        pos_beh = tf.einsum('ac,abc->abc', uid_beh, pos_beh)
        return tf.einsum('ajk,lk->aj', pos_beh, r)

    def _init_weights(self):
        all_weights = dict()

        initializer = tf.contrib.layers.xavier_initializer()
        all_weights['user_embedding'] = tf.Variable(initializer([self.n_users, self.emb_dim]),
                                                    name='user_embedding')
        all_weights['item_embedding'] = tf.Variable(initializer([self.n_items, self.emb_dim]),
                                                    name='item_embedding')
        all_weights['relation_embedding'] = tf.Variable(initializer([self.n_relations, self.emb_dim]),
                                                    name='relation_embedding')
        print('using xavier initialization')

        self.weight_size_list = [self.emb_dim] + [self.emb_dim] * max(self.n_layers, args.transfer_gnn_layer)

        for k in range(max(self.n_layers, args.transfer_gnn_layer)):
            all_weights['W_rel_%d%s' % (k,'clear')] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k + 1]]), name='W_rel_%d%s' % (k,'clear'))
            
            all_weights['W_gc_%d%s' % (k,'clear')] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k + 1]]), name='W_gc_%d%s' % (k,'clear'))
            all_weights['b_gc_%d%s' % (k,'clear')] = tf.Variable(
                initializer([1, self.weight_size_list[k + 1]]), name='b_gc_%d%s' % (k,'clear'))

            all_weights['W_bi_%d%s' % (k,'clear')] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k + 1]]), name='W_bi_%d%s' % (k,'clear'))
            all_weights['b_bi_%d%s' % (k,'clear')] = tf.Variable(
                initializer([1, self.weight_size_list[k + 1]]), name='b_bi_%d%s' % (k,'clear'))

            all_weights['W_rel_%d%s' % (k,'noise')] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k + 1]]), name='W_rel_%d%s' % (k,'noise'))
            
            all_weights['W_gc_%d%s' % (k,'noise')] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k + 1]]), name='W_gc_%d%s' % (k,'noise'))
            all_weights['b_gc_%d%s' % (k,'noise')] = tf.Variable(
                initializer([1, self.weight_size_list[k + 1]]), name='b_gc_%d%s' % (k,'noise'))

            all_weights['W_bi_%d%s' % (k,'noise')] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k + 1]]), name='W_bi_%d%s' % (k,'noise'))
            all_weights['b_bi_%d%s' % (k,'noise')] = tf.Variable(
                initializer([1, self.weight_size_list[k + 1]]), name='b_bi_%d%s' % (k,'noise'))
        return all_weights

       
    def denoise(self, origin_emb, target_emb):
        res_array = tf.expand_dims(tf.reduce_sum(tf.multiply(origin_emb,target_emb),axis=1),-1)*target_emb
        norm_num = tf.norm(target_emb, axis=1)*tf.norm(target_emb, axis=1)+1e-12
        clear_emb = res_array/tf.expand_dims(norm_num,-1)
        noise_emb = origin_emb - clear_emb
        if False:
            a = tf.cast(tf.reduce_sum(tf.multiply(origin_emb,target_emb),axis=1)>=0, tf.float32)
            clear_emb *= tf.expand_dims(a,-1)
        return clear_emb*0.1, noise_emb*0.1     
    
    def _split_A_hat(self, X):
        A_fold_hat = []

        fold_len = (self.n_users + self.n_items) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = self.n_users + self.n_items
            else:
                end = (i_fold + 1) * fold_len

            A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
        return A_fold_hat

    def _split_A_hat_node_dropout(self, X):
        A_fold_hat = []

        fold_len = (self.n_users + self.n_items) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = self.n_users + self.n_items
            else:
                end = (i_fold + 1) * fold_len

            temp = self._convert_sp_mat_to_sp_tensor(X[start:end])
            n_nonzero_temp = X[start:end].count_nonzero()
            A_fold_hat.append(self._dropout_sparse(temp, 1 - self.node_dropout[0], n_nonzero_temp))

        return A_fold_hat

    def _split_A_hat_node_without_dropout(self, X):
        A_fold_hat = []

        fold_len = (self.n_users + self.n_items) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = self.n_users + self.n_items
            else:
                end = (i_fold + 1) * fold_len

            temp = self._convert_sp_mat_to_sp_tensor(X[start:end])
            n_nonzero_temp = X[start:end].count_nonzero()
            A_fold_hat.append(self._dropout_sparse(temp, 1, n_nonzero_temp))

        return A_fold_hat

    def mess_drop(self, embs):
        return tf.nn.dropout(embs, 1 - self.mess_dropout[0])

    def gnns_transfer(self, allEmbed, A_fold_hat, layers, r, flag):
        ego_embeddings = allEmbed
        all_embeddings = [ego_embeddings]
        all_r = [r]
        for index in range(layers):            
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], all_embeddings[-1]))
            norm_embeddings = tf.concat(temp_embed, 0)
            norm_embeddings = tf.multiply(norm_embeddings, r)
            if args.encoder == 'lightgcn':                
                lightgcn_embeddings = norm_embeddings
                # embeddings_tmp = self.mess_drop(lightgcn_embeddings)  
                embeddings_tmp = lightgcn_embeddings         
            elif args.encoder == 'gccf':
                gccf_embeddings = tf.nn.leaky_relu(norm_embeddings)
                # embeddings_tmp = self.mess_drop(gccf_embeddings)
                embeddings_tmp = gccf_embeddings
            elif args.encoder == 'gcn':
                gcn_embeddings = tf.nn.leaky_relu(
                    tf.matmul(norm_embeddings, self.weights['W_gc_%d%s' % (index, flag)]) + self.weights[
                        'b_gc_%d%s' % (index, flag)])
                # embeddings_tmp = self.mess_drop(gcn_embeddings)
                embeddings_tmp = gcn_embeddings
            elif args.encoder == 'ghcf':
                ghcf_embeddings = tf.nn.leaky_relu(
                    tf.matmul(norm_embeddings, self.weights['W_gc_%d%s' % (index, flag)]))
                # embeddings_tmp = self.mess_drop(ghcf_embeddings)  
                embeddings_tmp = ghcf_embeddings
            elif args.encoder == 'ngcf':
                gcn_embeddings = tf.nn.leaky_relu(
                    tf.matmul(norm_embeddings, self.weights['W_gc_%d%s' % (index, flag)]) + self.weights[
                        'b_gc_%d%s' % (index, flag)])
                bi_embeddings = tf.multiply(ego_embeddings, gcn_embeddings)
                bi_embeddings = tf.nn.leaky_relu(
                    tf.matmul(bi_embeddings, self.weights['W_bi_%d%s' % (index, flag)]) + self.weights['b_bi_%d%s' % (index, flag)])
                # embeddings_tmp = self.mess_drop(gcn_embeddings + bi_embeddings)   
                embeddings_tmp = gcn_embeddings + bi_embeddings    
            r = tf.matmul(r, self.weights['W_rel_%d%s' % (index, flag)])        
            all_r.append(r)
            all_embeddings.append(embeddings_tmp)
        return all_embeddings

    def gnns(self, allEmbed, A_fold_hat, layers, r, flag):
        ego_embeddings = allEmbed
        all_embeddings = [ego_embeddings]
        all_r = [r]
        for index in range(layers):            
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], all_embeddings[-1]))
            norm_embeddings = tf.concat(temp_embed, 0)
            norm_embeddings = tf.multiply(norm_embeddings, r)
            if args.encoder == 'lightgcn':                
                lightgcn_embeddings = norm_embeddings
                embeddings_tmp = self.mess_drop(lightgcn_embeddings)            
            elif args.encoder == 'gccf':
                gccf_embeddings = tf.nn.leaky_relu(norm_embeddings)
                embeddings_tmp = self.mess_drop(gccf_embeddings)
            elif args.encoder == 'gcn':
                gcn_embeddings = tf.nn.leaky_relu(
                    tf.matmul(norm_embeddings, self.weights['W_gc_%d%s' % (index, flag)]) + self.weights[
                        'b_gc_%d%s' % (index, flag)])
                embeddings_tmp = self.mess_drop(gcn_embeddings)
            elif args.encoder == 'ghcf':
                ghcf_embeddings = tf.nn.leaky_relu(
                    tf.matmul(norm_embeddings, self.weights['W_gc_%d%s' % (index, flag)]))
                embeddings_tmp = self.mess_drop(ghcf_embeddings)                
            elif args.encoder == 'ngcf':
                gcn_embeddings = tf.nn.leaky_relu(
                    tf.matmul(norm_embeddings, self.weights['W_gc_%d%s' % (index, flag)]) + self.weights[
                        'b_gc_%d%s' % (index, flag)])
                bi_embeddings = tf.multiply(ego_embeddings, gcn_embeddings)
                bi_embeddings = tf.nn.leaky_relu(
                    tf.matmul(bi_embeddings, self.weights['W_bi_%d%s' % (index, flag)]) + self.weights['b_bi_%d%s' % (index, flag)])
                embeddings_tmp = self.mess_drop(gcn_embeddings + bi_embeddings)      
            r = tf.matmul(r, self.weights['W_rel_%d%s' % (index, flag)])        
            all_r.append(r)
            all_embeddings.append(embeddings_tmp)
        return all_embeddings, all_r
    
    def _create_gcn_embed(self):
        # node dropout.
        self.A_fold_hat_buy = self._split_A_hat_node_dropout(self.buy_adj)
        self.A_fold_hat_pv = self._split_A_hat_node_dropout(self.pv_adj)
        self.A_fold_hat_cart = self._split_A_hat_node_dropout(self.cart_adj)

        self.A_fold_hat_pwc = self._split_A_hat_node_dropout(self.pwc_adj)
        self.A_fold_hat_pwb = self._split_A_hat_node_dropout(self.pwb_adj)
        self.A_fold_hat_cwb = self._split_A_hat_node_dropout(self.cwb_adj)
        
        embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)

        u_g_embeddings = []
        i_g_embeddings = []
        
        r0 = tf.nn.embedding_lookup(self.weights['relation_embedding'], 0)
        r0 = tf.reshape(r0, [-1, self.emb_dim])

        r1 = tf.nn.embedding_lookup(self.weights['relation_embedding'], 1)
        r1 = tf.reshape(r1, [-1, self.emb_dim])

        r2 = tf.nn.embedding_lookup(self.weights['relation_embedding'], 2)
        r2 = tf.reshape(r2, [-1, self.emb_dim])
  
                      
        all_embeddings_pv, all_r0 = self.gnns(embeddings, self.A_fold_hat_pv, self.n_layers, r0, 'clear')
        all_embeddings_cart, all_r1 = self.gnns(embeddings, self.A_fold_hat_cart, self.n_layers, r1, 'clear')
        all_embeddings_buy, all_r2 = self.gnns(embeddings, self.A_fold_hat_buy, self.n_layers, r2, 'clear')

        _, all_pwc_noise = self.gnns(embeddings, self.A_fold_hat_pwc, self.n_layers, r0, 'noise')
        _, all_pwb_noise = self.gnns(embeddings, self.A_fold_hat_pwb, self.n_layers, r0, 'noise')
        _, all_cwb_noise = self.gnns(embeddings, self.A_fold_hat_cwb, self.n_layers, r1, 'noise')

        all_final_embs = [all_embeddings_pv, all_embeddings_cart, all_embeddings_buy]
        
        for idx, all_embedding in enumerate(all_final_embs):
            all_embedding = tf.stack(all_embedding, 1)
            all_embedding = tf.reduce_mean(all_embedding, axis=1, keepdims=False)
            u_g_embedding_tmp, i_g_embedding_tmp = tf.split(all_embedding, [self.n_users, self.n_items], 0)
            u_g_embeddings.append(u_g_embedding_tmp)
            i_g_embeddings.append(i_g_embedding_tmp)        
        
        all_r0=tf.reduce_mean(all_r0,0)
        all_r1=tf.reduce_mean(all_r1,0)
        all_r2=tf.reduce_mean(all_r2,0)

        all_pwc_noise=tf.reduce_mean(all_pwc_noise,0)
        all_pwb_noise=tf.reduce_mean(all_pwb_noise,0)
        all_cwb_noise=tf.reduce_mean(all_cwb_noise,0)
        
        return u_g_embeddings, i_g_embeddings, all_r0, all_r1, all_r2, all_pwc_noise, all_pwb_noise, all_cwb_noise            

    def create_non_sampling_loss(self):       
        # the transfer and main process 
        temps = []
        temps_mains = []
        for idx, ia_embs in enumerate(self.ia_embeddings):
            temps.append(tf.einsum('ab,ac->bc', ia_embs, ia_embs)\
               * tf.einsum('ab,ac->bc', self.uid[idx], self.uid[idx]))
            temps_mains.append(tf.einsum('ab,ac->bc', ia_embs, ia_embs)\
               * tf.einsum('ab,ac->bc', self.uid_main[idx], self.uid_main[idx]))
            
        # the unique process
        temps_noise = []
        for idx, ia_embs in enumerate(self.ia_unique_embs):
            temps_noise.append(tf.einsum('ab,ac->bc', ia_embs, ia_embs)\
               * tf.einsum('ab,ac->bc', self.noise_uid[idx], self.noise_uid[idx])) 
                   
        loss1 = 0
        loss2 = 0
        loss3 = 0
        losses = [loss1, loss2, loss3]
        
        self.pos_rs = [self.pos_rv, self.pos_rc, self.pos_rb]
        self.pos_mains = [self.pos_rv_main, self.pos_rc_main, self.pos_rb_main]
        self.rs = [self.r0, self.r1, self.r2]
        
        self.pos_beh_noise = [self.pos_pwc_noise, self.pos_pwb_noise, self.pos_cwb_noise]
        self.rs_noises = [self.r_pwc, self.r_pwb, self.r_cwb]
        for idx in range(len(losses)):
            # the main process loss
            losses[idx] += self.alpha[0]*self.wid[idx]*tf.reduce_sum(temps_mains[idx] * tf.matmul(self.rs[idx], self.rs[idx], transpose_a=True))
            losses[idx] += self.alpha[0]*tf.reduce_sum((1.0 - self.wid[idx]) * tf.square(self.pos_mains[idx]) - 2.0 * self.pos_mains[idx])
            
            # the transfer loss
            losses[idx] += self.alpha[1]*self.wid[idx]*tf.reduce_sum(temps[idx] * tf.matmul(self.rs[idx], self.rs[idx], transpose_a=True))
            losses[idx] += self.alpha[1]*tf.reduce_sum((1.0 - self.wid[idx]) * tf.square(self.pos_rs[idx]) - 2.0 * self.pos_rs[idx])

            # the unique loss of cart
            if idx == 2:
                losses[1] += self.alpha[2]*self.wid[idx]*tf.reduce_sum(temps_noise[idx] * tf.matmul(self.rs_noises[idx], self.rs_noises[idx], transpose_a=True))
                losses[1] += self.alpha[2]*tf.reduce_sum((1.0 - self.wid[idx]) * tf.square(self.pos_beh_noise[idx]) - 2.0 * self.pos_beh_noise[idx])  
            # the unique loss of view
            else:
                losses[0] += self.alpha[2]*self.wid[idx]*tf.reduce_sum(temps_noise[idx] * tf.matmul(self.rs_noises[idx], self.rs_noises[idx], transpose_a=True))
                losses[0] += self.alpha[2]*tf.reduce_sum((1.0 - self.wid[idx]) * tf.square(self.pos_beh_noise[idx]) - 2.0 * self.pos_beh_noise[idx])                               
      

        loss = self.coefficient[0] * losses[0] + self.coefficient[1] * losses[1] + self.coefficient[2] * losses[2]         
        regularizer = self.alpha[0]*(tf.nn.l2_loss(self.uid_main[-1])) +\
            self.alpha[1]*(tf.nn.l2_loss(self.uid[-1]) + tf.nn.l2_loss(self.ia_embeddings[-1])) +\
            self.alpha[2]*(tf.nn.l2_loss(self.noise_uid[-1])+tf.nn.l2_loss(self.ia_unique_embs[-1]))        

        emb_loss = self.decay * regularizer

        return loss, emb_loss

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)

    def _dropout_sparse(self, X, keep_prob, n_nonzero_elems):
        """
        Dropout for sparse tensors.
        """
        noise_shape = [n_nonzero_elems]
        random_tensor = keep_prob
        random_tensor += tf.random_uniform(noise_shape)
        dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
        pre_out = tf.sparse_retain(X, dropout_mask)

        return pre_out * tf.div(1., keep_prob)


def get_lables(temp_set,k=0.9999):
    max_item = 0
    item_lenth = []
    for i in temp_set:
        item_lenth.append(len(temp_set[i]))
        if len(temp_set[i]) > max_item:
            max_item = len(temp_set[i])
    item_lenth.sort()

    max_item = item_lenth[int(len(item_lenth) * k)-1]

    # print max_item
    for i in temp_set:
        if len(temp_set[i]) > max_item:
            temp_set[i] = temp_set[i][0:max_item]
        while len(temp_set[i]) < max_item:
            temp_set[i].append(n_items)
    return max_item, temp_set


def get_train_instances1(view_lable, cart_lable, buy_lable, pwc_lable, pwb_lable, cwb_lable):
    user_train, view_item, cart_item, buy_item = [], [], [], []
    pwc_item, pwb_item, cwb_item = [], [], []
    for i in buy_lable.keys():
        user_train.append(i)
        buy_item.append(buy_lable[i])
        if i not in view_lable.keys():
            view_item.append([n_items] * max_item_view)
        else:
            view_item.append(view_lable[i])

        if i not in cart_lable.keys():
            cart_item.append([n_items] * max_item_cart)
        else:
            cart_item.append(cart_lable[i])
            
        if i not in pwc_lable.keys():
            pwc_item.append([n_items] * max_item_pwc)
        else:
            pwc_item.append(pwc_lable[i])
            
        if i not in pwb_lable.keys():
            pwb_item.append([n_items] * max_item_pwb)
        else:
            pwb_item.append(pwb_lable[i])
            
        if i not in cwb_lable.keys():
            cwb_item.append([n_items] * max_item_cwb)
        else:
            cwb_item.append(cwb_lable[i])

    user_train = np.array(user_train)
    view_item = np.array(view_item)
    cart_item = np.array(cart_item)
    buy_item = np.array(buy_item)

    pwc_item = np.array(pwc_item)
    pwb_item = np.array(pwb_item)
    cwb_item = np.array(cwb_item)
    
    user_train = user_train[:, np.newaxis]
    return user_train, view_item, cart_item, buy_item, pwc_item, pwb_item, cwb_item

if __name__ == '__main__':

    random.seed(42)
    tf.set_random_seed(42)
    np.random.seed(42)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    log_dir = 'log/' + args.dataset + '/' + os.path.basename(__file__)
    
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    import datetime
    log_file = open(log_dir + '/log' + str(datetime.datetime.now()), 'w')

    def my_hook_out(text):
        log_file.write(text)
        log_file.flush()
        return 1, 0, text
    
    from print_hook import PrintHook
    ph_out = PrintHook()
    ph_out.Start(my_hook_out)
    
    print("Use gpu id:", args.gpu_id)
    for arg in vars(args):
        print(arg + '=' + str(getattr(args, arg)))
 
    # logger.saveDefault = True   
    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items

    """
        *********************************************************
        Generate the Laplacian matrix, where each entry defines the decay factor (e.g., p_ui) between two connected nodes.
        """
    pre_adj,pre_adj_pv,pre_adj_cart,pre_adj_mat_pwc,pre_adj_mat_pwb,pre_adj_mat_cwb = data_generator.get_adj_mat() #

    config['buy_adj'] = pre_adj
    config['pv_adj'] = pre_adj_pv
    config['cart_adj'] = pre_adj_cart
    config['pwc_adj'] = pre_adj_mat_pwc
    config['pwb_adj'] = pre_adj_mat_pwb
    config['cwb_adj'] = pre_adj_mat_cwb
    print('use the pre adjcency matrix')



    n_users, n_items = data_generator.n_users, data_generator.n_items

    train_items = np.load(data_generator.path + '/train_items.npy', allow_pickle='TRUE').item()
    pv_set = np.load(data_generator.path + '/pv_set.npy', allow_pickle='TRUE').item()
    cart_set = np.load(data_generator.path + '/cart_set.npy', allow_pickle='TRUE').item()

    pv_wo_cart_set = np.load(data_generator.path + '/pv_wo_cart.npy', allow_pickle='TRUE').item()
    pv_wo_buy_set = np.load(data_generator.path + '/pv_wo_buy.npy', allow_pickle='TRUE').item()
    cart_wo_buy_set = np.load(data_generator.path + '/cart_wo_buy.npy', allow_pickle='TRUE').item()
    
    max_item_buy, buy_lable = get_lables(train_items)
    max_item_view, view_lable = get_lables(pv_set)
    max_item_cart, cart_lable = get_lables(cart_set)

    max_item_pwc, pwc_lable = get_lables(pv_wo_cart_set)
    max_item_pwb, pwb_lable = get_lables(pv_wo_buy_set)
    max_item_cwb, cwb_lable = get_lables(cart_wo_buy_set)
    
    t0 = time()

    model = HPMR(max_item_view, max_item_cart, max_item_buy, max_item_pwc, max_item_pwb, max_item_cwb, data_config=config)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    sess.run(tf.global_variables_initializer())
    cur_best_pre_0 = 0.
    print('without pretraining.')

    run_time = 1

    loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []

    stopping_step = 0
    should_stop = False

    user_train1, view_item1, cart_item1, buy_item1, pwc_item1, pwb_item1, cwb_item1 = get_train_instances1(view_lable, cart_lable, buy_lable, pwc_lable, pwb_lable, cwb_lable)

    best_hr = 0
    base_save_path = './checkpoints/'+str(os.path.basename(__file__).split(".")[0])+'/'+args.dataset
    for epoch in range(args.epoch):

        shuffle_indices = np.random.permutation(np.arange(len(user_train1)))
        user_train1 = user_train1[shuffle_indices]
        view_item1 = view_item1[shuffle_indices]
        cart_item1 = cart_item1[shuffle_indices]
        buy_item1 = buy_item1[shuffle_indices]
        pwc_item1 = pwc_item1[shuffle_indices]
        pwb_item1 = pwb_item1[shuffle_indices]
        cwb_item1 = cwb_item1[shuffle_indices]

        t1 = time()
        loss, mf_loss, emb_loss = 0., 0., 0.

        n_batch = int(len(user_train1) / args.batch_size)

        for idx in range(n_batch):
            start_index = idx * args.batch_size
            end_index = min((idx + 1) * args.batch_size, len(user_train1))

            u_batch = user_train1[start_index:end_index]
            v_batch = view_item1[start_index:end_index]
            c_batch = cart_item1[start_index:end_index]
            b_batch = buy_item1[start_index:end_index]

            pwc_batch = pwc_item1[start_index:end_index]
            pwb_batch = pwb_item1[start_index:end_index]
            cwb_batch = cwb_item1[start_index:end_index]

            _, batch_loss, batch_mf_loss, batch_emb_loss = sess.run(
                [model.opt, model.loss, model.mf_loss, model.emb_loss],
                feed_dict={model.input_u: u_batch,
                           model.lable_buy: b_batch,
                           model.lable_view: v_batch,
                           model.lable_cart: c_batch,
                           model.lable_pwc: pwc_batch,
                           model.lable_pwb: pwb_batch,
                           model.lable_cwb: cwb_batch,
                           model.node_dropout: eval(args.node_dropout),
                           model.mess_dropout: eval(args.mess_dropout)})
            loss += batch_loss / n_batch
            mf_loss += batch_mf_loss / n_batch
            emb_loss += batch_emb_loss / n_batch

        if np.isnan(loss) == True:
            print('ERROR: loss is nan.')
            sys.exit()

        # print the test evaluation metrics each 10 epochs; pos:neg = 1:10.
        if (epoch + 1) % 10 != 0:
            if args.verbose > 0 and epoch % args.verbose == 0:
                perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f]' % (
                    epoch, time() - t1, loss, mf_loss, emb_loss)
                print(perf_str)
            continue

        t2 = time()
        users_to_test = list(data_generator.test_set.keys())
        ret = test(sess, model, users_to_test, drop_flag=True)
        if args.save_emb == 1:
            if ret['hit_ratio'][0] > best_hr:
                specific_fname = '/'+'final'+str(epoch)+'_'+str(args.encoder)
                best_hr = ret['hit_ratio'][0]
                if os.path.exists(base_save_path) != True:
                    os.makedirs(base_save_path)
                tmp_user = sess.run(model.ua_embeddings,feed_dict={
                            model.node_dropout: [0.] * args.gnn_layer,
                            model.mess_dropout: [0.] * args.gnn_layer})
                tmp_item = sess.run(model.ia_embeddings,feed_dict={
                            model.node_dropout: [0.] * args.gnn_layer,
                            model.mess_dropout: [0.] * args.gnn_layer})                                                                
                best_fname = base_save_path+specific_fname       
        t3 = time()

        loss_loger.append(loss)
        rec_loger.append(ret['recall'])
        pre_loger.append(ret['precision'])
        ndcg_loger.append(ret['ndcg'])
        hit_loger.append(ret['hit_ratio'])

        if args.verbose > 0:
            perf_str = 'Epoch %d [%.1fs + %.1fs]:, recall=[%.5f], ' \
                       'precision=[%.5f], hit=[%.5f], ndcg=[%.5f]' % \
                       (
                           epoch, t2 - t1, t3 - t2, ret['recall'][0],
                           ret['precision'][0], ret['hit_ratio'][0],
                           ret['ndcg'][0])
            print(perf_str)

            """
            *********************************************************
            Get the performance w.r.t. different sparsity levels.
            """
            if 0:
                users_to_test_list, split_state = data_generator.get_sparsity_split()

                for i, users_to_test in enumerate(users_to_test_list):
                    ret = test(sess, model, users_to_test, drop_flag=True)

                    final_perf = "recall=[%s], precision=[%s], hit=[%s], ndcg=[%s]" % \
                         ('\t'.join(['%.5f' % r for r in ret['recall']]),
                          '\t'.join(['%.5f' % r for r in ret['precision']]),
                          '\t'.join(['%.5f' % r for r in ret['hit_ratio']]),
                          '\t'.join(['%.5f' % r for r in ret['ndcg']]))
                    print(final_perf)
        

        cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best_pre_0,
                                                                    stopping_step, expected_order='acc', flag_step=5)

        # *********************************************************
        # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
        if should_stop == True:
            if args.save_emb == 1:
                best_hr = ret['hit_ratio'][0]
                print('best_hr =', best_hr)
                if os.path.exists(base_save_path) != True:
                    os.makedirs(base_save_path)
                np.save(best_fname+'_user.npy',tmp_user)
                np.save(best_fname+'_item.npy',tmp_item)
                print('Best user & item embeddings are saved!')                                                                         
            break

        # *********************************************************
        # save the user & item embeddings for pretraining.

    recs = np.array(rec_loger)
    pres = np.array(pre_loger)
    ndcgs = np.array(ndcg_loger)
    hit = np.array(hit_loger)

    best_rec_0 = max(recs[:, 0])
    idx = list(recs[:, 0]).index(best_rec_0)

    final_perf = "Best Iter=[%d]@[%.1f]\trecall=[%s], precision=[%s], hit=[%s], ndcg=[%s]" % \
                 (idx, time() - t0, '\t'.join(['%.5f' % r for r in recs[idx]]),
                  '\t'.join(['%.5f' % r for r in pres[idx]]),
                  '\t'.join(['%.5f' % r for r in hit[idx]]),
                  '\t'.join(['%.5f' % r for r in ndcgs[idx]]))
    print(final_perf)





