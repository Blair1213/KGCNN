# -*- coding: utf-8 -*-
# @Time    : 2021/9/16 下午9:29
# @Author  : xiaorui su
# @Email   :  suxiaorui19@mails.ucas.edu.cn
# @File    : multiAttention.py
# @Software : PyCharm


# -*- coding: utf-8 -*-

from keras.engine.topology import Layer
from keras import backend as K

# class AvgAggregator(Layer):
#     def __init__(self, activation: str ='relu', initializer='glorot_normal', regularizer=None,
#                  **kwargs):
#         super(AvgAggregator, self).__init__(**kwargs)
#         if activation == 'relu':
#             self.activation = K.relu
#         elif activation == 'tanh':
#             self.activation = K.tanh
#         else:
#             raise ValueError(f'`activation` not understood: {activation}')
#         self.initializer = initializer
#         self.regularizer = regularizer
#     def build(self, input_shape):
#         ent_embed_dim = input_shape[0][-1]
#         self.w = self.add_weight(name=self.name+'_w', shape=(ent_embed_dim, ent_embed_dim),
#                                  initializer=self.initializer, regularizer=self.regularizer)
#         self.b = self.add_weight(name=self.name+'_b', shape=(ent_embed_dim,), initializer='zeros')
#         super(SumAggregator, self).build(input_shape)



class SumAggregator(Layer):
    def __init__(self, activation: str ='relu', initializer='glorot_normal', regularizer=None,
                 **kwargs):
        super(SumAggregator, self).__init__(**kwargs)
        if activation == 'relu':
            self.activation = K.relu
        elif activation == 'tanh':
            self.activation = K.tanh
        else:
            raise ValueError(f'`activation` not understood: {activation}')
        self.initializer = initializer
        self.regularizer = regularizer

    def build(self, input_shape):
        ent_embed_dim = input_shape[0][-1]
        self.w = self.add_weight(name=self.name+'_w', shape=(ent_embed_dim, ent_embed_dim),
                                 initializer=self.initializer, regularizer=self.regularizer)
        self.b = self.add_weight(name=self.name+'_b', shape=(ent_embed_dim,), initializer='zeros')
        super(SumAggregator, self).build(input_shape)

    def call(self, inputs, **kwargs):
        entity, neighbor = inputs
        return self.activation(K.dot((entity + neighbor), self.w) + self.b)

    def compute_output_shape(self, input_shape):
        return input_shape[0]

class MultiAttention(Layer):
    def __init__(self, activation: str ='tanh', initializer='glorot_normal', regularizer=None,
                 **kwargs):
        super(MultiAttention,self).__init__(**kwargs)
        if activation == 'tanh':
            self.activation = K.tanh
        else:
            raise ValueError(f'`activation` not understood: {activation}')
        self.initializer = initializer
        self.regularizer = regularizer

    def build(self, input_shape):
        ent_embed_dim = input_shape[0][-1]
        rel_embed_dim = input_shape[1][-1]
        self.w_h = self.add_weight(name=self.name+'_wh', shape=(ent_embed_dim, ent_embed_dim),
                                 initializer=self.initializer, regularizer=self.regularizer)
        self.w_t = self.add_weight(name=self.name + '_wt', shape=(ent_embed_dim, ent_embed_dim),
                                   initializer=self.initializer, regularizer=self.regularizer)
        self.w_r = self.add_weight(name=self.name+'_wr', shape=(rel_embed_dim, ent_embed_dim),
                                 initializer=self.initializer, regularizer=self.regularizer)
        self.v_a = self.add_weight(name=self.name+'_va', shape=(rel_embed_dim*3, 1),
                                 initializer=self.initializer, regularizer=self.regularizer)
        super(MultiAttention,self).build(input_shape)

    def call(self,inputs,**kwargs):
        entity, relation, neighbor = inputs
        ##entity [none, 1,dim]
        ##relation [none,hop,dim_relation]
        ##neighbor [none,hop,dim]
        print("multi attention")
        hop_number = relation.shape[-2]

        f_entity = K.zeros_like(relation[:,:,0])   ##[?,hop]
        f_entity = K.reshape(f_entity,(-1,relation.shape[1],1)) ##[?,hop,1]

        entity_hat = K.batch_dot(f_entity,entity,[2,1])  ##[?,hop,dim]
        head_entity = K.dot(entity_hat,self.w_h) ##[?,hop_number, entity_dim]


        rel = K.dot(relation,self.w_r) ##[none,hop_number, entity_dim]
        tail_entity = K.dot(neighbor,self.w_t)
        print(head_entity.shape)
        head_rel = K.concatenate([head_entity,rel],axis=-1)
        head_rel_tail = K.concatenate([head_rel,tail_entity],axis=-1) ##[hop_number,entity_dim*3]


        output = K.softmax(K.dot(K.tanh(head_rel_tail),self.v_a),axis=-2)
        print("output of multiattention")
        print(output.shape)

        return output #[?,hop_number, 1]



