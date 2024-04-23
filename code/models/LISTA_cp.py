#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
file  : LISTA_cp.py
author: Xiaohan Chen
email : chernxh@tamu.edu
last_modified : 2018-10-21

Implementation of Learned ISTA with weight coupling.
"""

import numpy as np
import tensorflow as tf
import utils.train

from utils.tf import shrink_free
from models.LISTA_base import LISTA_base

class LISTA_cp (LISTA_base):

    """
    Implementation of learned ISTA with weight coupling constraint.
    """
    def __init__ (self, A, T, lam, untied, coord, scope,adapt,bias,normalize,function_type):
        """
        :prob:  : Instance of Problem class, describing problem settings.
        :T      : Number of layers (depth) of this LISTA model.
        :lam    : Initial value of thresholds of shrinkage functions.
        :untied : Whether weights are shared within layers.
        """
        self._A   = A.astype (np.float32)
        self._T   = T
        self._lam = lam
        self._M   = self._A.shape [0]
        self._N   = self._A.shape [1]

        self._scale = 1.001 * np.linalg.norm (A, ord=2)**2
        self._theta = (self._lam / self._scale).astype(np.float32)
        if coord:
            self._theta = np.ones ((self._N, 1), dtype=np.float32) * self._theta

        self._untied = untied
        self._coord  = coord
        self._scope  = scope
        self._adapt = adapt
        self._use_bias = bias
        self._normalize = normalize
        self._function_type = function_type
        """ Set up layers."""
        self.setup_layers()


    def setup_layers(self):
        """
        Implementation of LISTA model proposed by LeCun in 2010.

        :prob: Problem setting.
        :T: Number of layers in LISTA.
        :returns:
            :layers: List of tuples ( name, xh_, var_list )
                :name: description of layers.
                :xh: estimation of sparse code at current layer.
                :var_list: list of variables to be trained seperately.

        """
        Ws_    = []
        thetas_ = []
        if(self._use_bias):
            bias_ = []

        W = (np.transpose (self._A) / self._scale).astype (np.float32)

        with tf.variable_scope (self._scope, reuse=False) as vs:
            # constant
            self._kA_ = tf.constant (value=self._A, dtype=tf.float32)

            if not self._untied: # tied model
                Ws_.append (tf.get_variable (name='W', dtype=tf.float32,
                                             initializer=W))
                Ws_ = Ws_ * self._T

            for t in range (self._T):
                thetas_.append (tf.get_variable (name="theta_%d"%(t+1),
                                                 dtype=tf.float32,
                                                 initializer=self._theta))
                if self._untied: # untied model
                    Ws_.append (tf.get_variable (name="W_%d"%(t+1),
                                                 dtype=tf.float32,
                                                 initializer=W))
                if self._use_bias:
                    bias_.append(tf.get_variable(name="bias_%d"%(t+1),
                                                 dtype=tf.float32,
                                                 initializer=0.0))

        # Collection of all trainable variables in the model layer by layer.
        # We name it as `vars_in_layer` because we will use it in the manner:
        # vars_in_layer [t]
        if(self._use_bias):
            self.vars_in_layer = list(zip(Ws_,thetas_,bias_))
        else:
            self.vars_in_layer = list (zip (Ws_, thetas_))


    def inference (self, y_, x0_=None):
        xhs_  = [] # collection of the regressed sparse codes
        function_type = self._function_type
        if(self._normalize):
            stand_norm = 6.0
            y_norm = tf.norm(y_,axis=0)
            y_ = y_ * stand_norm / (y_norm+0.000001)

        if x0_ is None:
            batch_size = tf.shape (y_) [-1]
            xh_ = tf.zeros (shape=(self._N, batch_size), dtype=tf.float32)
        else:
            xh_ = x0_
        xhs_.append (xh_)

        with tf.variable_scope (self._scope, reuse=True) as vs:
            for t in range (self._T):
                #W_, theta_ = self.vars_in_layer [t]
                if(self._use_bias):
                    W_,theta_,bias__ = self.vars_in_layer[t]
                else:
                    W_, theta_ = self.vars_in_layer [t]
                    bias__ = 0

                res_ = y_ - tf.matmul (self._kA_, xh_)
                if(function_type == 0):
                    bais = tf.sqrt(tf.reduce_sum((tf.matmul(W_,res_)**2),axis= 0))/50.0
                elif(function_type == 1):
                    bais = tf.reduce_sum(tf.abs(tf.matmul(W_,res_)),axis= 0)/100.0
                elif(function_type == 2):
                    bais = tf.sqrt(tf.reduce_sum((res_**2),axis= 0))/50.0
                elif(function_type == 3):
                    bais = tf.reduce_sum(tf.abs(res_),axis= 0)/100.0
                #bais = tf.sqrt(tf.reduce_sum((tf.matmul(W_,res_)**2),axis= 0))/50.0
                if(self._adapt):
                    xh_ = shrink_free (xh_ + tf.matmul (W_, res_), theta_*bais+bias__)
                else:
                    xh_ = shrink_free (xh_ + tf.matmul (W_, res_), theta_)
                if(self._normalize):
                    xh_title = xh_ * (y_norm+0.000001) / stand_norm
                    xhs_.append (xh_title)
                else:
                    xhs_.append  (xh_)

        return xhs_

