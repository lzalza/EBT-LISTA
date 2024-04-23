#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This code is implemented on the base of https://github.com/VITA-Group/ALISTA

import os , sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # BE QUIET!!!!

# timing
import time
from datetime import timedelta

from config import get_config
import utils.prob as problem
import utils.data as data
import utils.train as train

import numpy as np
import tensorflow as tf
try :
    from sklearn.feature_extraction.image \
            import extract_patches_2d, reconstruct_from_patches_2d
except Exception as e :
    pass


def setup_model(config , **kwargs) :
    untiedf = 'u' if config.untied else 't'
    coordf = 'c' if config.coord  else 's'
    use_bias = True if config.SNR != 'inf' and config.bias else False
    if config.net == 'LISTA' :
        """LISTA"""
        config.model = ("LISTA_T{T}_lam{lam}_{untiedf}_{coordf}_{exp_id}"
                        .format (T=config.T, lam=config.lam, untiedf=untiedf,
                                 coordf=coordf, exp_id=config.exp_id))
        from models.LISTA import LISTA
        model = LISTA (kwargs['A'], T=config.T, lam=config.lam,
                       untied=config.untied, coord=config.coord,
                       scope=config.scope)


    if config.net == 'LISTA_cp' :
        """LISTA-CP"""
        config.model = ("LISTA_cp_T{T}_lam{lam}_{untiedf}_{coordf}_{exp_id}"
                        .format (T=config.T, lam=config.lam, untiedf=untiedf,
                                 coordf=coordf, exp_id=config.exp_id))
        from models.LISTA_cp import LISTA_cp
        model = LISTA_cp (kwargs['A'], T=config.T, lam=config.lam,
                          untied=config.untied, coord=config.coord,
                          scope=config.scope,adapt = config.adaptive, bias = use_bias, normalize = config.normalize, function_type = config.function)

    if config.net == 'LISTA_ss' :
        """LISTA-SS"""
        config.model = ("LISTA_ss_T{T}_lam{lam}_p{p}_mp{mp}_"
                        "{untiedf}_{coordf}_{exp_id}"
                        .format (T=config.T, lam=config.lam, p=config.percent,
                                 mp=config.max_percent, untiedf=untiedf,
                                 coordf=coordf, exp_id=config.exp_id))
        from models.LISTA_ss import LISTA_ss
        model = LISTA_ss (kwargs['A'], T=config.T, lam=config.lam,
                          percent=config.percent, max_percent=config.max_percent,
                          untied=config.untied , coord=config.coord,
                          scope=config.scope)

    if config.net == 'LISTA_cpss' :
        """LISTA-CPSS"""
        config.model = ("LISTA_cpss_T{T}_lam{lam}_p{p}_mp{mp}_"
                        "{untiedf}_{coordf}_{exp_id}"
                        .format (T=config.T, lam=config.lam, p=config.percent,
                                 mp=config.max_percent, untiedf=untiedf,
                                 coordf=coordf, exp_id=config.exp_id))
        from models.LISTA_cpss import LISTA_cpss
        model = LISTA_cpss (kwargs['A'], T=config.T, lam=config.lam,
                            percent=config.percent, max_percent=config.max_percent,
                            untied=config.untied , coord=config.coord,
                            scope=config.scope, adapt = config.adaptive, bias = use_bias, normalize = config.normalize, function_type = config.function)


    if config.net == "ALISTA":
        """ALISTA"""
        config.model = ("ALISTA_T{T}_lam{lam}_p{p}_mp{mp}_{W}_{coordf}_{exp_id}"
                        .format(T=config.T, lam=config.lam,
                                p=config.percent, mp=config.max_percent,
                                W=os.path.basename(config.W),
                                coordf=coordf, exp_id=config.exp_id))
        W = np.load(config.W)
        print("Pre-calculated weight W loaded from {}".format(config.W))
        from models.ALISTA import ALISTA
        model = ALISTA(kwargs['A'], T=config.T, lam=config.lam, W=W,
                       percent=config.percent, max_percent=config.max_percent,
                       coord=config.coord, scope=config.scope, adapt = config.adaptive, bias = use_bias, normalize = config.normalize, function_type = config.function)


        # set up encoder
        from models.AtoW_grad import AtoW_grad
        encoder = AtoW_grad(config.M, config.N, config.eT, Binit=kwargs["Binit"],
                            eta=config.eta, loss=config.encoder_loss,
                            Q=kwargs["Q"], scope=config.encoder_scope)
        # set up decoder
        from models.ALISTA_robust import ALISTA_robust
        decoder = ALISTA_robust(M=config.M, N=config.N, T=config.T,
                                percent=config.percent, max_percent=config.max_percent,
                                coord=config.coord, scope=config.decoder_scope)

        model_desc = ("robust_" + config.encoder + '_' + config.decoder +
                     "_elr{}_dlr{}_psmax{}_psteps{}_{}"
                     .format(config.encoder_lr, config.decoder_lr,
                             config.psigma_max, config.psteps, config.exp_id))
        model_dir = os.path.join(config.expbase, model_desc)
        config.resfn = os.path.join(config.resbase, model_desc)
        if not os.path.exists(model_dir):
            if config.test:
                raise ValueError("Testing folder {} not existed".format(model_dir))
            else:
                os.makedirs(model_dir)

        config.enc_load = os.path.join(config.expbase, config.encoder)
        config.dec_load = os.path.join(config.expbase, config.decoder.replace("_robust", ""))
        config.encoderfn = os.path.join(model_dir, config.encoder)
        config.decoderfn = os.path.join(model_dir, config.decoder)
        return encoder, decoder


    config.modelfn = os.path.join(config.expbase, config.model)
    config.resfn = os.path.join(config.resbase, config.model)
    print ("model disc:", config.model)

    return model


############################################################
######################   Training    #######################
############################################################

def run_train(config) :
    if config.task_type == "sc":
        run_sc_train(config)
    


def run_sc_train(config) :
    """Load problem."""
    if not os.path.exists(config.probfn):
        print(config.probfn)
        raise ValueError ("Problem file not found.")
    else:
        p = problem.load_problem(config.probfn)

    """Set up model."""
    model = setup_model (config, A=p.A)

    """Set up input."""
    config.SNR = np.inf if config.SNR == 'inf' else float (config.SNR)
    y_, x_, y_val_, x_val_ = (
        train.setup_input_sc (
            config.test, p, config.tbs, config.vbs, config.fixval,
            config.supp_prob, config.SNR, config.magdist, **config.distargs))

    """Set up training."""
    stages = train.setup_sc_training (
            model, y_, x_, y_val_, x_val_, None,
            config.init_lr, config.decay_rate, config.lr_decay)


    tfconfig = tf.ConfigProto (allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    with tf.Session (config=tfconfig) as sess:
        # graph initialization
        sess.run (tf.global_variables_initializer ())
        print(sess.run(tf.norm(y_,axis=0)))
        print(sess.run(tf.norm(y_,axis=1)))
        # start timer
        start = time.time ()

        # train model
        model.do_training(sess, stages, config.modelfn, config.scope,
                          config.val_step, config.maxit, config.better_wait)

        # end timer
        end = time.time ()
        elapsed = end - start
        print ("elapsed time of training = " + str (timedelta (seconds=elapsed)))

    # end of run_sc_train




############################################################
######################   Testing    ########################
############################################################

def run_test (config):
    if config.task_type == "sc":
        run_sc_test (config)

def run_sc_test (config) :
    """
    Test model.
    """

    """Load problem."""
    if not os.path.exists (config.probfn):
        raise ValueError ("Problem file not found.")
    else:
        p = problem.load_problem (config.probfn)

    """Load testing data."""
    xt = np.load (config.xtest)
    #print(xt)
    """Set up input for testing."""
    config.SNR = np.inf if config.SNR == 'inf' else float (config.SNR)
    input_, label_ = (
        train.setup_input_sc (config.test, p, xt.shape [1], None, False,
                              config.supp_prob, config.SNR,
                              config.magdist, **config.distargs))

    """Set up model."""
    model = setup_model (config , A=p.A)
    xhs_ = model.inference (input_, None)

    """Create session and initialize the graph."""
    tfconfig = tf.ConfigProto (allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    with tf.Session (config=tfconfig) as sess:
        # graph initialization
        sess.run (tf.global_variables_initializer ())
        # load model
        #a,the = sess.run ([a,the])
        model.load_trainable_variables (sess , config.modelfn)
        #a,the = sess.run ([a,the])
        #print(a)
        #print(the)
        nmse_denom = np.sum (np.square (xt))
        supp_gt = xt != 0

        lnmse  = []
        lspar  = []
        lsperr = []
        lflspo = []
        lflsne = []

        # test model
        for xh_ in xhs_ :
            xh = sess.run (xh_ , feed_dict={label_:xt})

            # nmse:
            loss = np.sum (np.square (xh - xt))
            nmse_dB = 10.0 * np.log10 (loss / nmse_denom)
            print (nmse_dB)
            lnmse.append (nmse_dB)

            supp = xh != 0.0
            # intermediate sparsity
            spar = np.sum (supp , axis=0)
            lspar.append (spar)

            # support error
            sperr = np.logical_xor(supp, supp_gt)
            lsperr.append (np.sum (sperr , axis=0))

            # false positive
            flspo = np.logical_and (supp , np.logical_not (supp_gt))
            lflspo.append (np.sum (flspo , axis=0))

            # false negative
            flsne = np.logical_and (supp_gt , np.logical_not (supp))
            lflsne.append (np.sum (flsne , axis=0))
    print(lnmse)
    res = dict (nmse=np.asarray  (lnmse),
                spar=np.asarray  (lspar),
                sperr=np.asarray (lsperr),
                flspo=np.asarray (lflspo),
                flsne=np.asarray (lflsne))

    np.savez (config.resfn , **res)
    # end of test





############################################################
#######################    Main    #########################
############################################################

def main ():
    # parse configuration
    config, _ = get_config()
    # set visible GPUs
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu
    if config.test:
        run_test (config)
    else:
        run_train (config)
    # end of main

if __name__ == "__main__":
    main ()

