from __future__ import print_function
import sys
sys.path.append("../../..")
from tests.tutorials import test_tutorials

import os
import re
import warnings
import imp
import shutil
import time
import argparse
import traceback
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import sys

temp_dir = 'tmp_notebook'
IPYTHON_VERSION = 4
TIME_OUT = 10*60

def _eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def _test_tutorial_nb(tutorial):
    """Run tutorial jupyter notebook to catch any execution error.

    Parameters
    ----------
    tutorial : str
        tutorial name in folder/tutorial format
    """

    tutorial_dir = os.path.join(os.path.dirname(__file__), 'straight_dope')
    tutorial_path = os.path.join(*([tutorial_dir] + tutorial.split('/')))
    tutorial_dir = os.path.dirname(tutorial_path)

    # see env variable docs in the doc string of the file
    kernel = os.getenv('MXNET_TUTORIAL_TEST_KERNEL', None)

    working_dir = os.path.join(*([temp_dir] + tutorial.split('/')))

    print("Cleaning and setting up temp directory '{}'".format(working_dir))
    shutil.rmtree(temp_dir, ignore_errors=True)

    errors = []
    notebook = None
    if not os.path.isdir(working_dir):
        os.makedirs(working_dir)
    try:
        notebook = nbformat.read(tutorial_path + '.ipynb', as_version=IPYTHON_VERSION)
        # Adding a small delay to allow time for sockets to be freed
        # stop-gap measure to battle the 1000ms linger of socket hard coded
        # in the kernel API code
        time.sleep(1.1)
        if kernel is not None:
            eprocessor = ExecutePreprocessor(timeout=TIME_OUT, kernel_name=kernel)
        else:
            eprocessor = ExecutePreprocessor(timeout=TIME_OUT)
        nb, _ = eprocessor.preprocess(notebook, {'metadata': {'path': tutorial_dir}})
    except Exception as err:
        err_msg = str(err)
        errors.append(err_msg)
    finally:
        if notebook is not None:
            output_file = os.path.join(working_dir, "output.txt")
            nbformat.write(notebook, output_file)
            output_nb = open(output_file, mode='r')
            for line in output_nb:
                if "Warning:" in line:
                    errors.append("Warning:\n"+line)
        if len(errors) > 0:
            _eprint('\n'.join(errors))
            return False
        return True

def _set_epochs(tutorial, num_epochs):
    tutorial_dir = os.path.join(os.path.dirname(__file__), 'straight_dope')
    tutorial_path = os.path.join(*([tutorial_dir] + tutorial.split('/'))) + ".ipynb"
    tutorial_dir = os.path.dirname(tutorial_path)
    
    # Read the notebook and set epochs to num_epochs
    with open(tutorial_path, 'r') as f:
        notebook = f.read()
    modified_notebook = re.sub(r'epochs\s+=\s+[0-9]+', 'epochs = 1', notebook)
    
    # Write the modified notebook into a .swap file
    swap_file_path = tutorial_path + ".swap"
    with open(swap_file_path, 'w') as f:
        f.write(modified_notebook)
    
    # Replace the notebook with the .swap file
    try:
        shutil.move(swap_file_path, tutorial_path)
    except shutil.Error:
        _eprint("Failed moving %s to %s" % (swap_file_path, tutorial_path))
        return False

    return True


#def test_ndarray():
#    assert _test_tutorial_nb('chapter01_crashcourse/ndarray')

#def test_linear_algebra():
#    assert _test_tutorial_nb('chapter01_crashcourse/linear-algebra')

#def test_probability():
#    assert _test_tutorial_nb('chapter01_crashcourse/probability')

#def test_ndarray():
#    assert _test_tutorial_nb('chapter01_crashcourse/ndarray')

#def test_autograd():
#    assert _test_tutorial_nb('chapter01_crashcourse/autograd')

# Chapter 2

#def test_linear_regression_scratch():
#    assert _test_tutorial_nb('chapter02_supervised-learning/linear-regression-scratch')

#def test_logistic_regression_gluon():
#    assert _test_tutorial_nb('chapter02_supervised-learning/logistic-regression-gluon')

#def test_logistic_regression_gluon():
#    assert _test_tutorial_nb('chapter02_supervised-learning/logistic-regression-gluon')

#def test_softmax_regression_scratch():
#    assert _test_tutorial_nb('chapter02_supervised-learning/softmax-regression-scratch')

#def test_softmax_regression_gluon():
#    assert _test_tutorial_nb('chapter02_supervised-learning/softmax-regression-gluon')

#def test_regularization_scratch():
#    assert _test_tutorial_nb('chapter02_supervised-learning/regularization-scratch')

#def test_regularization_gluon():
#    assert _test_tutorial_nb('chapter02_supervised-learning/regularization-gluon')

#def test_perceptron():
#    assert _test_tutorial_nb('chapter02_supervised-learning/perceptron')

# Chapter 3

#def test_mlp_scratch():
#    assert _test_tutorial_nb('chapter03_deep-neural-networks/mlp-scratch')

#def test_mlp_gluon():
#    assert _test_tutorial_nb('chapter03_deep-neural-networks/mlp-gluon')

#def test_mlp_dropout_scratch():
#    assert _test_tutorial_nb('chapter03_deep-neural-networks/mlp-dropout-scratch')

#def test_mlp_dropout_gluon():
#    assert _test_tutorial_nb('chapter03_deep-neural-networks/mlp-dropout-gluon')

#def test_plumbing():
#    assert _test_tutorial_nb('chapter03_deep-neural-networks/plumbing')

#def test_custom_layer():
#    assert _test_tutorial_nb('chapter03_deep-neural-networks/custom-layer')

#def test_serialization():
#    assert _test_tutorial_nb('chapter03_deep-neural-networks/serialization')

# Chapter 4

#def test_cnn_scratch():
#    assert _test_tutorial_nb('chapter04_convolutional-neural-networks/cnn-scratch')

#def test_cnn_gluon():
#    assert _test_tutorial_nb('chapter04_convolutional-neural-networks/cnn-gluon')

#def test_deep_cnns_alexnet():
#    assert _test_tutorial_nb('chapter04_convolutional-neural-networks/deep-cnns-alexnet')

#def test_very_deep_nets_vgg():
#    assert _test_tutorial_nb('chapter04_convolutional-neural-networks/very-deep-nets-vgg')

#def test_cnn_batch_norm_scratch():
#    assert _test_tutorial_nb('chapter04_convolutional-neural-networks/cnn-batch-norm-scratch')

#def test_cnn_batch_norm_gluon():
#    assert _test_tutorial_nb('chapter04_convolutional-neural-networks/cnn-batch-norm-gluon')

# Chapter 5

def test_simple_rnn():
    tutorial = 'chapter05_recurrent-neural-networks/simple-rnn'
    _set_epochs(tutorial, 1)
    assert _test_tutorial_nb('chapter05_recurrent-neural-networks/simple-rnn')

def test_lstm_scratch():
    tutorial = 'chapter05_recurrent-neural-networks/lstm-scratch'
    _set_epochs(tutorial, 1)
    assert _test_tutorial_nb('chapter05_recurrent-neural-networks/lstm-scratch')

def test_gru_scratch():
    tutorial = 'chapter05_recurrent-neural-networks/gru-scratch'
    _set_epochs(tutorial, 1)
    assert _test_tutorial_nb('chapter05_recurrent-neural-networks/gru-scratch')

def test_rnns_gluon():
    tutorial = 'chapter05_recurrent-neural-networks/rnns-gluon'
    _set_epochs(tutorial, 1)
    assert _test_tutorial_nb('chapter05_recurrent-neural-networks/rnns-gluon')

# chapter 6

def test_optimization_intro():
    assert _test_tutorial_nb('chapter06_optimization/optimization-intro')

def test_gd_sgd_scratch():
    assert _test_tutorial_nb('chapter06_optimization/gd-sgd-scratch')

def test_gd_sgd_gluon():
    assert _test_tutorial_nb('chapter06_optimization/gd-sgd-gluon')

def test_momentum_scratch():
    assert _test_tutorial_nb('chapter06_optimization/momentum-scratch')

def test_momentum_gluon():
    assert _test_tutorial_nb('chapter06_optimization/momentum-gluon')

def test_adagrad_scratch():
    assert _test_tutorial_nb('chapter06_optimization/adagrad-scratch')

def test_adagrad_gluon():
    assert _test_tutorial_nb('chapter06_optimization/adagrad-gluon')

def test_rmsprop_scratch():
    assert _test_tutorial_nb('chapter06_optimization/rmsprop-scratch')

def test_rmsprop_gluon():
    assert _test_tutorial_nb('chapter06_optimization/rmsprop-gluon')

def test_adadelta_scratch():
    assert _test_tutorial_nb('chapter06_optimization/adadelta-scratch')

def test_adadelta_gluon():
    assert _test_tutorial_nb('chapter06_optimization/adadelta-gluon')

def test_adam_scratch():
    assert _test_tutorial_nb('chapter06_optimization/adam-scratch')

def test_adam_gluon():
    assert _test_tutorial_nb('chapter06_optimization/adam-gluon')

# Chapter 7

def test_hybridize():
    assert _test_tutorial_nb('chapter07_distributed-learning/hybridize')

# Chapter 8

def test_object_detection():
    assert _test_tutorial_nb('chapter08_computer-vision/object-detection')

def test_fine_tuning():
    assert _test_tutorial_nb('chapter08_computer-vision/fine-tuning')

def test_visual_question_answer():
    assert _test_tutorial_nb('chapter08_computer-vision/visual-question-answer')

# Chapter 9

def test_tree_lstm():
    assert _test_tutorial_nb('chapter09_natural-language-processing/tree-lstm')

# Chapter 11

def test_intro_recommender_systems():
    assert _test_tutorial_nb('chapter11_recommender-systems/intro-recommender-systems')

# Chapter 12

def test_lds_scratch():
    assert _test_tutorial_nb('chapter12_time-series/lds-scratch')

def test_issm_scratch():
    assert _test_tutorial_nb('chapter12_time-series/issm-scratch')

# Chapter 14

def test_igan_intro():
    assert _test_tutorial_nb('chapter14_generative-adversarial-networks/gan-intro')

def test_dcgan():
    assert _test_tutorial_nb('chapter14_generative-adversarial-networks/dcgan')

def test_pixel2pixel():
    assert _test_tutorial_nb('chapter14_generative-adversarial-networks/pixel2pixel')

# Chapter 18

def test_bayes_by_backprop():
    assert _test_tutorial_nb('chapter18_variational-methods-and-uncertainty/bayes-by-backprop')

