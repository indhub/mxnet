import sys
sys.path.append("../../..")
from tests.tutorials import test_tutorials

import os
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

def _test_tutorial_nb(tutorial):
    """Run tutorial jupyter notebook to catch any execution error.

    Parameters
    ----------
    tutorial : str
        tutorial name in folder/tutorial format
    """

    tutorial_dir = os.path.join(os.path.dirname(__file__), 'straight_dope')
    tutorial_path = os.path.join(*([tutorial_dir] + tutorial.split('/')))

    # see env variable docs in the doc string of the file
    kernel = os.getenv('MXNET_TUTORIAL_TEST_KERNEL', None)
    no_cache = os.getenv('MXNET_TUTORIAL_TEST_NO_CACHE', False)

    working_dir = os.path.join(*([temp_dir] + tutorial.split('/')))

    if no_cache == '1':
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
        nb, _ = eprocessor.preprocess(notebook, {'metadata': {'path': working_dir}})
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
            print('\n'.join(errors))
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

def test_linear_regression_scratch():
    assert _test_tutorial_nb('chapter02_supervised-learning/linear-regression-scratch')

def test_logistic_regression_gluon():
    assert _test_tutorial_nb('chapter02_supervised-learning/logistic-regression-gluon')

def test_logistic_regression_gluon():
    assert _test_tutorial_nb('chapter02_supervised-learning/logistic-regression-gluon')

def test_softmax_regression_scratch():
    assert _test_tutorial_nb('chapter02_supervised-learning/softmax-regression-scratch')

def test_softmax_regression_gluon():
    assert _test_tutorial_nb('chapter02_supervised-learning/softmax-regression-gluon')

def test_regularization_scratch():
    assert _test_tutorial_nb('chapter02_supervised-learning/regularization-scratch')

def test_regularization_gluon():
    assert _test_tutorial_nb('chapter02_supervised-learning/regularization-gluon')

def test_perceptron():
    assert _test_tutorial_nb('chapter02_supervised-learning/perceptron')

#def test_()
#    assert _test_tutorial_nb('')

