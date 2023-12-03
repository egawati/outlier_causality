import unittest

from ocular.utils import transform_data_list_to_numpy
from ocular.utils import sampling_data
from ocular.utils import get_outlier_event_timestamp
from ocular.utils import get_slide_number_to_explain_outlier

import numpy as np
import pandas as pd 
import datetime

from ocular.initialization import scm_initialization
from ocular.causal_model import dag

class TestUtils(unittest.TestCase):
    def setUp(self):
        nodes = [('X0', 'X2'), ('X1', 'X2'), ('X2', 'X3'), ('X2', 'X4'), ('X3', 'X5')]
        self.features = ['X0', 'X1', 'X2', 'X3', 'X4', 'X5']
        self.causal_graph = dag.CausalGraph(nodes, self.features)

        n_samples = 100
        init_data = pd.DataFrame(np.random.randint(0,100,size=(n_samples, len(self.features))), columns=self.features)

        self.fm_types = {node : 'LinearModel' for node in self.features}
        self.noise_types = {node : 'AdditiveNoise' for node in self.features}

        self.m_samples = 75

        self.models, self.noise_samples, soutlier_scorers, sorted_nodes = scm_initialization(init_data, 
                                                             self.causal_graph, 
                                                             self.fm_types, 
                                                             self.noise_types, 
                                                             self.m_samples)
        # self.assertEqual(self.models['X2'][0]._causal_mechanism.n_features_in_, 2)
        # self.assertEqual(self.models['X3'][0]._causal_mechanism.n_features_in_, 1)
        # self.assertEqual(self.models['X4'][0]._causal_mechanism.n_features_in_, 1)
        # self.assertEqual(self.models['X5'][0]._causal_mechanism.n_features_in_, 1)

    def test_sampling_data(self):
        num_data = 40
        data = pd.DataFrame(np.random.randint(0,100,size=(num_data, len(self.features))), columns=self.features)
        m_samples = 30
        samples = sampling_data(data, m_samples)
        self.assertEqual(samples.shape[0], m_samples)
        self.assertEqual(samples.shape[1], data.shape[1])
        self.assertEqual(samples.shape[1], len(self.features))

    def test_get_outlier_event_timestamp(self):
        basic_time = datetime.datetime.now()
        outlier = {'values' : pd.DataFrame(np.random.randint(0,100,size=(1, len(self.features))), columns=self.features),
                   'event_ts' : basic_time - datetime.timedelta(seconds=10)}
        outlier_ts = get_outlier_event_timestamp(outlier)
        self.assertEqual(outlier_ts, outlier['event_ts'])

    def test_get_slide_number_to_explain_outlier(self):
        basic_time = datetime.datetime.now()
        active_slides = {0 : {'start_ts' : basic_time - datetime.timedelta(seconds=5), 'end_ts' : basic_time} , 
                         1 : {'start_ts' : basic_time, 'end_ts': basic_time + datetime.timedelta(seconds=5)},
                         2 : {'start_ts' : basic_time + datetime.timedelta(seconds=5), 'end_ts': basic_time + datetime.timedelta(seconds=10)}
                        }
        outlier_ts = basic_time + datetime.timedelta(seconds=6)
        slide_num = get_slide_number_to_explain_outlier(outlier_ts, active_slides)
        self.assertEqual(slide_num, 1)

    def test_get_slide_number_to_explain_outlier_2(self):
        basic_time = datetime.datetime.now()
        active_slides = {0 : {'start_ts' : basic_time - datetime.timedelta(seconds=5), 'end_ts' : basic_time} , 
                         1 : {'start_ts' : basic_time, 'end_ts': basic_time + datetime.timedelta(seconds=5)},
                         2 : {'start_ts' : basic_time + datetime.timedelta(seconds=5), 'end_ts': basic_time + datetime.timedelta(seconds=10)}
                        }
        outlier_ts = basic_time
        slide_num = get_slide_number_to_explain_outlier(outlier_ts, active_slides)
        self.assertEqual(slide_num, 0)

    def test_get_slide_number_to_explain_outlier3(self):
        active_slides = {3: {'start_ts': 1681528848.515157, 'end_ts': 1681528863.515157}, 
                       4: {'start_ts': 1681528863.51516, 'end_ts': 1681528878.51516}, 
                       5: {'start_ts': 1681528878.515162, 'end_ts': 1681528893.515162}, 
                       6: {'start_ts': 1681528893.515166, 'end_ts': 1681528908.515166}, 
                       7: {'start_ts': 1681528908.515168, 'end_ts': 1681528923.515168}}
        outlier_ts = 1681528913.3299546
        slide_num = get_slide_number_to_explain_outlier(outlier_ts, active_slides)
        self.assertEqual(slide_num, 6)

    def test_get_slide_number_to_explain_outlier4(self):
        active_slides = {1: {'start_ts': 1681770840.837686, 'end_ts': 1681770855.837686}, 
                         2: {'start_ts': 1681770900.837689, 'end_ts': 1681770915.837689}, 
                         3: {'start_ts': 1681770915.837691, 'end_ts': 1681770930.837691}, 
                         4: {'start_ts': 1681770930.837694, 'end_ts': 1681770945.837694}, 
                         5: {'start_ts': 1681770960.8377, 'end_ts': 1681770975.8377}}
        outlier_ts = 1681770979.8657744
        slide_num = get_slide_number_to_explain_outlier(outlier_ts, active_slides)
        self.assertEqual(slide_num, 5)