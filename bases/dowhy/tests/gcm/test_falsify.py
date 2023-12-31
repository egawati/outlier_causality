#  Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.

import networkx as nx
import numpy as np
import pandas as pd
from flaky import flaky

from dowhy.datasets import generate_random_graph
from dowhy.gcm.falsify import (
    FALSIFY_N_TESTS,
    FALSIFY_N_VIOLATIONS,
    _PermuteNodes,
    validate_causal_minimality,
    validate_graph,
    validate_lmc,
    validate_parental_dsep,
    validate_pd,
)
from dowhy.gcm.independence_test.generalised_cov_measure import generalised_cov_based
from dowhy.gcm.independence_test.kernel import kernel_based
from dowhy.gcm.ml import create_linear_regressor
from tests.gcm.independence_test.test_kernel import _generate_categorical_data


def _gcm_linear(X, Y, Z=None):
    return generalised_cov_based(
        X, Y, Z=Z, prediction_model_X=create_linear_regressor, prediction_model_Y=create_linear_regressor
    )


def _generate_simple_non_linear_data() -> pd.DataFrame:
    X = np.random.normal(loc=0, scale=1, size=5000)
    Y = X**2 + np.random.normal(loc=0, scale=1, size=5000)
    Z = np.exp(-Y) + np.random.normal(loc=0, scale=1, size=5000)

    return pd.DataFrame(data=dict(X=X, Y=Y, Z=Z))


@flaky(max_runs=1)
def test_given_exclude_original_order_when_generating_permutations_then_return_correct_permutations():
    num_nodes = np.random.randint(1, 10)
    perms = set()
    G = generate_random_graph(n=num_nodes)
    perm_gen = _PermuteNodes(G, exclude_original_order=True, n_permutations=-1)
    for perm in perm_gen:
        perms.add(perm)
        assert list(G.nodes) != list(perm.nodes)

    assert len(perms) == np.math.factorial(num_nodes) - 1


@flaky(max_runs=1)
def test_given_not_exclude_original_order_when_generating_permutations_then_return_correct_permutations():
    num_nodes = np.random.randint(1, 10)
    found_orig_perm = False
    perms = set()
    G = generate_random_graph(n=num_nodes)
    perm_gen = _PermuteNodes(G, exclude_original_order=False, n_permutations=-1)
    for perm in perm_gen:
        perms.add(perm)
        if list(G.nodes) == list(perm.nodes):
            found_orig_perm = True

    assert found_orig_perm
    assert len(perms) == np.math.factorial(num_nodes)


@flaky(max_runs=5)
def test_given_correct_collider_when_validating_graph_then_report_no_violations():
    # collider: X->Z<-Y
    true_dag = nx.DiGraph([("X", "Z"), ("Y", "Z")])
    X = np.random.normal(size=500)
    Y = np.random.normal(size=500)
    Z = 2 * X + 3 * Y + np.random.normal(size=500)
    data = pd.DataFrame(data=dict(X=X, Y=Y, Z=Z))
    summary = validate_graph(
        true_dag,
        data,
        causal_graph_reference=true_dag,
        methods=(validate_lmc, validate_pd, validate_parental_dsep),
        independence_test=_gcm_linear,
        conditional_independence_test=_gcm_linear,
    )

    assert summary["validate_lmc"][FALSIFY_N_VIOLATIONS] == 0
    assert summary["validate_lmc"][FALSIFY_N_TESTS] == 2
    assert summary["validate_pd"][FALSIFY_N_VIOLATIONS] == 0
    assert summary["validate_pd"][FALSIFY_N_TESTS] == 2
    assert summary["validate_parental_dsep"][FALSIFY_N_VIOLATIONS] == 0
    assert summary["validate_parental_dsep"][FALSIFY_N_TESTS] == 2


@flaky(max_runs=5)
def test_given_wrong_collider_when_validating_graph_then_report_violations():
    # collider: X->Z<-Y
    true_dag = nx.DiGraph([("X", "Z"), ("Y", "Z")])
    dag = nx.DiGraph([("X", "Y"), ("Z", "Y")])
    X = np.random.normal(size=500)
    Y = np.random.normal(size=500)
    Z = 2 * X + 3 * Y + np.random.normal(size=500)
    data = pd.DataFrame(data=dict(X=X, Y=Y, Z=Z))
    summary = validate_graph(
        dag,
        data,
        causal_graph_reference=true_dag,
        methods=(validate_lmc, validate_pd, validate_parental_dsep),
        independence_test=_gcm_linear,
        conditional_independence_test=_gcm_linear,
    )

    assert summary["validate_lmc"][FALSIFY_N_VIOLATIONS] == 2
    assert summary["validate_lmc"][FALSIFY_N_TESTS] == 2
    assert summary["validate_pd"][FALSIFY_N_VIOLATIONS] == 1
    assert summary["validate_pd"][FALSIFY_N_TESTS] == 2
    assert summary["validate_parental_dsep"][FALSIFY_N_VIOLATIONS] == 2
    assert summary["validate_parental_dsep"][FALSIFY_N_TESTS] == 2


@flaky(max_runs=5)
def test_given_correct_chain_when_validating_graph_then_report_no_violations():
    # chain: X->Y->Z
    true_dag = nx.DiGraph([("X", "Y"), ("Y", "Z")])
    X = np.random.normal(size=500)
    Y = 0.6 * X + np.random.normal(size=500)
    Z = Y + np.random.normal(size=500)
    data = pd.DataFrame(data=dict(X=X, Y=Y, Z=Z))
    summary = validate_graph(
        true_dag,
        data,
        causal_graph_reference=true_dag,
        methods=(validate_lmc, validate_pd, validate_parental_dsep),
        independence_test=_gcm_linear,
        conditional_independence_test=_gcm_linear,
    )

    assert summary["validate_lmc"][FALSIFY_N_VIOLATIONS] == 0
    assert summary["validate_lmc"][FALSIFY_N_TESTS] == 1
    assert summary["validate_pd"][FALSIFY_N_VIOLATIONS] == 0
    assert summary["validate_pd"][FALSIFY_N_TESTS] == 3
    assert summary["validate_parental_dsep"][FALSIFY_N_VIOLATIONS] == 0
    assert summary["validate_parental_dsep"][FALSIFY_N_TESTS] == 1


@flaky(max_runs=5)
def test_given_wrong_chain_when_validating_graph_then_report_violations():
    # chain: X->Y->Z
    true_dag = nx.DiGraph([("X", "Y"), ("Y", "Z")])
    dag = nx.DiGraph([("X", "Z"), ("Z", "Y")])
    X = np.random.normal(size=500)
    Y = 0.6 * X + np.random.normal(size=500)
    Z = Y + np.random.normal(size=500)
    data = pd.DataFrame(data=dict(X=X, Y=Y, Z=Z))
    summary = validate_graph(
        dag,
        data,
        causal_graph_reference=true_dag,
        methods=(validate_lmc, validate_pd, validate_parental_dsep),
        independence_test=_gcm_linear,
        conditional_independence_test=_gcm_linear,
    )

    assert summary["validate_lmc"][FALSIFY_N_VIOLATIONS] == 1
    assert summary["validate_lmc"][FALSIFY_N_TESTS] == 1
    assert summary["validate_pd"][FALSIFY_N_VIOLATIONS] == 0
    assert summary["validate_pd"][FALSIFY_N_TESTS] == 3
    assert summary["validate_parental_dsep"][FALSIFY_N_VIOLATIONS] == 1
    assert summary["validate_parental_dsep"][FALSIFY_N_TESTS] == 1


@flaky(max_runs=5)
def test_given_empty_DAG_and_data_when_validating_graph_then_report_no_violations():
    # Empty graph
    dag = nx.DiGraph()
    data = pd.DataFrame()
    summary = validate_graph(
        dag,
        data,
        causal_graph_reference=dag,
        methods=(validate_lmc, validate_pd, validate_parental_dsep),
        independence_test=_gcm_linear,
        conditional_independence_test=_gcm_linear,
    )
    assert summary["validate_lmc"][FALSIFY_N_VIOLATIONS] == 0
    assert summary["validate_lmc"][FALSIFY_N_TESTS] == 0
    assert summary["validate_pd"][FALSIFY_N_VIOLATIONS] == 0
    assert summary["validate_pd"][FALSIFY_N_TESTS] == 0
    assert summary["validate_parental_dsep"][FALSIFY_N_VIOLATIONS] == 0
    assert summary["validate_parental_dsep"][FALSIFY_N_TESTS] == 0


@flaky(max_runs=5)
def test_given_correct_full_DAG_when_validating_graph_then_report_no_violations():
    # Full DAG
    dag = nx.DiGraph([("X0", "X1"), ("X0", "X2"), ("X0", "X3"), ("X1", "X2"), ("X1", "X3"), ("X2", "X3")])
    X0 = np.random.normal(size=500)
    X1 = 0.5 * X0 + np.random.normal(size=500)
    X2 = 1.2 * X0 + 0.7 * X1 + np.random.normal(size=500)
    X3 = 0.6 * X0 + 0.8 * X1 + 1.3 * X2 + np.random.normal(size=500)
    data = pd.DataFrame(data=dict(X0=X0, X1=X1, X2=X2, X3=X3))
    summary = validate_graph(
        dag,
        data,
        causal_graph_reference=dag,
        methods=(validate_lmc, validate_pd, validate_parental_dsep),
        independence_test=_gcm_linear,
        conditional_independence_test=_gcm_linear,
    )

    assert summary["validate_lmc"][FALSIFY_N_VIOLATIONS] == 0
    assert summary["validate_lmc"][FALSIFY_N_TESTS] == 0
    assert summary["validate_pd"][FALSIFY_N_VIOLATIONS] == 0
    assert summary["validate_pd"][FALSIFY_N_TESTS] == 6
    assert summary["validate_parental_dsep"][FALSIFY_N_VIOLATIONS] == 0
    assert summary["validate_parental_dsep"][FALSIFY_N_TESTS] == 0


@flaky(max_runs=5)
def test_given_correct_single_node_when_validating_graph_then_report_no_violations():
    # DAG with single node
    dag = nx.DiGraph()
    dag.add_node("X")
    data = pd.DataFrame(data=dict(X=np.random.normal(size=500)))
    summary = validate_graph(
        dag,
        data,
        causal_graph_reference=dag,
        methods=(validate_lmc, validate_pd, validate_parental_dsep),
        independence_test=_gcm_linear,
        conditional_independence_test=_gcm_linear,
    )

    assert summary["validate_lmc"][FALSIFY_N_VIOLATIONS] == 0
    assert summary["validate_lmc"][FALSIFY_N_TESTS] == 0
    assert summary["validate_pd"][FALSIFY_N_VIOLATIONS] == 0
    assert summary["validate_pd"][FALSIFY_N_TESTS] == 0
    assert summary["validate_parental_dsep"][FALSIFY_N_VIOLATIONS] == 0
    assert summary["validate_parental_dsep"][FALSIFY_N_TESTS] == 0


@flaky(max_runs=5)
def test_given_correct_single_edge_when_validating_graph_then_report_no_violations():
    # DAG with single edge
    dag = nx.DiGraph([("X0", "X1")])
    X0 = np.random.normal(size=500)
    X1 = 2 * X0 + np.random.normal(size=500)
    data = pd.DataFrame(data=dict(X0=X0, X1=X1))
    summary = validate_graph(
        dag,
        data,
        causal_graph_reference=dag,
        methods=(validate_lmc, validate_pd, validate_parental_dsep),
        independence_test=_gcm_linear,
        conditional_independence_test=_gcm_linear,
    )

    assert summary["validate_lmc"][FALSIFY_N_VIOLATIONS] == 0
    assert summary["validate_lmc"][FALSIFY_N_TESTS] == 0
    assert summary["validate_pd"][FALSIFY_N_VIOLATIONS] == 0
    assert summary["validate_pd"][FALSIFY_N_TESTS] == 1
    assert summary["validate_parental_dsep"][FALSIFY_N_VIOLATIONS] == 0
    assert summary["validate_parental_dsep"][FALSIFY_N_TESTS] == 0


@flaky(max_runs=5)
def test_given_wrong_single_edge_when_validating_graph_then_report_violations():
    # DAG with single edge
    true_dag = nx.DiGraph([("X0", "X1")])
    dag = nx.DiGraph()
    dag.add_node("X0")
    dag.add_node("X1")
    X0 = np.random.normal(size=500)
    X1 = 2 * X0 + np.random.normal(size=500)
    data = pd.DataFrame(data=dict(X0=X0, X1=X1))
    summary = validate_graph(
        dag,
        data,
        causal_graph_reference=true_dag,
        methods=(validate_lmc, validate_pd, validate_parental_dsep),
        independence_test=_gcm_linear,
        conditional_independence_test=_gcm_linear,
    )

    assert summary["validate_lmc"][FALSIFY_N_VIOLATIONS] == 2
    assert summary["validate_lmc"][FALSIFY_N_TESTS] == 2
    assert summary["validate_pd"][FALSIFY_N_VIOLATIONS] == 0
    assert summary["validate_pd"][FALSIFY_N_TESTS] == 0
    assert summary["validate_parental_dsep"][FALSIFY_N_VIOLATIONS] == 2
    assert summary["validate_parental_dsep"][FALSIFY_N_TESTS] == 2


@flaky(max_runs=5)
def test_given_correct_categorical_when_validating_graph_then_report_no_violations():
    # chain: X->Z->Y
    dag = nx.DiGraph([("X", "Z"), ("Z", "Y")])
    X, Y, Z = _generate_categorical_data()
    data = pd.DataFrame(data=dict(X=X, Y=Y, Z=Z))
    summary = validate_graph(
        dag,
        data,
        causal_graph_reference=dag,
        methods=(validate_lmc, validate_pd, validate_parental_dsep),
        independence_test=kernel_based,
        conditional_independence_test=kernel_based,
    )

    assert summary["validate_lmc"][FALSIFY_N_VIOLATIONS] == 0
    assert summary["validate_lmc"][FALSIFY_N_TESTS] == 1
    assert summary["validate_pd"][FALSIFY_N_VIOLATIONS] == 0
    assert summary["validate_pd"][FALSIFY_N_TESTS] == 3
    assert summary["validate_parental_dsep"][FALSIFY_N_VIOLATIONS] == 0
    assert summary["validate_parental_dsep"][FALSIFY_N_TESTS] == 1


@flaky(max_runs=5)
def test_given_non_minimal_DAG_when_validating_causal_minimality_then_report_violations():
    true_dag = nx.DiGraph([("X0", "Y"), ("X1", "Y")])
    true_dag.add_node("X2")
    given_dag = true_dag.copy()
    given_dag.add_edge("X2", "Y")

    X0 = np.random.normal(size=500)
    X1 = np.random.normal(size=500)
    X2 = np.random.normal(size=500)
    Y = 2 * X0 + 3 * X1 + np.random.normal(size=500)
    data = pd.DataFrame(data=dict(X0=X0, X1=X1, X2=X2, Y=Y))
    summary = validate_graph(
        given_dag,
        data,
        methods=validate_causal_minimality,
        independence_test=_gcm_linear,
        conditional_independence_test=_gcm_linear,
    )

    assert summary["validate_causal_minimality"][FALSIFY_N_VIOLATIONS] == 1
    assert summary["validate_causal_minimality"][FALSIFY_N_TESTS] == 3


@flaky(max_runs=5)
def test_given_minimal_DAG_when_validating_causal_minimality_then_report_no_violations():
    dag = nx.DiGraph([("X0", "Y"), ("X1", "Y")])
    X0 = np.random.normal(size=500)
    X1 = np.random.normal(size=500)
    X2 = np.random.normal(size=500)
    Y = 2 * X0 + 3 * X1 + np.random.normal(size=500)
    data = pd.DataFrame(data=dict(X0=X0, X1=X1, X2=X2, Y=Y))
    summary = validate_graph(
        dag,
        data,
        methods=validate_causal_minimality,
        independence_test=_gcm_linear,
        conditional_independence_test=_gcm_linear,
    )

    assert summary["validate_causal_minimality"][FALSIFY_N_VIOLATIONS] == 0
    assert summary["validate_causal_minimality"][FALSIFY_N_TESTS] == 2
