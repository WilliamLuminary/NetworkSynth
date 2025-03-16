import os
import pickle

import numpy as np
import pytest

from handlers.attributes_calculator import AttributesCalculator, NewAttributesCalculator


def compare_dicts_of_floats(dict_a: dict[int, float], dict_b: dict[int, float], abs_tol: float = 1e-7):
    assert set(dict_a.keys()) == set(dict_b.keys())
    for key, val_a in dict_a.items():
        val_b = dict_b[key]
        assert val_a == pytest.approx(val_b, abs=abs_tol)


def compare_dicts_of_dicts_of_floats(dict_a: dict[int, dict[int, float]], dict_b: dict[int, dict[int, float]],
                                     abs_tol: float = 1e-7):
    assert set(dict_a.keys()) == set(dict_b.keys())
    for key, subdict_a in dict_a.items():
        subdict_b = dict_b[key]
        assert set(subdict_a.keys()) == set(subdict_b.keys())
        for subkey, val_a in subdict_a.items():
            val_b = subdict_b[subkey]
            assert val_a == pytest.approx(val_b, abs=abs_tol)


def compare_dicts_of_lists_of_floats(dict_a: dict[int, list[float]], dict_b: dict[int, list[float]],
                                     atol: float = 1e-7):
    assert set(dict_a.keys()) == set(dict_b.keys())
    for key, list_a in dict_a.items():
        list_b = dict_b[key]
        arr_a = np.array(sorted(list_a))
        arr_b = np.array(sorted(list_b))
        assert np.allclose(arr_a, arr_b, atol=atol)


def test_attributes_calculators_equivalence(load_weighted_test_nx_graph):
    graph = load_weighted_test_nx_graph
    old_calc = AttributesCalculator().analyze(graph)
    new_calc = NewAttributesCalculator().analyze(graph)
    assert old_calc.average_degree == pytest.approx(new_calc.average_degree, abs=1e-7)
    assert old_calc.average_edge_length == pytest.approx(new_calc.average_length, abs=1e-7)
    compare_dicts_of_floats(old_calc.degree_distribution, new_calc.degree_distribution)
    compare_dicts_of_dicts_of_floats(old_calc.degree_transition_probs, new_calc.degree_transition_probs)
    compare_dicts_of_lists_of_floats(old_calc.degree_edge_lengths, new_calc.degree_lengths)
    compare_dicts_of_lists_of_floats(old_calc.degree_angle_diffs, new_calc.degree_angles)
    print("All calculator attributes matched (within numerical tolerance) between old and new implementations.")


def test_pickle_io(load_unweighted_test_nx_graph):
    sample_graph = load_unweighted_test_nx_graph
    attr_cal = AttributesCalculator()
    attr_cal.analyze(sample_graph)
    pickle_file_path = os.path.join("..", "data", "attr_dict.pkl")
    _test_dump(attr_cal, pickle_file_path)
    loaded_data = _test_dump_type(pickle_file_path)
    _test_reconstruction(attr_cal, loaded_data)


def _test_reconstruction(attr_cal, loaded_data):
    reconstructed_attr_cal = AttributesCalculator(**loaded_data)
    assert attr_cal == reconstructed_attr_cal


def _test_dump(attr_cal, pickle_file_path):
    with open(pickle_file_path, "wb") as f:
        import dataclasses
        # noinspection PyTypeChecker
        pickle.dump(dataclasses.asdict(attr_cal), f)
    assert os.path.exists(pickle_file_path)


def _test_dump_type(pickle_file_path):
    with open(pickle_file_path, "rb") as f:
        load_file = pickle.load(f)
    from typing import Dict
    assert isinstance(load_file, Dict)
    return load_file
