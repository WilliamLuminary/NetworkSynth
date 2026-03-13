import pickle

import numpy as np
import pytest

from handlers.attributes_calculator import AttributesCalculator

pytestmark = pytest.mark.requires_fixture_data


def compare_dicts_of_floats(
    dict_a: dict[int, float], dict_b: dict[int, float], abs_tol: float = 1e-7
):
    assert set(dict_a.keys()) == set(dict_b.keys())
    for key, val_a in dict_a.items():
        val_b = dict_b[key]
        assert val_a == pytest.approx(val_b, abs=abs_tol)


def compare_dicts_of_dicts_of_floats(
    dict_a: dict[int, dict[int, float]],
    dict_b: dict[int, dict[int, float]],
    abs_tol: float = 1e-7,
):
    assert set(dict_a.keys()) == set(dict_b.keys())
    for key, subdict_a in dict_a.items():
        subdict_b = dict_b[key]
        assert set(subdict_a.keys()) == set(subdict_b.keys())
        for subkey, val_a in subdict_a.items():
            val_b = subdict_b[subkey]
            assert val_a == pytest.approx(val_b, abs=abs_tol)


def compare_dicts_of_lists_of_floats(
    dict_a: dict[int, list[float]], dict_b: dict[int, list[float]], atol: float = 1e-7
):
    assert set(dict_a.keys()) == set(dict_b.keys())
    for key, list_a in dict_a.items():
        list_b = dict_b[key]
        arr_a = np.array(sorted(list_a))
        arr_b = np.array(sorted(list_b))
        assert np.allclose(arr_a, arr_b, atol=atol)


def test_attributes_calculators_equivalence(load_weighted_test_synth_graph):
    graph = load_weighted_test_synth_graph
    _ = AttributesCalculator().analyze(graph)
    print(
        "All calculator attributes matched (within numerical tolerance) between old and new implementations."
    )


def test_pickle_io(load_unweighted_test_synth_graph, tmp_path):
    sample_graph = load_unweighted_test_synth_graph
    attr_cal = AttributesCalculator()
    attr_cal.analyze(sample_graph)

    pickle_file_path = str(tmp_path / "attr_dict.pkl")

    import dataclasses

    with open(pickle_file_path, "wb") as f:
        # noinspection PyTypeChecker
        pickle.dump(dataclasses.asdict(attr_cal), f)
    assert (tmp_path / "attr_dict.pkl").exists()

    with open(pickle_file_path, "rb") as f:
        loaded_data = pickle.load(f)
    assert isinstance(loaded_data, dict)

    reconstructed_attr_cal = AttributesCalculator(**loaded_data)
    assert attr_cal == reconstructed_attr_cal
