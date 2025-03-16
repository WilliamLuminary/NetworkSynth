import os
import pickle

from handlers.attributes_calculator import AttributesCalculator


def test_pickle_dict(load_unweighted_test_nx_graph):
    sample_graph = load_unweighted_test_nx_graph
    attr_cal = AttributesCalculator()
    attr_cal.analyze(sample_graph)
    with open("../data/attr_dict.pkl", "wb") as f:
        import dataclasses
        # noinspection PyTypeChecker
        pickle.dump(dataclasses.asdict(attr_cal), f)
    assert os.path.exists("../data/attr_dict.pkl")

    with open("../data/attr_dict.pkl", "rb") as f:
        load_file = pickle.load(f)
    from typing import Dict
    assert isinstance(load_file, Dict)

    reconstructed_attr_cal = AttributesCalculator(**load_file)
    assert attr_cal == reconstructed_attr_cal
