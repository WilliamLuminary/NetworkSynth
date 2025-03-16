import os
import pickle
from typing import Dict

from handlers import AttributesCalculator




def test_pickle_dump_graph_attr_agent(load_graph_from_pickle):
    sample_graph = load_graph_from_pickle
    agent = AttributesCalculator()
    agent.analyze(sample_graph)
    with open("attr_dict.pkl", "wb") as f:
        # noinspection PyTypeChecker
        pickle.dump(agent, f)

    assert os.path.exists("attr_dict.pkl")
    with open("attr_dict.pkl", "rb") as f:
        load_file = pickle.load(f)
    assert isinstance(load_file, Dict)
