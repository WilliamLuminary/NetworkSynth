import os
from collections import deque
from typing import Dict

from config import Config, Config2, DataType
from handlers import DataAgent


def _load_graphs_pkl(folder: str) -> List[nx.Graph]:
    for file in os.listdir(folder):
        if file.endswith('.pkl') and 'network' in file:
            with open(os.path.join(folder, file), 'rb') as f:
                content = pickle.load(f)
                return [content] if isinstance(content, nx.Graph) else content
    return []


def find_directories(base_dir: str, sub_folders=('synthetic', 'origin', 'original'), max_depth: int = 3) -> Dict[
    str, Dict]:
    name_networks_dict = {}
    queue = deque([(base_dir, 0, '')])  # (path, depth, rel_path)

    while queue:
        current_dir, depth, rel_path = queue.popleft()

        if os.path.basename(current_dir).startswith(('.', '__')):
            continue

        found = []
        tmp_dict = {}
        for entry in os.listdir(current_dir):
            if entry in sub_folders:
                key = 'original' if entry == 'origin' else entry
                full_path = os.path.join(current_dir, entry)
                tmp_dict[key] = _load_graphs_pkl(full_path)
                found.append(entry)
        if tmp_dict:
            name_networks_dict[rel_path] = tmp_dict

        if not found and depth < max_depth:
            for entry in os.listdir(current_dir):
                entry_path = os.path.join(current_dir, entry)
                if os.path.isdir(entry_path):
                    new_rel = os.path.join(rel_path, entry) if rel_path else entry
                    queue.append((entry_path, depth + 1, new_rel))
        elif found:
            name_networks_dict[rel_path]['path'] = current_dir

    return name_networks_dict


def load_data(result_dir: str) -> Dict[str, Dict]:
    base_output_dir = Config.BASE_OUTPUT_PATH
    base_directory = os.path.abspath(os.path.join(base_output_dir, result_dir))
    return find_directories(base_directory, ('synthetic', 'origin', 'original'))


ConfigSample.initialize()
if __name__ == '__main__':
    data = load_data('results_20250202_013241')

# def _load_graphs_pkl(folder: str) -> List[nx.Graph]:
#     for file in os.listdir(folder):
#         if file.endswith('.pkl') and 'network' in file:
#             with open(os.path.join(folder, file), 'rb') as f:
#                 content = pickle.load(f)
#                 return [content] if isinstance(content, nx.Graph) else content
#     return []
#
#
# def find_directories(base_dir: str, sub_folders=('synthetic', 'origin', 'original'), max_depth: int = 3) -> Dict[
#     str, Dict]:
#     name_networks_dict = {}
#     queue = deque([(base_dir, 0, '')])  # (path, depth, rel_path)
#
#     while queue:
#         current_dir, depth, rel_path = queue.popleft()
#
#         if os.path.basename(current_dir).startswith(('.', '__')):
#             continue
#
#         found = []
#         tmp_dict = {}
#         for entry in os.listdir(current_dir):
#             if entry in sub_folders:
#                 key = 'original' if entry == 'origin' else entry
#                 full_path = os.path.join(current_dir, entry)
#                 tmp_dict[key] = _load_graphs_pkl(full_path)
#                 found.append(entry)
#         if tmp_dict:
#             name_networks_dict[rel_path] = tmp_dict
#
#         if not found and depth < max_depth:
#             for entry in os.listdir(current_dir):
#                 entry_path = os.path.join(current_dir, entry)
#                 if os.path.isdir(entry_path):
#                     new_rel = os.path.join(rel_path, entry) if rel_path else entry
#                     queue.append((entry_path, depth + 1, new_rel))
#         elif found:
#             name_networks_dict[rel_path]['path'] = current_dir
#
#     return name_networks_dict
#
#
# def load_data(result_dir: str) -> Dict[str, Dict]:
#     base_output_dir = Config.BASE_OUTPUT_PATH
#     base_directory = os.path.abspath(os.path.join(base_output_dir, result_dir))
#     return find_directories(base_directory, ('synthetic', 'origin', 'original'))

# ConfigSample.initialize()
# if __name__ == '__main__':
#     data = load_data('results_20250202_013241')
#
#     for set_name, dataset in data.items():
#         batch_processor = MultifractalBatchProcessor.from_dict(dataset).process()
