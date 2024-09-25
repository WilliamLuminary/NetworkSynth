import traceback

from tqdm import tqdm

from src.run import generate_and_process_graphs

set_names = ['A']
resolutions = ['10kX']
# resolutions = ['10kX', '15kX', '20kX', '30kX', '40kX', '50kX', '60kX', '80kX', '100kX']
# set_names = ['C', 'D']
view = False  # False by default
sample = 10
network_num = 300
base_path = '/content/drive/MyDrive/vis/Results0925'

total_combinations = len(set_names) * len(resolutions)

failure = []
failure_details = []
with tqdm(total=total_combinations, desc="Processing Sets and Resolutions", ncols=50,
          bar_format="{l_bar}{bar} | {n_fmt}/{total_fmt}") as pbar:
    for set_name in set_names:
        for resolution in resolutions:
            print(f"\n{'-' * 20} Processing {set_name}-{resolution} {'-' * 20}")
            try:
                generate_and_process_graphs(set_name, resolution, view, base_path=base_path, graph_sample=sample,
                                            num_iterations=network_num)
            except Exception as e:
                error_message = traceback.format_exc()
                print(f"\nSkipping {set_name}-{resolution} due to: {e}\nDetails:\n{error_message}")
                failure_details.append((set_name, resolution, error_message))
                failure.append((set_name, resolution))
                continue
            pbar.update(1)
