import os
from datetime import datetime

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

INPUT_PATH = os.path.join(BASE_DIR, 'data')

OUTPUT_PATH = os.path.join(BASE_DIR, 'data', f'Results {current_time}')

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)
