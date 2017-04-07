import os
import sys
from pathlib import Path

base_dir = Path(__file__).parents[3]
sys.path.append(os.path.join(base_dir,'pupil_plugins_shared'))

from screen_tracker import Screen_Tracker