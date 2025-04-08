from pathlib import Path
import scipy.io as sio

dir_path = Path('/Users/ecm081000/Library/CloudStorage/Box-Box/Python/Projects/swarm-contrastive-decomposition/data/input')
signal_list = list(dir_path.glob('*.mat'))

for signal_file in signal_list:
    print(f'Loading signal from {signal_file}')
    mat_data = sio.loadmat(signal_file)

    signal = mat_data['signal']

print(mat_data)