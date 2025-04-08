import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import scipy.io as sio
from torch.utils.data import Dataset, DataLoader

from config.structures import set_random_seed, Config
from models.scd import SwarmContrastiveDecomposition
from processing.postprocess import save_results

set_random_seed(seed=42)

torch.cuda.empty_cache() 

def train(path):
    device = "cuda"  
    acceptance_silhouette = 0.85
    extension_factor = 150
    low_pass_cutoff = None
    high_pass_cutoff = None
    start_time = 0  #seconds
    end_time = -1   #seconds (-1 is to end of file)
    max_iterations = 250
    sampling_frequency = 4000
    peel_off_window_size_ms = 20   # ms
    output_final_source_plot = True
    use_coeff_var_fitness = True
    remove_bad_fr = False

    config = Config(
        device=device,
        acceptance_silhouette=acceptance_silhouette,
        extension_factor=extension_factor,
        low_pass_cutoff=low_pass_cutoff,
        high_pass_cutoff=high_pass_cutoff,
        sampling_frequency=sampling_frequency,
        start_time=start_time,
        end_time=end_time,
        max_iterations=max_iterations,
        peel_off_window_size_ms=peel_off_window_size_ms,
        output_final_source_plot=output_final_source_plot,
        use_coeff_var_fitness=use_coeff_var_fitness,
        remove_bad_fr=remove_bad_fr,
    )

    # Load data
    if path.suffix == ".mat":
        mat = sio.loadmat(str(path))
        neural_data = mat["signal"]["data"][0, 0]  # channels, time points
        
    elif path.suffix == ".npy":
        npy_data = np.load(path)
        neural_data = npy_data
    else:
        raise ValueError(
            "Data format not supported. Please provide data in .mat or .npy format."
        )

    #The model expects [time, channels] and our data is saved as [channels, time]
    # neural_data = neural_data.T

    
    if config.end_time == -1:
        neural_data = neural_data[ 
            config.start_time * sampling_frequency :, :
        ]

    else:
        neural_data = neural_data[ 
            config.start_time * sampling_frequency : config.end_time * sampling_frequency, :
        ]


    #convert neural data over to tensor
    neural_data = (
            torch.from_numpy(neural_data).t().to(device=device, dtype=torch.float32)
        )  # time, channels
    print(neural_data.shape)

    # Initiate the model and run
    model = SwarmContrastiveDecomposition()
    predicted_timestamps, dictionary = model.run(neural_data, config)

    return dictionary, predicted_timestamps


if __name__ == "__main__":
    HOME = Path.cwd().joinpath("data", "input_chunked")
    
    # Loop through all files that match your naming pattern
    path = list(HOME.glob("S241_1_signal_1.mat"))[0]
    file_name = path.stem  # Get the file name without the extension
    
    # Set output path for each file
    output_path = (
        Path(str(HOME).replace("input_chunked", "output_chunked"))
        .joinpath(file_name)
        .with_suffix(".pkl")
    )
    
    dictionary, _ = train(path)  # Pass each file to train function

    save_results(output_path, dictionary)
    print(f"Saved results to {output_path}")





