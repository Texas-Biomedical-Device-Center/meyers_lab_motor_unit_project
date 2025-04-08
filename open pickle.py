import torch
import pickle

# File path to your pickled data (replace with the correct path)
file_path = r"C:\Users\AXB240058\Box\Python\Projects\swarm-contrastive-decomposition\data\output\S241_3_signal 2.pkl"

# Open the pickle file and load the content
with open(file_path, "rb") as f:
    # Load the data using pickle first to get the raw data
    results = pickle.load(f)

# At this point, `results` should be a dictionary or a complex object that contains PyTorch tensors.

# Now, ensure that any tensor inside `results` is moved to CPU






