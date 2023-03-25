def load_dataset(data_path):
    with open(data_path, 'rb') as f:
        data_samples = pickle.load(f)
    return data_samples

