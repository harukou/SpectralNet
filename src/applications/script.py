import sys, os
# add directories in src/ to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))

# import run_net and get_data
from spectralnet import run_net
from core.data import get_data
from new_dset.concentric2 import generate_circle2
from collections import defaultdict

# SELECT GPU
os.environ['CUDA_VISIBLE_DEVICES'] ='0'

params = defaultdict(lambda: None)
# define hyperparameters
general_params = {

		'dset': 'new_dset',                  # dataset: reuters / mnist
        'val_set_fraction': 0.1,            # fraction of training set to use as validation
        'precomputedKNNPath': '',           # path for precomputed nearest neighbors (with indices and saved as a pickle or h5py file)
        'siam_batch_size': 128,             # minibatch size for siamese net

	# data generation parameters
        'train_set_fraction': 0.7,       # fraction of the dataset to use for training
        'noise_sig': 0.1,               # variance of the gaussian noise applied to x
        'n': 1500,                      # number of total points in dataset
        # training parameters
        'n_clusters': 2,
        'use_code_space': False,
        'affinity': 'full',
        'n_nbrs': 2,
        'scale_nbr': 2,
        'spec_ne': 300,
        'spec_lr': 1e-3,
        'spec_patience': 30,
        'spec_drop': 0.1,
        'batch_size': 128,
        'batch_size_orthonorm': 128,
        'spec_reg': None,
        'arch': [
            {'type': 'softplus', 'size': 50},
            {'type': 'BatchNormalization'},
            {'type': 'softplus', 'size': 50},
            {'type': 'BatchNormalization'},
            {'type': 'softplus', 'size': 50},
            {'type': 'BatchNormalization'},
            ],
        'use_all_data': True,
    }
params.update(general_params)
    
# load dataset
x_train, x_test, y_train, y_test = generate_circle2(n=1500, noise_sigma=0.1, 
															train_set_fraction=0.7)
new_dataset_data = (x_train, x_test, y_train, y_test)




# preprocess dataset
data = get_data(params, new_dataset_data)

# run spectral net
x_spectralnet, y_spectralnet = run_net(data, params)

import plot_2d
plot_2d.process(x_spectralnet, y_spectralnet, data, params)
