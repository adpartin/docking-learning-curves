"""
A batch prcoessing code that calls main_data_split.py with the same set of parameters
but different datasets.

For example:
python src/batch_lrn_crv.py --splitdir data/docking_data_march_30/ml.3CLPro_pocket1_dock.splits --datapath data/docking_data_march_30/ml.3CLPro_pocket1_dock.parquet --n_splits 40 --n_shards 10 --par_jobs 40
"""
import warnings
warnings.filterwarnings('ignore')

import os
import sys
from pathlib import Path
import argparse
from glob import glob
from time import time

import numpy as np
from joblib import Parallel, delayed

# File path
filepath = Path(__file__).resolve().parent
import main_lrn_crv
from datasplit.split_getter import get_unq_split_ids

parser = argparse.ArgumentParser()
parser.add_argument('-sd', '--splitdir', required=True, default=None, type=str,
                    help='Full path to data splits (default: None).')
parser.add_argument('-ns', '--n_splits', default=10, type=int,
                    help='Use a subset of splits (default: 10).')
parser.add_argument('--par_jobs', default=1, type=int, 
                    help=f'Number of joblib parallel jobs (default: 1).')
args, other_args = parser.parse_known_args()

# 'splitdir' is also required for the function main_lrn_crv()
other_args.extend(['--splitdir', args.splitdir]) 

# Number of parallel jobs
par_jobs = int( args.par_jobs )
assert par_jobs > 0, f"The arg 'par_jobs' must be at least 1 (got {par_jobs})"

# Data file names
split_pattern = f'1fold_s*_*_id.csv'
splitdir = Path( args.splitdir ).resolve()
all_split_files = glob( str(Path(splitdir, split_pattern)) )

# unq = [Path(path).name.split('1fold_s')[1].split('_')[0] for i, path in enumerate(all_split_files)]
# unq_split_ids = np.unique(unq)
unq_split_ids = get_unq_split_ids(all_split_files)

# Determine n_splits
n_splits = np.min([ len(unq_split_ids), args.n_splits ])

# Main func designed primarily for joblib Parallel
def gen_splits(split_id, *args):
    main_lrn_crv.main([ '--split_id', str(split_id), *args ]) 

# Main execution
t0 = time()
if par_jobs > 1:
    # https://joblib.readthedocs.io/en/latest/parallel.html
    results = Parallel(n_jobs=par_jobs, verbose=1)(
            delayed(gen_splits)(split_id, *other_args) for split_id in unq_split_ids[:n_splits] )
            # delayed(gen_splits)(dfile, *other_args) for dfile in dfiles )
else:
    for i, split_id in enumerate(unq_split_ids):
        print('Processing split_id', split_id)
        gen_splits(split_id, *other_args)
    
t_end = time() - t0
print('Runtime {:.2f} mins'.format( t_end/60 ))
print('Done.')

