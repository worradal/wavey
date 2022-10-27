from argparse import ArgumentParser
from os import makedirs
from os.path import join

import yaml

from data import Data

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('config_fpath')
    args = parser.parse_args()
    with open(args.config_fpath, 'r') as fp:
        configs = yaml.safe_load(fp)
    
    spectrum_dir = configs['spectrum_dir']
    weight_file = configs.get('weight_file')
    num_time_points = int(configs.get('number_of_time_points'))
    baseline_correction_method = configs.get('baseline_correction_method', None)
    baseline_correction_configs = configs.get('baseline_correction_configs')
    out_dir = configs['out_dir']
    makedirs(out_dir, exist_ok=True)
    
    original_data_fpath = join(out_dir, 'original_data.csv')
    transformed_data_fpath = join(out_dir, 'transformed_data.csv')

    spectral_data = Data(in_dir=spectrum_dir, num_time_points=num_time_points)
    print('Saving original data to ', original_data_fpath)
    spectral_data.save_to(original_data_fpath)

    if baseline_correction_method is not None:
        spectral_data.baseline_correct(
            method=baseline_correction_method, 
            configs=baseline_correction_configs)
        baseline_corrected_data_fpath = join(out_dir, 'baseline_corrected_data.csv')
        print('Saving baseline corrected data to ', baseline_corrected_data_fpath)
        spectral_data.save_to(baseline_corrected_data_fpath)

    spectral_data.fourier_transform()
    if weight_file is not None:
        spectral_data.weight(fpath=weight_file)
    spectral_data.inverse_fourier_transform()
    print('Saving transformed data to ', transformed_data_fpath)
    spectral_data.save_to(transformed_data_fpath)

