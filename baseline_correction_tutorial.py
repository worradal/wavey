from argparse import ArgumentParser

from matplotlib import pyplot as plt

from data import Data


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('spectrum_dir', help='Path of directory containing all spectra files')
    parser.add_argument('-wf', '--weight_file', required=False, help='Path to weight file')
    parser.add_argument('-nt', '--num_time', type=int,required=True, help='Number of time runs per run')
    
    args = parser.parse_args()
    spectral_data = Data(in_dir=args.spectrum_dir, num_time_points=args.num_time)

    for spectrum_id in [0, 1, 2]:
        plt.plot(list(range(len(spectral_data.y[:, spectrum_id]))), 
                 spectral_data.y[:, spectrum_id])
        plt.title(f'sample id {spectrum_id}')
        plt.show()
    
    baseline_configs = {
        'lambda_':  1e5,
        'stop_ratio': 1e-6, 
        'max_iters': 10000
    }
    spectral_data.baseline_correct(method='arpls', configs=baseline_configs)
    for spectrum_id in [0, 1, 2]:
        plt.plot(list(range(len(spectral_data.y[:, spectrum_id]))), 
                 spectral_data.y[:, spectrum_id], c='red')
        plt.title(f'sample id {spectrum_id} corrected')
        plt.show()
    



    

