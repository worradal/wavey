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

    for time_id in range(spectral_data.y.shape[-1]):
        plt.plot(spectral_data.x, spectral_data.y[:, time_id])
        plt.title(f'Time id {time_id}')
        plt.show()
    
    
    spectral_data.fourier_transform()
    for time_id in range(spectral_data.y.shape[-1]):
        plt.plot(spectral_data.x, spectral_data.y[:, time_id], c='red')
        plt.title(f'Time id after FT {time_id}')
        plt.show()
    
    spectral_data.weight(fpath='/Users/himaghnabhattacharjee/Documents/Research/MES/example_weights.csv')
    for time_id in range(spectral_data.y.shape[-1]):
        plt.plot(spectral_data.x, spectral_data.y[:, time_id], c='green')
        plt.title(f'Time id after FT and weighting {time_id}')
        plt.show()
    
    spectral_data.inverse_fourier_transform()
    spectral_data.weight(fpath='/Users/himaghnabhattacharjee/Documents/Research/MES/example_weights.csv')
    for time_id in range(spectral_data.y.shape[-1]):
        plt.plot(spectral_data.x, spectral_data.y[:, time_id])
        plt.title(f'Time id after FT and weighting and r-FFT {time_id}')
        plt.show()


    

