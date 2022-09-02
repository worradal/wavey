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

    for shift in range(spectral_data.y.shape[0]):
        plt.plot(list(range(len(spectral_data.y[shift, :]))), spectral_data.y[shift, :])
        plt.title(f'shift id {shift}')
        plt.show()
    
    
    spectral_data.fourier_transform()
    for shift in range(spectral_data.y.shape[0]):
        plt.plot(list(range(len(spectral_data.y[shift, :]))), spectral_data.y[shift, :])
        plt.title(f'after FT shift id {shift}')
        plt.show()
    
    spectral_data.weight(fpath='/Users/himaghnabhattacharjee/Documents/Research/MES/example_weights.csv')
    for shift in range(spectral_data.y.shape[0]):
        plt.plot(list(range(len(spectral_data.y[shift, :]))), spectral_data.y[shift, :])
        plt.title(f'after FT and weight shift id {shift}')
        plt.show()
    
    spectral_data.inverse_fourier_transform()
    for shift in range(spectral_data.y.shape[0]):
        plt.plot(list(range(len(spectral_data.y[shift, :]))), spectral_data.y[shift, :])
        plt.title(f'after FT shift and r-FFT id {shift}')
        plt.show()


    

