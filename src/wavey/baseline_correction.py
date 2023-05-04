"""Baseline correction"""
import numpy as np
from numpy.linalg import norm
from scipy import sparse
from scipy.sparse import linalg


class ARPLS:
    """Implements the Asymmetrically reweighted penalized least square method from [1]
    
    References
    [1]: Baek, S.-J., Park, A., Ahn, Y.-J. & Choo, J.
        Baseline correction using asymmetrically reweighted penalized least squares smoothing. 
        Analyst 140, 250–257 (2014).
  
    """
    def __init__(self, lambda_: float) -> None:
        self.lambda_ = lambda_
    
    def get_baseline(self, y: np.ndarray, stop_ratio: float=1e-6, max_iters: int=10, full_output=False)-> np.ndarray:
        L = len(y)
        diag = np.ones(L - 2)
        D = sparse.spdiags([diag, -2*diag, diag], [0, -1, -2], L, L - 2)
        H = self.lambda_ * D.dot(D.T)  # The transposes are flipped w.r.t the Algorithm on pg. 252
        w = np.ones(L)
        W = sparse.spdiags(w, 0, L, L)

        current_ratio = 1
        num_iters = 0
        while current_ratio > stop_ratio:
            z = linalg.spsolve(W + H, W * y)
            d = y - z
            dn = d[d < 0]
            m = np.mean(dn)
            s = np.std(dn)
            w_new = 1 / (1 + np.exp(2 * (d - (2*s - m))/s))
            current_ratio = norm(w_new - w) / norm(w)
            w = w_new
            W.setdiag(w)  # Do not create a new matrix, just update diagonal values
            
            num_iters += 1
            if num_iters > max_iters:
                print('Maximum number of iterations exceeded')
                break

        if full_output:
            info = {'num_iters': num_iters, 'final_ratio': current_ratio}
            return z, d, info
        else:
            return z
    

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from scipy.signal import savgol_filter
    
    # Generate some data
    x = np.linspace(0, 100, 1000)
    y = 2 * np.sin(x) + 0.1 * x + 0.1 * np.random.randn(1000)
    
    # Apply baseline correction
    arpls = ARPLS(1e6)
    y_corrected, d, info = arpls.get_baseline(y, full_output=True)
    
    # Plot the results
    plt.figure()
    plt.plot(x, y, label='Original')
    plt.plot(x, y_corrected, label='Corrected')
    plt.plot(x, savgol_filter(y, 51, 3), label='Savitzky-Golay')
    plt.legend()
    plt.show()
    print(info)