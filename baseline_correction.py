"""Baseline correction"""
import numpy as np

from wavey.spectrum import Spectrum

class IAsLS:
    def __init__(self, p, lambda_, lambda_1) -> None:
        self.p = p
        self.lambda_ = lambda_
        self.lambda_1 = lambda_1
    
    def get_baseline(self, spectrum: Spectrum) -> np.ndarray:
        y0 = spectrum.intensities
        z0 = self._get_second_order_fit(y0)

        def _get_residuals(y, z):
            return y - z

        def _update_w(residuals) -> np.ndarray:
            w = []
            for resid in residuals:
                if resid < 0:
                    w.append(1-self.p)
                else:
                    w.append(self.p)
            return np.array(w)
        
        curr_w = _update_w(_get_residuals(y0, z0))
        zi = _solve_zi(w, y0) #####
        new_w = _update_w(_get_residuals(y0, zi))
        

        


