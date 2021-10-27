import numpy as np

class L2_Norm_Squared(object):
    def __init__(self, coeff=1.):
        self.coeff = coeff

    def func_eval(self, x):
        """! Compute the function value of the \f$\ell_1\f$ - norm
        
        Parameters
        ---------- 
        @param x : input vector
            
        Returns
        ---------- 
        @retval : function value
        """
        return self.coeff*np.sum(np.square(x))

    def prox_eval(self, x, prox_param):
        """! Compute the proximal operator of the \f$\ell_2^2\f$ - norm

        \f$ prox_{\lambda \|.\|_2^2} = {arg\min_z}\left\{\|.\|_2^2 + \frac{1}{2\lambda}\|z - x\|^2\right\} \f$
        
        Parameters
        ---------- 
        @param w : input vector
        @param prox_param : penalty paramemeter
            
        Returns
        ---------- 
        @retval : output vector
        """
        prox_param *= self.coeff
        norm_x = np.linalg.norm(x, ord=2)
        return np.maximum(1 - prox_param/norm_x, 0) * x