import numpy as np

class Empty_Function(object):
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
        return 0

    def prox_eval(self, x, prox_param):
        """! Compute the proximal operator of the \f$\ell_1\f$ - norm

        \f$ prox_{\lambda \|.\|_1} = {arg\min_x}\left\{\|.\|_1 + \frac{1}{2\lambda}\|x - w\|^2\right\} \f$
        
        Parameters
        ---------- 
        @param w : input vector
        @param prox_param : penalty paramemeter
            
        Returns
        ---------- 
        @retval : output vector
        """
        return x
