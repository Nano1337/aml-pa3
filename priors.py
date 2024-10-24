import jax
import jax.numpy as np

'''
A Gaussian prior with parameters
    * mean_prior: a D-dimensional vector
    * precision_prior: the precision matrix is a scaled identity matrix, with this parameter being the scalar
'''
def gaussian_prior(mean_prior, precision_prior):

    def log_prior(mu):
        diff = mu - mean_prior
        lp = -0.5 * precision_prior * np.sum(diff**2)
        jax.debug.print("[DEBUG] Gaussian Prior Log Probability: {}", lp)
        return lp
    #
    return log_prior
#

'''
A gamma prior with parameters
    * shape_prior >= 1
    * rate_prior >= 1
'''
def gamma_prior(shape_prior, rate_prior):

    def log_prior(precision):
        lp = (shape_prior - 1) * np.log(precision) - rate_prior * precision
        jax.debug.print("[DEBUG] Gamma Prior Log Probability: {}", lp)
        return lp
    #
    return log_prior
#

'''
A Wishart prior with parameters
    * prec_scale_prior: the inverse of the scale matrix (thus, no need to invert anything)
    * dof_prior: strength of prior, should be at least D+1 for a D x D matrix input
'''
def wishart_prior(prec_scale_prior, dof_prior):

    def log_prior(L):
        precision = L @ L.T
        D = precision.shape[0]
        log_det_precision = 2 * np.sum(np.log(np.diag(L)))
        trace_term = np.trace(prec_scale_prior @ precision)
        lp = 0.5 * ((dof_prior - D - 1) * log_det_precision - trace_term)
        print(f"[DEBUG] Wishart Prior Log Probability: {lp}")
        return lp
    #
    return log_prior
#

'''
A symmetric Dirichlet prior: all entries of the concentration vector are assigned concentration_prior
'''
def dirichlet_prior(concentration_prior):

    def log_prior(mass):
        lp = (concentration_prior - 1) * np.sum(np.log(mass))
        print(f"[DEBUG] Dirichlet Prior Log Probability: {lp}")
        return lp
    #
    return log_prior
#






