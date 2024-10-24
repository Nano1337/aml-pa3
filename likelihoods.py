import jax
import jax.numpy as np

'''
Log likelihood for 1D Gaussian:
    * Data matrix X supplied as input, should be referenced in the inner function
    * model_parameters should be a dictionary with keys "mean" and "precision", whose values are the mean and precision parameter
    * reduce indicates whether we should return the summed log likelihood across data items in X, or return the per-item log likelihoods as a vector
'''
def gaussian_1d_likelihood(X):

    def log_lik_func(model_parameters, reduce=True):
        mean = model_parameters['mean']
        precision = model_parameters['precision']
        variance = 1.0 / precision  # Ensure precision > 0 due to transformations

        diff = X - mean
        log_liks = -0.5 * np.log(2 * np.pi * variance) - 0.5 * precision * diff**2

        if reduce:
            result = np.sum(log_liks)
            jax.debug.print("[DEBUG] Gaussian 1D Log Likelihood Sum: {}", result)
            return result
        else:
            jax.debug.print("[DEBUG] Gaussian 1D Log Likelihoods: {}", log_liks)
            return log_liks
    #
    return log_lik_func
#

'''
Log likelihood for multivariate Gaussian. X and reduce are the same as in gaussian_1d_likelihood.
model_parameters should be a dictionary with keys "mean" and "precision":
    * The mean is a D-vector.
    * The precision is a lower triangular D x D matrix, representing the Cholesky factor. The actual precision is formed by: precision @ precision.T

'''
def gaussian_likelihood(X):

    def log_lik_func(model_parameters, reduce=True):
        mean = model_parameters['mean']
        precision = model_parameters['precision']
        
        D = mean.shape[0]
        diff = X - mean
        
        # Compute log determinant of precision matrix
        log_det_precision = 2 * np.sum(np.log(np.diag(precision)))
        jax.debug.print("[DEBUG] Log Determinant of Precision: {}", log_det_precision)

        # compute quadratic term 
        quad_term = np.sum((diff @ precision) * diff, axis=-1)
        jax.debug.print("[DEBUG] Quadratic Term: {}", quad_term)
        
        # Compute log likelihood
        log_lik = 0.5 * (D * np.log(2 * np.pi) - log_det_precision - quad_term)
        jax.debug.print("[DEBUG] Log Likelihood: {}", log_lik)
        
        if reduce:
            return np.sum(log_lik)
        else:
            return log_lik
    #
    return log_lik_func
#

'''
Log likelihood for a mixture model.
The likelihood_func should be the result of gaussian_likelihood(X) (in principle, any location-scale likelihood)
model_parameters is a bit more complicated here. Make the following assumption:
    * The mixture weights - a vector - is accessed via key "mixture".
    * The individual components of the mixture can be accessed via key "components". This gives a list where each item is another dictionary,
      consisting of "mean" and "precision", whose parameters should match the above log likelihood functions.

NOTE: implement logsumexp for numerical stability.
'''
def mixture_likelihood(likelihood_func):

    def log_lik_func(model_parameters, reduce=True):
        mixture_weights = model_parameters['mixture']
        components = model_parameters['components']

        log_liks = []
        for i, component in enumerate(components):
            component_log_lik = likelihood_func(component, reduce=False)
            log_liks.append(component_log_lik)
            jax.debug.print("[DEBUG] Component {} Log Likelihoods: {}", i, component_log_lik)

        log_liks = np.array(log_liks)
        log_weights = np.log(mixture_weights)
        jax.debug.print("[DEBUG] Log Weights: {}", log_weights)

        # Use logsumexp trick for numerical stability
        max_log_lik = np.max(log_weights[:, np.newaxis] + log_liks, axis=0)
        jax.debug.print("[DEBUG] Max Log Likelihood for LogSumExp: {}", max_log_lik)

        per_sample_log_liks = max_log_lik + np.log(np.sum(
            np.exp(log_weights[:, np.newaxis] + log_liks - max_log_lik),
            axis=0
        ))

        if reduce:
            result = np.sum(per_sample_log_liks)
            jax.debug.print("[DEBUG] Mixture Log Likelihood Sum: {}", result)
            return result
        else:
            jax.debug.print("[DEBUG] Mixture Log Likelihoods: {}", per_sample_log_liks)
            return per_sample_log_liks
    #
    return log_lik_func
#





