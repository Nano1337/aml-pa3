import jax
import jax.numpy as np
from jax.random import normal as randn
from jax.scipy.linalg import cho_factor # necessary for log-det Jacobian of discrete distribution (mixture weights)

from utils import *

'''
This is an abstract class for parameter transformation.
'''
class ParamTransformation:
    def __init__(self):
        pass
    #

    '''
    Take variational parameters as input, and output the transformed mean. This "transformation" will be handled by the subclass.
    Assume variational_param is a 2 x D array consisting of the mean in the first row, and diagonal covariance in the second row.
    '''
    def mean(self, variational_param):
        return self.transformation(variational_param[0])
    #

    '''
    Take variational parameters as input, draw a sample from the Gaussian, and transform the draw. This "transformation" will be handled by the subclass.
    Assume variational_param is a 2 x D array consisting of the mean in the first row, and diagonal covariance in the second row.
    '''
    def sample(self, variational_param, key):
        mean, var = variational_param
        eps = randn(key, mean.shape)
        return self.transformation(mean + np.sqrt(var) * eps)
    #
#

'''
For mean vectors, transformations are identity mappings.
'''
class MeanParamTransformation(ParamTransformation):
    def __init__(self):
        super().__init__()
    #

    '''
    Transform from unconstrained to constrained model parameters.
    '''
    def transformation(self, unconstrained_params):
        transformed = unconstrained_params
        print(f"[DEBUG] Mean Transformation Output: {transformed}")
        return transformed
    #

    '''
    Compute the log-determinant of the Jacobian of the inverse mapping (from constrained parameters to unconstrained parameters).
    '''
    def ldj(self, constrained_params):
        ldj = 0.0
        print(f"[DEBUG] Mean Transformation LDJ: {ldj}")
        return ldj
    #
#

'''
For scalar-valued precision parameters, ensure strictly positive.
'''
class Precision1DParamTransformation(ParamTransformation):
    def __init__(self):
        super().__init__()
    
    def transformation(self, unconstrained_params):
        transformed = np.exp(unconstrained_params)
        print(f"[DEBUG] Precision1D Transformation Output: {transformed}")
        return transformed
    
    def ldj(self, constrained_params):
        ldj = np.log(constrained_params)
        print(f"[DEBUG] Precision1D LDJ: {ldj}")
        return ldj

    def sample(self, variational_param, key):  # Ensure 'key' is accepted
        return super().sample(variational_param, key)
    #
#

'''
For a precision matrix, ensure we have a positive definite matrix.
'''
class PrecisionParamTransformation(ParamTransformation):
    def __init__(self):
        super().__init__()
    #

    '''
    Assume that unconstrained parameters is a vector that contains the entries of a lower triangular D x D matrix.
    The first D entries of unconstrained_params should correspond to the diagonal. These need to be strictly positive.
    The rest should be the off-diagonal entries.
    Use assemble_precision to take in the diagonal & off-diagonal entries, and return a lower-triangular matrix.
    '''
    def transformation(self, unconstrained_params):
        D = int(np.sqrt(2 * len(unconstrained_params) + 0.25) - 0.5)
        diag = np.exp(unconstrained_params[:D])
        off_diag = unconstrained_params[D:] 
        precision_matrix = assemble_precision(diag, off_diag)
        print(f"[DEBUG] Precision Transformation Output:\n{precision_matrix}")
        return precision_matrix
    #

    '''
    Assume that constrained_params is a lower triangular matrix with strictly positive entries on the diagonal.
    '''
    def ldj(self, constrained_params):
        ldj = np.sum(np.log(np.diag(constrained_params)))
        print(f"[DEBUG] Precision Transformation LDJ: {ldj}")
        return ldj
    #
#

'''
For mixture weights - a discrete probability distribution - ensure the entries are strictly positive, and sum to 1.
'''
class MixtureParamTransformation(ParamTransformation):
    def __init__(self):
        super().__init__()
    #

    def transformation(self, unconstrained_params):
        transformed = jax.nn.softmax(unconstrained_params)
        print(f"[DEBUG] Mixture Transformation Output: {transformed}")
        return transformed
    #

    '''
    This is arguably the trickiest log-determinant Jacobian to reason about. Think carefully before coding this up.
    '''
    def ldj(self, constrained_params):
        K = len(constrained_params)
        ldj = -np.sum(constrained_params * np.log(constrained_params))
        print(f"[DEBUG] Mixture Transformation LDJ: {ldj}")
        return ldj
    #
#


# Utility function to create pytree of transformations
def create_transformation_pytree(model_structure):
    def map_fn(leaf):
        if isinstance(leaf, dict):
            if 'mean' in leaf and 'precision' in leaf:
                transformations = {
                    'mean': MeanParamTransformation(),
                    'precision': PrecisionParamTransformation() if leaf['precision'].ndim > 1 else Precision1DParamTransformation()
                }
                print(f"[DEBUG] Created Transformation for Leaf: {transformations}")
                return transformations
        elif isinstance(leaf, str) and leaf == 'mixture':
            transformation = MixtureParamTransformation()
            print(f"[DEBUG] Created Mixture Transformation: {transformation}")
            return transformation
        return leaf

    transformation_pytree = jax.tree_map(map_fn, model_structure)
    print(f"[DEBUG] Transformation PyTree: {transformation_pytree}")
    return transformation_pytree





