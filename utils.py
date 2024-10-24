import jax
from jax import random
import jax.numpy as np

prng_key = random.key(0)

'''
Initialize the PRNG with unique `seed`.
'''
def init_prng(seed):
    global prng_key
    prng_key = random.PRNGKey(seed)
    return prng_key
#

'''
Whenever you call randint or randn, you need to pass in as the first argument a call to this function.
This will advance the PRNG.
'''
def grab_prng():
    global prng_key
    _,prng_key = random.split(prng_key)
    return prng_key
#

'''
Assemble a precision matrix, given (1) its diagonal entries (1D array), and (2) its off-diagonal entries (1D array).
'''
def assemble_precision(diagonal, off_diagonal):
    D = diagonal.shape[0]
    return np.diag(diagonal) + np.zeros((D, D)).at[np.tril_indices(D, k = -1)].set(off_diagonal)
#
