import jax
import jax.numpy as np
from jax import random, vmap, grad

'''
The evidence lower bound. Takes in 3 arguments:
    * log_priors should be a dictionary that contains the priors (each a callable function) for all model parameters.
    * transformations should be a dictionary that contains the transformations (each an instance of ParamTransformation subclass) for all model parameters.
    * log_likelihood should be a callable function that evaluates the log likelihood, given a set of model parameters
Importantly, the dictionaries for log_priors and transformations should be structured in the exact same way.
'''
def ELBO(log_priors, transformations, log_likelihood):

    '''
    Function for actually computing the ELBO. Need to do the following:
        (1) Compute entropy of variational distribution (this is simple).
        (2) For each model parameter, draw from its corresponding variational distribution, and transform it to the appropriate constrained space.
        (3) Evaluate the corresponding log prior for each drawn model parameter.
        (4) Evaluate the corresponding log-determinant Jacobian for each drawn model parameter.
        (5) Last, compute the log likelihood, given all drawn model parameters.
    This function takes in as input the set of variational parameters. Here, again, this should be a dictionary formatted identical to the above (log_priors and transformations).
    But the entries ("leaf" nodes, in PyTree terminology) should be arrays consisting of the variational parameters, in correspondence with model parameters.
    '''
    def evidence_lower_bound(variational_parameters, keys):
        # Step 1: Compute entropy of the variational distribution (untransformed)
        entropy_untransformed = jax.tree_util.tree_reduce(
            lambda acc, x: acc + 0.5 * np.sum(np.log(2 * np.pi * np.e * x[1])),
            variational_parameters, 
            0.0
        )
        print(f"[DEBUG] Entropy (Untransformed): {entropy_untransformed}")

        # Step 2: Compute sum of log-determinant Jacobians from transformations
        log_det_jacobian_sum = 0.0
        for param_name, transformation in transformations.items():
            if isinstance(transformation, dict):
                # Handle nested transformations (e.g., for compound parameters)
                for sub_name, sub_transformation in transformation.items():
                    # Extract corresponding variational parameter (assuming variational_parameters[param_name] is a 2-row array)
                    # Typically, the first row is the mean, second is variance
                    variational_param = variational_parameters[param_name][0]
                    log_det_jacobian = sub_transformation.ldj(transformation[sub_name].transformation(variational_param))
                    log_det_jacobian_sum += log_det_jacobian
                    print(f"[DEBUG] Log-Det Jacobian for {param_name}.{sub_name}: {log_det_jacobian}")
            else:
                # Handle simple transformations
                variational_param = variational_parameters[param_name][0]
                transformed_param = transformation.transformation(variational_param)
                log_det_jacobian = transformation.ldj(transformed_param)
                log_det_jacobian_sum += log_det_jacobian
                print(f"[DEBUG] Log-Det Jacobian for {param_name}: {log_det_jacobian}")

        print(f"[DEBUG] Sum of Log-Det Jacobians: {log_det_jacobian_sum}")

        # Step 3: Compute the corrected entropy
        # According to the change of variables formula, entropy should include the log-det Jacobians
        entropy_corrected = entropy_untransformed + log_det_jacobian_sum
        print(f"[DEBUG] Entropy (Corrected with Jacobians): {entropy_corrected}")

        # Step 4: Draw samples from the variational distribution and apply transformations
        samples = jax.tree_map(
            lambda t, vp, k: t.sample(vp, k),
            transformations, 
            variational_parameters,
            keys
        )
        print(f"[DEBUG] Samples: {samples}")

        # Step 5: Compute log priors for each transformed sample
        log_prior_sum = 0.0
        for log_prior, sample in zip(jax.tree_leaves(log_priors), jax.tree_leaves(samples)):
            current_log_prior = log_prior(sample)
            log_prior_sum += current_log_prior
            print(f"[DEBUG] Log Prior ({log_prior.__name__}): {current_log_prior}")
        print(f"[DEBUG] Log Prior Sum: {log_prior_sum}")

        # Step 6: Compute log likelihood for the transformed samples
        log_likelihood_sum = log_likelihood(samples)
        print(f"[DEBUG] Log Likelihood Sum: {log_likelihood_sum}")

        # Step 7: Combine all components to compute the ELBO
        elbo = log_prior_sum + log_likelihood_sum - entropy_corrected
        print(f"[DEBUG] ELBO: {elbo}")

        return elbo

    # Attach transformations to evidence_lower_bound for access in optimize
    evidence_lower_bound.transformations = transformations
    
    return evidence_lower_bound
#

'''
Your optimization loop. Maximize the ELBO (elbo) with respect to the variational parameters, starting at initial_variational_params.
T is the number of optimization steps to take
init_step_size is the initial step size for the gradient descent scheme
alpha is in correspondence with the ADVI paper notation, representing the coefficient in the exponentially-weighted average of squared gradients.
mc_samples is the number of MC samples to take in estimating the gradient.
For simplicity (at some computational expense), compute the gradient multiple times over different samples of the variational distribution, and then average the results.
'''
def optimize(initial_variational_params, elbo, T, init_step_size=0.001, alpha=0.1, mc_samples=1):
    """
    Optimization loop to maximize the ELBO using gradient-based methods.

    Args:
        initial_variational_params (dict): Initial variational parameters.
        elbo (callable): Function to compute the ELBO.
        T (int): Number of optimization steps.
        init_step_size (float, optional): Initial step size. Defaults to 1.0.
        alpha (float, optional): Coefficient for exponentially-weighted average of squared gradients. Defaults to 0.1.
        mc_samples (int, optional): Number of MC samples. Defaults to 1.

    Returns:
        dict: Optimized variational parameters.
    """
    # Initialize variational parameters
    variational_params = jax.tree_map(lambda x: x.copy(), initial_variational_params)
    print(f"[DEBUG] Initial Variational Parameters: {variational_params}")

    # Initialize exponentially-weighted average of squared gradients
    s = jax.tree_map(lambda x: np.zeros_like(x), variational_params)
    
    for t in range(T):
        # Generate a new random key for this iteration
        key = random.PRNGKey(t)
        
        # Generate a batch of keys for MC samples
        keys = random.split(key, mc_samples)
        
        # Vectorize ELBO computation over MC samples
        elbo_batch = vmap(lambda k: elbo(variational_params, generate_key_tree(k, elbo.transformations)))(keys)
        print(f"[DEBUG] ELBO Batch at step {t}: {elbo_batch}")
        
        # Compute mean ELBO over MC samples
        mean_elbo = np.mean(elbo_batch)
        print(f"[DEBUG] Mean ELBO at step {t}: {mean_elbo}")
        
        # Objective is to maximize ELBO, so minimize negative ELBO
        objective = -mean_elbo
        print(f"[DEBUG] Objective (Negative ELBO) at step {t}: {objective}")
        
        # Compute gradient of the objective
        grads = grad(lambda params: - np.mean(vmap(lambda k: elbo(params, generate_key_tree(k, elbo.transformations)))(keys)))(variational_params)
        
        # Compute norms of gradients to check for exploding gradients
        gradient_norm = jax.tree_util.tree_reduce(lambda acc, x: acc + np.sum(x**2), grads, 0.0) ** 0.5
        print(f"[DEBUG] Gradient Norm at step {t}: {gradient_norm}")
        if np.isnan(gradient_norm):
            print("[ERROR] Gradient Norm is NaN!")
        
        # Update exponentially-weighted average of squared gradients
        s = jax.tree_map(lambda s_val, g: (1 - alpha) * s_val + alpha * g**2, s, grads)
        
        # Compute adaptive learning rate
        step_size = init_step_size / np.sqrt(t + 1)
        print(f"[DEBUG] Step Size at step {t}: {step_size}")
        
        # Update variational parameters
        variational_params = jax.tree_map(
            lambda p, g, s_val: p + step_size * g / (np.sqrt(s_val) + 1e-8),
            variational_params, grads, s
        )
        
        # Print progress every 100 iterations
        if t % 100 == 0:
            print(f"Step {t}, Loss: {objective}")
    
    return variational_params
#

def generate_key_tree(key, tree):
    """
    Recursively splits a PRNG key to match the structure of the given tree.

    Args:
        key: A JAX PRNG key.
        tree: A pytree (e.g., dict) whose structure will be mirrored in the output keys.

    Returns:
        A pytree of PRNG keys matching the structure of `tree`.
    """
    if isinstance(tree, dict):
        num_splits = len(tree)
        split_keys = random.split(key, num_splits)
        return {k: generate_key_tree(split_keys[i], v) for i, (k, v) in enumerate(tree.items())}
    else:
        return key
