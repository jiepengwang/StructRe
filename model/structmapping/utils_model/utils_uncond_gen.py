import utils.utils_io as IOUtils
import numpy as np
import os

# Function to fit a multivariate Gaussian distribution
def fit_gaussian(z):
    mean = np.mean(z, axis=0)
    cov = np.cov(z, rowvar=False)  # Get covariance matrix
    return mean, cov

# Sampling from the Gaussian distribution
def sample_gaussian(mean, cov, num_samples=100):
    return np.random.multivariate_normal(mean, cov, num_samples)


def read_lis_part_code_only(path_shape_code):
    dict_code = IOUtils.read_h5_to_dict_recursively(path_shape_code)
    return dict_code

def read_lis_part_code(path_shape_code, is_load_presample=True):
    num_samples = 1000
    path_sample = f'{os.path.splitext(path_shape_code)[0]}_sample_1k.npy'
    
    if (
        is_load_presample
        and os.path.exists(path_sample)
        and os.path.getmtime(path_sample) >= os.path.getmtime(path_shape_code)
    ):
        lis_sample = np.load(path_sample)
        return lis_sample
    
    dict_code = IOUtils.read_h5_to_dict_recursively(path_shape_code)
    if not dict_code:
        raise ValueError(f'No shape codes found in {path_shape_code}')

    lis_values = list(dict_code.values())
    lis_values = np.concatenate(lis_values, axis=0).squeeze(1)

    mean, cov = fit_gaussian(lis_values)
    print(f'Mean shape code: {mean.shape}')
    print(f'Cov shape code: {cov.shape}')

    lis_sample = sample_gaussian(mean, cov, num_samples=num_samples)
    np.save(path_sample, lis_sample)


    return lis_sample
