  from scipy import sparse
  from scipy.stats import uniform
  
  def sparse_esn_reservoir(dim, spectral_radius, density, symmetric):
      """Creates a COO representation of a sparse ESN reservoir.
      Params:
          dim: int, dimension of the square reservoir matrix
          spectral_radius: float, largest eigenvalue of the reservoir matrix
          density: float, 0.1 corresponds to approx every tenth element
              being non-zero
          symmetric: specifies if matrix.T == matrix
      Returns:
          matrix: a square scipy.sparse.csr_matrix
      """
      rvs = uniform(loc=-1., scale=2.).rvs
      matrix = sparse.random(dim, dim, density=density, data_rvs=rvs)
      matrix = matrix.tocsr()
      if symmetric:
          matrix = sparse.triu(matrix)
          tril   = sparse.tril(matrix.transpose(), k=-1)
          matrix = matrix + tril
          # calc eigenvalues with scipy's lanczos implementation:
          eig, _ = sparse.linalg.eigsh(matrix, k=2, tol=1e-4)
      else:
          eig, _ = sparse.linalg.eigs(matrix, k=2, tol=1e-4)
      rho     = np.abs(eig).max()
      matrix  = matrix.multiply(1./rho)
      matrix  = matrix.multiply(spectral_radius)
      return matrix
