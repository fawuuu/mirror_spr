##### Code for "A Continuous-Time Mirror Descent Approach to Sparse Phase Retrieval" #####

# Define the function to generate data
# m: number of observations
# n: dimension of the signal x^*
# k: sparsity level
# sigma: measurement noise
gen_data_sparse = function(m, n, k, sigma = 0){
  # Generate normalized k-sparse signal vector
  x = rnorm(n)
  ind = sample(1:n, k)
  x[-ind] = 0
  x = x / sqrt(sum(x^2))
  
  # Generate measurement matrix
  A = matrix(rnorm(m*n), nrow = m, ncol = n)
  
  # Generate observations
  y = as.vector((A%*%x)^2) + rnorm(m, sd = sigma)
  
  return(list(A = A, x = x, y = y))
}

# Define the empirical risk, squared-magnitude-based loss
# x: signal vector
# A: measurement matrix
# y: vector of observations
wf_loss = function(x, A, y){
  m = length(y)
  return(sum((y - (A %*% x)^2)^2) / (4*m))
}

# Define the gradient, squared-magnitude-based loss 
# x: signal vector
# A: measurement matrix
# y: vector of observations
wf_grad = function(x, A, y){
  m = nrow(A)
  return(t(A) %*% (((A%*%x)^2 - y) * (A%*%x)) / m)
}

# Define the Bregman divergence associated to the hyperbolic entropy mirror map
# between two vectors x and y
# beta: mirror map parameter
breg = function(x, y, beta){
  beta = rep(beta, length(x))
  psi1 = sum(x * asinh(x / beta) - sqrt(x^2 + beta^2))
  psi2 = sum(y * asinh(y / beta) - sqrt(y^2 + beta^2))
  grad = asinh(y / beta)
  return(psi1 - psi2 - t(grad) %*% (x - y))
}

# Define HWF
# A: measurement matrix
# y: vector of observations
# ini: index for initialization (i_max)
# step: step size
# beta: parameter of mirror map, corresponds to initialization size
# eps: stopping criterion 
# iteration: maximum number of iterations
# iterates: TRUE/FALSE, whether to save all iterates (relevant for large n)
# x_star: if iterates=FALSE, compute L2 error and Bregman divergence in each iteration
hwf = function(A, y, ini, step = 0.1, beta = 1e-6, iteration = 100, iterates = TRUE, x_star = 0){
  m = nrow(A)
  n = ncol(A)
  
  # Create matrix to save iterates if iterates=TRUE
  if(iterates == TRUE){
    X = matrix(0, nrow = iteration, ncol = n)
  }else{ # Else, create matrix to save L2 errors and Bregman divergences
    X = matrix(0, nrow = iteration, ncol = 2)
  }  
  
  # Initialization 
  # Estimate of the signal size
  theta = sqrt(mean(y))
  
  u_cur = rep(sqrt(beta/2), n)
  v_cur = rep(sqrt(beta/2), n)
  
  u_cur[ini] = sqrt(sqrt(theta^2/12) + sqrt(theta^2/12 + beta^2/4)) 
  v_cur[ini] = sqrt(-sqrt(theta^2/12) + sqrt(theta^2/12 + beta^2/4))
  
  x_cur = u_cur^2 - v_cur^2
  # Save iterate if iterates=TRUE
  if(iterates == TRUE){
    X[1,] = x_cur
  }else{ # Else, compute L2 error and Bregman divergence
    X[1,1] = min(sqrt(sum((x_cur - x_star)^2)), sqrt(sum((x_cur + x_star)^2)))
    X[1,2] = min(breg(x_star, x_cur, beta), breg(-x_star, x_cur, beta))
  } 
  
  for(t in 2:iteration){
    # Gradient updates
    r = 2 * step * wf_grad(x_cur, A, y)
    
    u_cur = u_cur * (1 - r)
    v_cur = v_cur * (1 + r)
    x_cur = u_cur^2 - v_cur^2
    
    # Save iterate if iterates=TRUE
    if(iterates == TRUE){
      X[t,] = x_cur
    }else{ # Else, compute L2 error and Bregman divergence
      X[t,1] = min(sqrt(sum((x_cur - x_star)^2)), sqrt(sum((x_cur + x_star)^2)))
      X[t,2] = min(breg(x_star, x_cur, beta), breg(-x_star, x_cur, beta))
    } 
  }
  
  return(X)
}