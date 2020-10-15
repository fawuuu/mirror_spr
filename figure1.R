##### Code for Figure 1 ##### 

# Run experiment for Figure 1
n = 50000       # Dimension of signal
m = 1000        # Sample size
k = 10          # Sparsity level
sigma = 0       # Noise level
iter = 2000     # Number of iterations

set.seed(1)

# Generate data
data = gen_data_sparse(m, n, k, sigma)
# Estimate index for initialization
ini1 = sort(t(data$A^2) %*% data$y, index.return = TRUE)$ix[n]

# Run HWF with different values for beta
# Compute L2 error and Bregman divergence within algorithm due to large n
res1 = hwf(data$A, data$y, ini = ini1, step = 0.1, beta = 1e-6, 
           iteration = iter, iterates = FALSE, x_star = data$x)
res2 = hwf(data$A, data$y, ini = ini1, step = 0.1, beta = 1e-10, 
           iteration = iter, iterates = FALSE, x_star = data$x)
res3 = hwf(data$A, data$y, ini = ini1, step = 0.1, beta = 1e-14,
           iteration = iter, iterates = FALSE, x_star = data$x)

# Create plot for Figure 1
pdf(file = "plot_beta.pdf", width = 8, height = 3)

par(mfrow=c(1,2), mai = c(0.7, 0.7, 0.1, 0.1), bg = "transparent")
plot(log(res3[,1]), type = "l", lwd = 2, col = "red", ylab = "", xlab = "", xaxt = "n")
axis(1, at = 200*(0:10), label = 200*(0:10))
title(xlab = "Iteration t", ylab = "Relative error (log)", line = 2.2)
lines(log(res2[,1]), col = "blue", lwd = 2)
lines(log(res1[,1]), col = "black", lwd = 2)
legend('topright',
       legend=c(expression(paste(beta, "=", 10^{-6})), expression(paste(beta, "=", 10^{-10})),
                expression(paste(beta, "=", 10^{-14}))),
       col=c("black", "blue", "red"), lwd = 2, cex = 0.8, inset = c(0.025, 0.025), bty = "n")

plot(log(res3[,2]), type = "l", lwd = 2, col = "red", ylab = "", xlab = "", xaxt = "n")
axis(1, at = 200*(0:10), label = 200*(0:10))
title(xlab = "Iteration t", ylab = "Bregman divergence (log)", line = 2.2)
lines(log(res2[,2]), col = "blue", lwd = 2)
lines(log(res1[,2]), col = "black", lwd = 2)
legend('topright',
       legend=c(expression(paste(beta, "=", 10^{-6})), expression(paste(beta, "=", 10^{-10})),
                expression(paste(beta, "=", 10^{-14}))),
       col=c("black", "blue", "red"), lwd = 2, cex = 0.8, inset = c(0.025, 0.025), bty = "n")

dev.off()