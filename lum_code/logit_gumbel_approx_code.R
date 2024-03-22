library(ordinal)
library(matrixStats)

compute_normal_mix_gumbel_kl_for_x <- function(x, normal_means, normal_sds, mixing_prob) {
  log_mix_prob_comp <- 
    mapply(function(mean, sd, prob) 
      log(prob) + dnorm(x, mean, sd, log = T), 
      normal_means, normal_sds, mixing_prob)
  log_mix_prob <- apply(log_mix_prob_comp, 1, logSumExp)
  kl_prob <- rep(0, length(x))
  non_zero_inds <- which(dgumbel(x) > .Machine$double.neg.eps)
  kl_prob[non_zero_inds] <-
    dgumbel(x[non_zero_inds]) * (dgumbel(x[non_zero_inds], log = T) - 
                                   log_mix_prob[non_zero_inds])
  return(kl_prob)
}

compute_normal_mix_gumbel_kl <- function(normal_means, normal_sds, mixing_prob) {
  integrate(compute_normal_mix_gumbel_kl_for_x, lower = -Inf, upper = Inf,
            normal_means = normal_means, normal_sds = normal_sds, 
            mixing_prob = mixing_prob, rel.tol = .Machine$double.eps^0.75)$value
}


optimize_next_gaussian_mixture_comp <- 
  function(next_param_info, 
           prev_normal_means, prev_normal_sds, prev_mixing_prob) {
  
  next_mixing_prob <- c(prev_mixing_prob, next_param_info[3])
  next_mixing_prob <- next_mixing_prob / sum(next_mixing_prob)
  next_mixing_prob <- pmin(next_mixing_prob, 1)
  next_mixing_prob <- pmax(next_mixing_prob, 1e-9)
  next_mixing_prob <- next_mixing_prob / sum(next_mixing_prob)
  kl <- compute_normal_mix_gumbel_kl(c(prev_normal_means, next_param_info[1]),
                                     c(prev_normal_sds, next_param_info[2]),
                                     next_mixing_prob)
  return(kl)
}

get_optimal_gaussian_mixture <- function(K) {
  normal_means <- c(-digamma(1))
  normal_sds <- c(sqrt(pi^2 / 6))
  normal_mixing_prob <- c(1)
  for (k in 2:K) {
    optim_next_info <- 
      optim(c(rnorm(1), 1, 1), optimize_next_gaussian_mixture_comp,
            prev_normal_means = normal_means, prev_normal_sds = normal_sds,
            prev_mixing_prob = normal_mixing_prob, method = "L-BFGS-B",
            lower = c(-Inf, 0, 0), upper = c(Inf, Inf, 100))
    normal_means <- c(normal_means, optim_next_info$par[1])
    normal_sds <- c(normal_sds, optim_next_info$par[2])
    normal_mixing_prob <- c(normal_mixing_prob, optim_next_info$par[3])
    normal_mixing_prob <- normal_mixing_prob / sum(normal_mixing_prob)
    normal_mixing_prob <- pmin(normal_mixing_prob, 1)
    normal_mixing_prob <- pmax(normal_mixing_prob, 1e-9)
    normal_mixing_prob <- normal_mixing_prob / sum(normal_mixing_prob)
    kl = optim_next_info$value
  }
  return(list("means" = normal_means,
              "sds" = normal_sds,
              "weights" = normal_mixing_prob,
              "kl" = kl))
}

#test_possible_mixture <- lapply(2:10, function(k) {
#  print(k); 
#  possible_results <- lapply(1:10, function(i) get_optimal_gaussian_mixture(k))
#  possible_results[[which.min(sapply(possible_results, function(info) info$kl))]]})


