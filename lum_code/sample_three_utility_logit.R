library(Rcpp)
library(mvtnorm)
library(tidyverse)

sourceCpp("three_utility_logit_approx_helper_functions.cpp")

vec_log_sum_exp <- function(a, b) {

  pmax(a, b) + log(1 + exp(pmin(a, b) - pmax(a, b)))

}

calc_waic_logit_approx_three_utility <- function(
  chain_results, case_vote_m, block = F, block_vote = F) {

  if (block) {
    num_votes = nrow(case_vote_m)
  } else if (block_vote) {
    num_votes = ncol(case_vote_m)
  } else {
    num_votes = sum(!is.na(as.vector(case_vote_m)))
  }

  mean_prob <- rep(-Inf, num_votes)
  mean_log_prob <- rep(0, num_votes)
  log_prob_var <- rep(0, num_votes)
  num_iter = 0

  for (i in 1:length(chain_results)) {
    result = chain_results[[i]][[1]]
    alpha_inds_1 <- grep("alpha_1", colnames(result))
    alpha_inds_2 <- grep("alpha_2", colnames(result))
    delta_inds_1 <- grep("delta_1", colnames(result))
    delta_inds_2 <- grep("delta_2", colnames(result))
    beta_inds <- grep("beta", colnames(result))
    for (j in 1:nrow(result)) {
      print(j)
      leg_ideology <- result[j, beta_inds]
      mean_m_1 <- sweep(outer(leg_ideology, result[j, delta_inds_1], "-"), 
                        2, result[j, alpha_inds_1], "*")
      mean_m_2 <- sweep(outer(leg_ideology, result[j, delta_inds_2], "-"), 
                        2, result[j, alpha_inds_2], "*")
      justice_probs <- 1 / (1 + exp(-mean_m_1) + exp(-mean_m_2)) 
      justice_probs[justice_probs < 1e-9] =
        1e-9
      justice_probs[justice_probs > (1 - 1e-9)] =
        1 - 1e-9
      log_prob <- case_vote_m * log(justice_probs) +
        (1 - case_vote_m) * log(1 - justice_probs)
      if (block) {
        log_prob <- rowSums(log_prob, na.rm = T)
      } else if (block_vote) {
        log_prob <- colSums(log_prob, na.rm = T)
      } else {
        log_prob <- log_prob[!is.na(log_prob)]
      }
      mean_prob <- vec_log_sum_exp(mean_prob, log_prob)
      next_mean_log_prob = (num_iter * mean_log_prob + log_prob) / (num_iter + 1)
      log_prob_var = log_prob_var +
        (log_prob - mean_log_prob) * (log_prob - next_mean_log_prob)
      mean_log_prob = next_mean_log_prob
      num_iter = num_iter + 1    
    }
  }
  return(
    log(mean_prob / num_iter) -
      (log_prob_var) / (num_iter - 1))
}

init_data_rcpp_logit_approx <- function(
  vote_m, leg_pos_init, alpha_pos_init, delta_pos_init, 
  y_star_m_1_init, y_star_m_2_init, y_star_m_3_init, 
  label_v_m_1_init, label_v_m_2_init, label_v_m_3_init,
  total_iter) {
  
  if (!is.null(leg_pos_init)) {
    leg_pos_m <- 
      matrix(leg_pos_init, nrow = total_iter, ncol = nrow(vote_m), byrow = T)
  } else {
    leg_pos_m <- matrix(0, nrow = total_iter, ncol = nrow(vote_m))
  }
  
  if (!is.null(alpha_pos_init)) {
    alpha_pos_m <- 
      matrix(t(alpha_pos_init), nrow = total_iter, ncol = 2 * ncol(vote_m), byrow = T)
  } else {
    alpha_pos_m <- 
      matrix(rep(c(-1, 1), ncol(vote_m)), 
             nrow = total_iter, ncol = 2 * ncol(vote_m), byrow = T)
  }
  
  if (!is.null(delta_pos_init)) {
    delta_pos_m <- 
      matrix(t(delta_pos_init), nrow = total_iter, ncol = 2 * ncol(vote_m), byrow = T)
  } else {
    delta_pos_m <- 
      matrix(0, nrow = total_iter, ncol = 2 * ncol(vote_m), byrow = T)
  }
  
  if (!is.null(y_star_m_1_init)) {
    y_star_m_1 <- y_star_m_1_init
    y_star_m_2 <- y_star_m_2_init
    y_star_m_3 <- y_star_m_3_init
  } else {
    y_star_info <- init_y_star_m(vote_m)
    y_star_m_1 <- y_star_info[[1]]
    y_star_m_2 <- y_star_info[[2]]
    y_star_m_3 <- y_star_info[[3]]
  }
  
  if (!is.null(label_v_m_1_init)) {
    label_v_m_1 <- label_v_m_1_init
    label_v_m_2 <- label_v_m_2_init
    label_v_m_3 <- label_v_m_3_init
  } else {
    label_v_m_1 <- matrix(0, nrow = nrow(vote_m), ncol = ncol(vote_m))
    label_v_m_2 <- matrix(0, nrow = nrow(vote_m), ncol = ncol(vote_m))
    label_v_m_3 <- matrix(0, nrow = nrow(vote_m), ncol = ncol(vote_m))
  }
  
  all_params_draw <- cbind(leg_pos_m, alpha_pos_m, delta_pos_m)
  beta_start_ind = 0;
  alpha_start_ind = nrow(vote_m);
  alpha_2_start_ind = alpha_start_ind + ncol(vote_m);
  delta_start_ind = alpha_2_start_ind + ncol(vote_m);
  delta_2_start_ind = delta_start_ind + ncol(vote_m);
  
  return(list(all_params_draw, y_star_m_1, y_star_m_2, 
              y_star_m_3, label_v_m_1, label_v_m_2, 
              label_v_m_3, beta_start_ind,
              alpha_start_ind, alpha_2_start_ind,
              delta_start_ind, delta_2_start_ind))
}

sample_three_utility_logit_approx_rcpp <- function(
  vote_m, leg_mean, leg_s, alpha_mean, alpha_cov_s,
  delta_mean, delta_cov_s, 
  normal_approx_mean, normal_approx_sd, normal_approx_prob,
  num_iter = 2000, start_iter = 0, keep_iter = 1,
  leg_pos_init = NULL, alpha_pos_init = NULL, delta_pos_init = NULL,
  y_star_m_1_init = NULL, y_star_m_2_init = NULL, y_star_m_3_init = NULL,
  label_v_m_1_init = NULL, label_v_m_2_init = NULL, label_v_m_3_init = NULL,
  pos_ind = 0, neg_ind = 0, start_val = NULL) {
  
  total_iter = (num_iter - start_iter) %/% keep_iter
  init_info <- init_data_rcpp_logit_approx(
    vote_m, leg_pos_init, alpha_pos_init, delta_pos_init, 
    y_star_m_1_init, y_star_m_2_init, y_star_m_3_init, 
    label_v_m_1_init, label_v_m_2_init, label_v_m_3_init,
    total_iter)
  
  if (!is.null(start_val)) {
    init_info[[1]][1,] <- start_val
  }
  
  draw_info <- sample_three_utility_logit_approx(
    vote_m, init_info[[1]], init_info[[2]], init_info[[3]], init_info[[4]],
    init_info[[5]], init_info[[6]], init_info[[7]],
    init_info[[8]], init_info[[9]], init_info[[10]], 
    init_info[[11]], init_info[[12]], leg_mean, leg_s, 
    alpha_mean, alpha_cov_s, delta_mean, delta_cov_s,
    normal_approx_mean, normal_approx_sd, normal_approx_prob,
    num_iter, start_iter, keep_iter, pos_ind - 1, neg_ind - 1)
  
  all_param_draw = draw_info[[1]]
  leg_names <- sapply(rownames(vote_m), function(name) {paste(name, "beta", sep = "_")})
  if (is.null(colnames(vote_m))) {
    colnames(vote_m) <- sapply(1:ncol(vote_m), function(i) {
      paste("vote", i, sep = "_")
    })
  }
  alpha_vote_names_1 <- sapply(colnames(vote_m), function(name) {
    paste(name, "alpha", "1", sep = "_")
  })
  alpha_vote_names_2 <- sapply(colnames(vote_m), function(name) {
    paste(name, "alpha", "2", sep = "_")
  })
  delta_vote_names_1 <- sapply(colnames(vote_m), function(name) {
    paste(name, "delta", "1", sep = "_")
  })
  delta_vote_names_2 <- sapply(colnames(vote_m), function(name) {
    paste(name, "delta", "2", sep = "_")
  })
  colnames(all_param_draw) <- 
    c(leg_names, alpha_vote_names_1, alpha_vote_names_2, delta_vote_names_1, delta_vote_names_2)
  
  return(c(list("param_draws" = all_param_draw), draw_info[-1]))
}

#load("gumbel_vb_approx_info.Rdata")
#load("data_file")
#chain_run <- sample_three_utility_logit_approx_rcpp(
#  house_votes_m, 0, 1, c(0, 0), diag(2) * 25,
#  c(-2, 10), diag(2) * 10,
#  gumbel_kl_approx[[1]], gumbel_kl_approx[[2]],
#  gumbel_kl_approx[[3]],
#  num_iter = 500000, start_iter = 300000, keep_iter = 10)


