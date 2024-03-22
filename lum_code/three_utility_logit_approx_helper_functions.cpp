#include <RcppArmadillo.h>
#include <cmath>
#include <RcppDist.h>
#include <mvtnorm.h>
#include <RcppArmadilloExtensions/sample.h>
//Code from RcppTN: https://github.com/olmjo/RcppTN/blob/master/src/rtn1.cpp
#include "rtn1.h"

using namespace Rcpp;
using namespace arma;
using namespace std;

//[[Rcpp::depends(RcppArmadillo, RcppDist, mvtnorm)]]

const double pi2 = pow(datum::pi,2);
const double TWOPI = 6.283185307179586;

double log_sum_exp(double a, double b) {
  return(max(a, b) + log1p(exp(min(a, b) - max(a, b))));
}

double log_1p2exp(double a, double b) {
  double max_val = max(a, b);
  double min_val = min(a, b);
  if (max_val > 700) {
    return(max_val + log1p(exp(min_val - max_val) + exp(-max_val)));
  }
  return(log1p(exp(max_val) + exp(min_val)));
}

double sample_three_utility_logit_approx_beta(
    rowvec y_star_m_1, rowvec y_star_m_3, 
    rowvec alpha_v_1, rowvec alpha_v_2,
    rowvec delta_v_1, rowvec delta_v_2,
    uvec approx_label_v_1, uvec approx_label_v_3,
    double beta_mean, double beta_s,
    vec normal_approx_means, vec normal_approx_sd) { 
  
  y_star_m_1 = y_star_m_1 - alpha_v_1 % delta_v_1 - 
    normal_approx_means(approx_label_v_1).t();
  y_star_m_1 /= normal_approx_sd(approx_label_v_1).t();
  y_star_m_3 = y_star_m_3 - alpha_v_2 % delta_v_2 - 
    normal_approx_means(approx_label_v_3).t();
  y_star_m_3 /= normal_approx_sd(approx_label_v_3).t();

  
  alpha_v_1 /= normal_approx_sd(approx_label_v_1).t();
  alpha_v_2 /= normal_approx_sd(approx_label_v_3).t();
  
  double post_var = 1.0 / pow(beta_s, 2) + 
    dot(alpha_v_1, alpha_v_1) + dot(alpha_v_2, alpha_v_2);
  double post_mean = beta_mean / pow(beta_s, 2) - 
    dot(alpha_v_1, y_star_m_1) - 
    dot(alpha_v_2, y_star_m_3);

  return(randn() / sqrt(post_var) + post_mean / post_var);
}

vec sample_three_utility_logit_approx_matched_alpha(
    vec y_star_m_1, vec y_star_m_3,  
    vec beta_v, vec delta_v,
    uvec approx_label_v_1, uvec approx_label_v_3,
    vec alpha_mean_v, mat alpha_cov_s,
    vec delta_mean_v, mat delta_cov_s,
    vec normal_approx_means, vec normal_approx_sd) {
  
  vec beta_diff_v_1 = (beta_v - delta_v(0)) / normal_approx_sd(approx_label_v_1);
  vec beta_diff_v_2 = (beta_v - delta_v(1)) / normal_approx_sd(approx_label_v_3);
  
  mat post_cov = alpha_cov_s.i();
  post_cov(0, 0) += dot(beta_diff_v_1, beta_diff_v_1);
  post_cov(1, 1) += dot(beta_diff_v_2, beta_diff_v_2);
  
  vec post_mean = solve(alpha_cov_s, alpha_mean_v);
  post_mean(0) -= dot(beta_diff_v_1, 
                      (y_star_m_1 - normal_approx_means(approx_label_v_1)) / 
                        normal_approx_sd(approx_label_v_1));
  post_mean(1) -= dot(beta_diff_v_2, 
                      (y_star_m_3 - normal_approx_means(approx_label_v_3)) / 
                        normal_approx_sd(approx_label_v_3));
  post_mean = solve(post_cov, post_mean);
  
  double sample_order_up_prob = 
    R::pnorm(0, post_mean(0), sqrt(1.0 / post_cov(0,0)), false, true) +
    R::pnorm(0, post_mean(1), sqrt(1.0 / post_cov(1,1)), true, true) +
    as_scalar(dmvnorm(delta_v.t(), delta_mean_v, delta_cov_s, true));
  double sample_order_down_prob = 
    R::pnorm(0, post_mean(0), sqrt(1.0 / post_cov(0,0)), true, true) +
    R::pnorm(0, post_mean(1), sqrt(1.0 / post_cov(1,1)), false, true) +
    as_scalar(dmvnorm(delta_v.t(), -delta_mean_v, delta_cov_s, true));
  
  double log_sample_prob = sample_order_up_prob - 
    log_sum_exp(sample_order_up_prob, sample_order_down_prob);
  double match_var = (log(randu()) < log_sample_prob) * 2 - 1;
    
  vec out_v(3);
  if (match_var == 1) {
    out_v(0) = rtn1(post_mean(0), 1.0 / sqrt(post_cov(0, 0)), 
          0, datum::inf);
    out_v(1) = rtn1(post_mean(1), 1.0 / sqrt(post_cov(1, 1)), 
          -datum::inf, 0);
  } else {
    out_v(0) = rtn1(post_mean(0), 1.0 / sqrt(post_cov(0, 0)), 
          -datum::inf, 0);
    out_v(1) = rtn1(post_mean(1), 1.0 / sqrt(post_cov(1, 1)), 
          0, datum::inf);
  }
  out_v(2) = match_var;  
  
  return(out_v);
}

vec sample_three_utility_logit_approx_matched_delta(
    vec y_star_m_1, vec y_star_m_3, 
    vec alpha_v, vec beta_v, double match_var,
    uvec approx_label_v_1, uvec approx_label_v_3,
    vec delta_mean_v, mat delta_cov_s,
    vec normal_approx_means, vec normal_approx_sd) {
  
  y_star_m_1 += (alpha_v(0) * beta_v - normal_approx_means(approx_label_v_1));
  y_star_m_1 /= normal_approx_sd(approx_label_v_1);
  y_star_m_3 += (alpha_v(1) * beta_v - normal_approx_means(approx_label_v_3));
  y_star_m_3 /= normal_approx_sd(approx_label_v_3);
  vec std_alpha_0 = alpha_v(0) / normal_approx_sd(approx_label_v_1);
  vec std_alpha_1 = alpha_v(1) / normal_approx_sd(approx_label_v_3);
  

  mat post_cov = delta_cov_s.i();
  post_cov(0, 0) += dot(std_alpha_0, std_alpha_0);
  post_cov(1, 1) += dot(std_alpha_1, std_alpha_1);
  
  vec post_mean = match_var * solve(delta_cov_s, delta_mean_v);
  post_mean(0) += dot(std_alpha_0, y_star_m_1);
  post_mean(1) += dot(std_alpha_1, y_star_m_3);
  return(rmvnorm(1, solve(post_cov, post_mean),
                 post_cov.i()).t());
}

vec flip_signs(
  vec vote_v, vec alpha_val, vec beta_val, vec delta_val, double match_var,
  vec alpha_mean_v, mat alpha_cov_s, vec delta_mean_v, mat delta_cov_s) {
  
  double curr_prob = 0;
  double flip_prob = 0;
  
  for (int i = 0; i < beta_val.n_elem; i++) {
    vec mean_v_curr = -alpha_val % (beta_val(i) - delta_val);
    vec mean_v_flip = alpha_val % (beta_val(i) + delta_val);
    
    curr_prob -= log_1p2exp(mean_v_curr(0), mean_v_curr(1));
    flip_prob -= log_1p2exp(mean_v_flip(0), mean_v_flip(1));
    if (vote_v(i) < 0.5) {
      curr_prob += log_sum_exp(mean_v_curr(0), mean_v_curr(1)); 
      flip_prob += log_sum_exp(mean_v_flip(0), mean_v_flip(1));
    }
  }
  
  if (log(randu()) < (flip_prob - curr_prob)) {
    return(-join_vert(alpha_val, delta_val)); 
  }
  return(join_vert(alpha_val, delta_val));
}

unsigned int sample_approx_label_v(
  double y_star_m_val, double alpha_val, double beta_val, double delta_val,
  vec normal_approx_means, vec normal_approx_sd, vec normal_approx_prob) {
  
  vec log_prob(normal_approx_means.n_elem, fill::zeros);
  double norm_factor = -datum::inf;
  double mean_val = alpha_val * (beta_val - delta_val);
  for (int k = 0; k < normal_approx_means.n_elem; k++) {
    log_prob(k) = log(normal_approx_prob(k)) +
      R::dnorm(y_star_m_val, normal_approx_means(k) - mean_val, 
            normal_approx_sd(k), true);
    norm_factor = log_sum_exp(norm_factor, log_prob(k));
  }
  vec prob = exp(log_prob - norm_factor);
  uvec opts = linspace<uvec>(0, normal_approx_means.n_elem - 1, normal_approx_means.n_elem);
  uvec draw = Rcpp::RcppArmadillo::sample(opts, 1, false, prob);
  return(draw(0));
}

vec sample_y_star_m_na(double mean_m_1, double mean_m_2, 
                       vec normal_approx_means, vec normal_approx_sd) {
  vec out_v(3, fill::randn);
  out_v = out_v % normal_approx_sd;
  out_v(0) += normal_approx_means(0) - mean_m_1;
  out_v(1) += normal_approx_means(1);
  out_v(2) += normal_approx_means(2) - mean_m_2;
  return(out_v);
}

vec sample_y_star_m_yea(vec y_star_yea, double mean_m_1, double mean_m_2,
                        vec normal_approx_means, vec normal_approx_sd) {
  
  y_star_yea(0) = 
    rtn1(normal_approx_means(0) - mean_m_1, normal_approx_sd(0), 
         -datum::inf, y_star_yea(1));
  y_star_yea(1) = 
    rtn1(normal_approx_means(1), normal_approx_sd(1), 
         max(y_star_yea(0), y_star_yea(2)), datum::inf);
  y_star_yea(2) = 
    rtn1(normal_approx_means(2) - mean_m_2, normal_approx_sd(2), 
         -datum::inf, y_star_yea(1));
  return(y_star_yea);
}

vec sample_y_star_m_no(vec y_star_no, double mean_m_1, double mean_m_2,
                       vec normal_approx_means, vec normal_approx_sd) {
  
  if (y_star_no(2) < y_star_no(1)) {
    y_star_no(0) = 
      rtn1(normal_approx_means(0) - mean_m_1, normal_approx_sd(0), 
           y_star_no(1), datum::inf);
  } else {
    y_star_no(0) = normal_approx_sd(0) * randn() + 
      normal_approx_means(0) - mean_m_1;
  }
  
  y_star_no(1) = 
    rtn1(normal_approx_means(1), normal_approx_sd(1), 
         -datum::inf, max(y_star_no(0), y_star_no(2)));
  
  if (y_star_no(0) < y_star_no(1)) {
    y_star_no(2) = 
      rtn1(normal_approx_means(2) - mean_m_2, normal_approx_sd(2), 
           y_star_no(1), datum::inf);
  } else {
    y_star_no(2) = normal_approx_sd(2) * randn() + 
      normal_approx_means(2) - mean_m_2;  
  }
  return(y_star_no);
}

vec sample_y_star_m(vec y_star_vec, double vote, double alpha_1, double alpha_2,
                    double leg_pos, double delta_1, double delta_2,
                    vec normal_approx_means, vec normal_approx_sd) {
  
  vec out_vec(3);
  double mean_m_1 = alpha_1 * (leg_pos - delta_1);
  double mean_m_2 = alpha_2 * (leg_pos - delta_2);
  if (vote == 1) {
    out_vec = sample_y_star_m_yea(y_star_vec, mean_m_1, mean_m_2, 
                                  normal_approx_means, normal_approx_sd);
  } else {
    out_vec = sample_y_star_m_no(y_star_vec, mean_m_1, mean_m_2, 
                                 normal_approx_means, normal_approx_sd);
  }
  return(out_vec);
}

// [[Rcpp::export]]
List sample_three_utility_logit_approx(
    mat vote_m, mat all_param_draws, 
    mat y_star_m_1, mat y_star_m_2, mat y_star_m_3, 
    umat label_m_1, umat label_m_2, umat label_m_3,
    int leg_start_ind, int alpha_v_1_start_ind, int alpha_v_2_start_ind, 
    int delta_v_1_start_ind, int delta_v_2_start_ind, 
    double leg_mean, double leg_sd, vec alpha_mean_v, mat alpha_cov_s,
    vec delta_mean_v, mat delta_cov_s, 
    vec normal_approx_means, vec normal_approx_sd, vec normal_approx_prob,
    int num_iter, int start_iter, 
    int keep_iter, int pos_ind, int neg_ind) {
  
  
  vec current_param_val_v = all_param_draws.row(0).t();
  for (int i = 0; i < num_iter; i++) {
    if (i % 100 == 0) {
      Rcout << i << "\n";
    }
    
    for (int j = 0; j < vote_m.n_rows; j++) {
      for (int k = 0; k < vote_m.n_cols; k++) {
        if (!is_finite(vote_m(j, k))) {
          continue;
        }
        
        label_m_1(j, k) = sample_approx_label_v(
          y_star_m_1(j, k), current_param_val_v(alpha_v_1_start_ind + k), 
          current_param_val_v(leg_start_ind + j), 
          current_param_val_v(delta_v_1_start_ind + k),
          normal_approx_means, normal_approx_sd, normal_approx_prob);
        
        label_m_2(j, k) = sample_approx_label_v(
          y_star_m_2(j, k), 0, 0, 0,
          normal_approx_means, normal_approx_sd, normal_approx_prob);
        
        label_m_3(j, k) = sample_approx_label_v(
          y_star_m_3(j, k), current_param_val_v(alpha_v_2_start_ind + k), 
          current_param_val_v(leg_start_ind + j), 
          current_param_val_v(delta_v_2_start_ind + k),
          normal_approx_means, normal_approx_sd, normal_approx_prob);
      }
    }
    
    for (int j = 0; j < vote_m.n_rows; j++) {
      for (int k = 0; k < vote_m.n_cols; k++) {
        if (!is_finite(vote_m(j, k))) {
          continue;
        }
        vec y_star_vec = {y_star_m_1(j, k), 
                          y_star_m_2(j, k), 
                          y_star_m_3(j, k)};
        uvec label_v = {
          label_m_1(j, k),
          label_m_2(j, k),
          label_m_3(j, k)};
        vec out_v = sample_y_star_m(
          y_star_vec, vote_m(j, k), 
          current_param_val_v(alpha_v_1_start_ind + k),
          current_param_val_v(alpha_v_2_start_ind + k),
          current_param_val_v(leg_start_ind + j), 
          current_param_val_v(delta_v_1_start_ind + k), 
          current_param_val_v(delta_v_2_start_ind + k),
          normal_approx_means(label_v), normal_approx_sd(label_v));
        y_star_m_1(j, k) = out_v(0);  
        y_star_m_2(j, k) = out_v(1);
        y_star_m_3(j, k) = out_v(2);
      }
    }
    
    for (unsigned int j = 0; j < vote_m.n_rows; j++) {
      uvec current_ind = {j};
      uvec interested_inds = find_finite(vote_m.row(j).t());
      current_param_val_v(leg_start_ind + j) =
        sample_three_utility_logit_approx_beta(
          y_star_m_1.submat(current_ind, interested_inds),
          y_star_m_3.submat(current_ind, interested_inds),
          current_param_val_v(alpha_v_1_start_ind + interested_inds).t(),
          current_param_val_v(alpha_v_2_start_ind + interested_inds).t(),
          current_param_val_v(delta_v_1_start_ind + interested_inds).t(),
          current_param_val_v(delta_v_2_start_ind + interested_inds).t(),
          label_m_1.submat(current_ind, interested_inds).t(),
          label_m_3.submat(current_ind, interested_inds).t(),
          leg_mean, leg_sd, normal_approx_means, normal_approx_sd);
    }
    
    vec match_var_v(vote_m.n_cols);
    for (unsigned int j = 0; j < vote_m.n_cols; j++) {
      uvec current_ind = {j};
      uvec interested_inds = find_finite(vote_m.col(j));
      vec delta_v = {current_param_val_v(delta_v_1_start_ind + j),
                     current_param_val_v(delta_v_2_start_ind + j)};
      vec out_v =
        sample_three_utility_logit_approx_matched_alpha(
          y_star_m_1.submat(interested_inds, current_ind), 
          y_star_m_3.submat(interested_inds, current_ind),  
          current_param_val_v(leg_start_ind + interested_inds), 
          delta_v, label_m_1.submat(interested_inds, current_ind),
          label_m_3.submat(interested_inds, current_ind),
          alpha_mean_v, alpha_cov_s,
          delta_mean_v, delta_cov_s,
          normal_approx_means, normal_approx_sd); 
      
      current_param_val_v(alpha_v_1_start_ind + j) = out_v(0);
      current_param_val_v(alpha_v_2_start_ind + j) = out_v(1);
      match_var_v(j) = out_v(2);
    }
    
    for (unsigned int j = 0; j < vote_m.n_cols; j++) {
      uvec current_ind = {j};
      uvec interested_inds = find_finite(vote_m.col(j));
      vec alpha_v = {current_param_val_v(alpha_v_1_start_ind + j),
                     current_param_val_v(alpha_v_2_start_ind + j)};
      vec out_v =
        sample_three_utility_logit_approx_matched_delta(
          y_star_m_1.submat(interested_inds, current_ind), 
          y_star_m_3.submat(interested_inds, current_ind),
          alpha_v, current_param_val_v(leg_start_ind + interested_inds), 
          match_var_v(j), label_m_1.submat(interested_inds, current_ind),
          label_m_3.submat(interested_inds, current_ind),
          delta_mean_v, delta_cov_s, normal_approx_means, normal_approx_sd); 
      current_param_val_v(delta_v_1_start_ind + j) = out_v(0);
      current_param_val_v(delta_v_2_start_ind + j) = out_v(1);
    }

    if (i > 0 && ((i + 1) % 5 == 0)) {
      for(unsigned int j = 0; j < vote_m.n_cols; j++) {
        uvec current_ind = {j};
        uvec interested_inds = find_finite(vote_m.col(j));
        vec alpha_v = {current_param_val_v(alpha_v_1_start_ind + j),
                       current_param_val_v(alpha_v_2_start_ind + j)};
        vec delta_v = {current_param_val_v(delta_v_1_start_ind + j),
                       current_param_val_v(delta_v_2_start_ind + j)};
        vec out_v = flip_signs(
            vote_m.submat(interested_inds, current_ind), 
            alpha_v, current_param_val_v(leg_start_ind + interested_inds), 
            delta_v, match_var_v(j), 
            alpha_mean_v, alpha_cov_s, delta_mean_v, delta_cov_s); 
        
        current_param_val_v(alpha_v_1_start_ind + j) = out_v(0);
        current_param_val_v(alpha_v_2_start_ind + j) = out_v(1);
        current_param_val_v(delta_v_1_start_ind + j) = out_v(2);
        current_param_val_v(delta_v_2_start_ind + j) = out_v(3);
      }
    }
    
    if (pos_ind > -1 && (current_param_val_v(leg_start_ind + pos_ind) < 0)) {
      current_param_val_v = -current_param_val_v;
    }
    
    if (neg_ind > -1 && pos_ind < 0 && 
         (current_param_val_v(leg_start_ind + neg_ind) > 0)) {
      current_param_val_v = -current_param_val_v;
    }
    
    int post_burn_i = i - start_iter + 1;
    if (i >= start_iter && (fmod(post_burn_i, keep_iter) == 0)) {
      int keep_iter_ind = post_burn_i / keep_iter - 1;
      all_param_draws.row(keep_iter_ind) = current_param_val_v.t();
    }
  }
  
  return(List::create(Named("param_draws") = all_param_draws, 
                      Named("y_star_m_1") = y_star_m_1, 
                      Named("y_star_m_2") = y_star_m_2, 
                      Named("y_star_m_3") = y_star_m_3,
                      Named("label_m_1") = label_m_1,
                      Named("label_m_2") = label_m_2,
                      Named("label_m_3") = label_m_3));
}

