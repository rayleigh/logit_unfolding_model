//
// This Stan program defines a simple model, with a
// vector of values 'y' modeled as normally distributed
// with mean 'mu' and standard deviation 'sigma'.
//
// Learn more about model development with Stan at:
//
//    http://mc-stan.org/users/interfaces/rstan.html
//    https://github.com/stan-dev/rstan/wiki/RStan-Getting-Started
//

// The input data is a vector 'y' of length 'N'.
data {
  int<lower=0> num_members;
  int<lower=0> num_issues;
  int<lower=0> num_votes;
  array[num_votes] int vote_v;
  array[num_votes] int vote_member_v;
  array[num_votes] int vote_issue_v;
  matrix[2, 2] alpha_cov;
  vector[2] delta_mu;
  matrix[2, 2] delta_cov;
}

// The parameters accepted by the model. Our model
// accepts two parameters 'mu' and 'sigma'.
parameters {
  vector[num_members] beta_v;
  matrix[num_issues, 2] alpha_m;
  matrix[num_issues, 2] delta_m;
  // vector[num_issues] alpha_1;
  // vector[num_issues] alpha_3;
  // vector[num_issues] delta_1;
  // vector[num_issues] delta_3;
}

// The model to be estimated. We model the output
// 'y' to be normally distributed with mean 'mu'
// and standard deviation 'sigma'.
model {
  
  beta_v ~ std_normal();
  for (j in 1:num_issues) {
    if (alpha_m[j, 1] > 0) {
      alpha_m[j, 1] ~ normal(0, alpha_cov[1,1]) T[0,];
      alpha_m[j, 2] ~ normal(0, alpha_cov[2,2]) T[,0];
      delta_m[j] ~ multi_normal(delta_mu, delta_cov);
    } else {
      alpha_m[j, 1] ~ normal(0, alpha_cov[1,1]) T[,0];
      alpha_m[j, 2] ~ normal(0, alpha_cov[2,2]) T[0,];
      delta_m[j] ~ multi_normal(-delta_mu, delta_cov);
    }
  }
  
  for (n in 1:num_votes) {
    real mean_1 = -alpha_m[vote_issue_v[n], 1] * 
      (beta_v[vote_member_v[n]] - delta_m[vote_issue_v[n], 1]);
    real mean_3 = -alpha_m[vote_issue_v[n], 2] * 
      (beta_v[vote_member_v[n]] - delta_m[vote_issue_v[n], 2]);
    target += -log(1 + exp(mean_1) + exp(mean_3));
    if (vote_v[n] == 0) {
      target += log_sum_exp(mean_1, mean_3);
    }
  }
}

