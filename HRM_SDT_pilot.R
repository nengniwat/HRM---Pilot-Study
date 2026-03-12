###############################################################################
##  HRM-SDT ANALYSIS  — v3 (fully revised, all bugs fixed)
##  Study:  Analytic Rubric Validation — Thai EFL Argumentative Writing
##  Input:  simulated_ratings_long.csv  (or replace with real data)
##  Model:  Hierarchical Rater Model with Signal Detection Theory
##            Level 1 (Rater model) : Latent-class SDT  (DeCarlo et al., 2011)
##            Level 2 (Item model)  : Partial Credit Model (PCM)
##  Estimation: Fully Bayesian via MCMC (JAGS + R2jags)
##
##  Rubric criteria (4 items):
##    Item 1 : Organization
##    Item 2 : Supporting Details – Advantages
##    Item 3 : Supporting Details – Disadvantages
##    Item 4 : Language Use
##
##  References:
##    DeCarlo, L. T., Kim, Y. K., & Johnson, M. S. (2011). JEM, 48(3), 333-357.
##    Patz, R. J., Junker, B. W., Johnson, M. S., & Mariano, L. T. (2002). JASA.
###############################################################################


# -----------------------------------------------------------------------------
# 0.  PACKAGE SETUP
# -----------------------------------------------------------------------------
install.packages(c("R2jags", "coda", "ggplot2", "dplyr", "tidyr"),
                 repos = "https://cloud.r-project.org")

library(R2jags)
library(coda)
library(ggplot2)
library(dplyr)
library(tidyr)
install.packages("R2jags", repos = "https://cloud.r-project.org")
library(R2jags)
# Redirect all console output to a log file
sink("HRM_SDT_output_log.txt", split = TRUE)

# -----------------------------------------------------------------------------
# 1.  SIMULATE DATA  (skip this section if loading real data)
#
#     Generates:  simulated_ratings_long.csv
#     Columns:    student_id, item_id, rater_id, score  (scores 1..K)
#
#     To use REAL data: comment out this entire section and set DATA_FILE
#     to your own CSV path in Section 2. The CSV must have these columns:
#       student_id, rater_id, item_id, score
# -----------------------------------------------------------------------------
set.seed(2025)

# Study parameters
N <- 150   # students
J <- 4     # rubric criteria (items)
R <- 3     # raters
K <- 5     # score categories (1..K)

# True parameters
theta_true <- rnorm(N, 0, 1)
delta_true <- c(-0.3, 0.5, 0.6, -0.2)   # sum ~= 0; last absorbed
tau_true   <- c(-1.5, -0.5, 0.5, 1.5)   # K-1 = 4 ordered thresholds

c_true <- matrix(
  c(-1.5, -0.5,  0.5,  1.5,   # Rater 1: neutral
    -2.5, -0.3,  0.3,  2.5,   # Rater 2: lenient (wider spread)
    -1.5, -0.5,  1.2,  2.8),  # Rater 3: severe at high end
  nrow = R, byrow = TRUE
)
d_true <- c(1.5, 1.2, 1.0)   # detection (consistency) parameters

# PCM helper (scalar)
pcm_probs_scalar <- function(theta_i, delta_j, tau, K) {
  lp <- theta_i - delta_j - tau
  cp <- plogis(lp)
  p  <- numeric(K)
  p[1] <- 1 - cp[1]
  for (k in 2:(K - 1)) p[k] <- cp[k - 1] - cp[k]
  p[K] <- cp[K - 1]
  pmax(p, 1e-10)
}

# SDT helper (scalar)
sdt_probs_scalar <- function(xi, c_r, d_r) {
  mu <- (xi - 1) * d_r
  cp <- pnorm(c_r - mu)
  p  <- c(cp[1], diff(cp), 1 - cp[length(cp)])
  pmax(p, 1e-10)
}

# Simulate ratings
sim_rows <- list()
for (i in seq_len(N)) {
  for (j in seq_len(J)) {
    xi_ij <- sample(seq_len(K), 1,
                    prob = pcm_probs_scalar(theta_true[i], delta_true[j],
                                           tau_true, K))
    for (r in seq_len(R)) {
      score <- sample(seq_len(K), 1,
                      prob = sdt_probs_scalar(xi_ij, c_true[r, ], d_true[r]))
      sim_rows[[length(sim_rows) + 1]] <- data.frame(
        student_id = i, item_id = j, rater_id = r, score = score
      )
    }
  }
}

df_sim <- do.call(rbind, sim_rows)
write.csv(df_sim, "simulated_ratings_long.csv", row.names = FALSE)

cat("Simulated data written to: simulated_ratings_long.csv\n")
cat(sprintf("  Rows: %d   Columns: %s\n\n",
            nrow(df_sim), paste(names(df_sim), collapse = ", ")))


# -----------------------------------------------------------------------------
# 2.  LOAD DATA FROM CSV
#
#     Change DATA_FILE to your own path when using real data.
#     Required columns: student_id, rater_id, item_id, score
# -----------------------------------------------------------------------------
DATA_FILE <- "simulated_ratings_long.csv"   # <- change path here for real data

cat("Loading data from:", DATA_FILE, "\n")
df <- read.csv(DATA_FILE)

# Normalize short column names if present (e.g. student -> student_id)
rename_map <- c(student = "student_id", item = "item_id", rater = "rater_id")
for (old in names(rename_map)) {
  if (old %in% names(df) && !rename_map[old] %in% names(df))
    names(df)[names(df) == old] <- rename_map[old]
}

# Verify required columns
required_cols <- c("student_id", "rater_id", "item_id", "score")
missing_cols  <- setdiff(required_cols, names(df))
if (length(missing_cols) > 0) {
  stop(paste("Missing columns in CSV:", paste(missing_cols, collapse = ", ")))
}

# Auto-detect study dimensions
N <- length(unique(df$student_id))
J <- length(unique(df$item_id))
R <- length(unique(df$rater_id))
K <- max(df$score)

cat(sprintf("  Students (N) = %d\n", N))
cat(sprintf("  Criteria (J) = %d\n", J))
cat(sprintf("  Raters   (R) = %d\n", R))
cat(sprintf("  Categories K = %d  (scores 1..%d)\n\n", K, K))

# Re-index IDs to 1..N, 1..J, 1..R
df$student_id <- as.integer(factor(df$student_id))
df$item_id    <- as.integer(factor(df$item_id))
df$rater_id   <- as.integer(factor(df$rater_id))

# Build 3D array X[i, j, r]
X <- array(NA_integer_, dim = c(N, J, R))
for (row in seq_len(nrow(df))) {
  X[df$student_id[row], df$item_id[row], df$rater_id[row]] <- df$score[row]
}

n_missing <- sum(is.na(X))
if (n_missing > 0) {
  cat(sprintf("  Note: %d missing ratings detected (%.1f%%) -- handled as NA in JAGS\n\n",
              n_missing, n_missing / length(X) * 100))
} else {
  cat("  Complete rating design -- no missing data.\n\n")
}

# Item labels
if ("item_name" %in% names(df)) {
  item_labels <- levels(factor(df$item_name))
} else {
  item_labels <- paste0("Item_", seq_len(J))
}


# -----------------------------------------------------------------------------
# 3.  DESCRIPTIVE CHECKS
# -----------------------------------------------------------------------------
cat("====================================================\n")
cat("  DESCRIPTIVE STATISTICS\n")
cat("====================================================\n")

cat("\nRater descriptives:\n")
for (r in seq_len(R)) {
  scores_r <- X[, , r][!is.na(X[, , r])]
  cat(sprintf("  Rater %d : mean = %.3f   SD = %.3f   n = %d\n",
              r, mean(scores_r), sd(scores_r), length(scores_r)))
}

cat("\nInter-rater agreement:\n")
for (r1 in seq_len(R - 1)) {
  for (r2 in (r1 + 1):R) {
    # FIX: vectorise over i and j to avoid 3D indexing issues
    v1   <- as.vector(X[, , r1])
    v2   <- as.vector(X[, , r2])
    both <- !is.na(v1) & !is.na(v2)
    diff_vec <- abs(v1[both] - v2[both])
    cat(sprintf("  Rater %d vs Rater %d :  exact = %.1f%%   within-1 = %.1f%%\n",
                r1, r2,
                mean(diff_vec == 0) * 100,
                mean(diff_vec <= 1) * 100))
  }
}

cat("\nScore frequency (all raters combined):\n")
print(table(c(X), dnn = "score"))

cat("\nMean score by rubric criterion:\n")
for (j in seq_len(J)) {
  cat(sprintf("  Item %d (%s) : %.3f\n",
              j, item_labels[j], mean(X[, j, ], na.rm = TRUE)))
}


# -----------------------------------------------------------------------------
# 4.  JAGS MODEL: HRM-SDT
#
#  Level 2 -- PCM:
#    cumP[i,j,xi] = logistic(theta_i - delta_j - tau_xi)
#    ideal_p[i,j,k] = differenced cumulative probs
#    xi[i,j] ~ dcat(ideal_p[i,j, 1:K])
#
#  Level 1 -- SDT:
#    mu_perc[i,j,r] = (xi[i,j] - 1) * d[r]
#    obs_cum[i,j,r,k] = phi(c[r,k] - mu_perc)
#    obs_p[i,j,r,k]   = differenced obs_cum
#    X[i,j,r] ~ dcat(obs_p[i,j,r, 1:K])
#
#  Identification constraints:
#    theta ~ N(0,1)        -- fixes mean and variance of ability
#    sum(delta) = 0        -- sum-to-zero for item difficulty
#    tau sorted            -- ordered score thresholds
#    d = exp(log_d)        -- positivity for detection parameter
#    c sorted per rater    -- ordered response criteria
# -----------------------------------------------------------------------------
hrm_sdt_jags <- "
model {

  ## PRIORS

  for (i in 1:N) {
    theta[i] ~ dnorm(0, 1)
  }

  for (j in 1:(J-1)) {
    delta[j] ~ dnorm(0, 0.1)
  }
  delta[J] <- -sum(delta[1:(J-1)])

  for (xi in 1:(K-1)) {
    tau_raw[xi] ~ dnorm(0, 0.5)
  }
  tau[1:(K-1)] <- sort(tau_raw[1:(K-1)])

  for (r in 1:R) {
    log_d[r] ~ dnorm(0.3, 1)
    d[r]     <- exp(log_d[r])
  }

  for (r in 1:R) {
    for (k in 1:(K-1)) {
      c_raw[r, k] ~ dnorm(0, 0.5)
    }
    c[r, 1:(K-1)] <- sort(c_raw[r, 1:(K-1)])
  }

  ## LIKELIHOOD

  for (i in 1:N) {
    for (j in 1:J) {

      ## Level 2: PCM ideal rating
      for (xi in 1:(K-1)) {
        cumP[i, j, xi] <- ilogit(theta[i] - delta[j] - tau[xi])
      }
      ideal_p[i, j, 1] <- 1 - cumP[i, j, 1]
      for (xi in 2:(K-1)) {
        ideal_p[i, j, xi] <- cumP[i, j, xi-1] - cumP[i, j, xi]
      }
      ideal_p[i, j, K] <- cumP[i, j, K-1]
      xi[i, j] ~ dcat(ideal_p[i, j, 1:K])

      ## Level 1: SDT observed ratings
      for (r in 1:R) {
        mu_perc[i, j, r] <- (xi[i, j] - 1) * d[r]
        for (k in 1:(K-1)) {
          obs_cum[i, j, r, k] <- phi(c[r, k] - mu_perc[i, j, r])
        }
        obs_p[i, j, r, 1] <- obs_cum[i, j, r, 1]
        for (k in 2:(K-1)) {
          obs_p[i, j, r, k] <- obs_cum[i, j, r, k] - obs_cum[i, j, r, k-1]
        }
        obs_p[i, j, r, K] <- 1 - obs_cum[i, j, r, K-1]
        X[i, j, r] ~ dcat(obs_p[i, j, r, 1:K])
      }
    }
  }
}
"


# -----------------------------------------------------------------------------
# 5.  JAGS DATA, INITS, AND SETTINGS
# -----------------------------------------------------------------------------
jags_data <- list(X = X, N = N, J = J, R = R, K = K)

make_inits <- function(seed_offset = 0) {
  set.seed(100 + seed_offset)
  list(
    theta   = rnorm(N, 0, 0.5),
    delta   = c(rnorm(J - 1, 0, 0.3), NA_real_),   # delta[J] is derived
    tau_raw = sort(rnorm(K - 1, 0, 0.5)),
    log_d   = rnorm(R, 0.3, 0.2),
    c_raw   = matrix(sort(rnorm(R * (K - 1), 0, 0.5)), nrow = R, ncol = K - 1)
  )
}

inits <- lapply(seq_len(3), make_inits)

params_monitor <- c("theta", "delta", "tau", "d", "c", "xi")

# MCMC settings
# PILOT_MODE = TRUE  -> fast (~5 min), for testing
# PILOT_MODE = FALSE -> full (~30-60 min), for final results
PILOT_MODE <- TRUE

if (PILOT_MODE) {
  n_iter   <- 5000;  n_burnin <- 2000;  n_thin <- 3
  cat("\n>>> PILOT mode (5,000 iter). Set PILOT_MODE=FALSE for full analysis.\n\n")
} else {
  n_iter   <- 30000;  n_burnin <- 10000;  n_thin <- 10
  cat("\n>>> FULL analysis mode (30,000 iter).\n\n")
}


# -----------------------------------------------------------------------------
# 6.  RUN JAGS
# -----------------------------------------------------------------------------
model_file <- tempfile(fileext = ".txt")
writeLines(hrm_sdt_jags, model_file)

cat("Fitting HRM-SDT in JAGS...\n")
t0 <- proc.time()

fit <- jags(
  data               = jags_data,
  inits              = inits,
  parameters.to.save = params_monitor,
  model.file         = model_file,
  n.chains           = 3,
  n.iter             = n_iter,
  n.burnin           = n_burnin,
  n.thin             = n_thin,
  progress.bar       = "text"
)

cat(sprintf("Done in %.1f seconds.\n", (proc.time() - t0)["elapsed"]))


# -----------------------------------------------------------------------------
# 7.  CONVERGENCE DIAGNOSTICS
# -----------------------------------------------------------------------------
cat("\n====================================================\n")
cat("  CONVERGENCE DIAGNOSTICS\n")
cat("====================================================\n")

mcmc_fit <- as.mcmc(fit)
rhat_all <- gelman.diag(mcmc_fit, multivariate = FALSE)$psrf[, "Point est."]

key_pnames <- c(
  paste0("delta[", seq_len(J), "]"),
  paste0("tau[",   seq_len(K - 1), "]"),
  paste0("d[",     seq_len(R), "]"),
  paste0("c[", rep(seq_len(R), each = K - 1), ",",
               rep(seq_len(K - 1), R), "]")
)
key_pnames <- key_pnames[key_pnames %in% names(rhat_all)]

cat("\nR-hat for key parameters (target: all < 1.1):\n")
print(round(rhat_all[key_pnames], 3))

pct_conv <- mean(rhat_all < 1.1, na.rm = TRUE) * 100
cat(sprintf("\nConverged: %.1f%% of all parameters\n", pct_conv))
if (pct_conv >= 95) cat(">>> ACCEPTABLE\n") else cat(">>> WARNING: increase n_iter\n")

cat("\nEffective sample sizes (target: > 100):\n")
print(round(effectiveSize(mcmc_fit)[key_pnames], 0))


# -----------------------------------------------------------------------------
# 8.  RESULTS SUMMARY
# -----------------------------------------------------------------------------
cat("\n====================================================\n")
cat("  PARAMETER ESTIMATES\n")
cat("====================================================\n")

summ <- fit$BUGSoutput$summary

## Item difficulty
cat("\n[Item Difficulty -- delta]\n")
cat(sprintf("  %-26s  %7s  %7s  %7s\n", "Criterion", "Est", "2.5%", "97.5%"))
for (j in seq_len(J)) {
  pn <- paste0("delta[", j, "]")
  if (pn %in% rownames(summ))
    cat(sprintf("  %-26s  %7.3f  %7.3f  %7.3f\n",
                item_labels[j], summ[pn,"mean"], summ[pn,"2.5%"], summ[pn,"97.5%"]))
}

## Thresholds
cat("\n[Score Thresholds -- tau]\n")
cat(sprintf("  %-10s  %7s  %7s  %7s\n", "Threshold", "Est", "2.5%", "97.5%"))
for (xi in seq_len(K - 1)) {
  pn <- paste0("tau[", xi, "]")
  if (pn %in% rownames(summ))
    cat(sprintf("  tau_%-6d  %7.3f  %7.3f  %7.3f\n",
                xi, summ[pn,"mean"], summ[pn,"2.5%"], summ[pn,"97.5%"]))
}

## Rater detection
cat("\n[Rater Detection -- d (consistency/precision)]\n")
cat(sprintf("  %-10s  %7s  %7s  %7s  %-22s\n",
            "Rater", "Est", "2.5%", "97.5%", "Interpretation"))
for (r in seq_len(R)) {
  pn <- paste0("d[", r, "]")
  if (pn %in% rownames(summ)) {
    d_e    <- summ[pn, "mean"]
    interp <- ifelse(d_e > 1.5, "High consistency",
               ifelse(d_e > 1.0, "Moderate consistency", "Low consistency"))
    cat(sprintf("  Rater %-5d  %7.3f  %7.3f  %7.3f  %-22s\n",
                r, d_e, summ[pn,"2.5%"], summ[pn,"97.5%"], interp))
  }
}

## Rater criteria
cat("\n[Rater Response Criteria -- c_rk]\n")
for (r in seq_len(R)) {
  d_e <- summ[paste0("d[", r, "]"), "mean"]
  cat(sprintf("  Rater %d  (d = %.3f):\n", r, d_e))
  for (k in seq_len(K - 1)) {
    pn <- paste0("c[", r, ",", k, "]")
    if (pn %in% rownames(summ))
      cat(sprintf("    c_%d :  %.3f  [%.3f, %.3f]\n",
                  k, summ[pn,"mean"], summ[pn,"2.5%"], summ[pn,"97.5%"]))
  }
}


# -----------------------------------------------------------------------------
# 9.  RELATIVE CRITERIA (rc) -- RATER BIAS INDEX
# -----------------------------------------------------------------------------
cat("\n====================================================\n")
cat("  RELATIVE CRITERIA (rc = c / [d x (K-1)])\n")
cat("====================================================\n")

rc_df <- data.frame(rater = integer(), cut   = integer(),
                    c_est = numeric(), d_est = numeric(),
                    rc    = numeric(), opt   = numeric(),
                    dev   = numeric())

for (r in seq_len(R)) {
  d_e <- summ[paste0("d[", r, "]"), "mean"]
  cat(sprintf("  Rater %d (d = %.3f):\n", r, d_e))
  for (k in seq_len(K - 1)) {
    pn  <- paste0("c[", r, ",", k, "]")
    c_e <- summ[pn, "mean"]
    rc  <- c_e / (d_e * (K - 1))
    opt <- (2 * k - (K - 1)) / (2 * (K - 1))
    flag <- ifelse(abs(rc - opt) > 0.15, " *", "")
    cat(sprintf("    Cut %d: rc = %.3f  (optimal = %.3f)%s\n", k, rc, opt, flag))
    rc_df <- rbind(rc_df, data.frame(rater = r, cut = k, c_est = c_e,
                                     d_est = d_e, rc = rc, opt = opt,
                                     dev = rc - opt))
  }
}
cat("\n  * deviation > 0.15 from optimal = potential rater bias\n")


# -----------------------------------------------------------------------------
# 10. STUDENT ABILITY ESTIMATES
# -----------------------------------------------------------------------------
cat("\n====================================================\n")
cat("  STUDENT ABILITY ESTIMATES (theta)\n")
cat("====================================================\n")

theta_est <- summ[paste0("theta[", seq_len(N), "]"), "mean"]
theta_se  <- summ[paste0("theta[", seq_len(N), "]"), "sd"]

cat(sprintf("  Mean theta = %.3f  (SD = %.3f)\n", mean(theta_est), sd(theta_est)))
cat(sprintf("  Range: %.3f  to  %.3f\n", min(theta_est), max(theta_est)))

theta_df <- data.frame(
  student_id  = seq_len(N),
  theta_mean  = theta_est,
  theta_sd    = theta_se,
  theta_lower = summ[paste0("theta[", seq_len(N), "]"), "2.5%"],
  theta_upper = summ[paste0("theta[", seq_len(N), "]"), "97.5%"]
)

write.csv(theta_df, "ability_estimates.csv", row.names = FALSE)
cat("\n  Ability estimates saved to: ability_estimates.csv\n")


# -----------------------------------------------------------------------------
# 11. PLOTS
# -----------------------------------------------------------------------------
cat("\nGenerating plots...\n")

## Plot 1: Student ability distribution
p_ability <- ggplot(theta_df, aes(x = theta_mean)) +
  geom_histogram(bins = 25, fill = "#2C3E6B", color = "white", linewidth = 0.3) +
  geom_vline(xintercept = mean(theta_est), color = "#E8A838",
             linewidth = 1, linetype = "dashed") +
  labs(title    = "Distribution of Estimated Student Ability (theta)",
       subtitle = sprintf("N = %d   Mean = %.3f   SD = %.3f",
                          N, mean(theta_est), sd(theta_est)),
       x = "Estimated theta (posterior mean)", y = "Count") +
  theme_bw(base_size = 12)
ggsave("plot_01_ability_distribution.png", p_ability, width = 6, height = 4, dpi = 150)

## Plot 2: Caterpillar plot (bottom & top students)
theta_sorted <- theta_df %>% arrange(theta_mean)
n_show       <- min(30, N)
theta_show   <- bind_rows(
  head(theta_sorted, n_show %/% 2),
  tail(theta_sorted, n_show %/% 2)
)

p_caterpillar <- ggplot(theta_show,
                        aes(x = reorder(factor(student_id), theta_mean),
                            y = theta_mean)) +
  geom_errorbar(aes(ymin = theta_lower, ymax = theta_upper),
                color = "grey70", width = 0, linewidth = 0.5) +
  geom_point(color = "#2C3E6B", size = 2) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey50") +
  coord_flip() +
  labs(title    = "Ability Estimates -- Bottom & Top Students",
       subtitle = "Points = posterior mean; bars = 95% credible interval",
       x = "Student ID", y = "Estimated theta") +
  theme_bw(base_size = 11)
ggsave("plot_02_ability_caterpillar.png", p_caterpillar, width = 6, height = 6, dpi = 150)

## Plot 3: Score distribution by rater
score_df <- df %>%
  mutate(rater_lab = paste0("Rater ", rater_id))

p_scores <- ggplot(score_df, aes(x = factor(score), fill = rater_lab)) +
  geom_bar(color = "white", linewidth = 0.3) +
  facet_wrap(~ rater_lab) +
  scale_fill_manual(values = c("#2C3E6B", "#5BA89E", "#E8A838")) +
  labs(title    = "Score Frequency Distribution by Rater",
       subtitle = "Central tendency = under-use of categories 1 and 5",
       x = "Score Category", y = "Frequency") +
  theme_bw(base_size = 12) +
  theme(legend.position = "none")
ggsave("plot_03_score_distributions.png", p_scores, width = 8, height = 4, dpi = 150)

## Plot 4: Rater criteria profiles
c_plot_df <- do.call(rbind, lapply(seq_len(R), function(r) {
  do.call(rbind, lapply(seq_len(K - 1), function(k) {
    pn <- paste0("c[", r, ",", k, "]")
    data.frame(rater = paste0("Rater ", r), cut = k,
               c_est = summ[pn, "mean"],
               c_lo  = summ[pn, "2.5%"],
               c_hi  = summ[pn, "97.5%"])
  }))
}))

p_criteria <- ggplot(c_plot_df, aes(x = cut, color = rater, fill = rater)) +
  geom_ribbon(aes(ymin = c_lo, ymax = c_hi), alpha = 0.18, color = NA) +
  geom_line(aes(y = c_est), linewidth = 1.1) +
  geom_point(aes(y = c_est), size = 2.8) +
  scale_x_continuous(breaks = seq_len(K - 1),
                     labels = paste0("Cut ", seq_len(K - 1))) +
  scale_color_manual(values = c("#2C3E6B", "#5BA89E", "#E8A838")) +
  scale_fill_manual(values  = c("#2C3E6B", "#5BA89E", "#E8A838")) +
  labs(title    = "Rater Response Criteria Profiles (c_rk)",
       subtitle = "Shaded band = 95% credible interval",
       x = "Response Cut", y = "Criteria value (logit)",
       color = NULL, fill = NULL) +
  theme_bw(base_size = 12) +
  theme(legend.position = "bottom")
ggsave("plot_04_rater_criteria.png", p_criteria, width = 7, height = 5, dpi = 150)

## Plot 5: Relative criteria deviation
p_rc <- ggplot(rc_df, aes(x = factor(cut), y = dev,
                           fill = paste0("Rater ", rater))) +
  geom_col(position = "dodge", color = "white", linewidth = 0.3) +
  geom_hline(yintercept = c(-0.15, 0.15), linetype = "dashed",
             color = "red", linewidth = 0.6) +
  scale_fill_manual(values = c("#2C3E6B", "#5BA89E", "#E8A838")) +
  labs(title    = "Relative Criteria Deviation from Optimal (rc - optimal)",
       subtitle = "Red dashed lines = +/-0.15 threshold for potential rater bias",
       x = "Response Cut", y = "Deviation", fill = NULL) +
  theme_bw(base_size = 12) +
  theme(legend.position = "bottom")
ggsave("plot_05_rc_deviation.png", p_rc, width = 7, height = 4.5, dpi = 150)

cat("Plots saved.\n")


# -----------------------------------------------------------------------------
# 12. SAVE ALL OUTPUTS
# -----------------------------------------------------------------------------
save(fit, jags_data, summ, df, X, theta_df, rc_df,
     file = "HRM_SDT_results.RData")

write.csv(
  data.frame(parameter = rownames(summ), summ),
  file      = "HRM_SDT_posterior_summary.csv",
  row.names = FALSE
)

cat("\n====================================================\n")
cat("  ALL DONE -- OUTPUT FILES\n")
cat("====================================================\n")
cat("  HRM_SDT_results.RData\n")
cat("  HRM_SDT_posterior_summary.csv\n")
cat("  ability_estimates.csv\n")
cat("  plot_01_ability_distribution.png\n")
cat("  plot_02_ability_caterpillar.png\n")
cat("  plot_03_score_distributions.png\n")
cat("  plot_04_rater_criteria.png\n")
cat("  plot_05_rc_deviation.png\n")
cat("\n")
cat("  Next steps:\n")
cat("  1. Check R-hat < 1.1 for all key parameters\n")
cat("  2. Run traceplot(as.mcmc(fit)) to inspect chain mixing\n")
cat("  3. Set PILOT_MODE = FALSE for full analysis\n")
cat("  4. Replace DATA_FILE path for real data\n")


# 1. Parameter estimates
print(round(summ[c(
  paste0("delta[", 1:J, "]"),
  paste0("tau[", 1:(K-1), "]"),
  paste0("d[", 1:R, "]"),
  paste0("c[", rep(1:R, each=K-1), ",", rep(1:(K-1), R), "]")
), c("mean","sd","2.5%","97.5%","Rhat","n.eff")], 3))

# 2. Convergence summary
cat(sprintf("Converged: %.1f%% of parameters (R-hat < 1.1)\n", pct_conv))

# 3. Relative criteria table
print(rc_df)

# 4. Ability summary
cat(sprintf("Mean theta = %.3f  SD = %.3f  Range = %.3f to %.3f\n",
            mean(theta_est), sd(theta_est), min(theta_est), max(theta_est)))

# Stop redirecting output
sink()
cat("Log saved to: HRM_SDT_output_log.txt\n")

# Save key results to a readable text file
out <- file("HRM_SDT_results_summary.txt", open = "wt")

sink(out)

cat("====================================================\n")
cat("  HRM-SDT RESULTS SUMMARY\n")
cat("====================================================\n\n")

# Convergence
cat(sprintf("Convergence: %.1f%% of parameters R-hat < 1.1\n\n", pct_conv))

# Key parameter table
cat("KEY PARAMETER ESTIMATES\n")
key_rows <- c(
  paste0("delta[", 1:J, "]"),
  paste0("tau[",   1:(K-1), "]"),
  paste0("d[",     1:R, "]"),
  paste0("c[", rep(1:R, each=K-1), ",", rep(1:(K-1), R), "]")
)
key_rows <- key_rows[key_rows %in% rownames(summ)]
print(round(summ[key_rows, c("mean","sd","2.5%","97.5%","Rhat","n.eff")], 3))

# Relative criteria
cat("\nRELATIVE CRITERIA (rc)\n")
print(rc_df)

# Ability
cat(sprintf("\nSTUDENT ABILITY\n"))
cat(sprintf("  Mean = %.3f   SD = %.3f   Range = %.3f to %.3f\n",
            mean(theta_est), sd(theta_est), min(theta_est), max(theta_est)))

sink()
close(out)
cat("Saved to: HRM_SDT_results_summary.txt\n")