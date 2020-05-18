library(rstanarm)
library(ggplot2)
library(dplyr)
library(bayesplot)

setwd("C:/Users/andra/Documents/Univerza/1.letnik/MLDS 1/MLDS-homework/HW7")
set.seed(1)

filepath <- 'dataset.csv'
data <- read.table(filepath, sep=",", dec=".", header=T, colClasses=c("factor", "numeric", "numeric"))
summary(data)

# Normalize data
data_norm <- data
data_norm$Angle <- data_norm$Angle / max(data_norm$Angle)
data_norm$Distance <- data_norm$Distance / max(data_norm$Distance)

fit_full <- stan_glm(Made ~ ., data = data_norm, 
                 family = binomial(link = "logit"),
                 iter = 10000, seed = 1,
                 cores = 2, algorithm = "sampling",
                 chains = 10)

samp_idx <- sample(x = 1:nrow(data_norm), size = 50, replace = F)
samp <- data_norm[samp_idx,]

fit_samp <- stan_glm(Made ~ ., data = samp, 
                 family = binomial(link = "logit"),
                 iter = 10000, seed = 1,
                 cores = 2, algorithm = "sampling",
                 chains = 10)

# Reduce number of samples
post_full <- as.data.frame(fit_full)[1:20000,]
post_samp <- as.data.frame(fit_samp)[1:20000,]

# Scatter plot: samples from posterior
plt2 <- mcmc_scatter(post_full, pars = c("Distance", "Angle"), alpha = .1, size = 1) + 
          stat_density2d(color = "red", size = .05) + theme_bw() + 
          ggtitle("Samples from posterior distribution - full dataset")

ggsave("full.pdf", plot = plt2, width = 5, dpi = 600)

plt3 <- mcmc_scatter(post_samp, pars = c("Distance", "Angle"), alpha = .1, size = 1) + 
          stat_density2d(color = "red", size = .05) + theme_bw() +
          ggtitle("Samples from posterior distribution - random subset")

ggsave("subset.pdf", plot = plt3, width = 5, dpi = 600)

# Estimate importance of distance w.r.t. angle
dist_importance <- mean(post_full$Distance < post_full$Angle)
dist_importance_samp <- mean(post_samp$Distance < post_samp$Angle)
dist_importance
dist_importance_samp

# Confidence interval for angle coefficient
angle_int <- posterior_interval(fit_full, pars = "Angle", prob = .95)
angle_int_samp <- posterior_interval(fit_samp, pars = "Angle", prob = .95)
angle_int
angle_int_samp
