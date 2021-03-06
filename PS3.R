# PROBLEM SET 3: LOGISTIC REGRESSION ########################
# Exercise 1:

# Load necessary libraries
pacman::p_load(ggplot2, tidyverse, plyr, dplyr, MASS, caret, ggpubr, maxLik)

# Exercise 1

# Set.seed()
seed <- 40
set.seed(seed)

# Parameters
n <- 1000
beta <- c(-2, 0.1, 1)
X1.min <- 18
X1.max <- 60
X2.P1 <- 0.5

# Define Data Gnerating function
data.gen <- function(n, beta, X1.min, X1.max, X2.P1) {
  X0 <- rep(1, n)
  X1 <- sort(runif(n, X1.min, X1.max))
  X2 <- rbinom(n, 1, X2.P1)
  X <- cbind(X0, X1, X2)
  logodds <- X %*% beta
  pi_x <- 1 / (1 + exp(-logodds))
  y <- rbinom(n, 1, prob = pi_x)
  data <- cbind.data.frame(X, logodds, pi_x, y)
  return(data)
}

# a)
# Generate the training data
data.train <- data.gen(n, beta, X1.min, X1.max, X2.P1)
head(data.train)

X.train <- cbind(data.train$X0, data.train$X1, data.train$X2)
head(X.train)

# Generate the test data
data.test <- data.gen(n, beta, X1.min, X1.max, X2.P1)
head(data.test)

X.test <- cbind(data.test$X0, data.test$X1, data.test$X2)
head(X.test)

## Explore the data
# Scatterplot of ys on X1
ggplot(data.train, aes(x = X1, y = y, colour = y)) +
  geom_point()

# ... ys on X2
ggplot(data.train, aes(x = X2, y = y, colour = y)) +
  geom_point()

# Distribution of X1
ggplot(data.train, aes(X1)) +
  geom_histogram()

# Distribution of X2
ggplot(data.train, aes(X2)) +
  geom_histogram()

# b)
# Estimate betas via ML (Logistic Regression)
Logit <- glm(y ~ X1 + X2, data = data.train, family = binomial(link = "logit"))
summary(Logit)
(beta.logit <- Logit$coefficients)

# Compared with manually doing ML

loglike<-function(b) {# the likelihood function for the logit model
  ll <- sum(-data.train$y * log(1 + exp(-(X.train %*% b))) -
              (1 - data.train$y) * log(1 + exp(X.train %*% b)))
  return(ll)
}

# Initialize estimation procedure
estim <- maxBFGS(loglike, finalHessian = TRUE, start = c(0, 0, 1))
beta.ML <- estim$estimate # give out parameter estimates
beta.ML
(beta.logit <- Logit$coefficients)

# Standard Error of the Coefficients
estim.hess <- estim$hessian
# the optimization routine returns the hessian matrix at the last iteration.
Cov <- -(solve(estim.hess))
# the covariance matrix is the (negative) inverse of the hessian matrix.
sde.ML <- sqrt(diag(Cov))#the standard errors are the square root of the diagonal of the inverse Hessian.
sde.ML
stdEr(Logit)

# Prepare some Data for plotting later
# "True" log odds and probs (never observe in real life)
logodds.true.train <- data.train$logodds
probs.true.train <- data.train$pi_x

# "True" log odds and probs for test data
logodds.true.test <- data.test$logodds
probs.true.test <- data.test$pi_x

# Construct 95% CI for the estimated probs

# "Manual" way
beta.ML.lower <- beta.ML - 1.96 * sde.ML
beta.ML.upper <- beta.ML + 1.96 * sde.ML
logodds.ML.train <- X.train %*% beta.ML
logodds.ML.train.lower <- X.train %*% beta.ML.lower
logodds.ML.train.upper <- X.train %*% beta.ML.upper

probs.ML.train <- 1 / (1 + exp(-logodds.ML.train))
probs.ML.train.lower <- 1 / (1 + exp(-logodds.ML.train.lower))
probs.ML.train.upper <- 1 / (1 + exp(-logodds.ML.train.upper))

logodds.ML.test <- X.test %*% beta.ML
logodds.ML.test.lower <- X.test %*% beta.ML.lower
logodds.ML.test.upper <- X.test %*% beta.ML.upper

probs.ML.test <- 1 / (1 + exp(-logodds.ML.test))
probs.ML.test.lower <- 1 / (1 + exp(-logodds.ML.test.lower))
probs.ML.test.upper <- 1 / (1 + exp(-logodds.ML.test.upper))

# Fit the model again to the data.train
logit.train <- predict(Logit, data.train, se = T)
logodds.fit.train <- logit.train$fit
probs.fit.train <- 1 / (1 + exp(-logodds.fit.train))
lower.train <- logodds.fit.train - 1.96 * logit.train$se.fit # lower bound
upper.train <- logodds.fit.train + 1.96 * logit.train$se.fit # upper bound
lower.train <- 1 / (1 + exp(-lower.train))
upper.train <- 1 / (1 + exp(-upper.train))

# Estimated marginal effect of X1
dx1.train <- beta.logit[2] * probs.fit.train * (1 - probs.fit.train)

# Use the estimated betas to fit the test data
logit.test <- predict(Logit, newdata = data.test, se = T)
logodds.fit.test <- logit.test$fit

# Check if it's correct
head(logodds.fit.test)
head(X.test %*% beta.logit)

lower.test <- logodds.fit.test - 1.96 * logit.test$se.fit # lower bound of log odds
upper.test <- logodds.fit.test + 1.96 * logit.test$se.fit # upper bound...
(probs.fit.test <- 1 / (1 + exp(-logodds.fit.test)))
lower.test <- 1 / (1 + exp(-lower.test))
upper.test <- 1 / (1 + exp(-upper.test))

# MSE and AVE
y.pred.train <-  c()
y.pred.test <- c()
threshold <- 0.65

for (i in 1:n) {
  if (probs.fit.train[i] >= threshold) {
    y.pred.train[i] <- 1
  }
  else {
    y.pred.train[i] <- 0
  }
  if (probs.fit.test[i] >= threshold) {
    y.pred.test[i] <- 1
  }
  else {
    y.pred.test[i] <- 0
  }
}
(MSE <- sum(y.pred.train != data.train$y) / length(data.train$y))
(AVE <- sum(y.pred.test != data.test$y) / length(data.test$y))

# For Confusion Matrices we factor the ys
y.train.cfm <- factor(data.train$y, levels = c(0, 1), labels = c("Pos", "Neg"))
y.test.cfm <- factor(data.test$y, levels = c(0, 1), labels = c("Pos", "Neg"))
y.pred.train.cfm <- factor(y.pred.train, levels = c(0, 1), labels = c("Pos", "Neg"))
y.pred.test.cfm <- factor(y.pred.test, levels = c(0, 1), labels = c("Pos", "Neg"))

(cfm.train <- (table(y.pred.train.cfm, y.train.cfm,
               dnn = c("Predicted", "True"))))
(addmargins(cfm.train))
addmargins(prop.table(cfm.train))
addmargins(prop.table(cfm.train, 2))

(cfm.test <- (table(y.pred.test.cfm, y.test.cfm,
               dnn = c("Predicted", "True"))))
(addmargins(cfm.test))
addmargins(prop.table(cfm.test))
addmargins(prop.table(cfm.test, 2))

# c) Interpretation

# Logodss of the first observation and P(Y_1 = 1)
logit.train$fit[1] # "automatic"
X.train[1, ] %*% beta.logit # "manually"
# Odss of the first obs
(odds_1 <- exp(logit.train$fit[1]))
# With an odds of 1.7:1 the first observation is in group 1
# Or in other words, the probability that Y_1 = 1 is:
(pi_x_1 <- 1 / (1 + exp(-logit.train$fit[1]))) # P(Y_1 = 1) is around 64%
                                               # gives an odds of ~ 64/36 ~ 1.7)
# Marginal effect:
# one unit increase in beta_1 increases the odds by e^beta_1 - 1 %
(exp(beta.logit[2]) - 1) * 100
logodds_1_new <- beta.logit[1] * X.train[1, 1] +
beta.logit[2] * (X.train[1, 2] + 1) +
beta.logit[3] * X.train[1, 3]
odds_1_new <- exp(logodds_1_new)
(odds_1_new - odds_1) / odds_1 *100

(dx1_1 <- beta.logit[2] * pi_x_1 * (1 - pi_x_1)) * 100
pi_x_1_new <- 1 / (1 + exp(-logodds_1_new))
(pi_x_1_new - pi_x_1) * 100

####################################################

# d)
plt.data <- data.frame(X1.train = data.train$X1,
                       X1.test = data.test$X1,
                       X2.train = factor(data.train$X2),
                       X2.test = factor(data.test$X2),
                       y.train = data.train$y,
                       y.test = data.test$y,
                       probs.train = probs.fit.train,
                       probs.ML.train,
                       probs.test = probs.fit.test,
                       probs.ML.test,
                       probs.true.train,
                       probs.true.test,
                       upper.train,
                       probs.ML.train.upper,
                       lower.train,
                       probs.ML.train.lower,
                       upper.test,
                       probs.ML.test.upper,
                       lower.test,
                       probs.ML.test.lower,
                       dx1.train)

head(plt.data)
str(plt.data)
ggplot(plt.data, aes(X1.train)) +
  geom_line(aes(y = probs.fit.train, colour = X2.train), cex = 1) +
  geom_ribbon(aes(ymin = lower.train,
                  ymax = upper.train,
                  colour = X2.train),
              alpha = 0.1) +
  geom_point(aes(y = y.train), size = 0.3) +
  geom_point(aes(y = probs.true.train), shape = "x", cex = 0.8)
logit.train.plt <- ggplot(plt.data, aes(X1.train)) +
  geom_line(aes(y = probs.fit.train, colour = X2.train), cex = 1) +
  geom_ribbon(aes(ymin = lower.train,
                  ymax = upper.train,
                  colour = X2.train),
              alpha = 0.1) +
  geom_point(aes(y = y.train), size = 0.3) 
ggplot(plt.data, aes(X1.train)) +
  geom_line(aes(y = probs.ML.train, colour = X2.train), cex = 1) +
  geom_ribbon(aes(ymin = probs.ML.train.lower,
                  ymax = probs.ML.train.upper,
                  colour = X2.train),
              alpha = 0.1) +
  geom_point(aes(y = y.train), size = 0.3) +
  geom_point(aes(y = probs.true.train), shape = "x", cex = 0.8)

ML.train.plt <- ggplot(plt.data, aes(X1.train)) +
  geom_line(aes(y = probs.ML.train, colour = X2.train), cex = 1) +
  geom_ribbon(aes(ymin = probs.ML.train.lower,
                  ymax = probs.ML.train.upper,
                  colour = X2.train),
              alpha = 0.1) +
  geom_point(aes(y = y.train), size = 0.3)

ggplot(plt.data, aes(X1.test)) +
  geom_line(aes(y = probs.fit.test, colour = X2.test), cex = 1) +
  geom_ribbon(aes(ymin = lower.test,
                  ymax = upper.test,
                  colour = X2.test),
              alpha = 0.1) +
  geom_point(aes(y = y.test), size = 0.3) +
  geom_point(aes(y = probs.true.test), shape = "x", cex = 0.8)

logit.test.plt <- ggplot(plt.data, aes(X1.test)) +
  geom_line(aes(y = probs.fit.test, colour = X2.test), cex = 1) +
  geom_ribbon(aes(ymin = lower.test,
                  ymax = upper.test,
                  colour = X2.test),
              alpha = 0.1) +
  geom_point(aes(y = y.test), size = 0.3)

ggplot(plt.data, aes(X1.test)) +
  geom_line(aes(y = probs.ML.test, colour = X2.test), cex = 1) +
  geom_ribbon(aes(ymin = probs.ML.test.lower,
                  ymax = probs.ML.test.upper,
                  colour = X2.test),
              alpha = 0.1) +
  geom_point(aes(y = y.test), size = 0.3) +
  geom_point(aes(y = probs.true.test), shape = "x", cex = 0.8)

ML.test.plt <- ggplot(plt.data, aes(X1.test)) +
  
  geom_line(aes(y = probs.ML.test, colour = X2.test), cex = 1) +
  geom_ribbon(aes(ymin = probs.ML.test.lower,
                  ymax = probs.ML.test.upper,
                  colour = X2.test),
              alpha = 0.1) +
  geom_point(aes(y = y.test), size = 0.3)

ggarrange(logit.train.plt, ML.train.plt)
ggarrange(logit.test.plt, ML.test.plt)
rlang::last_error()
# Plot the marginal effect of X1
ggplot(plt.data, aes(X1.train)) +
  geom_line(aes(y = dx1.train, colour = X2.train), cex = 1)

#############################################################

# What about QDA?
frame.qda <- cbind.data.frame(X0 = data.train$X0, X1 = data.train$X1,
                              X2 = data.train$X2, y = factor(data.train$y, 
                                              levels = c("0", "1"),
                                              labels = c("Pos", "Neg")))
table(frame.qda$y)
head(frame.qda)

ggplot(frame.qda, aes(x = X1, y = X2, colour = y)) + 
  geom_point()

qda <- qda(y ~ X1 + X2, data = frame.qda)
qda

qda.fit <- predict(qda, frame.qda)$class
qda.plt.frame <- cbind.data.frame(frame.qda, qda.fit)
ggplot(qda.plt.frame, aes(x = X1, y = X2, colour = y)) + 
  geom_point(aes(shape = qda.fit))
cfm.qda <- table(qda.fit, frame.qda$y, dnn = c("Predicted", "True"))
cfm.qda
addmargins(cfm.qda)
###############################################################


# Exercise 2
# Parameters
set.seed(seed)
nsim <- 100 # number of simulations
n <- 1000
beta <- c(-2, 0.1, 1)
X1.min <- 18
X1.max <- 60
X2.P1 <- 0.5

# Modified DGP
data.gen <- function(n, beta, X1.min, X1.max, X2.P1) {
  X0 <- rep(1, n)
  X1 <- sort(runif(n, X1.min, X1.max))
  X2 <- rbinom(n, 1, X2.P1)
  X <- cbind(X0, X1, X2)
  logodds <- X %*% beta
  pi_x <- 1 / (1 + exp(-logodds))
  y <- rep(0, n)
  for (i in 1:n) {
    if (pi_x[i] >= min(probs.fit.train)) {
      y[i] <- 1
    }
    else {
      y[i] <- 0
    }
  }
  data <- cbind.data.frame(X, logodds, pi_x, y)
  return(data)
}

# Define a function to get MSE and AVE from a model
geterr <- function(threshold) {
  # Generate data
  data.train <- data.gen(n, beta, X1.min, X1.max, X2.P1)
  data.test <- data.gen(n, beta, X1.min, X1.max, X2.P1) 
  X.train <- cbind(data.train$X0, data.train$X1, data.train$X2)
  X.test <- cbind(data.test$X0, data.test$X1, data.test$X2)
  
  # Get logistic estimates and fitted values
  Logit <- glm(y ~ X1 + X2, data = data.train, family = binomial)
  logit.train <- predict(Logit, data.train, se = T)
  logodds.fit.train <- logit.train$fit
  probs.fit.train <- 1 / (1 + exp(-logodds.fit.train))
  
  logit.test <- predict(Logit, newdata = data.test, se = T)
  logodds.fit.test <- logit.test$fit
  probs.fit.test <- 1 / (1 + exp(-logodds.fit.test))
  
  # MSE and AVE
  y.pred.train <-  c()
  y.pred.test <- c()
  
  for (i in 1:n) {
    if (probs.fit.train[i] >= threshold) {
      y.pred.train[i] <- 1
    }
    else {
      y.pred.train[i] <- 0
    }
    if (probs.fit.test[i] >= threshold) {
      y.pred.test[i] <- 1
    }
    else {
      y.pred.test[i] <- 0
    }
  }
  # Training error
  MSE <- sum(y.pred.train != data.train$y) / length(data.train$y)
  
  # Testing error
  AVE <- sum(y.pred.test != data.test$y) / length(data.test$y)
  errs <- cbind.data.frame(MSE, AVE)
  
  return(errs)
}
geterr(0.8)

# Time for simulations
# Simulate and store the interested values
errlist <- replicate(nsim, geterr(0.5), simplify = F)
MSEs <- c()
AVEs <- c()
for (i in 1:nsim) {
  MSEs[i] <- errlist[[i]]$MSE
  AVEs[i] <- errlist[[i]]$AVE
}
MSEs
AVEs
plt.data.2 <- cbind.data.frame(sim = 1:nsim, err = MSEs, type = "MSEs") %>%
  rbind(data.frame(sim = 1:nsim, err = AVEs, type = "AVEs"))

ggplot(plt.data.2, aes(sim, err, colour = type)) + 
  geom_point()
ggplot(plt.data.2, aes(y = err, colour = type)) + 
  geom_boxplot()
ggplot(plt.data.2, aes(x = err, colour = type)) + 
  geom_density()


# change the lower bound of unif can lead to a more "full" logistic curve
