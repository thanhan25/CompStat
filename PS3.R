# PROBLEM SET 3: LOGISTIC REGRESSION ########################
# Exercise 1:

# Load necessary libraries
pacman::p_load(ggplot2, tidyverse, MASS, caret, ggpubr, maxLik)

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
  y <- rep(0, n)
  pi_x <- 1 / (1 + exp(-(X %*% beta)))
  y <- rbinom(n, 1, prob = pi_x)
  data <- cbind.data.frame(X, y)
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

# Explore the data
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

loglike<-function(b) # the likelihood function for the logit model
{
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
sde <- sqrt(diag(Cov))#the standard errors are the square root of the diagonal of the inverse Hessian. 
sde
stdEr(Logit)

# Prepare some Data for plotting later
# "True" log odds and probs (never observe in real life)
logodds.true.train <- X.train %*% beta 
probs.true.train <- 1 / (1 + exp(-logodds.true.train))

# "True" log odds and probs for test data
logodds.true.test <- X.test %*% beta 
probs.true.test <- 1 / (1 + exp(-logodds.true.test))

# Construct 95% CI for the estimated probs
# Fit the model again to the data.train
logit.train <- predict(Logit, data.train, se = T)
logodds.fit.train <- logit.train$fit
probs.fit.train <- 1 / (1 + exp(-logodds.fit.train))
lower.train <- logodds.fit.train - 1.96 * logit.train$se.fit # lower bound
upper.train <- logodds.fit.train + 1.96 * logit.train$se.fit # upper bound
lower.train <- 1 / (1 + exp(-lower.train)) 
upper.train <- 1 / (1 + exp(-upper.train))

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

(cfm.train <- (table(y.pred.train, data.train$y,
               dnn = c("Predicted", "True"))))
(addmargins(cfm.train))
addmargins(prop.table(cfm.train))
addmargins(prop.table(cfm.train, 2))

(cfm.test <- (table(y.pred.test, data.test$y,
               dnn = c("Predicted", "True"))))
(addmargins(cfm.test))
addmargins(prop.table(cfm.test))
addmargins(prop.table(cfm.test, 2))

# c)

# d)
plt.data <- data.frame(X1.train = data.train$X1,
                       X1.test = data.test$X1,
                       X2.train = factor(data.train$X2),
                       X2.test = factor(data.test$X2),
                       y.train = data.train$y,
                       y.test = data.test$y,
                       probs.train = probs.fit.train,
                       probs.test = probs.fit.test,
                       probs.true.train, 
                       probs.true.test,
                       upper.train,
                       lower.train,
                       upper.test,
                       lower.test) 

head(plt.data)  
str(plt.data)

ggplot(plt.data, aes(X1.train)) +
  geom_line(aes(y = probs.fit.train, colour = X2.train), size = 2) + 
  geom_ribbon(aes(ymin = lower.train, 
                  ymax = upper.train, 
                  colour = X2.train), 
              alpha = 0.5) +
  geom_point(aes(y = y.train)) + 
  geom_point(aes(y = probs.true.train), size = 0.1)

ggplot(plt.data, aes(X1.test)) +
  geom_line(aes(y = probs.fit.test, colour = X2.test), size = 2) + 
  geom_ribbon(aes(ymin = lower.test, ymax = upper.test, colour = X2.test), alpha = 0.5) +
  geom_point(aes(y = y.test)) + 
  geom_point(aes(y = probs.true.test), size = 0.1)


###############################################################


# Exercise 2

# MSE and AVE

# change the lower bound of unif can lead to a more "full" logistic curve












