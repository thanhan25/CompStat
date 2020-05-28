# PROBLEM SET 3: LOGISTIC REGRESSION ########################
# Exercise 1:

# Load necessary libraries
pacman::p_load(ggplot2, tidyverse, MASS, caret, ggpubr)

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
  X1 <- runif(n, X1.min, X1.max)
  X2 <- rbinom(n, 1, X2.P1)
  X <- cbind(X0, X1, X2)
  y <- rep(0, n)
  y.pi <- 1 / (1 + exp(-(X %*% beta)))
  for (i in 1:n) {
    y[i] <- rbinom(1, 1, y.pi[i])
  }
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
Logit <- glm(y ~ X1 + X2, data = data.train, family = binomial)
summary(Logit)
(beta.logit <- Logit$coefficients)

# Calculate the estimated log-odds and probabilities
(logits.train <- Logit$fitted.values) # log odds
(probs.logit.train <- 1 / (1 + exp(-logits.train))) # Pr(Y=1|X)

# "True" log odds and probs (never observe in real life)
true.logits.train <- X.train %*% beta 
true.probs.train <- 1 / (1 + exp(-true.logits.train))

logits.test <- predict(Logit, newdata = data.test, se = T)

# Check if it's correct
head(logits.test$fit)
head(X.test %*% beta.logit)

# Construct 95% CI for the estimates
lower <- logits.test$fit - 1.96 * logits.test$se.fit # lower bound of log odds
upper <- logits.test$fit + 1.96 * logits.test$se.fit # upper bound...
(probs.logit.test <- 1 / (1 + exp(-logits.test$fit)))
lower <- 1 / (1 + exp(-lower)) 
upper <- 1 / (1 + exp(-upper))

# "True" log odds and probs for test data
true.logits.test <- X.test %*% beta 
true.probs.test <- 1 / (1 + exp(-true.logits.test))

# MSE and AVE
y.pred.train <-  c()
y.pred.test <- c()
for (i in 1:n) {
  if (probs.logit.train[i] >= 0.68) {
    y.pred.train[i] <- 1
  }
  else {
    y.pred.train[i] <- 0
  }
  if (probs.logit.test[i] >= 0.6) {
    y.pred.test[i] <- 1
  }
  else {
    y.pred.test[i] <- 0
  }
}
(err.train <- sum(y.pred.train != data.train$y) / length(data.train$y))
(cfm <- table(y.pred.train, data.train$y))
(table(y.pred.test, data.test$y))
(prop.table(cfm))
# c)

# d)
plt.data <- data.frame(X1.train = data.train$X1,
                       X1.test = data.test$X1,
                       X2.train = factor(data.train$X2),
                       X2.test = factor(data.test$X2),
                       y.train = data.train$y,
                       y.test = data.test$y,
                       probs.train = probs.logit.train,
                       probs.test = probs.logit.test,
                       true.probs.train, 
                       true.probs.test,
                       upper,
                       lower) 

head(plt.data)  
str(plt.data)

ggplot(plt.data, aes(X1.train, colour = X2.train)) +
  stat_smooth(aes(x = X1.train, y = y.train), 
              method = "glm", 
              method.args = list(family = "binomial"), 
              se = T,
              lwd = 2) +
  geom_point(aes(y = y.train)) + 
  geom_point(aes(y = true.probs.train), size = 0.5)

ggplot(plt.data, aes(X1.test)) +
  geom_line(aes(y = probs.logit.test, colour = X2.test), size = 2) + 
  geom_ribbon(aes(ymin = lower, ymax = upper, colour = X2.test), alpha = 0.5) +
  geom_point(aes(y = y.test)) + 
  geom_point(aes(y = true.probs.test), size = 0.1)


###############################################################


# Exercise 2

# MSE and AVE

# change the lower bound of unif can lead to a more "full" logistic curve












