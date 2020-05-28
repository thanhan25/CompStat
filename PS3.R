# PROBLEM SET 3: LOGISTIC REGRESSION ########################
# Exercise 1:

# Load necessary libraries
pacman::p_load(ggplot2, tidyverse, MASS, caret, ggpubr)

# Exercise 1

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

# Generate the training data
data.train <- data.gen(n, beta, X1.min, X1.max, X2.P1) %>%
  rename("X0.train" = "X0",
         "X1.train" = "X1",
         "X2.train" = "X2",
         "y.train" = "y")
head(data.train)

# Generate the test data
data.test <- data.gen(n, beta, X1.min, X1.max, X2.P1) %>%
  rename(#"X0.test" = "X0",
         "X1.test" = "X1",
         "X2.test" = "X2",
         "y.test" = "y")
head(data.test)

# Explore the data
# Scatterplot of ys on X1
ggplot(data.train, aes(x = X1.train, y = y.train, colour = y.train)) + 
  geom_point()

# ... ys on X2
ggplot(data.train, aes(x = X2.train, y = y.train, colour = y.train)) + 
  geom_point()

# Distribution of X1
ggplot(data.train, aes(X1.train)) + 
  geom_histogram()

# Distribution of X2
ggplot(data.train, aes(X2.train)) + 
  geom_histogram() 

# Estimate betas via ML (Logistic Regression)
Logistic <- glm(y.train ~ X1.train + X2.train, data = data.train, family = binomial)
summary(Logistic)

