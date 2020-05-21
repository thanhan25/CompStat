library(Metrics)

set.seed(50)

# Parameters
N <- 1000
beta.vec <- c(5, -0.5)
mux <- 0
sdx <- sqrt(1.5)
mu <- 0
sd <- sqrt(10)

# Define a data generating function
data.gen <- function(N, beta.vec, mux, sdx, mu, sd) {
  X1 <- rep(1, N)
  X2 <- rnorm(N, mux, sdx)
  X <- cbind(X1, X2)
  eps <- rnorm(N, mu, sd)
  y <- X %*% beta.vec + eps
  data <- list(y = y, X = X, constant = X1, X2 = X2)
  return(data)
}

# a) Generate a training sample
data.train <- data.gen(N, beta.vec, mux, sdx, mu, sd)
y.train <- data.train$y
X.train <- data.train$X

# b) Generate a test sample
data.test <- data.gen(N, beta.vec, mux, sdx, mu, sd)
y.test <- data.test$y
X.test <- data.test$X

# c) OLS Estimator
model.train <- lm(y.train ~ X.train - 1, data = data.train)
summary(model.train)
beta.hat = model.train$coefficients 

# d) MSE and prediction error
y.train.fit <- model.train$fitted.values
MSE.train <- mse(y.train, y.train.fit)

y.test.fit <- X.test %*% beta.hat 
MSE.test <- mse(y.test, y.test.fit)

# e)
mse.pol <- function(p, data.train, data.test) {
  X.poly.train <- matrix(NA, N, p + 1)
  X.poly.test <- matrix(NA, N, p + 1)
  Models <- list()
  beta.poly.hat <- list()
  MSE.poly.train <- c()
  MSE.poly.test <- c()
  
  for (i in 1: 5) {
    X.poly.train[, i] <- data.train$X2 ^ (i -1)
    X.poly.test[, i] <- data.test$X2 ^ (i -1)
    Models[[i]] <- lm(y.train ~ X.poly.train[, 1:i] - 1)
    beta.poly.hat[[i]] <- Models[[i]]$coefficients
    MSE.poly.train[i] <- mse(data.train$y, Models[[i]]$fitted.values)
    if (i == 1) {
      y.pol.test.pred <- X.poly.test[, 1:i] * beta.poly.hat[[i]]
    }
    else {
      y.pol.test.pred <- X.poly.test[, 1:i] %*% beta.poly.hat[[i]]
    }
    MSE.poly.test[i] <- mse(data.test$y, y.pol.test.pred)
  }
  return(list(MSE = MSE.poly.train, APE = MSE.poly.test))
}
mse.pol(4, data.train, data.test)


########################################################################
########################################################################

# Exercise 2: Simulation Study

# a) b) Repeat the simulation 1000 times and store the results
sim <- 1000
p <- 4

studies <- replicate(sim, mse.pol(p, 
                                  data.gen(N, beta.vec, mux, sdx, mu, sd), 
                                  data.gen(N, beta.vec, mux, sdx, mu, sd)),
                     simplify = F)
                                  
MSEs <- matrix(nrow = p + 1, ncol = sim)
APEs <- matrix(nrow = p + 1, ncol = sim)

for (i in 1:sim) {
  MSEs[, i] <- studies[[i]]$MSE
  APEs[, i] <- studies[[i]]$APE
}

MSE.sim <- rowMeans(MSEs)
APE.sim <- rowMeans(APEs)

for (i in 1:p+1) {
  plot(MSEs[i, ], type = "p", col = "red")
  lines(APEs[i, ], type = "p", col = "blue")
}

plot(MSE.sim, type = 'p', col = "red", ylim = c(10.35, 10.45))
lines(APE.sim, type = 'p', col = "blue")
