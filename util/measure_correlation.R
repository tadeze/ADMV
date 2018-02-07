library('fastmvn')
library(corrplot)
library(MASS)
# Simulate bivariate normal data
mu <- c(0,0)                         # Mean
Sigma <- matrix(c(1, .5, .5, 1), 2)  # Covariance matrix
# > Sigma
# [,1] [,2]
# [1,]  1.0  0.1
# [2,]  0.1  1.0

# Generate sample from N(mu, Sigma)
bivn <-  mvrnorm(5000, mu = mu, Sigma = Sigma )  # from Mass packagehead(bivn)                                      
# Calculate kernel density estimate
bivn.kde <- kde2d(bivn[,1], bivn[,2], n = 50)   # from MASS package
# Contour plot overlayed on heat map image of results
image(bivn.kde)       # from base graphics package
contour(bivn.kde, add = TRUE)     # from base graphics package

d=2
r= 0.8
An=300
mu_mt = matrix(rep(c(-3, 0, 3), d), ncol = d, nrow = 3)
sig_mt = list(diag(d) + r - diag(r, d),
              diag(d) + r - diag(r, d),
              diag(d) + r - diag(r, d))
w = c(1, 1, 1) / 3
gmm = data.frame(class = "nominal",
                 rmixn(
                   N,
                   mu = mu_mt,
                   sigma = sig_mt,
                   w = w,
                   isChol = F
                 ))
mu= 0
#CC =c(c(0,0),c(0,-2),c(0,-6),c(1,1),c(-1,-2),c(1,-2))
nom <-
  data.frame(class = "anomaly", rmvn(
    n = An,
    mu = rep(c(-1,-2),1),#rep(mu, d),
    sigma = diag(d) + r - diag(r, d),
    isChol = T
  ))

gmmx = rbind(gmm, nom)
#@png("2dmixture-with-anom-1d.png")
ggplot(gmmx,aes(X1, X2, color=class)) +   geom_density2d() + theme_bw() + geom_point()
ff = IsolationVolumeForest::IsolationTrees(gmmx[,-1])
sc = IsolationVolumeForest::AnomalyScore(gmmx[,-1],ff)
pROC::roc(gmmx[,1],sc$outF)
ggplot(gmmx,aes(X2, X1)) +   geom_density2d() + theme_bw() + geom_point() +
  geom_point(aes(X2,X1,color=class))

