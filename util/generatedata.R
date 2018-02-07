source('submitscript/util.R')
library('mvnfast')
## Generate  normasl of N(0,I) and anomalies of N(3,I) with 0.1 percent & 8 d 
datapath= "synthetic/"

for(n in 1:20){
	delta = sample(c(2,3,4),1)
    	df = util.generateData(n_normal = 2700, d = 8, delta = delta,d_rel = 3,n_anomaly = 300,corr = F )
        df$class[df$class==1] = "anomaly"
       df$class[df$class==0] = "nominal"
       normal_mixture = paste0(datapath,"synthetic_uncorrelated_delta_",delta,"_",n,"_.csv")
      write.table(df, normal_mixture,sep=",", row.names = F, quote = F)
      ## H2 Add additional 5 irrelevant dimension to make it harder. 
        x = matrix(rep(runif(3000, -1,1),5),ncol=5)
        df = cbind(df, x)
	normal_noise = paste0(datapath,"synthetic_noise_5_uncorr_delta_",delta,"_",n,"_.csv")
	util.writecsv(df,normal_noise)
}

##
d = 2
N = 2700
An = 300
for (n in 1:5) {
  for (r in c(0.4, 0.8, 1.2, 1.6)) {
    dx <-
      data.frame(class = "nominal", rmvn(
        n = N,
        mu = rep(0, d),
        sigma = diag(d) + r - diag(r, d),
        isChol = T
      ))
    mu <-  sample(c(2, 3, 4), 1)
    anom <-
      data.frame(class = "anomaly", rmvn(
        n = An,
        mu = rep(mu, d),
        sigma = diag(d) + r - diag(r, d),
        isChol = T
      ))
    dfn <- rbind(dx, anom)
    correlated_mixture = paste0(
      datapath,
      "synthetic_correlated_d_8_delta_",
      delta,
      "_rho_",
      r,
      "_iter_",
      n,
      ".csv"
    )
    
    util.writecsv(dfn, correlated_mixture)
    
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
                       isChol = T
                     ))
    mixture_df = rbind(gmm, anom)
    mixuture_data = paste0(datapath,
                           "synthetic_mixture_d_8_delta_",
                           delta,
                           "_rho_",
                           r,
                           "iter_",
                           n,
                           ".csv")
    util.writecsv(mixture_df, mixuture_data)
    
  }
  
}

## Mixture data test 
pp <- list()
mixture.data <-
  function(d=8, N=3000, anom.prop=0.1, r=0.4,n){
    
    #d=8
    #r= 0.8
    an= ceiling(N*anom.prop) #300
    
    mu_mt = matrix(rep(c(-3, 0, 3), d), ncol = d, nrow = 3)
    sig_mt = list(diag(d) + r - diag(r, d),
                  diag(d) + r - diag(r, d),
                  diag(d) + r - diag(r, d))
    w = c(1, 1, 1) / 3
    gmm = data.frame(class = "nominal",
                     rmixn(
                       N-an,
                       mu = mu_mt,
                       sigma = sig_mt,
                       w = w,
                       isChol = T
                     ))
    mus = sample(c(-2,4,5,-5,-6,1,-1,0,6),2,F)
    mu_anom = rep(mus, 4)
    anom <-
      data.frame(class = "anomaly", rmvn(
        n = an,
        mu = mu_anom , #rep(c(-2,3),4),#rep(mu, d),
        sigma = diag(d) + r - diag(r, d),
        isChol = T
      ))
    mixture_df = rbind(gmm, anom)
    #pp[[n]] <- ggplot(mixture_df,aes(X1, X2, color=class)) +   geom_density2d() + theme_bw() + geom_point()
    mixuture_data = paste0(datapath,
                           "synthetic_mixture_d_8_delta_",
                            mu_anom[1],"_",mu_anom[2],
                           
                           "_rho_",
                           r,
                           "iter_",
                           n,
                           ".csv")
    util.writecsv(mixture_df, mixuture_data)
  }
for (n in 1:5) {
  for (r in c(0.4, 0.8, 1.2, 1.6)) {
    mixture.data(r=r,n=n)
  }
}


