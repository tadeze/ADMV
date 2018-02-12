source('submitscript/util.R')
library('mvnfast')
datapath ="mixturedata/"
norm_vec <- function(x) sqrt(sum(x ^ 2))

mixture.data <-
  function(d=8, N=3000, anom.prop=0.1, r=0.4,b=2,c=0.6,n){
    
    #d=8
    #r= 0.8
    an= ceiling(N*anom.prop) #300
    
    mu_mt = matrix(rep(c(-3, 0, 3), d), ncol = d, nrow = 3)
    mu_mt[2,]= rep(c(3, -3), d/2)
    sig_mt = list(0.1*diag(d) + 0.2*r - diag(r, d),
                  0.1*diag(d) + 0.1*r - diag(r, d),
                  0.1*diag(d) + 0.2*r - diag(r, d))
    w = c(1, 1, 1) / 3
    gmm = data.frame(class = "nominal",
                     rmixn(
                       N-an,
                       mu = mu_mt,
                       sigma = sig_mt,
                       w = w,
                       isChol = T
                     ))
    #mus = sample(c(-2,4,5,-5,-6,1,-1,0,6),2,F)
    #mu_anom = rep(mus, 4)
    
    #mu = 0
    v = as.matrix(rep(c(1, -1), d / 2))
    mu_anom <-  b * v + as.matrix(diag(replicate(9, runif(d,-1, 1))))
    
    anom_dt <- rmvn(
      n = an,
      mu = mu_anom,
      #rep(mu, d),
      sigma = diag(d),
      # + r - diag(r, d),
      isChol = F
    )
    normalized_anom <-  anom_dt #/ apply(anom_dt, 1, norm_vec)
    nom <-
      data.frame(class = "anomaly", normalized_anom)
    
    # 
    # anom <-
    #   data.frame(class = "anomaly", rmvn(
    #     n = an,
    #     mu = mu_anom , #rep(c(-2,3),4),#rep(mu, d),
    #     sigma = diag(d) + r - diag(r, d),
    #     isChol = T
    #   ))
    mixture_df = rbind(gmm, nom)
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
  for (r in c(0.4,0.6, 0.8, 1.2)) {
    mixture.data(r=r,n=n,d=8,c=0.4,b=4)
  }
}
