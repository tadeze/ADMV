source('submitscript/util.R')
library('mvnfast')
datapath ="correlateddata/"

sample_mean <-
       function(d=8, lw = -3, up = 3){
       A <- rep(lw,d)
       B <- rep(up,d)
       alpha <- runif(1,0,1)
       return (A*alpha + B*(1-alpha))
  }       

generate.correlated <- 
  function(d=8, b=1.2,c=0.8, n=1,r){
	N = 2700
	an = 300
	dx <-
	  data.frame(class = "nominal", rmvn(
	    n = N,
	    mu = sample_mean(d=d), #diag(replicate(9, runif(d, -3,3))),
	    sigma = c*diag(d) + r - c*diag(r, d),
	    isChol = T
	  ))

	v = as.matrix(rep(c(1,-1),d/2))
	mu_anom <-  b*v + sample_mean(d=d) #as.matrix(diag(replicate(9, runif(d, -3,3))))
	anom <-
	  data.frame(class = "anomaly", rmvn(
	    n = an,
	    mu = mu_anom,
	    sigma = c*diag(d) + r -c*diag(r,d),
	    isChol = T
	  ))
	dfn <- rbind(dx, anom)

	correlated = paste0(datapath,
			       "synthetic_correlated_d_8_delta_",
			       r,"_",b,
			       "iter_",
			       n,
			       ".csv")
	util.writecsv(dfn,correlated)
}


for(i in 1:5){
	for(r in c(0.4, 0.6,0.8,1.2))
	generate.correlated(n=i,b=2, c=0.3,d=8,r)
}

