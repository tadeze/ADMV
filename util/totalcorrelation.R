## Compute total correlation and modality of the data. 
## C(x_1, X_2, ,) = \sum H(X_i) - H(x_1, ,,x_n) 
## H = 1/2 ln det 2pie\sigma 


entropy <- 
  function(sigma){
    entr = 0.5*log(2*pi*exp(1)*det(sigma))  
    return(entr)
  }
  

entropy.x <-
  function(sigma){
    return(0.5*log(2*pi*exp(1)*sigma*sigma))
  }



## Given a dataset 
dataset <- "../../kddexperiment/group2/abalone_benchmark_0556.csv"
dx = read.csv(dataset)