##############################################################
# util R
# Generate synthetic data
# n_normal: no of normal points
# d: dimension
# d_rel: relevant attribute for isolation
# delta: difficult level (deviation from the normal points)
# n_anomaly: no of anomaly in the data 
################################################################
#libloc='/nfs/guille/tgd/users/zemichet/R/x86_64-redhat-linux-gnu-library/3.2'
libloc=NULL
#.libPaths(libloc)

library('MASS')
library('mvtnorm',lib.loc=libloc)
#suppressMessages(require('IsolationForest',lib.loc=libloc))
suppressMessages(require('pROC',lib.loc=libloc))
suppressMessages(require('osu.common',lib.loc=libloc))

#compute standard error
util.se<-function(df) return((sd(df, na.rm=T)/sqrt(length(df))))
#compute 95% Normal two sided confidence interval.
util.ciZ <-function(df) return (util.se(df)*1.96)  #+- for lower and upper intervals
#return time Id
timeID <- function(){format(Sys.time(), "%Y%m%d%H%m%s") }





util.change_n_toparameter<-function(...,n)
{
	tmp<-list(...)
	param<-list()
	for(elm in tmp)
	{
		q=n%%elm
		n=n%/%elm
		param<-rbind(param,q)
	}
	return(unlist(rbind(param,n)))
}










util.generateData <- function(n_normal,d,d_rel,delta,n_anomaly,
			      corr=FALSE, covxy=0)
  
{
  
  mn_normal <- matrix(0,nrow=1,ncol=d)
  sg_normal <- diag(d)
 # sg <- diag(d)+covxy-diag(covxy,d)
  s_normal <- data.frame(class=0,rmvnorm(n_normal,mean=mn_normal,sigma=sg_normal))
 
  relFeatures=sample(d,d_rel)
  mn_anomaly <- matrix(c(rep(0,d)))
  mn_anomaly[relFeatures]=delta

sg_anomaly=  diag(d)

  if(corr==TRUE)
  {
    #covxy=0.9  #correlation coefficient, which cov(x,y) = rho*var(x)*var(y)
    
    #Modify the correlated relevant features with 0.9 covariance matrix
    sg_anomaly[relFeatures,relFeatures]=  sg_anomaly[relFeatures,relFeatures]+covxy-diag(covxy,length(relFeatures))
  
  #sg_anomaly = sg
  }
  
  s_anomaly <- data.frame(class=1,rmvnorm(n_anomaly,mean=mn_anomaly,sigma=sg_anomaly))
 #  s_anomaly$class <- 1
  dataset <- rbind(s_anomaly,s_normal)
  
 ## compute the density of the point.. and append to the generated dataset. 

  ##dataset$density <- dmvnorm(as.matrix(dataset[,-1]),mean=mn_normal,sigma=sg_normal)
  return(dataset)        
}

#New auc by random sampling  
#Assuming first column as score and second column as density
util.ORauc<-function(dataset)
{

#Mean sampel of p_norm(x_2) < P_norm(x_1)  where x_1 is anom and x_2 is normal 

 auc = mean(sample(dataset[dataset[,1]==1,2],100000L,TRUE) < sample(dataset[dataset[,1]==0,2],100000L,TRUE))
 return(auc)
    
}			      
  
#compute ROC based on pROC
util.roc<-function(evidence,score,level=c(1,0))
{
 library('pROC')
  #AUC result
  rcc <- roc(evidence,score,levels=level,direction=">")
  return(rcc)
}

# precision<-function()
# {
#   #top k precision
#   k <- ceiling(anom * 2)
#   topAnom <- df[order(df$score,decreasing=T),]
#   topK <- length(which(head(topAnom,k)$class>0))/anom
#   kprec <- data.frame(d,anom,n,drel,dlt,topK)
#   
# }
    
#PCA
util.pca<-function(X,alpha=0.95)
{
  #retain dimension above alpha% variance
  pcomp <- prcomp(X)
  vars <-pcomp$sdev^2
  vars <-vars/sum(vars)
  varcumsum<-cumsum(vars)
  k=max(which(varcumsum<=alpha))
  Xred<-pcomp$x[,1:k]
  return(Xred)
}

# Collect the ouput from the experiment
#@inputdir : input dir of the collected files
#@outputdir: output dir for the result

util.collectResult<-function(inputdir,outputdir)
{
	auc<-data.frame()
	for(fl in list.files(inputdir))
	{
	 
	try(ds <-read.csv(paste0(inputdir,fl)),silent=F)   
  	
 	 auc<-rbind(auc,ds)
  
	}

	write.table(auc,file=outputdir,sep=',',row.name=F,quote=F)
}


#custome experiments 

#@input dataset: dataset of the the first feature as ground truth feature.
#return auc score of the transformed data.
#Transform the dimension using pairwise subtraction nearby features
#apply iForest on the transformed data
util.transform<-function(dataset)
{
  trands<-dataset[,1]  #where first column is class
  dta<-dataset[,-1]
  dmn<-ncol(dta) 
  for(i in 1:(dmn/2))
  {
    trands<-cbind(trands,v=(dta[,2*i]-dta[,(2*i-1)]))
  }
  dataset=trands
  
#   #IsolationForest
#   rwSamp=max(256,0.1*nrow(dataset))
#   iff <- IsolationTrees(dataset[,-c(1,2)],ntree=100,hlim=100,rowSamp=T,nRowSamp=rwSamp)
#   sc <- AnomalyScore(dataset[,-c(1,2)],iff)
#   rcc<-roc(dataset[,1],sc$outF,levels=c(0,1))
#   
  return (util.iForest(as.data.frame(dataset)))
  #return(data.frame(filenName=basename(pth),auc=rcc$auc))
}



##Apply pca

#@input: dataset: dataframe where the first feature is class label
#@presv: amount of variance to preserv 1 indicates no reduction
#@return: auc score after applying iForest
#@
util.PCAiForest<-function(dataset,presv=1)
  
{
  
  #fname=paste0(pth,"")
  
  dataset=data.frame(dataset[,1],util.pca(dataset[,-1],presv))#apply pca
 if(is.null(dataset)) return(NULL) 
  return (util.iForest(as.data.frame(dataset)))  


}
#apply iForest and return auc of the result 
#@input: dataset: dataframe where first feature indicates class label and remaining numeric features
#@return: auc score of the result
#use subsampling of 
util.iForest<-function(dataset,minsample=512,tree=100,hlm=100)
{

  #IsolationForest
  ds_size<-nrow(dataset)
  rwSamp=max(minsample,0.1*ds_size)
  rwSamp = ifelse(rwSamp>ds_size,ceiling(ds_size/2),rwSamp)
  iff <- IsolationTrees(dataset[,-1],ntree=tree,rowSamp=T,nRowSamp=rwSamp,hlim=hlm)
  sc <- AnomalyScore(dataset[,-1],iff)
  rcc <- util.roc(dataset[,1],sc$outF) #,levels=c(0,1))
   return (rcc$auc)
}

util.iForest.roc<-function(dataset,minsample=256,tree=100,hlm=ceiling(log2(nrow(dataset))),levels=c(1,0),clean.train=FALSE)
{

  #IsolationForest
 # ds_size<-nrow(dataset)
 # rwSamp=max(minsample,0.1*ds_size)
 # rwSamp = ifelse(rwSamp>ds_size,ceiling(ds_size/2),rwSamp)
  if(clean.train==T){
  clean.ds <- dataset[dataset[,1]==levels[2],]
  iff <- IsolationTrees(clean.ds[,-1],ntree=tree,hlim=hlm,rowSamp=T,nRowSamp=minsample)
  } else{ 
 iff <- IsolationTrees(dataset[,-1],ntree=tree,hlim=hlm,rowSamp=T,nRowSamp=minsample)
}
  sc <- AnomalyScore(dataset[,-1],iff)
  rcc <- util.roc(dataset[,1],sc$outF,levels) #,levels=c(0,1))
  return (rcc$auc)
}


#Read from command line 
util.readcmd <-function()
{
args<-commandArgs(trailingOnly=TRUE)
return (args)
}

#write csv file 
util.writecsv<-function(df,fname)
{
 write.table(df,fname,sep=',',row.name=F,quote=F)
}



#Average precision at k 
apk <- function(k, actual, predicted)
{
  score <- 0.0
  cnt <- 0.0
  
  for (i in 1:min(k,length(predicted)))
  {
    if (predicted[i] %in% actual && !(predicted[i] %in% predicted[0:(i-1)]))
    {
      cnt <- cnt + 1
      score <- score + cnt/i 
    }
  }
  score <- score / min(length(actual), k)
  score
}

#' Compute the mean average precision at k
#'
#' This function computes the mean average precision at k
#' of two lists of sequences.
#'
#' @param k max length of predicted sequence
#' @param actual list of ground truth sets (vectors)
#' @param predicted list of predicted sequences (vectors)
#' @export
util.mapk <- function (k, actual, predicted)
{
  if( length(actual)==0 || length(predicted)==0) 
  {
    return(0.0)
  }
  
  scores <- rep(0, length(actual))
  for (i in 1:length(scores))
  {
    scores[i] <- apk(k, actual[[i]], predicted[[i]])
  }
  score <- mean(scores)
  score
}



if.clean<-function(ds,row_sample,clean.dt=FALSE,sampling=T,class_label =c(0,1))
{ 
  #set.seed(100)
  
  if(clean.dt==TRUE){
    clean.ds <- ds[ds[,1]==class_label[1],]
    iff = IsolationTrees(clean.ds[,-1],nRowSamp = row_sample,rowSamp=sampling,ntree= 100,hlim = 100)
  }else{
    iff = IsolationTrees(ds[,-1],nRowSamp = row_sample,rowSamp =sampling,ntree = 100,hlim=100)
    
  }
  
  sc = AnomalyScore(ds[,-1],iff)
  return(sc$outF)
  
}



trimmed.iforest<-function(ds,row_sample,sampling=T,trimm=FALSE,trimm.percent=20,
                          include_trimmed=TRUE
                          ,class_label=c(0,1))
{
  
  #set.seed(100)
  TRIM_ITERATION <- 10
  iff = IsolationTrees(ds[,-1],nRowSamp = row_sample,rowSamp=sampling,ntree= 100,hlim = 100)
  
  trimmed.ds <- ds
  
  if(trimm==TRUE)
  {
    trimm_no <- 0
    trimmed.index <- c()
    if(include_trimmed==FALSE) #include trimmed index in the iteration. 
    {
      
      while(trimm_no<trimm.percent)
      {
        
        
        sc = AnomalyScore(trimmed.ds[,-1],iff)
        top_k = sort(sc$outF,decreasing = T,index.return=T)$ix[1:ceiling(0.01*length(sc$outF))] #trim 1% at a time 
        # and remvoe the index. 
        trimmed.index = top_k
        trimmed.ds = trimmed.ds[-trimmed.index,]
        
        iff = IsolationTrees(trimmed.ds[,-1],nRowSamp = row_sample,rowSamp =sampling,ntree = 100,hlim=100)
        trimm_no= trimm_no +1
        #cat(" Iteration number ",trimm_no," With trainng data of ",length(sc$outF),"\n")
      }
      
    }else
    {
      iter_count <- 0
      while(iter_count<TRIM_ITERATION){
        sc = AnomalyScore(ds[,-1],iff)
        top_k = sort(sc$outF,decreasing = T,index.return=T)$ix[1:ceiling(trimm.percent*length(sc$outF)/100)]
        trimmed.index = union(trimmed.index,top_k)
        trimmed.ds = ds[-trimmed.index,]
        iff = IsolationTrees(trimmed.ds[,-1],nRowSamp = row_sample,rowSamp =sampling,ntree = 100,hlim=100)
        iter_count = iter_count +1
      }
      
    }
    
    
    
  }
  sc = AnomalyScore(ds[,-1],iff) #Finall score of trimmed train. 
  
  return(sc$outF)
  
  
}




