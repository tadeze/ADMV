## Analysis and plot of the result. 
library('dplyr')



plot_miss_value<-function(ld, iff,miss_ff=0.33, tt){
  
  
  ifor_filtered = iff[iff$miss_feature_prop==miss_ff_num,]
  loda_filtered = ld[ld$miss_feature_prop== miss_ff_num,]
  
  ylimt = c(min(min(ifor_filtered$mean_impute ,loda_filtered$mean_impute ))- 0.01,
            max(max(ifor_filtered$reduce),max(loda_filtered$reduce))+0.01)
  plot(ifor_filtered$miss_prop,ifor_filtered$mean_impute ,type='l',xlab='Proportion of missing features',ylab="auc",ylim=ylimt,main=tt)
  arrows(ifor_filtered$miss_prop,ifor_filtered$mean_impute  - ifor_filtered$impute_ci,
         ifor_filtered$miss_prop,ifor_filtered$mean_impute  + ifor_filtered$impute_ci,code=3,length=0.02,angle=90,col="black")
  lines(ifor_filtered$miss_prop,ifor_filtered$reduce,type='l',col='red')
  #lines(ifor_filtered$miss_prop,ifor_filtered$MICE_impute,type='l',col='red')
  arrows(ifor_filtered$miss_prop,ifor_filtered$reduce - ifor_filtered$reduce_ci,
         ifor_filtered$miss_prop,ifor_filtered$reduce + ifor_filtered$reduce_ci,code=3,length=0.02,angle=90,col='red')
  
  # Loda
  lines(loda_filtered$miss_prop,loda_filtered$mean_impute ,type='l',col="blue")
  arrows(loda_filtered$miss_prop,loda_filtered$mean_impute  - loda_filtered$impute_ci,
         loda_filtered$miss_prop,loda_filtered$mean_impute  + loda_filtered$impute_ci,code=3,length=0.02,angle=90,col="blue")
  lines(loda_filtered$miss_prop,loda_filtered$reduce,type='l',col='green')
  #lines(ifor_filtered$miss_prop,ifor_filtered$MICE_impute,type='l',col='red')
  arrows(loda_filtered$miss_prop,loda_filtered$reduce - loda_filtered$reduce_ci,
         loda_filtered$miss_prop,loda_filtered$reduce + loda_filtered$reduce_ci,code=3,length=0.02,angle=90,col='green')
  legend("topright",col=c('black','red','blue','green'),legend = c("IF","IF-MISS","LODA","LODA-MISS"),lty = c(1,1,1,1),cex=0.5)
}


groupby_common_keyk<-function(datax){
  ds = read.csv(datax,header=F)
  
  names(ds)<-c('ix','miss_prop','miss_feature_prop','auc_mean_impute','auc_reduced','auc_MICE_impute','e','algo')
  dx = ds %>% group_by(miss_prop, miss_feature_prop) %>% 
    summarise(mean_impute = mean(auc_mean_impute), mean_impute_ci=util.ciZ(auc_mean_impute),
              reduce = mean(auc_reduced),reduce_ci = util.ciZ(auc_reduced),
              MICE_impute=mean(auc_MICE_impute), MICE_impute_ci = util.ciZ(auc_MICE_impute),
              #mean_ap_miss = mean(ap_cm),mean_ap_miss_ci = util.ciZ(ap_cm),
              len = length(auc_mean_impute))
  write.table(dx,paste0(datax,"_summary.csv"), sep=",", row.names = F, quote=F)
}