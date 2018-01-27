## Analysis and plot of the result. 
library('dplyr')
library(gridExtra)
library(tidyr)

plot_miss_value<-function(ld, iff,bifor,miss_ff=0.33, tt){
  
  
  ifor_filtered = iff[iff$miss_feature_prop==miss_ff,]
  loda_filtered = ld[ld$miss_feature_prop== miss_ff,]
  bifor_filtered = bifor[bifor$miss_feature_prop==miss_ff,]
  ylimt = c(min(min(ifor_filtered$mean_impute) ,min(loda_filtered$reduce),min(bifor_filtered$reduce),min(loda_filtered$mean_impute))- 0.01,
            max(max(ifor_filtered$reduce),max(loda_filtered$reduce),max(ifor_filtered$MICE_impute))+0.01)
  
  plot(ifor_filtered$miss_prop,ifor_filtered$mean_impute ,type='l',xlab='Proportion of missing instance',ylab="auc",ylim=ylimt,main=tt)
  arrows(ifor_filtered$miss_prop,ifor_filtered$mean_impute  - ifor_filtered$mean_impute_ci,
         ifor_filtered$miss_prop,ifor_filtered$mean_impute  + ifor_filtered$mean_impute_ci,code=3,length=0.02,angle=90,col="black")
  lines(ifor_filtered$miss_prop,ifor_filtered$reduce,type='l',col='magenta')
  lines(ifor_filtered$miss_prop,ifor_filtered$reduce,type='p',col='magenta',pch=15)
  arrows(ifor_filtered$miss_prop,ifor_filtered$reduce - ifor_filtered$reduce_ci,
         ifor_filtered$miss_prop,ifor_filtered$reduce + ifor_filtered$reduce_ci,code=3,length=0.02,angle=90,col='red')
  
  #lines(ifor_filtered$miss_prop,ifor_filtered$M,type='l',col='red')
  lines(ifor_filtered$miss_prop,ifor_filtered$MICE_impute,type='l',col='orange',pch=16,lty=1)
  lines(ifor_filtered$miss_prop,ifor_filtered$MICE_impute,type='p',col='orange',pch=16)
  arrows(ifor_filtered$miss_prop,ifor_filtered$MICE_impute - ifor_filtered$MICE_impute_ci,
         ifor_filtered$miss_prop,ifor_filtered$MICE_impute + ifor_filtered$MICE_impute_ci,code=3,length=0.02,angle=90,col='orange')
  # Bagging 
  lines(bifor_filtered$miss_prop,bifor_filtered$reduce,type='l',col='green',lty=1)
  lines(bifor_filtered$miss_prop,bifor_filtered$reduce,type='p',col='green',pch=17,lty=1)
  arrows(bifor_filtered$miss_prop,bifor_filtered$reduce - bifor_filtered$reduce_ci,
         bifor_filtered$miss_prop,bifor_filtered$reduce + bifor_filtered$reduce_ci,code=1,length=0.02,angle=90,col='green')
  
  
  # Loda
  lines(loda_filtered$miss_prop,loda_filtered$mean_impute ,type='l',col="blue", lty=3,pch=2)
  lines(loda_filtered$miss_prop,loda_filtered$mean_impute ,type='p',col="blue", lty=3,pch=2)
  arrows(loda_filtered$miss_prop,loda_filtered$mean_impute  - loda_filtered$mean_impute_ci,
         loda_filtered$miss_prop,loda_filtered$mean_impute  + loda_filtered$mean_impute_ci,code=3,length=0.02,angle=90,col="black")
  lines(loda_filtered$miss_prop,loda_filtered$reduce,type='l',col='green',lty=3,pch=3)
  lines(loda_filtered$miss_prop,loda_filtered$reduce,type='p',col='cyan',lty=3,pch=3)
  #lines(ifor_filtered$miss_prop,ifor_filtered$MICE_impute,type='l',col='red')
  arrows(loda_filtered$miss_prop,loda_filtered$reduce - loda_filtered$reduce_ci,
         loda_filtered$miss_prop,loda_filtered$reduce + loda_filtered$reduce_ci,code=3,length=0.02,angle=90,col='green')
  lines(loda_filtered$miss_prop,loda_filtered$MICE_impute,type='l',col='red',lty=3,pch=4)
  lines(loda_filtered$miss_prop,loda_filtered$MICE_impute,type='p',col='red',lty=3,pch=4)
  arrows(loda_filtered$miss_prop,loda_filtered$MICE_impute - loda_filtered$MICE_impute_ci,
         loda_filtered$miss_prop,loda_filtered$MICE_impute + loda_filtered$MICE_impute_ci,code=3,length=0.02,angle=90,col='orange')
  
  
  legend("bottomleft",col=c('black','magenta','orange','green','blue','cyan','red'),
         legend = c("IF-mean","IF-propdist","IF-MICE","IF-bagg","LODA-mean","LODA-bagg","LODA-MICE"),
         lty = c(1,1,1,1,3,3,3),cex=0.5, pch=c(1,15,16,17,2,3,4))
}


groupby_common_keyk<-function(datax, outputdir="summary/"){
  ds = read.csv(datax,header=F)
  if(ncol(ds)==7)
    ds = cbind(ds,1) #Just to handle the last experiment run without including the ensemble column in the results. 
  names(ds)<-c('ix','miss_prop','miss_feature_prop','auc_mean_impute','auc_reduced','auc_MICE_impute','e','algo')
  dx = ds %>% group_by(miss_prop, miss_feature_prop) %>% 
    summarise(mean_impute = mean(auc_mean_impute), mean_impute_ci=util.ciZ(auc_mean_impute),
              reduce = mean(auc_reduced),reduce_ci = util.ciZ(auc_reduced),
              MICE_impute=mean(auc_MICE_impute), MICE_impute_ci = util.ciZ(auc_MICE_impute),
              #mean_ap_miss = mean(ap_cm),mean_ap_miss_ci = util.ciZ(ap_cm),
              len = length(auc_mean_impute))
  dir_path <- dirname(datax)
  dir.create(paste0(dir_path,"/",outputdir),showWarnings = F)
  write.table(dx,paste0(dir_path,"/",outputdir,"/",basename(datax),"_summary.csv"), sep=",", row.names = F, quote=F)
  #print(paste0(dir_path,outputdir,basename(datax),"_summary.csv"))
  return(dx)
}



mainpath = "../../kddexperiment/missingdata/benchmarkResult/"

# Summarize and save it
outputdir="summaryx"
for (ff in list.files(mainpath, pattern="*.csv")){
  print(ff)
  dx = groupby_common_keyk(paste0(mainpath,ff), outputdir=outputdir)
}
#datasets = c("skin", "magic.gamma", "spambase","") #"skin","synthetic"
datasets <- unique(sapply(list.files(paste0(mainpath,"/",outputdir), pattern = "*.csv"), function(x){
  strsplit(x,"_")[[1]][2]
}))
#dataset = datasets[1] #"synthetic"
#pdf("benchmarkresult.pdf", width = 11, height = 8)
#for(dataset in datasets){
#dataset = "skin"
all = data.frame()
gt <- list()

algorithm = c("ifor", "loda", "bifor","egmm")
#par(mfrow=c(2,2))
pdf("lineplot.pdf")

for (dataset in datasets) {
  
  for (algo in algorithm) {
    #rm(algo)
    file_name = list.files(
      path = paste0(mainpath, outputdir),
      pattern = paste0("^", algo, "_", dataset, "*")
    )
    if(length(file_name)<1) next
    full_path = paste0(mainpath, outputdir, "/", file_name)
    if (file.exists(full_path)) {
      print(full_path)
      assign(algo, read.csv(full_path))
      dxx <- gather_result(get(algo), file_name)
      all <- rbind(all, dxx)
      miss_ff = unique(ifor$miss_feature_prop)[2]
     
      #rm(get(algo))
    }
  }
  plot_miss_value(loda,ifor,bifor,miss_ff,dataset)
}

#loda<-read.csv(paste0(mainpath,'summary/loda_',dataset,'.preproc.csv_summary.csv'))
#bifor<-read.csv(paste0(mainpath,'summary/bifor_',dataset,'.preproc.csv_summary.csv'))


### Gather all data from all algorith

gather_result <- function(df, file_name){
  fname_split <- strsplit(file_name,"_")[[1]]
  algo_name = fname_split[1]
  df$dataset = fname_split[2]
  df_pp = df %>% gather(algorithm, auc, -c("dataset","miss_prop","miss_feature_prop","mean_impute_ci","reduce_ci","MICE_impute_ci","len"))
  # Rename factor names from "cond1" and "cond2" to "first" and "second"
  df_pp$algorithm[df_pp$algorithm=="mean_impute"] <- paste0(algo_name,"_mean_impute")
  df_pp$algorithm[df_pp$algorithm=="MICE_impute"] <- paste0(algo_name,"_MICE_impute")
  
  df_pp2  <- df_pp %>% gather(ci_name,ci,-c("dataset","miss_prop","miss_feature_prop","algorithm","auc","len"))
  
  df_pp2$ci_name[df_pp2$ci_name=="mean_impute_ci"] <- paste0(algo_name,"_mean_impute")
  df_pp2$ci_name[df_pp2$ci_name=="MICE_impute_ci"] <- paste0(algo_name,"_MICE_impute")
  if(algo_name=="ifor"){
      df_pp2$algorithm[df_pp$algorithm=="reduce"] <- paste0(algo_name,"_propdist")
      df_pp2$ci_name[df_pp2$ci_name=="reduce_ci"] <- paste0(algo_name,"_propdist")
  }else{
      df_pp2$algorithm[df_pp$algorithm=="reduce"] <- paste0(algo_name,"_bagg")
      df_pp2$ci_name[df_pp2$ci_name=="reduce_ci"] <- paste0(algo_name,"_bagg")
  }
  idx <- which(df_pp2$ci_name!=df_pp2$algorithm) # remove duplicate rows. 
  df_final <- df_pp2[-idx, ] 
  df_final <- df_final %>% subset(select=-c(ci_name))
  df_final$miss_feature_prop <- round(df_final$miss_feature_prop, 2)
  i=1
  df_final$indx=0
  for(missp in unique(df_final$miss_feature_prop)){
    df_final$indx[df_final$miss_feature_prop==missp] <-  i #df_final %>% group_by(miss_feature_prop) %>% mutate(miss_index=1)    
    i=i +1
  }

  return(df_final)
  # Barplot 
}

## Plot operation 
library(ggplot2)
all_filtered <- all[-which(all$algorithm %in% c("bifor_mean_impute","bifor_MICE_impute")),]
gp <- list()
i=1
colrs = c("#999999", "#E6AF00","#56B4E9","#EF0212","#341211","#00EEFF","#12F500", "#FF00EE", "#CC3CEE","#EA0101")
for(missprop in unique(all_filtered$miss_prop)){
  for(indx in unique(all_filtered$indx)){
  gp[[i]]=ggplot(all_filtered[all_filtered$miss_prop==missprop & all_filtered$indx==indx,], 
         aes(x=dataset, y=auc,fill=as.factor(algorithm)),ylim=c(0.6,0.9)) + 
  geom_bar(stat="identity",position=position_dodge()) + 
  geom_errorbar(aes(ymin=auc-ci, ymax=auc+ci), width=.2, position=position_dodge(.9)) + 
  ggtitle(paste0("Missing proportion of ",missprop," Index of ", indx)) + 
    scale_fill_manual(values=colrs)
    theme_bw()
  #print(missprop)
  i = i+1
  }
}
pdf("All_graphs_disp.pdf",width=13, height=6)

for(i in seq(length(gp))){
grid.arrange(gp[[i]])
}
dev.off()
## Plot along the bars
write.table(all,"benchmark_result_aggregated.csv",sep=",", row.names = F, quote=F)
