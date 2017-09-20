library('dplyr')
source('../experiments/util.R')

data="../experiments/summaryold/if_particle.csv"
library(ggplot2)
library(gridExtra)

plot_prop<-function(ld,iff,miss_ff_num=0.2,tt){
  
  
  df = iff[iff$num_miss_features==miss_ff_num,]
  dfl = ld[ld$num_miss_features== miss_ff_num,]
  
  ylimt = c(min(min(df$mean_auc,dfl$mean_auc))-0.01,max(max(df$mean_auc_miss),max(dfl$mean_auc_miss))+0.01)
  plot(df$anom_prop,df$mean_auc,type='l',xlab='Proportion of missing features',ylab="auc",ylim=ylimt,main=tt)
  arrows(df$anom_prop,df$mean_auc - df$auc_ci,
         df$anom_prop,df$mean_auc + df$auc_ci,code=3,length=0.02,angle=90,col="black")
  lines(df$anom_prop,df$mean_auc_miss,type='l',col='red')
  arrows(df$anom_prop,df$mean_auc_miss - df$mean_auc_miss_ci,
         df$anom_prop,df$mean_auc_miss + df$mean_auc_miss_ci,code=3,length=0.02,angle=90,col='red')
  
  # Loda
  lines(dfl$anom_prop,dfl$mean_auc,type='l',col="blue")
  arrows(dfl$anom_prop,dfl$mean_auc - dfl$auc_ci,
         dfl$anom_prop,dfl$mean_auc + dfl$auc_ci,code=3,length=0.02,angle=90,col="blue")
  lines(dfl$anom_prop,dfl$mean_auc_miss,type='l',col='green')
  arrows(dfl$anom_prop,dfl$mean_auc_miss - dfl$mean_auc_miss_ci,
         dfl$anom_prop,dfl$mean_auc_miss + dfl$mean_auc_miss_ci,code=3,length=0.02,angle=90,col='green')
  legend("topright",col=c('black','red','blue','green'),legend = c("IF","IF-MISS","LODA","LODA-MISS"),lty = c(1,1,1,1),cex=0.5)
}


plotconf<-function(ld,iff,anom_prop=0.1,tt){
  
  
  df = iff[iff$anom_prop==anom_prop,]
  dfl = ld[ld$anom_prop==anom_prop,]
  
  ylimt = c(min(min(df$mean_auc,dfl$mean_auc))-0.01,max(max(df$mean_auc_miss),max(dfl$mean_auc_miss))+0.01)
  plot(df$num_miss_features,df$mean_auc,type='l',xlab='Proportion of missing features',ylab="auc",ylim=ylimt,main=tt)
  arrows(df$num_miss_features,df$mean_auc - df$auc_ci,
         df$num_miss_features,df$mean_auc + df$auc_ci,code=3,length=0.02,angle=90,col="black")
  lines(df$num_miss_features,df$mean_auc_miss,type='l',col='red')
  arrows(df$num_miss_features,df$mean_auc_miss - df$mean_auc_miss_ci,
         df$num_miss_features,df$mean_auc_miss + df$mean_auc_miss_ci,code=3,length=0.02,angle=90,col='red')
  
  # Loda
    lines(dfl$num_miss_features,dfl$mean_auc,type='l',col="blue")
  arrows(dfl$num_miss_features,dfl$mean_auc - dfl$auc_ci,
         dfl$num_miss_features,dfl$mean_auc + dfl$auc_ci,code=3,length=0.02,angle=90,col="blue")
  lines(dfl$num_miss_features,dfl$mean_auc_miss,type='l',col='green')
  arrows(dfl$num_miss_features,dfl$mean_auc_miss - dfl$mean_auc_miss_ci,
         dfl$num_miss_features,dfl$mean_auc_miss + dfl$mean_auc_miss_ci,code=3,length=0.02,angle=90,col='green')
  legend("topright",col=c('black','red','blue','green'),legend = c("IF","IF-MISS","LODA","LODA-MISS"),lty = c(1,1,1,1),cex=0.5)
}


ci_style_width=0.01
plot_error <-function(dx,anom_prop=0.1){
  df = dx[dx$anom_prop==anom_prop,]
  p_auc=qplot(x=num_miss_features,y=mean_auc,data=df) +
    geom_line(aes(color="red"))+ 
    geom_errorbar(aes(x=num_miss_features,ymin=mean_auc-auc_ci,ymax=mean_auc+auc_ci),width=ci_style_width)+
    geom_errorbar(aes(x=num_miss_features,ymin=mean_auc_miss-mean_auc_miss_ci ,
                      ymax=mean_auc_miss+mean_auc_miss_ci),width=ci_style_width) + 
    geom_line(aes(y=mean_auc_miss,color='')) + 
    ylab("auc") + xlab("Number of missing features") + 
    ggtitle(datax)  +
    theme_bw()
  p_ap=qplot(x=num_miss_features,y=mean_ap,data=df) +
    geom_line(aes(color="AD with missing value"))+ 
    geom_errorbar(aes(x=num_miss_features,ymin=mean_ap-ap_ci,ymax=mean_ap+ap_ci),width=ci_style_width)+
    geom_errorbar(aes(x=num_miss_features,ymin=mean_ap_miss-mean_ap_miss_ci ,
                      ymax=mean_ap_miss+mean_ap_miss_ci),width=ci_style_width) + 
    geom_line(aes(y=mean_ap_miss,color='AD with missing value treated')) + 
    ylab("ap") + xlab("Proportion of missing features") + 
    ggtitle(datax)  +
    theme_bw()
    grid.arrange(nrow=2,p_auc,p_ap)  
}

plot_error_anom <-function(dx,miss_feature=2){
  df = dx[dx$num_miss_features==miss_feature,]

  p_auc=qplot(x=anom_prop,y=mean_auc,data=df) +
    geom_line(aes(color="With missing value"))+ 
    geom_errorbar(aes(x=anom_prop,ymin=mean_auc-auc_ci,ymax=mean_auc+auc_ci),width=ci_style_width)+
    geom_errorbar(aes(x=anom_prop,ymin=mean_auc_miss-mean_auc_miss_ci ,
                      ymax=mean_auc_miss+mean_auc_miss_ci),width=ci_style_width) + 
    geom_line(aes(y=mean_auc_miss,color='Missing treated')) + 
    ylab("auc") + xlab("Proportion of data with missing value") + 
    ggtitle(datax)  +
    theme_bw()
  p_ap=qplot(x=anom_prop,y=mean_ap,data=df) +
    geom_line(aes(color="AD with missing value"))+ 
    geom_errorbar(aes(x=anom_prop,ymin=mean_ap-ap_ci,ymax=mean_ap+ap_ci),width=ci_style_width)+
    geom_errorbar(aes(x=anom_prop,ymin=mean_ap_miss-mean_ap_miss_ci ,
                      ymax=mean_ap_miss+mean_ap_miss_ci),width=ci_style_width) + 
    geom_line(aes(y=mean_ap_miss,color='AD with missing value treated')) + 
    ylab("ap") + xlab("Proportion of data with missing value") + 
    ggtitle(datax)  +
    theme_bw()
  
  grid.arrange(nrow=2,p_auc,p_ap)  
}

groupby_common_key<-function(datax){
  ds = read.csv(datax)
  dx = ds %>% group_by(anom_prop,num_miss_features) %>% 
    summarise(mean_auc = mean(auc), auc_ci=util.ciZ(auc),
              mean_ap = mean(ap),ap_ci = util.ciZ(ap),
              mean_auc_miss=mean(auc_cm), mean_auc_miss_ci = util.ciZ(auc_cm),
              mean_ap_miss = mean(ap_cm),mean_ap_miss_ci = util.ciZ(ap_cm),len = length(auc))
}

## Group ensemble size 
yeast_grp = groupby_common_key('../experiments/summary/ensemble_loda_abalone.csv')


## plot with the R plot 

summary_data ="../experiments/summary/"
loda_files <- list.files(paste0(summary_data,"loda/"))
if_files <- list.files(paste0(summary_data,"if/"))
pdf("Anomaly-missing-R-all.pdf",width = 11,height=6)#,width=3,height=2)
par(mfrow=c(2,3))
for(ff in 1:length(loda_files)){

 ldd <- groupby_common_key(paste0(summary_data,"loda/",loda_files[ff]))
 iff<- groupby_common_key(paste0(summary_data,"if/",if_files[ff]))
 title = strsplit(strsplit(basename(if_files[ff]),"\\_")[[1]][2],"\\.")[[1]][1]
plotconf(ldd,iff,0.2,title)
}
dev.off()


## Plot with the ggplot graphs 
pdf("Anomaly-missing-value-results.pdf",width=11,height=8)
for(ff in list.files(data)){
  datax = paste0(data,ff)
  #print(datax)
  dx = groupby_common_key(datax)  
  plot_error(dx)
  miss_ff_num = unique(dx$num_miss_features)[2]
  #plot_error_anom(dx,miss_feature = miss_ff_num)

  #write.table(dx,paste0(data,"_summmary.csv"),sep=",",row.name
  #s = F,quote = F)
}
dev.off()

##################################### Handling the index in particle loda 
pf = read.csv('../experiments/summary/if/if_abalone.csv')
pl = read.csv('../experiments/summary/loda_particle.csv')
pl_sb = pl[,-c(1:3)]
pl_sb[,names(pf[,c(1:3)])] =pf[,c(1:3)] 

ldd = pl_sb%>%group_by(anom_prop,num_miss_features) %>% 
  summarise(mean_auc = mean(auc), auc_ci=util.ciZ(auc),
            mean_ap = mean(ap),ap_ci = util.ciZ(ap),
            mean_auc_miss=mean(auc_cm), mean_auc_miss_ci = util.ciZ(auc_cm),
            mean_ap_miss = mean(ap_cm),mean_ap_miss_ci = util.ciZ(ap_cm),len = length(auc))

iff = pf%>%group_by(anom_prop,num_miss_features) %>% 
  summarise(mean_auc = mean(auc), auc_ci=util.ciZ(auc),
            mean_ap = mean(ap),ap_ci = util.ciZ(ap),
            mean_auc_miss=mean(auc_cm), mean_auc_miss_ci = util.ciZ(auc_cm),
            mean_ap_miss = mean(ap_cm),mean_ap_miss_ci = util.ciZ(ap_cm),len = length(auc))


############### group by ensemble size #################### 
datapath = '../experiments/summary/ensemble_loda_abalone.csv'

ensemble_plot<-function(datapath){

ds = read.csv(datapath)

dx = ds %>% group_by(anom_prop,num_miss_features,ensemble_size) %>% 
  summarise(mean_auc = mean(auc), auc_ci=util.ciZ(auc),
            mean_ap = mean(ap),ap_ci = util.ciZ(ap),
            mean_auc_miss=mean(auc_cm), mean_auc_miss_ci = util.ciZ(auc_cm),
            mean_ap_miss = mean(ap_cm),mean_ap_miss_ci = util.ciZ(ap_cm),len = length(auc))
miss_features = unique(dx$num_miss_features)[2]
dx_filter = dx%>% filter(num_miss_features==miss_features)

qplot(x=ensemble_size,y=mean_auc,data=dx_filter, group=as.factor(anom_prop),color=as.factor(anom_prop)) + 
  geom_point() + geom_line()
}