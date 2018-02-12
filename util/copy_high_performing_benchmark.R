
pathdir="/nfs/guille/bugid/adams/meta_analysis/results_summaries/"
destination = "/nfs/guille/bugid/adams/ifTadesse/kddexperiment/group2/"
ff = list.files(pathdir)
aucs <- data.frame()
for (bench in ff){
  if(bench=="new_all" || bench=="all"){
   next 
  }
for(idx in 290:300){
  #idx = 300
  bench_name = paste0(pathdir,bench,"/auc_",bench,".csv")
  res = read.csv(bench_name,T)
  res = res[order(res$iforest,decreasing=T),]
 if(bench %in% c("particle","gas","yeast","synthetic", "yearp")) next 
 if(bench=="abalone"){
	 idx = idx +40
  }

  cat(bench,"_",res$bench.id[idx],res$iforest[idx],"\n")
  filename = paste0("/nfs/guille/bugid/adams/meta_analysis/benchmarks/",bench,"/",res$bench.id[idx],".csv")
  xx = read.csv(filename)
  #print(nrow(xx))
  aucs <- rbind(aucs,data.frame(res$bench.id[idx], res$iforest[idx], res$loda[idx],res$egmm[idx]))
  #cat(bench,"_",res$bench.id,"\n")
  file.copy(filename, destination)
}
}
writet.table(aucs,"dataset_used_summary.csv",row.names=F,quote=F)
#"/nfs/guille/bugid/adams/meta_analysis/results_summaries/abalone/auc_abalone.csv"
