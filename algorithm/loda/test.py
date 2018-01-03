import pandas as pd
import loda

NA = -9999.0
data_path = "synthetic100.csv"
#data_path = "../missingdata/experiments/anomaly/abalone/fullsamples/abalone_1.csv"
df = pd.read_csv(data_path)
print df.head(5)
train_data = df.ix[:,1:8].as_matrix()
#print nrow(train_data),ncol(train_data)
test_data = train_data.copy()
#test_data[0,[1,3]] = NA
#test_data[1,2] = NA
#test_data[5,2] = NA

ld = loda.Loda()
ld.train(train_x=train_data)
#print pvh.pvh
score = ld.score(test_data)
#ldd = loda(train_data,test=test_data,maxk=100,check_miss=True)
#lbl = df.ix[:,0]=="anomaly" # pd.Series(map(int,df.ix[:,0]!='nominal'))
#dx = cbind(lbl,ldd.nll)
#dy = cbind(lbl,loda(train_data,test=test_data,maxk=100,check_miss=False).nll)
#dx = cbind(dx,ldd.anomranks)
#print ldd.anomranks#,ldd.nll
#print ldd.nll

#print 1-fn_au_detection(dy)
#print 1-fn_au_detection(dx)
print score.nll


