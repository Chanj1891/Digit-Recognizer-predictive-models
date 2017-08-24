
############### DECISION TREE ALGORITHM 
###############################################################################
########################      training dataset      ###########################
##############      divided into training and validation      #################
###############################################################################

#clearing the environment
rm(list=ls())

#reading in file as a dataframe
windep <- read.csv("C:\Users\Jaish\Desktop\Practicum\optdigits-orig_windep.dat", header = FALSE, sep = " ")

#dropping 1026 column as it is a duplicate of 1025
windep <- windep[,-1026]

#setting seed for reproducibility
set.seed(190619890)

#training dataset fraction and splitting data into training and validation
sampsize = 0.8
sampindx = sample(1:nrow(windep), as.integer(sampsize*(nrow(windep))), replace=FALSE)

tr.ind <- windep[sampindx,]
te.ind <- windep[-sampindx,]

rm(sampsize,sampindx,windep)

#simple decision tree implementation with default parameters
library(rpart)
library(rpart.plot)

dtree <- rpart(V1025~., data=tr.ind, method="class")
#summary(dtree)
printcp(dtree)
rpart.plot(dtree, type=3, extra=101, fallen.leaves = TRUE)


tr.ind$pred <- predict(dtree, newdata=tr.ind, type="class")
table(tr.ind$V1025, tr.ind$pred)
tab <- table(tr.ind$V1025, tr.ind$pred)
1-(sum(diag(tab))/sum(tab))
#misclassification on training dataset 0.2623521

te.ind$pred <- predict(dtree, newdata=te.ind, type="class")
table(te.ind$V1025, te.ind$pred)
tab <- table(te.ind$V1025, te.ind$pred)
1-(sum(diag(tab))/sum(tab))
#misclassification on validation dataset 0.2583333 (Note: surprisingly lower! But, is it?)

#tweaking parameters to arrive at the optimal parameters
for(i in seq(1,10,by=1)){
  print(i/200)
  testree <- rpart(V1025~., data=tr.ind, method="class", control = rpart.control(cp = i/200))
  pred <- factor(matrix(nrow=nrow(te.ind),ncol=1))
  pred <- predict(testree, newdata = te.ind, type = "class")
  tab <- table(te.ind$V1025, pred)
  print(1-(sum(diag(tab))/sum(tab)))
  rm(testree,pred,tab)
}


###############################################################################
#########################      testing dataset      ###########################
###############################################################################
wdep <- read.csv("D:/Business Analytics/Summer 2016/Analytics Practicum/optdigits-orig_wdep.dat", header = FALSE, sep = " ")
wdep <- wdep[,-1026]

#predicting
wdep$pred <- predict(dtree, newdata=wdep, type="class")
table(wdep$V1025, wdep$pred)
tab <- table(wdep$V1025, wdep$pred)
1-(sum(diag(tab))/sum(tab))
#misclassification on testing dataset 0.2948038





#################################################################
##################### K-NEAREST NEIGHBOR CODE ###################
#################################################################

  train<-read.csv("<folder location>/optdigits-orig_cv_linear.dat",header=FALSE, sep=" ")
head(train)
train<-train[,1:1025]
tra<-read.csv("<folder location>/optdigits-orig_cv_linear.dat",header=FALSE, sep=" ")
head(tra)
tra<-tra[,1:1025]
library(digest) 
train1<-train[!duplicated(lapply(train, digest))]
##sampling the data to 1:1 ratio
sample<-sample(1:nrow(train),as.integer(nrow(train)*0.5),replace=FALSE)
k.build<-train[sample,]
dim(k.build)
k.test<-train[-sample,]
dim(k.test)
library(class)
##testing the model run
model <- knn(train = k.build,test = k.test,cl = k.build$V1025,k=3)
table(model,k.test$V1025)
mean(model==k.test$V1025)
##tabulating accuracy for different k values
k<-1:10
accuracy<-rep(0,10)
for(i in k){
  model <- knn(train = k.build,test = k.test,cl = k.build$V1025,k=i)
  accuracy[i]<-mean(model==k.test$V1025)}
plot(k,accuracy,type="b")
result=cbind(k,accuracy)
result

#################################################################
##################### J48 CLASSIFIER CODE #######################
#################################################################

install.packages('RWeka')
library(RWeka)

#reading in file as a dataframe
windep <- read.csv("<folder location>/optdigits-orig_tra_linear.dat", header = FALSE, sep = " ")
head(windep)
#dropping 1026 column as it is a duplicate of 1025
windep <- windep[,-1026]

#setting seed for reproducibility
set.seed(190619890)
dim(windep)
#training dataset fraction and splitting data into training and validation
sampsize = 0.8
sampindx = sample(1:nrow(windep), as.integer(sampsize*(nrow(windep))), replace=FALSE)
tr.ind <- windep[sampindx,]
te.ind <- windep[-sampindx,]

j48model <- J48(as.factor(V1025)~. , data=tr.ind)
j48model

result<-predict(j48model,newdata=te.ind)
table(result,te.ind$V1025)
##calculates the accuracy, generated accuracy of 84.49%
mean(result == te.ind$V1025)

############# Application of model on testing data ############

wdep <- read.csv("<folder location>/optdigits-orig_cv_linear.dat", header = FALSE, sep = " ")
wdep <- wdep[,-1026]

############# Prediction
wdep$pred <- predict(j48model, newdata=wdep, type="class")
table(wdep$V1025, wdep$pred)
tab <- table(wdep$V1025, wdep$pred)
mean(wdep$V1025 == wdep$pred)

#################################################################
##################### BAGGING TECHNIQUE #########################
#################################################################

install.packages('RWeka')
library(RWeka)

#reading in file as a dataframe
windep <- read.csv("<folder location>/optdigits-orig_tra_linear.dat", header = FALSE, sep = " ")
head(windep)
#dropping 1026 column as it is a duplicate of 1025
windep <- windep[,-1026]

#setting seed for reproducibility
set.seed(190619890)
dim(windep)
#training dataset fraction and splitting data into training and validation
sampsize = 0.8
sampindx = sample(1:nrow(windep), as.integer(sampsize*(nrow(windep))), replace=FALSE)
tr.ind <- windep[sampindx,]
te.ind <- windep[-sampindx,]

install.packages("ipred")
library(ipred)
bagmodel <- bagging(as.factor(V1025)~., data=tr.ind)
bagmodel

result<-predict(bagmodel,newdata=te.ind)
table(result,te.ind$V1025)
##calculates the accuracy, generated accuracy of 84.49%
mean(result == te.ind$V1025)

############# Application of model on testing data ############

wdep <- read.csv("<folder location>/optdigits-orig_cv_linear.dat", header = FALSE, sep = " ")
wdep <- wdep[,-1026]

############# Prediction
wdep$pred <- predict(bagmodel, newdata=wdep, type="class")
table(wdep$V1025, wdep$pred)
tab <- table(wdep$V1025, wdep$pred)
mean(wdep$V1025 == wdep$pred)