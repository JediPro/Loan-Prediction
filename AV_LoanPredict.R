
# Load libraries and images ####
library(caret)
library(car)
library(mice)
library(ROCR)
source("D:/R_Data/functions.R")
dsub=as.data.frame(matrix(nrow = nrow(test.raww), ncol = 2))
colnames(dsub)=c("Loan_ID","Loan_Status")
dsub$Loan_ID=test.raw$Loan_ID
dsub$Loan_Status=lrmod.p
write.csv(dsub, "Submission.csv", row.names = F)
setwd("D:/Datasets/AV_Loan Prediction")
load("D:/Datasets/AV_Loan Prediction/LoanPData.RData")
train.raw=read.csv("train.csv", header = T, sep = ",", na.strings = "")
test.raw=read.csv("test.csv", header = T, sep = ",", na.strings = "")
dtrain=train.raw
dtest=test.raw
resp=dtrain$Loan_Status
respn=ifelse(resp=='Y',1,0)
dfull=rbind(dtrain[,-13], dtest)

# Clean data ####
summary(dtrain)
summary(dtest)
table(dfull$Gender, dfull$Married, useNA = "ifany")
table(dfull$Married, dfull$Dependents, useNA = "always")

# Married
dfull$Married[is.na(dfull$Married) & dfull$Gender=="Male"] = "Yes"
dfull$Married[is.na(dfull$Married) & dfull$Gender=="Female"] = "No"


# Convert data before using ####
# ApplicantIncome
ggplot(dfull, aes(x=ApplicantIncome))+geom_density()
dfull$ApplicantIncome=log(dfull$ApplicantIncome+1)
ggplot(dfull, aes(x=ApplicantIncome))+geom_density()
# CoapplicantIncome
ggplot(dfull, aes(x=CoapplicantIncome))+geom_density()
dfull$CoapplicantIncome=log(dfull$CoapplicantIncome+1)
# Form new column due ot bimodal nature of distribution
dfull$CoAppIncLevel=ifelse(dfull$CoapplicantIncome>4,"High","Low")
dfull$CoAppIncLevel=as.factor(dfull$CoAppIncLevel)
# Loan Amount
densityplot(dfull$LoanAmount)
dfull$LoanAmount=log(dfull$LoanAmount+1)
# Loan Term
dfull$Loan_Amount_Term=recode(dfull$Loan_Amount_Term, "0:359='LT1'; 
                              else='MT1'")
dfull$Loan_Amount_Term=as.factor(dfull$Loan_Amount_Term)
# Credit History
dfull$Credit_History=as.factor(dfull$Credit_History)
# Add column for Loan term in months
dfull$LoanTerm=dfull2$Loan_Amount_Term

# Fill NAs ####
predmat=quickpred(dfull, mincor = 0.2)
predmat[,"Loan_ID"]=0
dtemp=mice(data = dfull, m=8,maxit = 8, seed = 713, predictorMatrix = predmat)
dimp=complete(dtemp, 8)
summary(dimp)

# Split datasets ####
mf=model.matrix(~., data = dimp[,!names(dimp) %in% 'Loan_ID'])
mf=as.data.frame(mf)
# Normalize Data
mn=as.data.frame(apply(mf, 2, function(x){(x-min(x))/(max(x)-min(x))}))

mtrain=mn[1:nrow(train.raw),]
mtest=mn[(nrow(train.raw)+1):nrow(mf),]

set.seed(713)
prt=sample(seq_len(nrow(mtrain)), 0.7*nrow(mtrain), replace = F)
dt=mtrain[prt,]
dv=mtrain[-prt,]
respf=ifelse(respn==1,"Y","N")
resp.t=respf[prt]
resp.v=respf[-prt]

# Fit XGBoost ####
set.seed(713)
cvCtrl=trainControl(method = "repeatedcv", number = 3, repeats = 3, verboseIter = T)
xgbmod=train(x=dt[,-1], y=as.factor(resp.t), method = "xgbLinear", trControl = cvCtrl, tuneLength = 3)
xgbmod.v=predict(xgbmod, newdata = dv[,-1], type = "prob")
rocc(resp.v, xgbmod.v$Y)
xgbmod.c=ifelse(xgbmod.v$Y>0.3,"Y","N")
sum(diag(table(resp.v, xgbmod.c)))/ length(xgbmod.c)
table(resp.v, xgbmod.c)
xgbmod.f=predict(xgbmod, newdata=mtest[,-1], type = "prob")
xgbmod.p=ifelse(xgbmod.f$Y,"Y","N")

# Logistic regression ####
set.seed(713)
cvCtrl=trainControl(method = "repeatedcv", number = 5, repeats = 5, verboseIter = T)
lrmod=train(x=dt[,-1], y=resp.t, method = "glm", trControl = cvCtrl, tuneLength = 5)
lrmod.r=predict(lrmod, newdata = dv, type = "prob")
rocc(resp.v, lrmod.r$Y)
lrmod.c=ifelse(lrmod.r$Y>0.35, "Y", "N")
sum(diag(table(lrmod.c, resp.v)))/ length(lrmod.c)
table(resp.v, lrmod.c)
lrmod.f=predict(lrmod, newdata=mtest, type = "prob")
lrmod.p=ifelse(lrmod.f$Y>0.45, "Y", "N")

# Neural Net ####
set.seed(713)
cvCtrl=trainControl(method = "repeatedcv", number = 5, repeats = 5, verboseIter = T)
nnmod=train(x=dt[,-1], y=resp.t, method = "nnet", trControl = cvCtrl, tuneLength = 5)
nnmod.v=predict(nnmod, newdata = dv, type = "prob")
rocc(resp.v,nnmod.v$Y)
nnmod.c=ifelse(nnmod.v$Y>0.3,"Y","N")
sum(diag(table(resp.v, nnmod.c)))/ length(nnmod.c)
table(resp.v, nnmod.c)
nnmod.f=predict(nnmod, newdata=mtest, type="prob")
nnmod.p=ifelse(nnmod.f$Y>0.35,"Y","N")

# SVM ####
set.seed(713)
cvCtrl=trainControl(method = "repeatedcv", number = 3, repeats = 3, verboseIter = T, classProbs = T)
svmmod=train(x=dt[,-1], y=resp.t, method = "svmRadial", trControl = cvCtrl, tuneLength = 3)
svmmod.v=predict(svmmod, newdata = dv[,-1], type="prob")
rocc(resp.v, svmmod.v$Y)
svmmod.c=ifelse(svmmod.v$Y>0.5,"Y","N")
sum(diag(table(resp.v, svmmod.c)))/ length(svmmod.c)
table(resp.v, svmmod.c)
svmmod.f=predict(svmmod, newdata=mtest[,-1], type="prob")
svmmod.p=ifelse(svmmod.f$Y>0.4, "Y", "N")

# Random Forest ####
set.seed(713)
cvCtrl=trainControl(method = "repeatedcv", number = 4, repeats = 4, verboseIter = T, classProbs = T)
rfmod=train(x=dt[,-1], y=resp.t, method = "rf", trControl = cvCtrl, tuneLength = 4)
rfmod.v=predict(rfmod, newdata = dv[,-1], type="prob")
rocc(resp.v, rfmod.v$Y)
rfmod.c=ifelse(rfmod.v$Y>0.4,"Y","N")
sum(diag(table(resp.v, rfmod.c)))/ length(rfmod.c)
table(resp.v, rfmod.c)
rfmod.f=predict(rfmod, newdata=mtest[,-1], type="prob")
rfmod.p=ifelse(rfmod.f$Y>0.5, "Y", "N")

# Model ensembling ####
ensm.a=as.data.frame(matrix(nrow = nrow(mtest), ncol = 5))
colnames(ensm.a)=c("XGB","LogR","NN","SVM","RF")
ensm.a$XGB=xgbmod.f$`1`
ensm.a$LogR=lrmod.f$Y
ensm.a$NN=nnmod.f$Y
ensm.a$SVM=svmmod.f$Y
ensm.a$RF=rfmod.f$Y
wgth=c(0.3,0.4,0.2,0.0,0.1)
ensm.f=wgth * ensm.a
ensm.p=apply(ensm.f,1,sum)
ensm.p=ifelse(ensm.p>0.6,"Y","N")
