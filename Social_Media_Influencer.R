#The caret package provides a consistent interface 
#into hundreds of machine learning algorithms 
#and provides useful convenience methods for 
#data visualization, data resampling, model tuning and model comparison etc.
install.packages('caret')
install.packages('e1071')
library(caret)
library(ggplot2)
library(kernlab)
library(dplyr)

train=read.csv(file="train.csv")
dim(train)
str(train)


#create a list of 80% of the rows of the original dataset
#we can use for training
set.seed(123)
validation_set<-createDataPartition(train$Choice, p=0.80, list=FALSE)
#View(validation_set)
#select 20% of the data for validation
validation<-train[-validation_set,]
#View(validation)

#use the remaining 80% of data to train and test the models
train<-train[validation_set,]

#dimensions of dataset
dim(train)

#types for each attribute
sapply(train,class)

#converting class label "Choice" to factor type from integer.
train$Choice=as.factor(train$Choice)

#checking the datatypes again
sapply(train,class)

#first 5 rows to get an idea of the data
head(train)

# as the label class "Choice" is categorical we check equal distribution
sum(train$Choice==0)
sum(train$Choice==1)
percentage <- prop.table(table(train$Choice) * 100)
cbind(freq=table(train$Choice), percentage=percentage)
p<-ggplot(train,aes(Choice)) 
p + geom_bar(fill="red")+ geom_text(stat='count',aes(label=..count..),vjust=-1)

#no changes are required as the number of 1's and 0's are almost equal.

#list the levels for the class label
levels(train$Choice)
#we can see there are two levels, making it a binary classification problem
summary(train)

#Prediction algorithms before any preprocessing
#Set-up the test harness to use 10-fold  cross validation.
#We will use 10-fold crossvalidation to estimate accuracy.
#This will split our dataset into 10 parts, train in 9 and test
#on 1 and release for all combinations of train-test splits.
#We will also repeat the process 3 times for each algorithm with different
#splits of the data into 10 groups, in an effort to get a more accurate estimate.

#Run algorithms using 10-fold cross validation
control<-trainControl(method="cv",number=10)
metric<-"Accuracy"


#Pre-Processing
#Detecting outliers and eliminating them
sapply(train,function(x) sum(is.na(x))) #checking for missing values
sapply(train, function(x) length(unique(x)))
sapply(train, class)
train3=train
#1
Q1 <- quantile(train3$A_follower_count, probs=c(.25, .75), na.rm = FALSE)
iqr1 <- IQR(train3$A_follower_count)
up <-  Q1[1]+1.5*iqr1 # Upper Range  
low<- Q1[1]-1.5*iqr1 # Lower Range
eliminated<- subset(train3,train3$A_follower_count > (Q1[1] - 1.5*iqr1) & train3$A_follower_count< (Q1[2]+1.5*iqr1))
#2
Q1 <- quantile(eliminated$A_listed_count, probs=c(.25, .75), na.rm = FALSE)
iqr1 <- IQR(eliminated$A_listed_count)
up <-  Q1[1]+1.5*iqr1 # Upper Range  
low<- Q1[1]-1.5*iqr1 # Lower Range
eliminated<- subset(eliminated,eliminated$A_listed_count > (Q1[1] - 1.5*iqr1) & eliminated$A_listed_count< (Q1[2]+1.5*iqr1))
#3
Q1 <- quantile(eliminated$A_mentions_sent, probs=c(.25, .75), na.rm = FALSE)
iqr1 <- IQR(eliminated$A_mentions_sent)
up <-  Q1[1]+1.5*iqr1 # Upper Range  
low<- Q1[1]-1.5*iqr1 # Lower Range
eliminated<- subset(eliminated,eliminated$A_mentions_sent > (Q1[1] - 1.5*iqr1) & eliminated$A_mentions_sent< (Q1[2]+1.5*iqr1))
#4
Q1 <- quantile(eliminated$A_retweets_sent, probs=c(.25, .75), na.rm = FALSE)
iqr1 <- IQR(eliminated$A_retweets_sent)
up <-  Q1[1]+1.5*iqr1 # Upper Range  
low<- Q1[1]-1.5*iqr1 # Lower Range
eliminated<- subset(eliminated,eliminated$A_retweets_sent > (Q1[1] - 1.5*iqr1) & eliminated$A_retweets_sent< (Q1[2]+1.5*iqr1))
#5
Q1 <- quantile(eliminated$A_posts, probs=c(.25, .75), na.rm = FALSE)
iqr1 <- IQR(eliminated$A_posts)
up <-  Q1[1]+1.5*iqr1 # Upper Range  
low<- Q1[1]-1.5*iqr1 # Lower Range
eliminated<- subset(eliminated,eliminated$A_posts > (Q1[1] - 1.5*iqr1) & eliminated$A_posts< (Q1[2]+1.5*iqr1))
#6
Q1 <- quantile(eliminated$B_follower_count, probs=c(.25, .75), na.rm = FALSE)
iqr1 <- IQR(eliminated$B_follower_count)
up <-  Q1[1]+1.5*iqr1 # Upper Range  
low<- Q1[1]-1.5*iqr1 # Lower Range
eliminated<- subset(eliminated,eliminated$A_follower_count > (Q1[1] - 1.5*iqr1) & eliminated$A_follower_count< (Q1[2]+1.5*iqr1))
#7
Q1 <- quantile(eliminated$B_retweets_sent, probs=c(.25, .75), na.rm = FALSE)
iqr1 <- IQR(eliminated$B_retweets_sent)
up <-  Q1[1]+1.5*iqr1 # Upper Range  
low<- Q1[1]-1.5*iqr1 # Lower Range
eliminated<- subset(eliminated,eliminated$B_retweets_sent > (Q1[1] - 1.5*iqr1) & eliminated$B_retweets_sent< (Q1[2]+1.5*iqr1))
#8
Q1 <- quantile(eliminated$B_listed_count, probs=c(.25, .75), na.rm = FALSE)
iqr1 <- IQR(eliminated$B_listed_count)
up <-  Q1[1]+1.5*iqr1 # Upper Range  
low<- Q1[1]-1.5*iqr1 # Lower Range
eliminated<- subset(eliminated,eliminated$B_listed_count > (Q1[1] - 1.5*iqr1) & eliminated$B_listed_count< (Q1[2]+1.5*iqr1))
#9
Q1 <- quantile(eliminated$B_mentions_sent, probs=c(.25, .75), na.rm = FALSE)
iqr1 <- IQR(eliminated$B_mentions_sent)
up <-  Q1[1]+1.5*iqr1 # Upper Range  
low<- Q1[1]-1.5*iqr1 # Lower Range
eliminated<- subset(eliminated,eliminated$B_mentions_sent > (Q1[1] - 1.5*iqr1) & eliminated$B_mentions_sent< (Q1[2]+1.5*iqr1))
#10
Q1 <- quantile(eliminated$B_posts, probs=c(.25, .75), na.rm = FALSE)
iqr1 <- IQR(eliminated$B_posts)
up <-  Q1[1]+1.5*iqr1 # Upper Range  
low<- Q1[1]-1.5*iqr1 # Lower Range
eliminated<- subset(eliminated,eliminated$B_posts > (Q1[1] - 1.5*iqr1) & eliminated$B_posts< (Q1[2]+1.5*iqr1))
#11
Q1 <- quantile(eliminated$B_network_feature_1, probs=c(.25, .75), na.rm = FALSE)
iqr1 <- IQR(eliminated$B_network_feature_1)
up <-  Q1[1]+1.5*iqr1 # Upper Range  
low<- Q1[1]-1.5*iqr1 # Lower Range
eliminated<- subset(eliminated,eliminated$B_network_feature_1 > (Q1[1] - 1.5*iqr1) & eliminated$B_network_feature_1< (Q1[2]+1.5*iqr1))
#12
Q1 <- quantile(train3$A_following_count, probs=c(.25, .75), na.rm = FALSE)
iqr1 <- IQR(train3$A_following_count)
up <-  Q1[1]+1.5*iqr1 # Upper Range  
low<- Q1[1]-1.5*iqr1 # Lower Range
eliminated<- subset(train3,train3$A_following_count > (Q1[1] - 1.5*iqr1) & train3$A_following_count< (Q1[2]+1.5*iqr1))
#13
Q1 <- quantile(eliminated$A_mentions_received, probs=c(.25, .75), na.rm = FALSE)
iqr1 <- IQR(eliminated$A_mentions_received)
up <-  Q1[1]+1.5*iqr1 # Upper Range  
low<- Q1[1]-1.5*iqr1 # Lower Range
eliminated<- subset(eliminated,eliminated$A_mentions_received > (Q1[1] - 1.5*iqr1) & eliminated$A_mentions_received< (Q1[2]+1.5*iqr1))
#14
Q1 <- quantile(eliminated$A_retweets_received, probs=c(.25, .75), na.rm = FALSE)
iqr1 <- IQR(eliminated$A_retweets_received)
up <-  Q1[1]+1.5*iqr1 # Upper Range  
low<- Q1[1]-1.5*iqr1 # Lower Range
eliminated<- subset(eliminated,eliminated$A_retweets_received > (Q1[1] - 1.5*iqr1) & eliminated$A_retweets_received< (Q1[2]+1.5*iqr1))
#15
Q1 <- quantile(eliminated$A_network_feature_3, probs=c(.25, .75), na.rm = FALSE)
iqr1 <- IQR(eliminated$A_network_feature_3)
up <-  Q1[1]+1.5*iqr1 # Upper Range  
low<- Q1[1]-1.5*iqr1 # Lower Range
eliminated<- subset(eliminated,eliminated$A_network_feature_3 > (Q1[1] - 1.5*iqr1) & eliminated$A_network_feature_3< (Q1[2]+1.5*iqr1))
#16
Q1 <- quantile(eliminated$A_network_feature_2, probs=c(.25, .75), na.rm = FALSE)
iqr1 <- IQR(eliminated$A_network_feature_2)
up <-  Q1[1]+1.5*iqr1 # Upper Range  
low<- Q1[1]-1.5*iqr1 # Lower Range
eliminated<- subset(eliminated,eliminated$A_network_feature_2 > (Q1[1] - 1.5*iqr1) & eliminated$A_network_feature_2< (Q1[2]+1.5*iqr1))
#17
Q1 <- quantile(eliminated$A_network_feature_1, probs=c(.25, .75), na.rm = FALSE)
iqr1 <- IQR(eliminated$A_network_feature_1)
up <-  Q1[1]+1.5*iqr1 # Upper Range  
low<- Q1[1]-1.5*iqr1 # Lower Range
eliminated<- subset(eliminated,eliminated$A_network_feature_1 > (Q1[1] - 1.5*iqr1) & eliminated$A_network_feature_1< (Q1[2]+1.5*iqr1))
#18
Q1 <- quantile(eliminated$B_following_count, probs=c(.25, .75), na.rm = FALSE)
iqr1 <- IQR(eliminated$B_following_count)
up <-  Q1[1]+1.5*iqr1 # Upper Range  
low<- Q1[1]-1.5*iqr1 # Lower Range
eliminated<- subset(eliminated,eliminated$B_following_count > (Q1[1] - 1.5*iqr1) & eliminated$B_following_count< (Q1[2]+1.5*iqr1))
#19
Q1 <- quantile(eliminated$B_mentions_received, probs=c(.25, .75), na.rm = FALSE)
iqr1 <- IQR(eliminated$B_mentions_received)
up <-  Q1[1]+1.5*iqr1 # Upper Range  
low<- Q1[1]-1.5*iqr1 # Lower Range
eliminated<- subset(eliminated,eliminated$B_mentions_received > (Q1[1] - 1.5*iqr1) & eliminated$B_mentions_received< (Q1[2]+1.5*iqr1))
#20
Q1 <- quantile(eliminated$B_retweets_sent, probs=c(.25, .75), na.rm = FALSE)
iqr1 <- IQR(eliminated$B_retweets_sent)
up <-  Q1[1]+1.5*iqr1 # Upper Range  
low<- Q1[1]-1.5*iqr1 # Lower Range
eliminated<- subset(eliminated,eliminated$B_retweets_sent > (Q1[1] - 1.5*iqr1) & eliminated$B_retweets_sent< (Q1[2]+1.5*iqr1))
#21
Q1 <- quantile(eliminated$B_network_feature_3, probs=c(.25, .75), na.rm = FALSE)
iqr1 <- IQR(eliminated$B_network_feature_3)
up <-  Q1[1]+1.5*iqr1 # Upper Range  
low<- Q1[1]-1.5*iqr1 # Lower Range
eliminated<- subset(eliminated,eliminated$B_network_feature_3 > (Q1[1] - 1.5*iqr1) & eliminated$B_network_feature_3< (Q1[2]+1.5*iqr1))
#22
Q1 <- quantile(eliminated$B_network_feature_2, probs=c(.25, .75), na.rm = FALSE)
iqr1 <- IQR(eliminated$B_network_feature_2)
up <-  Q1[1]+1.5*iqr1 # Upper Range  
low<- Q1[1]-1.5*iqr1 # Lower Range
eliminated<- subset(eliminated,eliminated$B_network_feature_2 > (Q1[1] - 1.5*iqr1) & eliminated$B_network_feature_2< (Q1[2]+1.5*iqr1))

set.seed(200)
fit.svm4<-train(Choice~.,data=eliminated,method="svmRadial",metric=metric, trControl=control) 

validation$Choice<-as.factor(validation$Choice)
str(validation$Choice)
predictions4<-predict(fit.svm4,validation)
confusionMatrix(predictions4,validation$Choice)
#accuracy=53.09

#accuracy is decreased after removing the ouliers so the outliers are not harmful for the prediction

#Identifying the columns which effect the outcome most
train4=train
Choice<-as.numeric(train4$Choice)
#finding correlation

cor.test(train4$A_follower_count,Choice, method = "pearson")
cor.test(train4$A_following_count,Choice, method = "pearson")
cor.test(train4$A_listed_count,Choice, method = "pearson")
cor.test(train4$A_mentions_received,Choice,method = "pearson")
cor.test(train4$A_mentions_sent,Choice, method = "pearson")
cor.test(train4$A_retweets_received,Choice,method = "pearson")
cor.test(train4$A_retweets_sent,Choice, method = "pearson")
cor.test(train4$A_posts,Choice, method = "pearson")
cor.test(train4$A_network_feature_1,Choice, method = "pearson")
cor.test(train4$A_network_feature_2,Choice, method = "pearson")
cor.test(train4$A_network_feature_3,Choice, method = "pearson")


cor.test(train4$B_follower_count,Choice, method = "pearson")
cor.test(train4$B_following_count,Choice, method = "pearson")
cor.test(train4$B_listed_count,Choice, method = "pearson")
cor.test(train4$B_mentions_received,Choice,method = "pearson")
cor.test(train4$B_mentions_sent,Choice, method = "pearson")
cor.test(train4$B_retweets_received,Choice,method = "pearson")
cor.test(train4$B_retweets_sent,Choice, method = "pearson")
cor.test(train4$B_posts,Choice, method = "pearson")
cor.test(train4$B_network_feature_1,Choice, method = "pearson")
cor.test(train4$B_network_feature_2,Choice, method = "pearson")
cor.test(train4$B_network_feature_3,Choice, method = "pearson")

#summarize the correlation
cor(train4[, sapply(train4, class) != "factor"])

Choice<-as.factor(train4$Choice)
str(train4$Choice)


#appending train4 set with the most correlated values
train4<-train4[,c(-2,-6,-5,-10,-11,-16,-17,-22,-14)]


dim(train4)
colnames(train4)
#########################################SVM################################
set.seed(200)
fit.svm<-train(Choice~.,data=train4,method="svmRadial",metric=metric, trControl=control)

validation$Choice<-as.factor(validation$Choice)
str(validation$Choice)
predictionsa<-predict(fit.svm,validation)
confusionMatrix(predictionsa,validation$Choice)
#accuracy  74.55

##########################logistic regression##################################
fit.glm<-train(Choice~.,data=train4,method="glm",metric=metric, trControl=control,family = "binomial")
validation$Choice<-as.factor(validation$Choice)
str(validation$Choice)
predictionsa<-predict(fit.glm,validation)
confusionMatrix(predictionsa,validation$Choice)
#accuracy 72.73


###############################Naves Bayes##########################
fit.nb<-train(Choice~.,data=train4,method="nb",metric=metric, trControl=control)
validation$Choice<-as.factor(validation$Choice)
str(validation$Choice)
predictionsa<-predict(fit.nb,validation)
confusionMatrix(predictionsa,validation$Choice)
#accuracy   70

############################################DESCION TREE##################################
library("rpart") 
install.packages("rpart.plot")
library("rpart.plot")
install.packages("rattle")
library("rattle")
library("RColorBrewer")
library(caret)

fit.dt <- rpart(Choice ~.,method="class",data=train4)
plot(fit.dt)
text(fit.dt, pretty=0)
fancyRpartPlot(fit.dt)
pred_Train = predict(fit.dt, train4, type="class")
table_mat<-table(pred_Train, train4$Choice)
table_mat

pred_Test = predict(fit.dt, validation, type = "class")
pred_Test
table_mat<-table(pred_Test, validation$Choice)
base_accuracy <- mean(pred_Test == validation$Choice)
base_accuracy
## 76.5

###########################################kNN ##########################

fit.knn<-train(Choice~.,data=train4,method="knn",metric=metric, trControl=control)
validation$Choice<-as.factor(validation$Choice)
str(validation$Choice)
predictionsa<-predict(fit.knn,validation)
confusionMatrix(predictionsa,validation$Choice) #73.36


results <- resamples(list(svm = fit.svm , glm = fit.glm , nb = fit.nb,knn=fit.knn))
summary(results)
dotplot(results)

######################################Ensemble Learning ###############################


###################################Averaging ##############################
#Predicting the probabilities
validation1=validation

#random forest##
fit.rf<-train(Choice~.,data=train4,method="rf",metric=metric, trControl=control)
validation$Choice<-as.factor(validation$Choice)
str(validation$Choice)
predictionsa<-predict(fit.rf,validation)
confusionMatrix(predictionsa,validation$Choice) #76

validation1$pred_rf_prob<-predict(fit.rf,validation1,type='raw')
validation1$pred_nb_prob<-predict(fit.nb,validation1,type='raw')
validation1$pred_lr_prob<-predict(fit.glm,validation1,type='raw')
validation1$pred_knn_prob<-predict(fit.knn,validation1,type='raw')

validation1$pred_svm_prob<-predict(fit.svm,validation1,type='raw')
validation1$pred_dt_prob<-predict(fit.dt,validation1,type='class')

str(validation1)

x <- list(col1 = as.numeric(validation1$pred_dt_prob), col2 = as.numeric(validation1$pred_svm_prob),col3 = as.numeric(validation1$pred_knn_prob),
          col4 = as.numeric(validation1$pred_lr_prob),col5= as.numeric(validation1$pred_nb_prob),col6 = as.numeric(validation1$pred_rf_prob))

validall <- as.data.frame(x)
str(validall)

validall$col1<-as.numeric(ifelse(validall$col1 ==1,0,1))
validall$col2<-as.numeric(ifelse(validall$col2 ==1,0,1))
validall$col3<-as.numeric(ifelse(validall$col3 ==1,0,1))
validall$col4<-as.numeric(ifelse(validall$col4 ==1,0,1))
validall$col5<-as.numeric(ifelse(validall$col5 ==1,0,1))
validall$col6<-as.numeric(ifelse(validall$col6 ==1,0,1))


str(validall)

#Taking average of predictions
validation1$pred_avg1<-(validall$col1 + validall$col2 + validall$col3 + validall$col4 + validall$col5 + validall$col6)/6
str(validation1)
validation1$pred_avg1<-ifelse(validation1$pred_avg1 > 0.5,1,0)
str(validation1)
accuracy <- mean(validation1$pred_avg1 == validation1$Choice)
accuracy #75.4

##################################Majority Voting ###########################################


tail(validall)

validation1$pred_max<-as.numeric(apply(validall,1,function(x) names(which.max(table(x)))))


str(validation1)
validation1$pred_max
accuracy <- mean(validation1$pred_max == validation1$Choice)
accuracy #75.4

####################################Weighted Average########################3
validation1$weight_pred_avg1<-(validall$col1*25 + validall$col2*10 + validall$col3*10 + validall$col4*25 + validall$col5*10 + validall$col6*25)/6
str(validation1)
validation1$weight_pred_avg1<-ifelse(validation1$weight_pred_avg1 > 10,1,0)
str(validation1)
accuracy <- mean(validation1$weight_pred_avg1 == validation1$Choice)
accuracy #76
################################bagging ##############################################

###Baagging crat
fit.treebag <- train(Choice~., data=train4, method="treebag", metric=metric, trControl=control)
validation$Choice<-as.factor(validation$Choice)
str(validation$Choice)
predictionsa<-predict(fit.treebag,validation)
confusionMatrix(predictionsa,validation$Choice)

bagging_results <- resamples(list(treebag=fit.treebag, rf=fit.rf))
summary(bagging_results)
dotplot(bagging_results)#75.4

################################# boosting #################################################

##########C50
fit.c50 <- train(Choice~., data=train4, method="C5.0", metric=metric, trControl=control)
validation$Choice<-as.factor(validation$Choice)
str(validation$Choice)
predictionsa<-predict(fit.c50,validation)
confusionMatrix(predictionsa,validation$Choice) #78.5

###Stochastic Gradient Boosting
fit.gbm <- train(Choice~., data=train4, method="gbm", metric=metric, trControl=control, verbose=FALSE)
validation$Choice<-as.factor(validation$Choice)
str(validation$Choice)
predictionsa<-predict(fit.gbm,validation)
confusionMatrix(predictionsa,validation$Choice) #78.09

boosting_results <- resamples(list(c5.0=fit.c50, gbm=fit.gbm))
summary(boosting_results)
dotplot(boosting_results)


bag_boost <- resamples(list(c5.0=fit.c50, gbm=fit.gbm,treebag=fit.treebag, rf=fit.rf))
summary(bag_boost)
dotplot(bag_boost)

