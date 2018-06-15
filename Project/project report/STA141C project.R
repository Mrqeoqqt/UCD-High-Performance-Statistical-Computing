# Apply xgboost to processed data #

##### import dataset #####
processed<- read.csv('C:/Users/MiaoWangqian/Desktop/processed.csv', header = TRUE)

##### import library #####
library(xgboost)

##### data preparation #####
idx <- sample(2,nrow(processed),replace = TRUE, prob = c(0.7,0.3))
train <- processed[idx == 1,]
test <- processed[idx == 2,]
y_train = train['shot_made_flag']
X_train = train[-grep('shot_made_flag', colnames(train))]
y_test = test['shot_made_flag']
X_test = test[-grep('shot_made_flag', colnames(test))]

##### train xgboost #####
dtrain <- xgb.DMatrix(data = data.matrix(X_train), label =  data.matrix(y_train))
bst <- xgboost(data = dtrain, max.depth = 5, eta = 1, nthread = 2, nround =50 , objective = "binary:logistic", verbose = 0)

##### prediction #####
y_pred <- predict(bst, data.matrix(X_test))
y_pred = as.numeric(y_pred > 0.5)
sens = sum(y_pred ==1 & y_test == 1)/sum(y_test == 1)
spec = sum(y_pred ==0 & y_test == 0)/sum(y_test == 0)
error = sum(y_pred != y_test)/length(y_pred)
results = c(sens, spec, error)
names(results) = c("Sensitivity", "Specificity", "Error Rate")
round(results, 4)
