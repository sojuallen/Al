library(lightgbm)
library(methods)

data("agaricus.train", package =  "lightgbm")
data("agaricus.test", package = "lightgbm")


train <- agaricus.train
test <- agaricus.test

# The loaded data is stored in sparseMatrix, and label is a numeric vector in {0,1}
class(train$label)
class(train$data)

## --------------------Basic Training using lightgbm----------------
# This is the basic usage of lightgbm you can put matrix in data field
# Note: we are putting in sparse matrix here, lightgbm naturally handles sparse input
# Use sparse matrix when your feature is sparse (e.g. when you are using one-hot encoding vector)

print("Training lightgbm with sparseMatrix")
bst <- lightgbm(data = train$data,
                label = train$label,
                num_leaves = 4, 
                learning_rate = 1, 
                nrounds = 2, 
                objective = "binary")


## alternatively, you can use put in dense matrix, i.e. basic R-matrix 

print("Training lightgbm with basic dense Matrix")
bst <- lightgbm(data = as.matrix(train$data),
                label = train$label,
                num_leaves = 4, 
                learning_rate = 1, 
                nrounds = 2, 
                objective = "binary")

## Or, using the input from a lgb.Dataset objecet, which stores label, data and other meta datas needed for advanced features
print("Training lightgbm with lgb.Dataset")
dtrain <- lgb.Dataset(data = train$data, label = train$label)
bst <- lightgbm(data = dtrain, 
                num_leaves = 4, 
                learning_rate = 1, 
                nrounds = 2, 
                objective = "binary")

# Verbose = 0,1,2
print("Train lightgbm with verbose 0, no message")
bst <- lightgbm(data = dtrain,
                num_leaves = 4,
                learning_rate = 1,
                nrounds = 2,
                objective = "binary",
                verbose = 0)

print("Train lightgbm with verbose 1, print evaluation metric")
bst <- lightgbm(data = dtrain,
                num_leaves = 4,
                learning_rate = 1,
                nrounds = 2,
                nthread = 2,
                objective = "binary",
                verbose = 1)

print("Train lightgbm with verbose 2, also print information about tree")
bst <- lightgbm(data = dtrain,
                num_leaves = 4,
                learning_rate = 1,
                nrounds = 2,
                nthread = 2,
                objective = "binary",
                verbose = 2)


#--------------------Basic prediction using lightgbm--------------
# You can do prediction using the following line
# You can put in Matrix, sparseMatrix, or lgb.Dataset
pred <- predict(bst, test$data)
err <- mean(as.numeric(pred > 0.5) != test$label)
print(paste("test-error=", err))

table(pred,test$label)


#--------------------Save and load models-------------------------
# Save model to binary local file
lgb.save(bst, "lightgbm.model")

# Load binary model to R
bst2 <- lgb.load("lightgbm.model")
pred2 <- predict(bst2, test$data)

# pred2 should be identical to pred
print(paste("sum(abs(pred2-pred))=", sum(abs(pred2 - pred))))


#--------------------Advanced features ---------------------------
# To use advanced features, we need to put data in lgb.Dataset
dtrain <- lgb.Dataset(data = train$data, label = train$label, free_raw_data = FALSE)
dtest <- lgb.Dataset(data = test$data, label = test$label, free_raw_data = FALSE)

