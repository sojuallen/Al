
dTrain <- data.frame(x = c('a','b','b','c','a','b','c','c','b','b','c'), 
                     y = c(1,2,1,2,1,2,1,1,2,1,1))
summary(dTrain)
lm_md <- lm(y~x, data = dTrain)
lm_md
summary(lm_md)

# Bad example - do not use data.matrix below
# you might get away with it sometime, but it is not a good practice in general
data.matrix(dTrain)
# Reason: model.matrix() does not sotre its one-hot plan in a convenient manner. It is only pulling the "contrasts" attribute plus
# examining the column names of the first encoding, but the levels identified are not conveniently represented. 
# Basically one can not ensure that the the same formula aplied to two different data sets are using the same encoding!!!

dTrain <- data.frame(x= c('a','b','c'), 
                     stringsAsFactors = FALSE)
encTrain <- stats::model.matrix(~x, dTrain)
print(encTrain)

dTest <- data.frame(x=c('b','c'),
                    stringsAsFactors = FALSE)
stats::model.matrix(~x,dTest)

# main goal - encode the variable and store its encoding plan in a conveniently re-usable form. 
# Beware - many other (python) encoding was one-off ported and fails to save the encoding plan for future use. 

# For example when using a machine learning implementation that is not completely R - centric:
# eg. xgboost. that requires data to be already encoded as a numeric matrix instead of a data.frame.


# set up example data set
library(titanic)
data(titanic_train)
str(titanic_train)
summary(titanic_train)

outcome <- 'Survived'
target <- 1
shouldBeCategorical <- c('PassengerId', 'Pclass', 'Parch')

for (v in shouldBeCategorical) {
  titanic_train[[v]] <- as.factor(titanic_train[[v]])
}
tooDetailed <- c("Ticket","Cabin","Name","PassengerId")
vars <- setdiff(colnames(titanic_train), c(outcome, tooDetailed))
dTrain <- titanic_train

summary(dTrain)

# design cross-validated modeling experiment: 
library(xgboost)
library(sigr)
library(WVPlots)
library(vtreat)

set.seed(100)
crossValPlan <- vtreat::kWayStratifiedY(nRows = nrow(dTrain), # vtreat - a statistically sound 'data.frame' Processor/Conditioner
                                        nSplits = 10, # Number of groups to split into
                                        dframe = dTrain, # original data frame
                                        y = dTrain[[outcome]])

evaluateModelingProcedure <- function(xMatrix, outcomeV, crossValPlan){
  preds <- rep(NA_real_, nrow(xMatrix))
  for (ci in crossValPlan){
    nrounds <- 1000
    cv <- xgb.cv( # Cross validation function of xgboost
      data = xMatrix[ci$train, ], # can be xgb.Dmatrix, matrix, or dgCMatrix
      label = outcomeV[ci$train],  # vector of response values - should be provided only if data is an R-matrix
      objective = 'binary:logistic', # customised objective function, returns gradient and second order gradient with given prediction and dtrain
      nrounds = nrounds, # max number of iterations
      verbose = 0, # boolean, TRUE or FALSE - print the statistics during the process
      nfold = 5 # the original dataset is randomly partitioned into nfold equal size subsamples
    )
    # nrounds <- which.min(cv$evaluation_lg$test_rmse_mean) # Regression
    # nrounds <- which.min(cv$evaluation_log$test_error_mean) # classification
    model <- xgboost(data = xMatrix[ci$train,],
                     label = outcomeV[ci$train],
                     objective = 'binary:logistic',
                     nrounds = nrounds,
                     verbose = 0)
    preds[ci$app]<- predict(model, xMatrix[ci$app,])
  }
  preds
}

# prefered way to encode data is to use the 'vtreat' package int the "no variable mode" 
set.seed(2000)
tplan <- vtreat::designTreatmentsZ( # design variable treatment with no outcome variable
                                    dframe = dTrain, # data frame to learn treatments from (training data)
                                    varlist = vars, # names of columns to treat (effective variables)
                                    minFraction = 0, # optional minimum frequency a categorical level must have to be converted to an indicator column.
                                    verbose = FALSE # Boolean, TRUE of FALSE - print the statistics during the process
                                    )
# restrict to common varaibles types 
sf <- tplan$scoreFrame
newvars <- sf$varName[sf$code %in% c("lev","clean","isBAD")]
trainVtreat <- as.matrix(vtreat::prepare(tplan, 
                                         dTrain, 
                                         varRestriction = newvars))

print(dim(trainVtreat))
print(colnames(trainVtreat))

dTrain$predVtreatZ <- evaluateModelingProcedure(trainVtreat, 
                                                dTrain[[outcome]] == target, 
                                                crossValPlan
                                                )

sigr::permTestAUC(d = dTrain,
                  modelName = 'predVtreatZ',
                  yName = outcome, 
                  yTarget = target)

WVPlots::ROCPlot(frame = dTrain,
                 xvar = "predVtreatZ",
                 truthVar = outcome, 
                 truthTarget = target, 
                 title = "vtreat encoder performance",
                 estimate_sig = TRUE)


###################################################################################
###################################################################################

# using caret package - ths also applies an encoding functionality properly split between training (caret::dummyVars()) and application (predict())

library(caret)
set.seed(2000)
f <- paste('~', paste(vars, collapse = ' + '))
encoder <- caret::dummyVars(as.formula(f), dTrain)
trainCaret <- predict(encoder, dTrain)
print(dim(trainCaret))
dTrain$predCaret <- evaluateModelingProcedure(trainCaret,
                                              dTrain[[outcome]]==target,
                                              crossValPlan)
sigr::permTestAUC(dTrain, 
                  'predCaret',
                  outcome, target)
WVPlots::ROCPlot(dTrain, 
                 'predCaret', 
                 outcome, target, 
                 'caret encoder performance',
                 estimate_sig = TRUE)

