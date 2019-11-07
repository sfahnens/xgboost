require(xgboost)
# load in the agaricus dataset
data(agaricus.train, package='xgboost')
data(agaricus.test, package='xgboost')
dtrain <- xgb.DMatrix(agaricus.train$data, label = agaricus.train$label)
dtest <- xgb.DMatrix(agaricus.test$data, label = agaricus.test$label)

nround <- 20
param <- list(max_depth=2, eta=.3, silent=1, nthread=2, objective='binary:logistic')

folds <- xgboost:::generate.cv.folds(5, length(agaricus.train$label), TRUE, agaricus.train$label, list(objective='binary:logistic'))

cat('running cross validation / default\n')
xgb.cv(param, dtrain, nround, folds = folds, metrics={'error'})

cat('running cross validation / source\n')
source <- xgboost::xgb.reconfigurableSource(dtrain, folds)
xgb.cv.source(param, source, nround, metrics={'error'})

