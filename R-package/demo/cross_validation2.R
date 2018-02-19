require(xgboost)
# load in the agaricus dataset
data(agaricus.train, package='xgboost')
data(agaricus.test, package='xgboost')
dtrain <- xgb.DMatrix(agaricus.train$data, label = agaricus.train$label)
dtest <- xgb.DMatrix(agaricus.test$data, label = agaricus.test$label)

nround <- 2
param <- list(max_depth=2, eta=1, silent=1, nthread=2, objective='binary:logistic')

cat('running cross validation\n')
# do cross validation, this will print result out as
# [iteration]  metric_name:mean_value+std_value
# std_value is standard deviation of the metric
xgb.cv(param, dtrain, nround, nfold=5, metrics={'error'})

folds <- xgboost:::generate.cv.folds(5, length(agaricus.train$label), TRUE, agaricus.train$label, list(objective='binary:logistic'))
slices <- xgb.cv2.makeSlices(dtrain, folds)

data_pairs <- lapply(1:5, function(i) {
  return(list(
    train = xgb.cv2.slicesToMatrix(slices, as.integer(1:5)[-i]),
    test = xgb.cv2.slicesToMatrix(slices, as.integer(i))
  ))
})

xgb.cv2(params = param, data_pairs = data_pairs, nround, metrics={'error'})
