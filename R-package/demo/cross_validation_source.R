require(xgboost)
# load in the agaricus dataset
data(agaricus.train, package='xgboost')
data(agaricus.test, package='xgboost')

set.seed(12456)
# sx <- sample(1:nrow(agaricus.train$data), 1000)
sx <- 1:nrow(agaricus.train$data)

dtrain <- xgboost::xgb.DMatrix(agaricus.train$data[sx, ], label = agaricus.train$label[sx])

nround <- 10
param <- list(max_depth=2, eta=.3, silent=1, nthread=2, objective='binary:logistic')
# param <- list(max_depth=1, eta=.3, silent=1, nthread=1, objective='binary:logistic', tree_method="hist")

set.seed(42)
folds <- xgboost:::generate.cv.folds(5, length(agaricus.train$label[sx]),
                                     TRUE, agaricus.train$label[sx], list(objective='binary:logistic'))

cat('running cross validation / default\n')
xgboost::xgb.cv(param, dtrain, nround, folds = folds, metrics={'error'})

cat('running cross validation / source\n')
source <- xgboost::xgb.reconfigurableSource(dtrain, folds)
xgboost::xgb.cv.source(param, source, nround, metrics={'error'})

cat('running learner source\n')
dtrain <- xgboost::xgb.DMatrix(source, active_folds=1:attr(source, "nfold"))
xgboost::xgb.train(
  params = param,
  data = dtrain,
  nrounds = nround,
  watchlist = list(train = dtrain),
  verbose = 1
)

# ---- repeated cv with from single same source

nfold <- 5
nrep <- 3
nsub <- 10

set.seed(42)
folds_rep <- xgboost:::generate.cv.folds(nfold*nrep*nsub, length(agaricus.train$label[sx]),
                                         TRUE, agaricus.train$label[sx], list(objective='binary:logistic'))
source_rep <- xgboost::xgb.reconfigurableSource(dtrain, folds_rep)

# rep <- 1
for(rep in 1:nrep) {
  cat('## running repeated cross validation:', rep,'\n')

  set.seed(rep*42)
  fold_idx <- sample.int(length(folds_rep))
  fold_groups <- split(fold_idx, 1:nfold)

  cat('running repeated cross validation / default\n')
  folds <- lapply(fold_groups, function(g) unlist(folds_rep[g]))
  xgboost::xgb.cv(param, dtrain, nround, folds = folds, metrics={'error'})

  cat('running repeated cross validation / source\n')
  xgboost::xgb.cv.source(param, source_rep, nround,
                         fold_groups=fold_groups, metrics={'error'})

  cat('\n')
}
