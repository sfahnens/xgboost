require(xgboost)
# load in the agaricus dataset
data(agaricus.train, package='xgboost')
data(agaricus.test, package='xgboost')
dtrain <- xgb.DMatrix(agaricus.train$data, label = agaricus.train$label)
dtest <- xgb.DMatrix(agaricus.test$data, label = agaricus.test$label)

nround <- 10
param <- list(max_depth=2, eta=1, silent=1, nthread=2, objective='binary:logistic')

cat('running cross validation\n')
# do cross validation, this will print result out as
# [iteration]  metric_name:mean_value+std_value
# std_value is standard deviation of the metric
xgb.cv(param, dtrain, nround, nfold=5, metrics={'error'})
xgb.cv_xx(param, dtrain, nround, nfold=5, metrics={'error'})


# dall <- xgboost:::xgb.get.DMatrix(dtrain, NULL, NA)
# bst_folds <- lapply(seq_along(folds), function(k) {
#   dtest  <- slice(dall, folds[[k]])
#   dtrain <- slice(dall, unlist(folds[-k]))
#   handle <- xgboost:::xgb.Booster.handle(param, list(dtrain, dtest))
#   list(dtrain = dtrain, bst = handle, watchlist = list(train = dtrain, test=dtest), index = folds[[k]])
# })
# 
# attach(xgboost)

folds <- xgboost:::generate.cv.folds(5, length(agaricus.train$label), TRUE, agaricus.train$label, list(objective='binary:logistic'))
slices <- xgb.cv2.makeSlices(dtrain, folds)

data_pairs <- eval(lapply(1:5, function(i) {
  cat(sprintf('make data pairs %s\n', i))
  return(list(
    train = xgb.cv2.slicesToMatrix(slices, as.integer(1:5)[-i]),
    test = xgb.cv2.slicesToMatrix(slices, as.integer(i))
  ))
}))

cat('running cv2\n')
xgb.cv2(params = param, data_pairs = data_pairs, nround, metrics={'error'})


## ---
xgb.cv_xx <- function(params=list(), data, nrounds, nfold, label = NULL, missing = NA,
                   prediction = FALSE, showsd = TRUE, metrics=list(),
                   obj = NULL, feval = NULL, stratified = TRUE, folds = NULL, 
                   verbose = TRUE, print_every_n=1L,
                   early_stopping_rounds = NULL, maximize = NULL, callbacks = list(), ...) {
  
  check.deprecation(...)
  
  params <- check.booster.params(params, ...)
  # TODO: should we deprecate the redundant 'metrics' parameter?
  for (m in metrics)
    params <- c(params, list("eval_metric" = m))
  
  check.custom.obj()
  check.custom.eval()
  
  #if (is.null(params[['eval_metric']]) && is.null(feval))
  #  stop("Either 'eval_metric' or 'feval' must be provided for CV")
  
  # Check the labels
  if ( (inherits(data, 'xgb.DMatrix') && is.null(getinfo(data, 'label'))) ||
       (!inherits(data, 'xgb.DMatrix') && is.null(label)))
    stop("Labels must be provided for CV either through xgb.DMatrix, or through 'label=' when 'data' is matrix")
  
  # CV folds
  if(!is.null(folds)) {
    if(!is.list(folds) || length(folds) < 2)
      stop("'folds' must be a list with 2 or more elements that are vectors of indices for each CV-fold")
    nfold <- length(folds)
  } else {
    if (nfold <= 1)
      stop("'nfold' must be > 1")
    folds <- generate.cv.folds(nfold, nrow(data), stratified, label, params)
  }
  
  # Potential TODO: sequential CV
  #if (strategy == 'sequential')
  #  stop('Sequential CV strategy is not yet implemented')
  
  # verbosity & evaluation printing callback:
  params <- c(params, list(silent = 1))
  print_every_n <- max( as.integer(print_every_n), 1L)
  if (!has.callbacks(callbacks, 'cb.print.evaluation') && verbose) {
    callbacks <- add.cb(callbacks, cb.print.evaluation(print_every_n, showsd = showsd))
  }
  # evaluation log callback: always is on in CV
  evaluation_log <- list()
  if (!has.callbacks(callbacks, 'cb.evaluation.log')) {
    callbacks <- add.cb(callbacks, cb.evaluation.log())
  }
  # Early stopping callback
  stop_condition <- FALSE
  if (!is.null(early_stopping_rounds) &&
      !has.callbacks(callbacks, 'cb.early.stop')) {
    callbacks <- add.cb(callbacks, cb.early.stop(early_stopping_rounds, 
                                                 maximize = maximize, verbose = verbose))
  }
  # CV-predictions callback
  if (prediction &&
      !has.callbacks(callbacks, 'cb.cv.predict')) {
    callbacks <- add.cb(callbacks, cb.cv.predict(save_models = FALSE))
  }
  # Sort the callbacks into categories
  cb <- categorize.callbacks(callbacks)
  
  
  # create the booster-folds
  dall <- xgb.get.DMatrix(data, label, missing)
  bst_folds <- lapply(seq_along(folds), function(k) {
    dtest  <- slice(dall, folds[[k]])
    
    dtrain <- slice(dall, unlist(folds[-k]))
    handle <- xgb.Booster.handle(params, list(dtrain, dtest))
    list(dtrain = dtrain, bst = handle, watchlist = list(train = dtrain, test=dtest), index = folds[[k]])
  })
  # a "basket" to collect some results from callbacks
  basket <- list()
  
  # extract parameters that can affect the relationship b/w #trees and #iterations
  num_class <- max(as.numeric(NVL(params[['num_class']], 1)), 1)
  num_parallel_tree <- max(as.numeric(NVL(params[['num_parallel_tree']], 1)), 1)
  
  # those are fixed for CV (no training continuation)
  begin_iteration <- 1
  end_iteration <- nrounds
  
  # synchronous CV boosting: run CV folds' models within each iteration
  for (iteration in begin_iteration:end_iteration) {
    
    for (f in cb$pre_iter) f()
    
    msg <- lapply(bst_folds, function(fd) {
      xgb.iter.update(fd$bst, fd$dtrain, iteration - 1, obj)
      xgb.iter.eval(fd$bst, fd$watchlist, iteration - 1, feval)
    })
    msg <- simplify2array(msg)
    bst_evaluation <- rowMeans(msg)
    bst_evaluation_err <- sqrt(rowMeans(msg^2) - bst_evaluation^2)
    
    for (f in cb$post_iter) f()
    
    if (stop_condition) break
  }
  for (f in cb$finalize) f(finalize = TRUE)
  
  # the CV result
  ret <- list(
    call = match.call(),
    params = params,
    callbacks = callbacks,
    evaluation_log = evaluation_log,
    niter = end_iteration,
    folds = folds
  )
  ret <- c(ret, basket)
  
  class(ret) <- 'xgb.cv.synchronous'
  invisible(ret)
}
environment(xgb.cv_xx) <- asNamespace('xgboost')



