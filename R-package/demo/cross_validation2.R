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
# xgb.cv_xx(param, dtrain, nround, nfold=5, metrics={'error'})

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
xgb.cv2b <- function(params=list(), data_pairs, nrounds, label = NULL, missing = NA,
                   prediction = FALSE, showsd = TRUE, metrics=list(),
                   obj = NULL, feval = NULL, verbose = TRUE, print_every_n=1L,
                   early_stopping_rounds = NULL, maximize = NULL, callbacks = list(), ...) {

  check.deprecation(...)

  params <- check.booster.params(params, ...)
  # TODO: should we deprecate the redundant 'metrics' parameter?
  for (m in metrics)
    params <- c(params, list("eval_metric" = m))

  check.custom.obj()
  check.custom.eval()

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
  bst_folds <- lapply(seq_along(data_pairs), function(k) {
    dtest  <- data_pairs[[k]]$test
    dtrain  <- data_pairs[[k]]$train
    handle <- xgb.Booster.handle(params, list(dtrain, dtest))
    list(dtrain = dtrain, bst = handle, watchlist = list(train = dtrain, test=dtest), dtest=dtest)
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

    # cat(sprintf("%s\n", iteration))
    msg <- lapply(bst_folds, function(fd) {
      train <- slice.xgb.DMatrix(fd$dtrain, 1:dim(fd$dtrain)[1])
      test <- slice.xgb.DMatrix(fd$dtest, 1:dim(fd$dtest)[1])

      watchlist <- list(train=train, test=test)

      # cat(sprintf("update fold\n"))
      # xgb.iter.update(fd$bst, fd$dtrain, iteration - 1, obj)
      xgb.iter.update(fd$bst, train, iteration - 1, obj)
      # cat(sprintf("eval fold\n"))
      xgb.iter.eval(fd$bst, watchlist, iteration - 1, feval)
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
    niter = end_iteration
  )
  ret <- c(ret, basket)

  class(ret) <- 'xgb.cv2.synchronous'
  invisible(ret)
}
environment(xgb.cv2b) <- asNamespace('xgboost')


cat('running cv2b\n')
xgb.cv2b(params = param, data_pairs = data_pairs, nround, metrics={'error'})

