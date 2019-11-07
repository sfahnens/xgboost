library(data.table)

dummy.data.table <- function(types = NULL,
                             nrow = 100,
                             ncol = 10) {
  col.double <- function() {
    return(rnorm(nrow))
  }

  col.integer <- function() {
    len <- sample(2:nrow, 1)
    vec <- sample(1:len, nrow, replace = TRUE)
    return(vec)
  }

  col.factor <- function() {
    return(as.factor(paste0("F", col.integer())))
  }

  # col.logic <- function() {
  #   vec <- col.integer()
  #   return(as.logical(vec > sample(min(vec):max(vec), 1)))
  # }

  funs <- c(d = col.double,
            i = col.integer,
            f = col.factor)
  if (is.null(types)) {
    funs <-
      c(col.integer, funs[sample(1:length(funs), ncol, replace = TRUE)])
  } else {
    funs <- funs[strsplit(types, "")[[1]]]
  }

  cols <- lapply(funs, function(fun)
    fun())
  cols <- unname(cols)
  return(setDT(cols))
}

dt <- dummy.data.table(nrow = 10000, ncol = 20)
# dt <- dummy.data.table("if", 10)
label <- runif(nrow(dt))

contr.onehot <- function(lvls, sparse) {
  return(contr.treatment(length(lvls), sparse=sparse, contrasts = FALSE))
}

fac.cols <- colnames(dt)[sapply(colnames(dt), function(col) { return(class(dt[[col]]) == "factor") })]
contrasts <- rep("contr.onehot", length(fac.cols))
names(contrasts) <- fac.cols
contrasts <- as.list(contrasts)

f <- reformulate(termlabels = c(colnames(dt), 0))
mm <-
  Matrix::sparse.model.matrix(
    f,
    dt,
    row.names = FALSE,
    contrasts = contrasts,
    verbose = TRUE
  )
ma <- xgboost::xgb.DMatrix(mm, label = label)

# copy directly
sb <- xgboost::xgb.reconfigurableSource(ma, list(1:nrow(mm)))
mb <- xgboost::xgb.DMatrix(sb, active_folds=1)

# construct fresh
sc <- xgboost::xgb.reconfigurableSource(dt, list(1:nrow(dt)), label=label)
mc <- xgboost::xgb.DMatrix(sb, active_folds=1)

xgboost::xgb.diff.DMatrix(ma, mb)
xgboost::xgb.diff.DMatrix(ma, mc)

# ---

folds.b <- xgboost:::generate.cv.folds(3,
                                         length(label),
                                         TRUE,
                                         label,
                                         list(objective='reg:linear'))

s2.a <- xgboost::xgb.reconfigurableSource(ma, folds.b)
s2.b <- xgboost::xgb.reconfigurableSource(ma, folds.b)

m2.a <- xgboost::xgb.DMatrix(s2.a, active_folds=as.integer(1, 3))
m2.b <- xgboost::xgb.DMatrix(s2.b, active_folds=as.integer(1, 3))
xgboost::xgb.diff.DMatrix(m2.a, m2.b)

