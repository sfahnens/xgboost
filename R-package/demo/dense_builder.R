library(xgboost)
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

dt <- dummy.data.table(nrow = 10000000, ncol = 20)
# dt <- dummy.data.table("if", 10)

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

label <- runif(nrow(mm))
a <- xgb.DMatrix(mm, label = label)

folds <- list(1:nrow(mm))

slices <- xgb.cv2.makeSlices(folds, a)
b <- xgb.cv2.slicesToMatrix(slices, as.integer(1))

xgb.diff.DMatrix(a, b)

system.time({
  slices2 <- xgb.cv2.makeSlices(folds, dt, label=label)
  c <- xgb.cv2.slicesToMatrix(slices2, as.integer(1))
})

xgb.diff.DMatrix(a, c)
