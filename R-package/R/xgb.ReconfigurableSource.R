

#' @rdname xgb.reconfigurableSource
#' @export
xgb.reconfigurableSource <- function(data, folds, label=NULL, weights=NULL) {
  if(inherits(data, "xgb.DMatrix")) {
    stopifnot(is.null(label))
    handle <- .Call(XGReconfigurableSourceCreateFromDMatrix_R, data, folds);
    cnames <- colnames(data)
  } else if(inherits(data, "data.frame")) {
    handle <- .Call(XGReconfigurableSourceCreateFromDataFrame_R, folds, data, label, weights);

    cnames <- c()
    for(cname in colnames(data)) {
      col <- data[[cname]]

      if(class(col) == "factor") {
        cnames <- c(cnames, paste0(cname, levels(col)))
      } else {
        cnames <- c(cnames, cname)
      }
    }
  } else {
    stop("unknown input for make slices")
  }
  attributes(handle) <- list(.cnames=cnames,
                             nfold = length(folds),
                             class="xgb.reconfigurableSource")
  return(handle)
}