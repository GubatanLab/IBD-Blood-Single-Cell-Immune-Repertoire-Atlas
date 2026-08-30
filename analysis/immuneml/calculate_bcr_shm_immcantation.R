suppressPackageStartupMessages({
  library(shazam)
})

args <- commandArgs(trailingOnly = TRUE)

get_arg <- function(flag, default = NULL) {
  idx <- match(flag, args)
  if (is.na(idx) || idx == length(args)) {
    return(default)
  }
  args[[idx + 1]]
}

input_rds <- get_arg("--input-rds", "C:/path/to/private-immuneml-results/BCR/IBDBCRonly.rds")
aligned_airr <- get_arg("--aligned-airr", NA_character_)
output_dir <- get_arg("--output-dir", "chain_bcr_immuneml")

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
shm_output <- file.path(output_dir, "bcr_shm_rates_by_patientID.csv")
airr_output <- file.path(output_dir, "bcr_shm_immcantation_unaligned_airr.tsv")
status_output <- file.path(output_dir, "bcr_shm_immcantation_status.txt")

infer_locus <- function(v_call) {
  out <- sub("^(IG[HKL]).*$", "\\1", as.character(v_call))
  out[!grepl("^IG[HKL]$", out)] <- NA_character_
  out
}

build_unaligned_airr <- function(rds_path) {
  obj <- readRDS(rds_path)
  meta <- obj$meta
  data <- obj$data
  rows <- list()

  for (i in seq_len(nrow(meta))) {
    sample <- as.character(meta$Sample[[i]])
    if (!sample %in% names(data)) {
      next
    }
    tab <- data[[sample]]
    n <- nrow(tab)
    if (n == 0) {
      next
    }

    rows[[length(rows) + 1]] <- data.frame(
      sequence_id = paste0(sample, "_", seq_len(n)),
      Sample = sample,
      SampleID = as.character(meta$SampleID[[i]]),
      PatientID = as.character(meta$PatientID[[i]]),
      Diagnosis1 = as.character(meta$Diagnosis1[[i]]),
      sequence = as.character(tab$Sequence),
      v_call = as.character(tab$V.name),
      d_call = as.character(tab$D.name),
      j_call = as.character(tab$J.name),
      duplicate_count = suppressWarnings(as.numeric(tab$Clones)),
      locus = infer_locus(tab$V.name),
      stringsAsFactors = FALSE
    )
  }

  airr <- do.call(rbind, rows)
  airr$duplicate_count[is.na(airr$duplicate_count) | airr$duplicate_count < 1] <- 1
  airr
}

write_status <- function(lines) {
  writeLines(lines, status_output)
}

summarize_unavailable <- function(airr, reason) {
  patients <- split(airr, airr$PatientID)
  rows <- lapply(names(patients), function(patient_id) {
    x <- patients[[patient_id]]
    data.frame(
      PatientID = patient_id,
      SampleID = paste(sort(unique(x$SampleID)), collapse = ";"),
      Diagnosis1 = paste(sort(unique(x$Diagnosis1)), collapse = ";"),
      n_sequences = nrow(x),
      n_weighted_sequences = sum(x$duplicate_count, na.rm = TRUE),
      n_IGH = sum(x$locus == "IGH", na.rm = TRUE),
      n_IGK = sum(x$locus == "IGK", na.rm = TRUE),
      n_IGL = sum(x$locus == "IGL", na.rm = TRUE),
      shm_rate = NA_real_,
      shm_rate_weighted = NA_real_,
      status = reason,
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, rows)
}

calculate_from_aligned_airr <- function(path) {
  db <- read.delim(path, stringsAsFactors = FALSE, check.names = FALSE)
  required <- c("PatientID", "sequence_alignment", "duplicate_count")
  missing <- setdiff(required, colnames(db))
  if (length(missing) > 0) {
    stop("Aligned AIRR table is missing required columns: ", paste(missing, collapse = ", "))
  }

  germline_col <- if ("germline_alignment_d_mask" %in% colnames(db)) {
    "germline_alignment_d_mask"
  } else if ("germline_alignment" %in% colnames(db)) {
    "germline_alignment"
  } else {
    stop("Aligned AIRR table needs germline_alignment_d_mask or germline_alignment.")
  }

  db <- db[nchar(db$sequence_alignment) > 0 & nchar(db[[germline_col]]) > 0, ]
  db$duplicate_count <- suppressWarnings(as.numeric(db$duplicate_count))
  db$duplicate_count[is.na(db$duplicate_count) | db$duplicate_count < 1] <- 1

  mut <- shazam::observedMutations(
    db,
    sequenceColumn = "sequence_alignment",
    germlineColumn = germline_col,
    frequency = TRUE,
    combine = TRUE,
    nproc = 1
  )

  rate_col <- grep("^mu_freq|_mu_freq|mutation_freq", colnames(mut), value = TRUE)[1]
  if (is.na(rate_col)) {
    stop("Could not identify mutation frequency column from observedMutations output.")
  }

  patients <- split(mut, mut$PatientID)
  rows <- lapply(names(patients), function(patient_id) {
    x <- patients[[patient_id]]
    w <- suppressWarnings(as.numeric(x$duplicate_count))
    w[is.na(w) | w < 1] <- 1
    rate <- suppressWarnings(as.numeric(x[[rate_col]]))
    data.frame(
      PatientID = patient_id,
      SampleID = if ("SampleID" %in% colnames(x)) paste(sort(unique(x$SampleID)), collapse = ";") else NA_character_,
      Diagnosis1 = if ("Diagnosis1" %in% colnames(x)) paste(sort(unique(x$Diagnosis1)), collapse = ";") else NA_character_,
      n_sequences = nrow(x),
      n_weighted_sequences = sum(w),
      n_IGH = if ("locus" %in% colnames(x)) sum(x$locus == "IGH", na.rm = TRUE) else NA_integer_,
      n_IGK = if ("locus" %in% colnames(x)) sum(x$locus == "IGK", na.rm = TRUE) else NA_integer_,
      n_IGL = if ("locus" %in% colnames(x)) sum(x$locus == "IGL", na.rm = TRUE) else NA_integer_,
      shm_rate = mean(rate, na.rm = TRUE),
      shm_rate_weighted = weighted.mean(rate, w, na.rm = TRUE),
      status = paste0("calculated_with_shazam_observedMutations_rate_column=", rate_col),
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, rows)
}

airr <- build_unaligned_airr(input_rds)
write.table(airr, airr_output, sep = "\t", quote = FALSE, row.names = FALSE)

if (!is.na(aligned_airr) && file.exists(aligned_airr)) {
  result <- calculate_from_aligned_airr(aligned_airr)
  write.csv(result, shm_output, row.names = FALSE)
  write_status(c(
    paste("Immcantation package: shazam", as.character(packageVersion("shazam"))),
    paste("Calculated SHM rates from aligned AIRR:", aligned_airr),
    paste("Output:", shm_output)
  ))
} else {
  reason <- "unavailable_missing_imgt_germline_alignment_required_by_Immcantation_shazam_observedMutations"
  result <- summarize_unavailable(airr, reason)
  write.csv(result, shm_output, row.names = FALSE)
  write_status(c(
    paste("Immcantation package: shazam", as.character(packageVersion("shazam"))),
    "SHM rates were not calculated because the current BCR files lack sequence_alignment and germline_alignment_d_mask/germline_alignment.",
    "The generated AIRR-like table is unaligned and ready for IgBLAST/Change-O preprocessing:",
    airr_output,
    "After IgBLAST/Change-O/CreateGermlines, rerun this script with --aligned-airr <aligned_airr.tsv>."
  ))
}

cat("Wrote", shm_output, "\n")
cat("Wrote", airr_output, "\n")
cat("Wrote", status_output, "\n")
