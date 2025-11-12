import os
import re
from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage
from rpy2.robjects.vectors import StrVector
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri

pandas2ri.activate()

class MetaFeatureExtractor:
    """
    Extract meta-features from regression datasets using R's ECoL package.

    Example:
    --------
    from meta_ir.features import MetaFeatureExtractor
    extractor = MetaFeatureExtractor(["data/machine_CPU.csv"])
    meta_df = extractor.meta_feature_extraction()
    """

    def __init__(self, data_sets):
        if not isinstance(data_sets, list):
            raise ValueError("data_sets must be a list of CSV file paths.")
        self.data_sets = data_sets

    def meta_feature_extraction(self):
        """Runs the ECoL meta-feature extraction pipeline via R."""
        if not self.data_sets:
            raise ValueError("No dataset paths provided to MetaFeatureExtractor.")

        r_data_sets = StrVector(self.data_sets)

        r_code = '''
        ecol <- function(data_sets){
            if (!requireNamespace("ECoL", quietly = TRUE)) install.packages("ECoL", repos="https://cloud.r-project.org")
            library(ECoL)
            result_list <- list()
            for (i in data_sets) {
                message("Processing: ", i)
                ds <- read.csv(i)
                ds_n <- basename(i)

                l <- linearity(target~ ., ds, summary=c("mean", "min", "max", "sd"))
                d <- dimensionality(target~ ., ds, summary=c("mean", "min", "max", "sd"))
                c <- correlation(target~ ., ds, summary=c("mean", "min", "max", "sd"))
                s <- smoothness(target~ ., ds, summary=c("mean", "min", "max", "sd"))

                n_row <- nrow(ds)
                n_col <- ncol(ds) - 1

                myList <- list(
                    n_row = n_row,
                    n_col = n_col,
                    l = l,
                    d = d,
                    c = c,
                    s = s
                )

                temp_df <- as.data.frame(t(unlist(myList)))
                names(temp_df) <- gsub("^([ldcs]\\\\.)+", "", names(temp_df), perl=TRUE)
                result_list[[length(result_list) + 1]] <- temp_df
            }
            result_df <- do.call(rbind, result_list)
            return(result_df)
        }
        '''

        powerpack = SignatureTranslatedAnonymousPackage(r_code, "powerpack")
        r_df = powerpack.ecol(r_data_sets)
        df = pandas2ri.rpy2py(r_df)

        # Python-side cleanup (backup cleaning in case R names are inconsistent)
        df.columns = [re.sub(r'^[ldcs]\.', '', c) for c in df.columns]

        return df


def extract_and_save(input_path, output_path):
    """Convenience wrapper to extract and save meta-features from a CSV."""
    extractor = MetaFeatureExtractor([input_path])
    meta_df = extractor.meta_feature_extraction()
    meta_df.to_csv(output_path, index=False)
    print(f"Meta-features saved to: {output_path}")
    return meta_df
