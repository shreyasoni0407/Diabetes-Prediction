library(caret)

data <- read.csv('project_dataset_5K.csv')

na_percentage <- colMeans(is.na(data))

columns_to_remove <- which(na_percentage > 0.5)

data_cleaned <- data[, -columns_to_remove]

for (i in 1:ncol(data_cleaned)) {
  if (sum(is.na(data_cleaned[, i])) > 0) {
    data_cleaned[is.na(data_cleaned[, i]), i] <- mean(data_cleaned[, i], na.rm = TRUE)
  }
}

# Calculate the variance for each column
variances <- apply(data_cleaned, 2, var)

# Identify columns with variance less than 0.3
columns_with_low_variance <- which(variances < 0.3)

# Remove these columns from the dataframe
data_final <- data_cleaned[, -columns_with_low_variance]

correlation_matrix <- cor(data_final[, -ncol(data_final)], use = "pairwise.complete.obs")
diag(correlation_matrix) <- 0
highly_correlated_columns <- c()
for (i in 1:(ncol(correlation_matrix)-1)) {
  for (j in (i+1):ncol(correlation_matrix)) {
    if (abs(correlation_matrix[i, j]) > 0.75) {
      highly_correlated_columns <- c(highly_correlated_columns, j)
    }
  }
}
highly_correlated_columns <- unique(highly_correlated_columns)
data_final_cleaned <- data_final[, -highly_correlated_columns, drop = FALSE]
write.csv(data_final_cleaned, 'final_cleaned_dataset.csv')
