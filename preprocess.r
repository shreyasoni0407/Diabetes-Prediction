# Load necessary libraries
library(missForest)

# Read the dataset
data <- read.csv('project_dataset_5K.csv')

num_columns <- ncol(data)

# Print the number of columns
print(num_columns)
# Calculate the percentage of missing values in each column
missing_percentage <- colMeans(is.na(data)) * 100

# Set a threshold for the percentage of missing values (e.g., 70%)
threshold <- 70

# Identify columns with missing values exceeding the threshold
columns_to_drop <- names(missing_percentage[missing_percentage > threshold])

# Remove columns with high percentage of missing values
data <- data[, !(names(data) %in% columns_to_drop)]

# Check for columns with remaining missing values
missing_columns <- colnames(data)[colSums(is.na(data)) > 0]

if (length(missing_columns) > 0) {
  # Impute missing values using missForest
  imputation_result <- missForest(data[,missing_columns])
  data[,missing_columns] <- imputation_result$ximp
  
  if (sum(is.na(data)) > 0) {
    print("Warning: Missing values still exist after imputation.")
  } else {
    print("Imputation successful: No missing values remaining.")
  }
} else {
  print("No missing values remaining after removing columns with high missingness.")
}

# Summary of the dataset
write.csv(data,"temp.csv")


# Calculate the correlation matrix of the remaining columns
correlation_matrix <- cor(data[, -ncol(data)])

correlation_threshold <- 0.8
highly_correlated_pairs <- which(abs(correlation_matrix) > correlation_threshold & correlation_matrix != 1, arr.ind = TRUE)

columns_to_remove <- character(0)

for (i in 1:nrow(highly_correlated_pairs)) {
  pair <- highly_correlated_pairs[i, ]
  col1 <- colnames(data)[pair[1]]
  col2 <- colnames(data)[pair[2]]
  
  if (!(col1 %in% columns_to_remove) & !(col2 %in% columns_to_remove)) {
    columns_to_remove <- c(columns_to_remove, col1)
  }
}

data <- data[, !(names(data) %in% columns_to_remove)]

if (length(columns_to_remove) > 0) {
  print(paste("Removed highly correlated columns:", paste(columns_to_remove, collapse = ", ")))
} else {
  print("No highly correlated columns found.")
}

write.csv(data, "temp.csv")


# Calculate the variance of each column
column_variances <- apply(data, 2, var)

# Set a threshold for minimum intra-variance
variance_threshold <- 0.3 # You can adjust this threshold as needed

# Identify columns with variance below the threshold
low_variance_columns <- names(column_variances[column_variances < variance_threshold])

# Remove columns with low variance
data <- data[, !(names(data) %in% low_variance_columns)]

# Print columns removed
if (length(low_variance_columns) > 0) {
  print(paste("Removed columns with low variance:", paste(low_variance_columns, collapse = ", ")))
} else {
  print("No columns found with low variance.")
}


for (col in colnames(data)[-ncol(data)]) {
  median_value <- median(data[[col]], na.rm = TRUE)
  data[[col]][data[[col]] > median_value * 3 | data[[col]] < median_value * 0.33] <- median_value
}

write.csv(data, "final_cleaned_dataset.csv")


