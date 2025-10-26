library(MASS) 
library(factoextra)
library(rsample)
library(ROSE)
library(smotefamily)
library(RWeka)
library(caret)
library(e1071)
library(class)
library(naivebayes)
library(randomForest)
library(mclust)
library(pROC)
library(Boruta)
library(glmnet)
data <- read.csv('final_cleaned_dataset.csv')
data$Class<- factor(data$Class)
str(data)
set.seed(101)
split <- initial_split(data, prop = 0.80, strata = Class)
tr <- training(split)
ts <- testing(split)
barplot(table(tr$Class))

#########################------------------sampling----------------------------###################################################

over_sampled_data <- ovun.sample(Class ~ ., data = tr , method = "over")$data
barplot(table(over_sampled_data$Class))

under_sampled_data <- ovun.sample(Class ~ ., data = tr , method = "under")$data
barplot(table(under_sampled_data$Class))

#########################-------------------------Feature Selection-------------------------###################################################

#-----------------------------------Boruta for oversample---------------------------------------

boruta_result <- Boruta(Class ~ ., data = over_sampled_data, doTrace = 2) 
boruta_result
selected_features <- getSelectedAttributes(boruta_result, withTentative = TRUE)
length(selected_features)
Boruta_oversample <- over_sampled_data[, selected_features]
str(Boruta_oversample)
Boruta_oversample$Class <- over_sampled_data$Class
num_columns <- ncol(Boruta_oversample)

# Print the number of columns
print(num_columns)

#----------------------------------Boruta for undersample------------------------------------------------------

boruta_result <- Boruta(Class ~ ., data = under_sampled_data, doTrace = 2) 
boruta_result
selected_features <- getSelectedAttributes(boruta_result, withTentative = TRUE)
length(selected_features)
Boruta_undersample <- under_sampled_data[, selected_features]
str(Boruta_undersample)
Boruta_undersample$Class <- under_sampled_data$Class

#-----------------------------elastic_Net for oversample-----------------
x_over <- model.matrix(Class ~ . - 1, data = over_sampled_data)
y_over <- over_sampled_data$Class
alpha <- 0.5 
cv_fit_over <- cv.glmnet(x_over, y_over, alpha=alpha, family="binomial")
lambda_min_over <- cv_fit_over$lambda.min
fit_over <- glmnet(x_over, y_over, alpha=alpha, lambda=lambda_min_over, family="binomial")
coef_over <- coef(fit_over, s=lambda_min_over)
selected_features_over <- rownames(coef_over)[coef_over[,1] != 0]
print(selected_features_over)
selected_features_under_excluding_first <- selected_features_over[-1]
ElasticNet_oversapmled <- over_sampled_data[, c(selected_features_under_excluding_first, "Class")]
str(ElasticNet_oversapmled)

#-----------------------------elastic net for undersample-------------------------------

x_over <- model.matrix(Class ~ . - 1, data = under_sampled_data)
y_over <- under_sampled_data$Class
alpha <- 0.5 
cv_fit_over <- cv.glmnet(x_over, y_over, alpha=alpha, family="binomial")
lambda_min_over <- cv_fit_over$lambda.min
fit_over <- glmnet(x_over, y_over, alpha=alpha, lambda=lambda_min_over, family="binomial")
coef_over <- coef(fit_over, s=lambda_min_over)
selected_features_over <- rownames(coef_over)[coef_over[,1] != 0]
print(selected_features_over)
selected_features_under_excluding_first <- selected_features_over[-1]
ElasticNet_Undersapmled <- under_sampled_data[, c(selected_features_under_excluding_first, "Class")]
str(ElasticNet_Undersapmled)

#--------------------------------lasso for oversampled data------------------------------

x_over <- model.matrix(Class ~ . - 1, data = over_sampled_data)
y_over <- over_sampled_data$Class
set.seed(101)
cv_lasso_over <- cv.glmnet(x_over, y_over, alpha = 1, family = "binomial")

coef_lasso_over <- coef(cv_lasso_over, s = "lambda.min")
selected_features_over <- rownames(coef_lasso_over)[coef_lasso_over[,1] != 0]
selected_features_over
selected_features_over_excluding_first <- selected_features_over[-1]
Lasso_Oversampled <- over_sampled_data[, c(selected_features_over_excluding_first, "Class")]
str(Lasso_Oversampled)



#----------------------------lasso for undersampled data----------------------------------
x_over <- model.matrix(Class ~ . - 1, data = under_sampled_data)
y_over <- under_sampled_data$Class
set.seed(101)
cv_lasso_under <- cv.glmnet(x_over, y_over, alpha = 1, family = "binomial")
coef_lasso_under <- coef(cv_lasso_under, s = "lambda.min")
selected_features_under <- rownames(coef_lasso_under)[coef_lasso_under[,1] != 0]
selected_features_under
selected_features_under_excluding_first <- selected_features_under[-1]
Lasso_Undersampled <- under_sampled_data[, c(selected_features_under_excluding_first, "Class")]
str(Lasso_Undersampled)



#########################################-----------------Classification-------------------#############################################################


#################---------BORUTA-------------------###################################

# ------------------ Boruta for lda over--------------
# Load necessary libraries
library(MASS) # For lda
library(pROC) # For ROC analysis

lda_model <- lda(Class ~ ., data = Boruta_oversample)
test_predictions <- predict(lda_model, newdata = ts)
conf_matrix <- table(test_predictions$class, ts$Class)
print(conf_matrix)
accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
cat("Overall Accuracy: ", round(accuracy * 100, 3), "%\n")
predicted_probabilities <- predict(lda_model, newdata = ts, type = "response")$posterior[,2]

roc_result <- roc(ts$Class, predicted_probabilities)
print(roc_result)

plot(roc_result, main="ROC Curve")


#-----------------Boruta for lda under----------------
lda_model <- lda(Class ~ ., data = Boruta_undersample)
test_predictions <- predict(lda_model, newdata = ts)
conf_matrix <- table(test_predictions$class, ts$Class)
print(conf_matrix)
accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
cat("Overall Accuracy: ", round(accuracy * 100, 3), "%\n")
predicted_probabilities <- predict(lda_model, newdata = ts, type = "response")$posterior[,2]

roc_result <- roc(ts$Class, predicted_probabilities)
print(roc_result)

plot(roc_result, main="ROC Curve")



#######################--------ElasticNet--------------################################

#-------------- Elastic net for lda over---------------------
lda_model <- lda(Class ~ ., ElasticNet_Undersapmled)
test_predictions <- predict(lda_model, newdata = ts)
conf_matrix <- table(test_predictions$class, ts$Class)
print(conf_matrix)
accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
cat("Overall Accuracy: ", round(accuracy * 100, 3), "%\n")
predicted_probabilities <- predict(lda_model, newdata = ts, type = "response")$posterior[,2]

roc_result <- roc(ts$Class, predicted_probabilities)
print(roc_result)

plot(roc_result, main="ROC Curve")

#------------------Elastic net for lda under------------------
lda_model <- lda(Class ~ ., ElasticNet_oversapmled)
test_predictions <- predict(lda_model, newdata = ts)
conf_matrix <- table(test_predictions$class, ts$Class)
print(conf_matrix)
accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
cat("Overall Accuracy: ", round(accuracy * 100, 3), "%\n")
predicted_probabilities <- predict(lda_model, newdata = ts, type = "response")$posterior[,2]

roc_result <- roc(ts$Class, predicted_probabilities)
print(roc_result)

plot(roc_result, main="ROC Curve")



############----------------------------Lasso----------------##################################


#--------------lda for lasso over-----------------------
lda_model <- lda(Class ~ ., Lasso_Oversampled)
test_predictions <- predict(lda_model, newdata = ts)
conf_matrix <- table(test_predictions$class, ts$Class)
print(conf_matrix)
accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
cat("Overall Accuracy: ", round(accuracy * 100, 3), "%\n")
predicted_probabilities <- predict(lda_model, newdata = ts, type = "response")$posterior[,2]

roc_result <- roc(ts$Class, predicted_probabilities)
print(roc_result)

plot(roc_result, main="ROC Curve")

#----------------lda for lasso under----------------------
lda_model <- lda(Class ~ ., Lasso_Undersampled)
test_predictions <- predict(lda_model, newdata = ts)
conf_matrix <- table(test_predictions$class, ts$Class)
print(conf_matrix)
accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
cat("Overall Accuracy: ", round(accuracy * 100, 3), "%\n")
predicted_probabilities <- predict(lda_model, newdata = ts, type = "response")$posterior[,2]

roc_result <- roc(ts$Class, predicted_probabilities)
print(roc_result)

plot(roc_result, main="ROC Curve")



