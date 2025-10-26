library(dplyr)
library(ggplot2)
library(lattice)
library(caret)
library(randomForest)
library(caret)
library(rsample)
library(ROSE)
library(Boruta)
library(FSelector)
library(glmnet)


data <- read.csv('final_cleaned_dataset.csv')


data$Class<- factor(data$Class)
str(data)


set.seed(101)
split <- initial_split(data, prop = 0.80, strata = Class)
tr <- training(split)
ts <- testing(split)
barplot(table(tr$Class))

over_sampled_data <- ovun.sample(Class ~ ., data = tr , method = "over")$data
barplot(table(over_sampled_data$Class))


under_sampled_data <- ovun.sample(Class ~ ., data = tr , method = "under")$data
barplot(table(under_sampled_data$Class))

############################-------------------Boruta for oversample---------------########################
boruta_result <- Boruta(Class ~ ., data = over_sampled_data, doTrace = 2) 
boruta_result
selected_features <- getSelectedAttributes(boruta_result, withTentative = TRUE)
length(selected_features)
Boruta_Oversampled <- over_sampled_data[, selected_features]
str(Boruta_Undersampled)
Boruta_Oversampled$Class <- over_sampled_data$Class

#-----------------Boruta for undersample-----------------------------------------------------------

boruta_result <- Boruta(Class ~ ., data = under_sampled_data, doTrace = 2) 
boruta_result
selected_features <- getSelectedAttributes(boruta_result, withTentative = TRUE)
length(selected_features)
Boruta_Undersampled <- under_sampled_data[, selected_features]
str(Boruta_Undersampled)
Boruta_Undersampled$Class <- under_sampled_data$Class

#--------------------------------lasso for oversampled data------------------------------

x_over <- model.matrix(Class ~ . - 1, data = over_sampled_data)
y_over <- over_sampled_data$Class
set.seed(101)
cv_lasso_over <- cv.glmnet(x_over, y_over, alpha = 1, family = "binomial")

coef_lasso_over <- coef(cv_lasso_over, s = "lambda.min")
selected_features_over <- rownames(coef_lasso_over)[coef_lasso_over[,1] != 0]
selected_features_over
selected_features_over_excluding_first <- selected_features_over[-1]
Lasso_Oversample <- over_sampled_data[, c(selected_features_over_excluding_first, "Class")]
str(Lasso_Oversample)

#----------------------------lasso for undersampled data----------------------------------
x_over <- model.matrix(Class ~ . - 1, data = under_sampled_data)
y_over <- under_sampled_data$Class
set.seed(101)
cv_lasso_under <- cv.glmnet(x_over, y_over, alpha = 1, family = "binomial")

coef_lasso_under <- coef(cv_lasso_under, s = "lambda.min")
selected_features_under <- rownames(coef_lasso_under)[coef_lasso_under[,1] != 0]
selected_features_under
selected_features_under_excluding_first <- selected_features_under[-1]
Lasso_Undersample <- under_sampled_data[, c(selected_features_under_excluding_first, "Class")]
str(Lasso_Undersample)


#-----------------------------elaastic net over-----------------
x_over <- model.matrix(Class ~ . - 1, data = over_sampled_data)
y_over <- over_sampled_data$Class
alpha <- 0.5 
cv_fit_over <- cv.glmnet(x_over, y_over, alpha=alpha, family="binomial")
lambda_min_over <- cv_fit_over$lambda.min
fit_over <- glmnet(x_over, y_over, alpha=alpha, lambda=lambda_min_over, family="binomial")
coef_over <- coef(fit_over, s=lambda_min_over)
selected_features_over <- rownames(coef_over)[coef_over[,1] != 0]
print(selected_features_over)
selected_features_under_excluding_first1 <- selected_features_over[-1]
ElasticNet_Oversample<- over_sampled_data[, c(selected_features_under_excluding_first1, "Class")]
str(ElasticNet_Oversample)

#-----------------------------elastic net under-------------------------------

x_over <- model.matrix(Class ~ . - 1, data = under_sampled_data)
y_over <- under_sampled_data$Class
alpha <- 0.5 
cv_fit_over <- cv.glmnet(x_over, y_over, alpha=alpha, family="binomial")
lambda_min_over <- cv_fit_over$lambda.min
fit_over <- glmnet(x_over, y_over, alpha=alpha, lambda=lambda_min_over, family="binomial")
coef_over <- coef(fit_over, s=lambda_min_over)
selected_features_over <- rownames(coef_over)[coef_over[,1] != 0]
print(selected_features_over)
selected_features_under_excluding_first2 <- selected_features_over[-1]
ElasticNet_UnderSample <- under_sampled_data[, c(selected_features_under_excluding_first2, "Class")]
str(ElasticNet_UnderSample)



#--------------------svm for boruta over------------------------------------
svm_model <- svm(Class ~ ., data =Boruta_Oversampled, kernel = "radial", cost = 1, probability = TRUE)
summary(svm_model)

predicted <- predict(svm_model, ts)
cm <- table(predicted, ts$Class)
print(cm)
accuracy <- sum(diag(cm)) / sum(cm)
cat("Overall Accuracy: ", round(accuracy * 100, 3), "%\n")
predicted_probabilities <- predict(svm_model, newdata = ts, type = "response")
roc_score=roc.curve(ts$Class, predicted_probabilities, levels = levels(ts$Class))
roc_score


#----------------svm for boruta under-------------------------------------------------------------------

svm_model <- svm(Class ~ ., data =Boruta_Undersampled, kernel = "radial", cost = 1, probability = TRUE)
predicted <- predict(svm_model, ts)
cm <- table(predicted, ts$Class)
print(cm)
accuracy <- sum(diag(cm)) / sum(cm)
cat("Overall Accuracy: ", round(accuracy * 100, 3), "%\n")
predicted_probabilities <- predict(svm_model, newdata = ts, type = "response")
roc_score=roc.curve(ts$Class, predicted_probabilities, levels = levels(ts$Class))
roc_score








#------------------------svm for lasso over-----------------------------
svm_model <- svm(Class ~ ., data =Lasso_Oversample, kernel = "radial", cost = 1, probability = TRUE)
predicted <- predict(svm_model, ts)
cm <- table(predicted, ts$Class)
print(cm)
accuracy <- sum(diag(cm)) / sum(cm)
cat("Overall Accuracy: ", round(accuracy * 100, 3), "%\n")
predicted_probabilities <- predict(svm_model, newdata = ts, type = "response")
roc_score=roc.curve(ts$Class, predicted_probabilities, levels = levels(ts$Class))
roc_score

#----------------------svm for lasso under------------------
svm_model <- svm(Class ~ ., data = Lasso_Undersample, kernel = "radial", cost = 1, probability = TRUE)
predicted <- predict(svm_model, ts)
cm <- table(predicted, ts$Class)
print(cm)
accuracy <- sum(diag(cm)) / sum(cm)
cat("Overall Accuracy: ", round(accuracy * 100, 3), "%\n")
predicted_probabilities <- predict(svm_model, newdata = ts, type = "response")
roc_score=roc.curve(ts$Class, predicted_probabilities, levels = levels(ts$Class))
roc_score











#------------------------svm for elastic net over-----------------------
svm_model <- svm(Class ~ ., data =ElasticNet_Oversample, kernel = "radial", cost = 1, probability = TRUE)
predicted <- predict(svm_model, ts)
cm <- table(predicted, ts$Class)
print(cm)
accuracy <- sum(diag(cm)) / sum(cm)
cat("Overall Accuracy: ", round(accuracy * 100, 3), "%\n")
predicted_probabilities <- predict(svm_model, newdata = ts, type = "response")
roc_score=roc.curve(ts$Class, predicted_probabilities, levels = levels(ts$Class))
roc_score

#------------------------svm for elastic net under-----------------------
svm_model <- svm(Class ~ ., data =ElasticNet_UnderSample, kernel = "radial", cost = 1, probability = TRUE)
predicted <- predict(svm_model, ts)
cm <- table(predicted, ts$Class)
print(cm)
accuracy <- sum(diag(cm)) / sum(cm)
cat("Overall Accuracy: ", round(accuracy * 100, 3), "%\n")
predicted_probabilities <- predict(svm_model, newdata = ts, type = "response")
roc_score=roc.curve(ts$Class, predicted_probabilities, levels = levels(ts$Class))
roc_score


