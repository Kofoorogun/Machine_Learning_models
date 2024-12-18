# Load necessary packages
library(dplyr)
library(tidyverse)
library(lubridate)
library(randomForest)
library(e1071)
library(caret)

# Load dataset
df <- read_csv("fraud test.csv")
# Remove some of the data, as it is too heavy
df1 <- df[1:100000, ]

# Print the modified dataset
print(df1)

# Checking the ratio of fraud transactions vs non fraud transactions
df1 %>% group_by(is_fraud) %>% summarize(n())

# Exploratory Data Analysis
df1 %>% group_by(category) %>% summarize(n(), Fraud_Rate=100*mean(is_fraud), max(amt), min(amt))

df2 <- df %>% filter(df$is_fraud == 0)
hist(df1$amt, breaks = 100, col = "green", main = "Histogram of CC charges for Non-Fraud Cases", xlab = "Values", ylab = "Frequency")

df3 <- df %>% filter(df$is_fraud == 1)
hist(df2$amt, breaks = 100, col = "purple", main = "Histogram of CC charges for Fraud Cases", xlab = "Values", ylab = "Frequency")

# Perform ANOVA
model <- aov(amt ~ is_fraud, data = df1)
summary(model)

# Creation of previous fraud transaction flag
df1$full_name <- paste0(df1$first, " ", df1$last)
min_date_of_fraud <- df1 %>% group_by(full_name) %>% filter(is_fraud==1) %>% summarize(min_date_fraud=min(trans_date_trans_time))
df1 <- left_join(df1, min_date_of_fraud, by = "full_name")
df1$previous_fraud <- 0
df1$previous_fraud[df1$trans_date_trans_time > df1$min_date_fraud] <- 1

df1$date <- as.Date(ceiling_date(dmy_hm(df1$trans_date_trans_time), "month"))
#Creation of Time based variables
df1 <- df1 %>%
  mutate(
    trans_hour = hour(as.POSIXct(trans_date_trans_time, format="%d/%m/%Y %H:%M")),
    trans_day = wday(as.Date(trans_date_trans_time, format="%d/%m/%Y"), label=TRUE),
    trans_month = month(as.Date(trans_date_trans_time, format="%d/%m/%Y"), label=TRUE),
    trans_weekend = ifelse(trans_day %in% c("Sat", "Sun"), 1, 0))

df1$full_name <- paste0(df1$first, " ", df1$last)

#Creation of Number of transactions of customer
df1 <- df1 %>%
  group_by(full_name) %>%
  mutate(num_of_transactions = row_number()) %>%
  ungroup()

#Creation of age of customer
df1$dob_year <- as.Date(df1$dob, format = "%d/%m/%Y")
df1$dob_year <- format(df1$dob_year, "%Y")
df1$year <- as.Date(dmy_hm(df1$trans_date_trans_time), format = "%d/%m/%Y")
df1$year <- format(df1$year, "%Y")
df1$age_of_user <- as.double(df1$dob_year) + as.double(df1$year)

#Creation of numerical features
df1$amt_sq <- df1$amt * df1$amt
df1$num_of_transactions_sq <- df1$num_of_transactions * df1$num_of_transactions
df1$age_of_user_sq <- df1$age_of_user * df1$age_of_user

# Splitting into train and test set using date
# Train set
train_prop <- 0.8
train <- df1 %>%
  sample_frac(train_prop)
# Test set
test <- df1 %>%
  anti_join(train, by = NULL)
nrow(train)
nrow(test)
# Variable selection
train <- train %>% select(merchant, category, amt, gender, city, job, is_fraud, previous_fraud, num_of_transactions, age_of_user, trans_hour, amt_sq, num_of_transactions_sq, age_of_user_sq)
test <- test %>% select(merchant, category, amt, gender, city, job, is_fraud, previous_fraud, num_of_transactions, age_of_user, trans_hour, amt_sq, num_of_transactions_sq, age_of_user_sq)
nrow(train)
nrow(test)

# Feature Standardization
train$amt<-(train$amt-mean(train$amt))/sd(train$amt)
train$num_of_transactions<-(train$num_of_transactions-mean(train$num_of_transactions))/sd(train$num_of_transactions)
train$age_of_user<-(train$age_of_user-mean(train$age_of_user))/sd(train$age_of_user)
train$amt_sq<-(train$amt_sq-mean(train$amt_sq))/sd(train$amt_sq)
train$num_of_transactions_sq<-(train$num_of_transactions_sq-mean(train$num_of_transactions_sq))/sd(train$num_of_transactions_sq)
train$age_of_user_sq<-(train$age_of_user_sq-mean(train$age_of_user_sq))/sd(train$age_of_user_sq)

test$num_of_transactions_sq<-(test$num_of_transactions_sq-mean(test$num_of_transactions_sq))/sd(test$num_of_transactions_sq)
test$age_of_user_sq<-(test$age_of_user_sq-mean(test$age_of_user_sq))/sd(test$age_of_user_sq)
test$amt_sq<-(test$amt_sq-mean(test$amt_sq))/sd(test$amt_sq)
test$amt<-(test$amt-mean(test$amt))/sd(test$amt)
test$num_of_transactions<-(test$num_of_transactions-mean(test$num_of_transactions))/sd(test$num_of_transactions)
test$age_of_user<-(test$age_of_user-mean(test$age_of_user))/sd(test$age_of_user)

# Select predictors and response variable
# predictors <- c("merchant","category","amt","gender","city","job","previous_fraud","num_of_transactions","age_of_user","trans_hour","amt_sq","num_of_transactions_sq","age_of_user_sq")
# response_variable <- "is_fraud"
X_train <- data.matrix(train[, setdiff(names(train), c("is_fraud"))])
y_train <- train$is_fraud

X_test <- data.matrix(test[, setdiff(names(test), c("is_fraud"))])
y_test <- test$is_fraud

# Random Forest Classifier
# rf_model <- randomForest(as.factor(y_train) ~ X_train, data = train)
rf_model <- randomForest(
  x = X_train, 
  y = as.factor(y_train), 
  ntree = 100,  # Number of trees in the forest
  mtry = sqrt(ncol(X_train)),  # Number of variables randomly sampled as candidates at each split
  importance = TRUE  # Calculate variable importance
)

# Support Vector Machine Classifier
# svm_model <- svm(is_fraud ~ ., data = train)
svm_model <- svm(
  x = X_train, 
  y = as.factor(y_train), 
  kernel = "radial",  # Radial basis function kernel
  cost = 1,           # Cost parameter
  gamma = 0.1         # Gamma parameter
)

# k-Nearest Neighbors Classifier
# knn_model <- knn(as.factor(is_fraud) ~ ., data = train, method = "knn")
train_control <- trainControl(method = "cv", number = 5)  # 5-fold cross-validation
# Train the KNN model
knn_model <- train(
  X_train, as.factor(y_train),
  method = "knn",
  trControl = train_control,
  tuneGrid = expand.grid(k = 1:10)  # Tune k from 1 to 10
)

# Logistic Regression
# log_model <- glm(as.factor(is_fraud) ~ ., data = train)

# Make predictions
rf_pred <- predict(rf_model, X_test)
svm_pred <- predict(svm_model, X_test)
knn_pred <- predict(knn_model, X_test)

# Evaluate performance
# Define evaluation function
evaluate <- function(actual, predicted) {
  confusion_matrix <- table(actual, predicted)
  accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
  return(list(accuracy, confusion_matrix))
}

# Evaluate performance of each classifier
rf_accuracy <- evaluate(y_test, rf_pred)
svm_accuracy <- evaluate(y_test, svm_pred)
knn_accuracy <- evaluate(y_test, knn_pred)


# Print results
print("Random Forest Classifier Accuracy:")
print(rf_accuracy)
print("Support Vector Machine Classifier Accuracy:")
print(svm_accuracy)
print("k-Nearest Neighbors Classifier Accuracy:")
print(knn_accuracy)

