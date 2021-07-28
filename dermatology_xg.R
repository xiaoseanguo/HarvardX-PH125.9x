##########################################################
# Install Packages
##########################################################

if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")

if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(e1071)) install.packages("e1071", repos = "http://cran.us.r-project.org")

if(!require(rpart)) install.packages("rpart", repos = "http://cran.us.r-project.org")
if(!require(ipred)) install.packages("ipred", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
if(!require(gbm)) install.packages("gbm", repos = "http://cran.us.r-project.org")

if(!require(smotefamily)) install.packages("smotefamily", repos = "http://cran.us.r-project.org")

##########################################################
# Libraries
##########################################################

library(data.table) # Reading data
library(tidyverse) # Wrangling data & graphing

library(caret) # Modeling
library(e1071) # Helper functions

library(rpart) # Training decision tree
library(ipred) # Training bagged decision tree
library(randomForest) # Training random forest
library(gbm) # Training gradient boost

library(smotefamily) # Sampling data for balanced class

##########################################################
# 1. Download Dataset
##########################################################

# Dermatology dataset:
# https://archive.ics.uci.edu/ml/machine-learning-databases/dermatology/
# https://archive.ics.uci.edu/ml/machine-learning-databases/dermatology/dermatology.data


# Download dataset description file
dl0 <- tempfile()
download.file("https://archive.ics.uci.edu/ml/machine-learning-databases/dermatology/dermatology.names", dl0)

# Read text file with dataset description
di <- readLines(dl0)

# View the dataset information text file
print(di)

# Download dataset
dl <- tempfile()
download.file("https://archive.ics.uci.edu/ml/machine-learning-databases/dermatology/dermatology.data", dl)

# Read dataset as dataframe & add column names according to the description file
df0 <- fread(file = dl, 
            col.names = c("C01_erythema", 
                          "C02_scaling", 
                          "C03_definite_borders", 
                          "C04_itching", 
                          "C05_koebner_phenomenon", 
                          "C06_polygonal_papules", 
                          "C07_follicular_papules", 
                          "C08_oral_mucosal_involvement", 
                          "C09_knee_and_elbow_involvement", 
                          "C10_scalp_involvement", 
                          "C11_family_history", 
                          "H12_melanin_incontinence", 
                          "H13_eosinophils_in_infiltrate", 
                          "H14_PNL_infiltrate", 
                          "H15_papillary_dermis_fibrosis", 
                          "H16_exocytosis", 
                          "H17_acanthosis", 
                          "H18_hyperkeratosis", 
                          "H19_parakeratosis", 
                          "H20_rete_ridges_clubbing", 
                          "H21_rete_ridges_elongation", 
                          "H22_suprapapillary_epidermis_thinning", 
                          "H23_spongiform_pustule", 
                          "H24_munro_microabcess", 
                          "H25_focal_hypergranulosis", 
                          "H26_granular_layer_disappearance", 
                          "H27_basal_layer_vacuolisation_and_damage", 
                          "H28_spongiosis", 
                          "H29_retes_sawtooth_appearance", 
                          "H30_follicular_horn_plug", 
                          "H31_perifollicular_parakeratosis", 
                          "H32_inflammatory_monoluclear_inflitrate", 
                          "H33_band_like_infiltrate", 
                          "C34_age", 
                          "D0_diagnosis") )

# Examine structure of raw data
str(df0, give.attr = FALSE)

##########################################################
# 2. Preprocessing
##########################################################

# 2.1 Transform Predictors
##########################################################

# Copy and convert the data set to tibble
df <- tibble(df0)

# Convert "?" symbol in C34_age to NA
df$C34_age <- na_if(df$C34_age, '?')

# Check the class of each column in df 
sapply(df, class)

# Convert data type of predictors C34_age, C11_family_history, and D0_diagnosis
df <- df %>% mutate(C34_age = as.numeric(df$C34_age),
                    C11_family_history = as.logical(df$C11_family_history),
                    D0_diagnosis = as.factor(df$D0_diagnosis))

# Move D0_diagnosis column (outcome column) to front 
df <- relocate(df, D0_diagnosis)


# 2.2 Create Training & Testing Sets
##########################################################

# Partition data so 20% of data is Validation set
# Set the seed for reproducibility
set.seed(100)

# Create partition index with original proportions of diagnosis
test_index <- createDataPartition(y = df$D0_diagnosis, 
                                  times = 1, p = 0.2, list = FALSE)
# Create training set
train_set <- df[-test_index,]

# Create test set
test_set <- df[test_index,]


# 2.3 Impute Missing Values
##########################################################

# Create function for imputing missing values
# Argument data_set is the dataset with missing values
imputing_data <- function(data_set){
  
  # Transform data type to int to create dummy variables 
  dummy_value <- dummyVars(~., data = data_set[, -1]) %>%
    predict(newdata = data_set[, -1])
  
  # Calculate values missing in data set
  missing_value <- preProcess(dummy_value, method = 'bagImpute') %>%
    predict(dummy_value)
  
  # Round and impute the missing age values 
  data_set$C34_age <- missing_value[, 35] %>% round()
  
  # Return the data set with imputed values
  return(data_set)
}

# Impute missing values into training set
train_set <- imputing_data(train_set)

# Impute missing values into testing set
test_set <- imputing_data(test_set)



##########################################################
# 3. Exploratory Data Analysis (on Training Set)
##########################################################

# 3.1 Basic Data Exploration
##########################################################

# Display number of observations 
train_set %>% nrow()

# Display number of unique diagnosis 
train_set$D0_diagnosis %>% n_distinct()

# Display number of observations for each diagnosis
train_set %>% group_by(D0_diagnosis) %>% count()

# Display range of each predictor 
train_set[, -1]%>%
  mutate(C11_family_history = as.numeric(C11_family_history)) %>%
  summary(digit = 0) %>% .[c(1, 6),] %>% tibble() %>% t() 


# 3.2 Visualize Data
##########################################################

# Create list of diagnosis names for plot legends
diagnosis <- c("1 Psoriasis",
            "2 Seborrheic Dermatitis",
            "3 Lichen Planus",
            "4 Pityriasis Rosea",
            "5 Chronic Dermatitis",
            "6 Pityriasis Rubra Pilaris")

# Create a ggplot layer for formatting legends
legend_layer <- list(theme(legend.position="bottom"),
                     guides(fill = guide_legend(nrow = 1)),
                     scale_fill_discrete(name = "Diagnosis", 
                                         labels = diagnosis))

# Reshape training set for graphing
train_set_plot <- train_set %>% gather(symptom, value, -D0_diagnosis)

# Plot the count of each diagnosis 
train_set %>% 
  ggplot(aes(D0_diagnosis, fill = D0_diagnosis)) + 
  geom_bar() +
  legend_layer +
  ggtitle('Count of Each Diagnosis') +
  xlab('Diagnosis') +
  ylab('Count') 

# Plot the distribution of values for each predictor
train_set_plot %>%
  ggplot(aes(value, fill = D0_diagnosis)) +
  geom_histogram(data=subset(train_set_plot, symptom == 'C34_age'), 
                 binwidth = 20) +
  geom_histogram(data=subset(train_set_plot, symptom != 'C34_age'), 
                 binwidth = 1) +
  facet_wrap(~symptom, scales = "free_x") + 
  legend_layer +
  ggtitle('Distribution of Predictor Values') +
  xlab('Value') +
  ylab('Count') 

# Plot the distribution of each diagnosis for each predictor
train_set_plot %>%
  ggplot(aes(D0_diagnosis, value, fill = D0_diagnosis)) +
  geom_boxplot() +
  facet_wrap(~symptom, scales = "free") + 
  legend_layer +
  ggtitle('Distribution of Diagnosis Classes') +
  xlab('Diagnosis') +
  ylab('Value') 


##########################################################
# 4. Build and Analyze Models  
##########################################################

# 4.1 Function to Evaluate the Model
##########################################################

# Create function to find the final model's accuracy and kappa
# Argument fit_model is the train object produced by each model algorithm
fit_results <- function(fit_model){
  
  # Extract the results table from train object
  fit_model$results %>%
    
    # Find the model with the highest Kappa
    .[which.max(.$Kappa),] %>%
    
    # Extract the accuracy and kappa values
    select('Accuracy', 'Kappa')
}


# 4.2 Cross Validation Parameters
##########################################################

# Set cross validation parameters
control <- trainControl(method = 'repeatedcv',
                        # 5 folds
                        number =5, 
                        # Repeat 5 times
                        repeats = 5,
                        # Save predictions from model with optimal parameters
                        savePredictions = 'final')


# 4.3 Decision Tree Model (rpart)
##########################################################

# Plot default parameters on decision tree model
train(D0_diagnosis~.,
      data = train_set,
      method = 'rpart',
      metric = 'Kappa',
      trControl = control)

# Set up tunegrid
grid_dt <- expand.grid(cp = c(0, 0.05, 0.1, 0.15, 0.2))

# Tune decision tree model parameters 
# Set the seed for reproducibility
set.seed(100)
fit_dt <- train(D0_diagnosis~.,
                data = train_set,
                method = 'rpart',
                metric = 'Kappa',
                tuneGrid = grid_dt,
                trControl = control)

# Plot the tuning of parameters
plot(fit_dt, main = 'DT Param Tuning')

# Extract the final model parameters 
fit_dt$bestTune

# Evaluate model
summary_tbl <- cbind(tibble(Model = "DT"),
                     fit_results(fit_dt))

# Show summary table
summary_tbl %>% knitr::kable()


# 4.4 Bagged Tree Model (ipred)
##########################################################

# Train bagged tree model (bagged tree has no parameters)
# Set the seed for reproducibility
set.seed(100)
fit_bt <- train(D0_diagnosis~.,
                data = train_set,
                method = 'treebag',
                metric = 'Kappa',
                trControl = control)

# Evaluate model
summary_tbl <- rbind(summary_tbl, 
                     cbind(tibble(Model = "BT"),
                     fit_results(fit_bt)))

# Show summary table
summary_tbl %>% knitr::kable()


# 4.5 Random Forest Model (randomForest)
##########################################################

# Plot default parameters on random forest model
train(D0_diagnosis~.,
      data = train_set,
      method = 'rf',
      metric = 'Kappa',
      trControl = control)

# Set up tunegrid
grid_rf <- expand.grid(mtry = c(2, 4, 6, 8, 10))

# Tune random forest model parameters 
# Set the seed for reproducibility
set.seed(100)
fit_rf <- train(D0_diagnosis~.,
                data = train_set,
                method = 'rf',
                metric = 'Kappa',
                tuneGrid = grid_rf,
                trControl = control)

# Plot the tuning of parameters
plot(fit_rf, main = 'RF Param Tuning')

# Extract the final model parameters 
fit_rf$bestTune

# Evaluate model
summary_tbl <- rbind(summary_tbl, 
                     cbind(tibble(Model = "RF"),
                           fit_results(fit_rf)))

# Show summary table
summary_tbl %>% knitr::kable()


# 4.6 Gradient Boost Model (gbm)
##########################################################

# Examine default parameters on gradient boost model
train(D0_diagnosis~.,
      data = train_set,
      method = 'gbm',
      metric = 'Kappa',
      trControl = control,
      verbose = FALSE)

# Set up tunegrid
grid_gb <- expand.grid(n.trees = c(40, 50, 60), 
                       interaction.depth = c(1, 2, 3),
                       shrinkage = c(0.06, 0.08, 0.1),
                       n.minobsinnode = c(6, 8, 10))

# Tune gradient boost model parameters
# Set the seed for reproducibility
set.seed(100)
fit_gb <- train(D0_diagnosis~.,
                data = train_set,
                method = 'gbm',
                metric = 'Kappa',
                tuneGrid = grid_gb,
                trControl = control,
                verbose = FALSE)

# Plot the tuning of parameters
plot(fit_gb, main = 'GB Param Tuning')

# Extract the final model parameters 
fit_gb$bestTune

# Evaluate model
summary_tbl <- rbind(summary_tbl, 
                     cbind(tibble(Model = "GB"),
                           fit_results(fit_gb)))

# Show summary table
summary_tbl %>% knitr::kable()


# 4.7 Balance Unbalanced Classification with Up-sample
##########################################################

# Resample from training set to create a sample with balanced classes
# Set the seed for reproducibility
set.seed(100)
train_set_upsample <- upSample(x = train_set[, -1],
                               y = train_set$D0_diagnosis, 
                               yname = 'D0_diagnosis') %>%
  
  # Move D0_diagnosis column to front 
  relocate(D0_diagnosis)


# 4.8 Decision Tree Model (rpart) with Up-sample
##########################################################

# Tune decision tree model parameters 
# Set the seed for reproducibility
set.seed(100)
fit_dt_us <- train(D0_diagnosis~.,
                   data = train_set_upsample,
                   method = 'rpart',
                   metric = 'Kappa',
                   tuneGrid = grid_dt,
                   trControl = control)

# Plot the tuning of parameters
plot(fit_dt_us, main = 'DTUS Param Tuning')

# Extract the final model parameters 
fit_dt_us$bestTune

# Evaluate model
summary_tbl <- rbind(summary_tbl, 
                     cbind(tibble(Model = "DTUS"),
                           fit_results(fit_dt_us)))

# Show summary table
summary_tbl %>% knitr::kable()


# 4.9 Bagged Tree Model (ipred) with Up-sample
##########################################################

# Train bagged tree model (bagged tree has no parameters)
# Set the seed for reproducibility
set.seed(100)
fit_bt_us <- train(D0_diagnosis~.,
                   data = train_set_upsample,
                   method = 'treebag',
                   metric = 'Kappa',
                   trControl = control)

# Evaluate model
summary_tbl <- rbind(summary_tbl, 
                     cbind(tibble(Model = "BTUS"),
                           fit_results(fit_bt_us)))

# Show summary table
summary_tbl %>% knitr::kable()


# 4.10 Random Forest Model (randomForest) with Up-sample
##########################################################

# Tune random forest model parameters 
# Set the seed for reproducibility
set.seed(100)
fit_rf_us <- train(D0_diagnosis~.,
                   data = train_set_upsample,
                   method = 'rf',
                   metric = 'Kappa',
                   tuneGrid = grid_rf,
                   trControl = control)

# Plot the tuning of parameters
plot(fit_rf_us, main = 'RFUS Param Tuning')

# Extract the final model parameters 
fit_rf_us$bestTune

# Evaluate model
summary_tbl <- rbind(summary_tbl, 
                     cbind(tibble(Model = "RFUS"),
                           fit_results(fit_rf_us)))

# Show summary table
summary_tbl %>% knitr::kable()


# 4.11 Gradient Boost Model (gbm) with Up-sample
##########################################################

# Tune gradient boost model parameters
# Set the seed for reproducibility
set.seed(100)
fit_gb_us <- train(D0_diagnosis~.,
                   data = train_set_upsample,
                   method = 'gbm',
                   metric = 'Kappa',
                   tuneGrid = grid_gb,
                   trControl = control,
                   verbose = FALSE)

# Plot the tuning of parameters
plot(fit_gb_us, main = 'GBUS Param Tuning')

# Extract the final model parameters 
fit_gb_us$bestTune

# Evaluate model
summary_tbl <- rbind(summary_tbl, 
                     cbind(tibble(Model = "GBUS"),
                           fit_results(fit_gb_us)))

# Show summary table
summary_tbl %>% knitr::kable()


# 4.12 Variable Importance 
##########################################################

# Place train objects from all the models in a list
model_ls <- list(fit_dt,
                 fit_bt,
                 fit_rf,
                 fit_gb,
                 fit_dt_us,
                 fit_bt_us,
                 fit_rf_us,
                 fit_gb_us)

# Create function to calculate variable importance
# Argument fit_model is the train object produced by the model algorithms
imp_fun <- function(fit_model){
  
  # Calculate the variable importance
  varImp(fit_model)$importance %>%
    
    # Convert the importance vector to data frame
    as.data.frame()%>%
    
    # Transpose the data frame so predictors are rows
    rownames_to_column() %>%
    
    # Add the column names
    `colnames<-`(c('Predictors', 'Importance')) %>%
    
    # Arrange predictors so the order is the same (alphabetical) for all models 
    arrange(Predictors)
}

# Calculate variable importance for all models
imp_data <- sapply(model_ls, imp_fun)

# Combine variable importance data frames into a single matrix
imp <- cbind(imp_data[, 1]$Predictors,
             imp_data[, 1]$Importance,
             imp_data[, 2]$Importance,
             imp_data[, 3]$Importance,
             imp_data[, 4]$Importance,
             imp_data[, 5]$Importance,
             imp_data[, 6]$Importance,
             imp_data[, 7]$Importance,
             imp_data[, 8]$Importance)

# Insert column names to the variable importance matrix
colnames(imp) <- c('predictors',
                   '1 dt',
                   '3 bt',
                   '5 rf',
                   '7 gb',
                   '2 dt_us',
                   '4 bt_us',
                   '6 rf_us',
                   '8 gb_us')

# Reshape variable importance matrix for graphing
imp_plot <- imp %>% as_tibble %>%
  gather(model, importance, -predictors)

# Graph variable importance
imp_plot %>% ggplot(aes(reorder(predictors, desc(predictors)), 
                        importance, fill = model)) +
  geom_col() +
  coord_flip() +
  facet_grid(~model) +
  ggtitle('Variable Importance ') +
  xlab('Predictors') +
  ylab('Importance') +
  theme(axis.text.x = element_blank(),
        legend.position = "none")


##########################################################
# 5. Make Predictions
##########################################################

# 5.1 Predict Test Set 
##########################################################

# Create function to predict the test set and make confusion matrix
# Argument fit_model is the train object produced by the model algorithms
pred_cm <- function(fit_model){

  # Predict the diagnosis in the test set
  predict(fit_model, test_set) %>%
    
    # Compare predicted to actual diagnosis to build the confusion matrix 
    confusionMatrix(test_set$D0_diagnosis)
}

# Predict test set with decision tree model
predict_dt <- pred_cm(fit_dt)

# Predict test set with bagged tree model
predict_bt <- pred_cm(fit_bt)

# Predict test set with random forest model
predict_rf <- pred_cm(fit_rf)

# Predict test set with gradient boost model
predict_gb <- pred_cm(fit_gb)

# Predict test set with up-sampled decision tree model
predict_dt_us <- pred_cm(fit_dt_us)

# Predict test set with up-sampled bagged tree model
predict_bt_us <- pred_cm(fit_bt_us)

# Predict test set with up-sampled random forest model
predict_rf_us <- pred_cm(fit_rf_us)

# Predict test set with up-sampled gradient boost model
predict_gb_us <- pred_cm(fit_gb_us)


# 5.2 Evaluate Prediction Accuracy 
##########################################################

# Combine confusion matrix results from all models into single matrix
results_data <- rbind(predict_dt$overall[1:4],
                      predict_bt$overall[1:4],
                      predict_rf$overall[1:4],
                      predict_gb$overall[1:4],
                      predict_dt_us$overall[1:4],
                      predict_bt_us$overall[1:4],
                      predict_rf_us$overall[1:4],
                      predict_gb_us$overall[1:4]) %>%
  # Convert to tibble
  as_tibble()

# List model names 
models <- c('1 dt',
            '3 bt',
            '5 rf',
            '7 gb',
            '2 dt_us',
            '4 bt_us',
            '6 rf_us',
            '8 gb_us')

# Combine model names with result statistics
results <- cbind(models, results_data) 

# Display the overall accuracy 
results %>% knitr::kable()

# Plot the overall prediction accuracy for each model 
results %>% 
  ggplot(aes(models, Accuracy, fill = models)) + 
  geom_col() +
  geom_errorbar(aes(ymin=AccuracyLower, ymax=AccuracyUpper)) +
  ggtitle('Overall Accuracy') +
  xlab('Model') +
  theme(legend.position = "none")


# 5.3 Evaluate Prediction Accuracy by Class
##########################################################

# Create function to plot calculated statistics by each diagnosis class
# Argument col_num is the column number that corresponds to a specific stat
# Argument plot_title is the title given to the plot 
stat_by_class <- function(col_num, plot_title){
  
  # Combine stat by class from all models into single matrix
  stat_bc_data <- cbind(predict_dt$byClass[,col_num],
                        predict_bt$byClass[,col_num],
                        predict_rf$byClass[,col_num],
                        predict_gb$byClass[,col_num],
                        predict_dt_us$byClass[,col_num],
                        predict_bt_us$byClass[,col_num],
                        predict_rf_us$byClass[,col_num],
                        predict_gb_us$byClass[,col_num]) %>%
    # Convert to tibble
    as_tibble()
  
  # Add col names
  colnames(stat_bc_data) <- models
  
  # Add row names
  stat_bc <- cbind(diagnosis,stat_bc_data)
  
  # Reshape stat by class for graphing
  stat_bc_plot <- stat_bc %>% gather(model, value, -diagnosis)
  
  # Plot stat by class 
  stat_bc_plot %>%
    ggplot(aes(model, value, fill = model)) +
    geom_col() +
    facet_wrap(~diagnosis) + 
    scale_fill_discrete(name = "Model") +
    ggtitle(plot_title) +
    theme(axis.title.x=element_blank(),
          axis.text.x=element_blank(),
          axis.ticks.x=element_blank())
}

# Plot balanced accuracy by class
stat_by_class(11, 'Balanced Accuracy by Class')

# Plot F1 score by class
stat_by_class(7, 'F1 Score by Class')

# Plot sensitivity & recall by class
stat_by_class(1, 'Sensitivity and recall by Class')

# Plot specificity by class
stat_by_class(2, 'Specificity by Class')

# Plot precision by class
stat_by_class(5, 'Precision by Class')


# 5.4 Evaluate Prediction Errors 
##########################################################

# Show errors from decision tree model
predict_dt$table %>% knitr::kable(caption = 'Decision Tree', row.names = TRUE)

# Show errors from bagged tree model
predict_bt$table %>% knitr::kable(caption = 'Bagged Tree', row.names = TRUE)

# Show errors from random forest model
predict_rf$table %>% knitr::kable(caption = 'Random Forest', row.names = TRUE)

# Show errors from gradient boost model
predict_gb$table %>% knitr::kable(caption = 'Gradient Boost', row.names = TRUE)

# Show errors from up-sampled decision tree model
predict_dt_us$table %>% knitr::kable(caption = 'Up-Sampled Decision Tree', row.names = TRUE)

# Show errors from up-sampled bagged tree model
predict_bt_us$table %>% knitr::kable(caption = 'Up-Sampled Bagged Tree', row.names = TRUE)

# Show errors from up-sampled random forest model
predict_rf_us$table %>% knitr::kable(caption = 'Up-Sampled Random Forest', row.names = TRUE)

# Show errors from up-sampled gradient boost model
predict_gb_us$table %>% knitr::kable(caption = 'Up-Sampled Gradient Boost', row.names = TRUE)



