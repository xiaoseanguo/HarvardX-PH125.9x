##########################################################
# Install libraries
##########################################################

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)
library(lubridate)


##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

# Download the MovieLens 10M dataset
dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

# Extract the ratings data frame
ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

# Extract the movies data frame
movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# Convert data type on movies data frame (using R 4.0 or later)
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))

# Combine movies and ratings data frames
movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

# Remove unused objects 
rm(dl, ratings, movies, test_index, temp, movielens, removed)

# Structure of edx
str(edx)


##########################################################
# 1. Data Wrangling (Edx Set)
##########################################################

# Check the class of each column in edx 
sapply(edx, class)

# Create function to construct required predictors and convert data types
# Argument data_clean is the dataset (edx or validation) to be processed 
cleaning_data <- function(data_clean){
  data_clean %>% 
    
    # Convert data frame to tibble
    tibble() %>%
    
    # Convert each column to appropriate class
    mutate(userId = as.factor(userId),
           movieId = as.factor(movieId),
           timestamp = as_datetime(timestamp),
           genres = as.factor(genres)) %>%
    
    # Extract the year when the user viewed the movie from timestamp column
    mutate(viewDate = year(timestamp)) %>%
    
    # Extract the movie's release year from title column and convert to numeric
    mutate(releaseDate = str_extract(title, "\\([12]\\d{3}\\)") %>% 
             str_remove_all(., "[\\)\\(]") %>% 
             as.numeric()) %>% 
    
    # Extract the year when each user watched their first movie in this data set
    group_by(userId) %>% 
    mutate(view1stDate = min(viewDate)) %>%
    
    # Create "movieMaturity" as the time from release date to watch date
    mutate(movieMaturity = viewDate - releaseDate) %>%
    
    # Create "userMaturity" as the time from user's 1st movie to current movie
    mutate(userMaturity = viewDate - view1stDate) %>%
    
    # Remove timestamp column
    mutate(timestamp = NULL)
}

# Preprocess the edx data set
edx_clean <- cleaning_data(edx)


##########################################################
# 2. Exploratory Data Analysis (Edx set)
##########################################################

# 2.1 Size of Edx Dataset
##########################################################

# Display the predictors and their data type in the cleaned data set  
sapply(edx_clean, class)

# Display number of ratings (given by specific user to specific movie)
nrow(edx_clean)

# Display number of unique users 
n_distinct(edx_clean$userId)

# Display number of unique movies 
n_distinct(edx_clean$movieId)

# Display ratings values
unique(edx_clean$rating) %>% sort()

# Display number of unique genre combinations 
n_distinct(edx_clean$genres)

# Display range of movie release years 
summary(edx_clean$releaseDate)

# Display range of user's 1st movie view years 
summary(edx_clean$view1stDate)

# Display range of movie maturity years 
summary(edx_clean$movieMaturity)

# Display range of user maturity years 
summary(edx_clean$userMaturity)


# 2.2 Distribution of ratings for each predictor
##########################################################

# Distribution of ratings (total number of each rating)
edx_clean %>% ggplot(aes(rating)) + 
  geom_bar(color = I("black")) +
  ggtitle('Distribution of Ratings') +
  xlab('Ratings: Value') +
  ylab('Ratings: Count') 

# Distribution of average ratings based on movie (number of movies for each rating) 
edx_clean %>% group_by(movieId) %>%
  summarise(mean_rating = mean(rating)) %>%
  ggplot(aes(mean_rating)) +
  geom_histogram(bins = 30, color = I("black")) +
  ggtitle('Distribution of Average Ratings Based on Movie: Movie Quality') +
  xlab('Average Ratings of Movies') +
  ylab('Ratings: Count') 

# Distribution of average ratings based on user (number of users for each rating) 
edx_clean %>% group_by(userId) %>%
  summarise(mean_rating = mean(rating)) %>%
  ggplot(aes(mean_rating)) +
  geom_histogram(bins = 30, color = I("black")) +
  ggtitle('Distribution of Average Ratings Based on User: User Harshness') +
  xlab('Average Ratings of Users') +
  ylab('Ratings: Count') 

# Distribution of average ratings based on genre (number of genres for each rating) 
edx_clean %>% group_by(genres) %>%
  summarise(mean_rating = mean(rating)) %>%
  ggplot(aes(mean_rating)) +
  geom_histogram(bins = 30, color = I("black")) +
  ggtitle('Distribution of Average Ratings Based on Genre: Genre Preference') +
  xlab('Average Ratings of Genres') +
  ylab('Ratings: Count') 

# Distribution of average ratings based on release year (number of release year for each rating) 
edx_clean %>% group_by(releaseDate) %>%
  summarise(mean_rating = mean(rating)) %>%
  ggplot(aes(mean_rating)) +
  geom_histogram(bins = 25, color = I("black")) +
  ggtitle('Distribution of Average Ratings Based on Release Year: Movie Age') +
  xlab('Average Ratings of Release Years') +
  ylab('Ratings: Count') 

# Distribution of average ratings based on number of years since movie release (movie maturity)
edx_clean %>% group_by(movieMaturity) %>%
  summarise(mean_rating = mean(rating)) %>%
  ggplot(aes(mean_rating)) +
  geom_histogram(bins = 25, color = I("black")) +
  ggtitle('Distribution of Average Ratings Based on Time Since Movie Release') +
  xlab('Average Ratings of Time Since Movie Release') +
  ylab('Ratings: Count') 

# Distribution of average ratings based on number of years since user first rated (user maturity)
edx_clean %>% group_by(userMaturity) %>%
  summarise(mean_rating = mean(rating)) %>%
  ggplot(aes(mean_rating)) +
  geom_histogram(bins = 25, color = I("black")) +
  ggtitle("Distribution of Average Ratings Based on Time Since User's 1st Rating") +
  xlab("Average Ratings of Time Since User's 1st Rating") +
  ylab('Ratings: Count') 


# 2.3 Prevalence of values within each predictor
##########################################################

# Number of ratings a movie has
edx_clean %>% group_by(movieId) %>%
  summarise(count_rating = n()) %>%
  ggplot(aes(count_rating)) +
  geom_histogram(bins = 30, color = I("black")) +
  scale_x_log10() +
  ggtitle('Number of Ratings a Movie Has: Movie Popularity') +
  xlab('Number of Times Movie Was Rated') +
  ylab('Ratings: Count') 

# Number of ratings a user has submitted 
edx_clean %>% group_by(userId) %>%
  summarise(count_rating = n()) %>%
  ggplot(aes(count_rating)) +
  geom_histogram(bins = 30, color = I("black")) +
  scale_x_log10() +
  ggtitle('Number of Ratings a User Provided: User Engagement') +
  xlab('Number of Times User Rated') +
  ylab('Ratings: Count') 

# Number of ratings in each Genre Combination 
edx_clean %>% group_by(genres) %>%
  summarise(count_rating = n()) %>%
  ggplot(aes(count_rating)) +
  geom_histogram(bins = 30, color = I("black")) +
  scale_x_log10(breaks = scales::trans_breaks("log10", function(x) 10^x),
                labels = scales::trans_format("log10", scales::math_format(10^.x))) +
  ggtitle('Number of Ratings Between Genres') +
  xlab('Number of Times Any Movie in a Specific Genre Was Rated') +
  ylab('Ratings: Count') 

# Number of ratings vs each release year
edx_clean %>% ggplot(aes(releaseDate)) +
  geom_bar(color = I("black")) +
  xlim(1910, 2010) +
  scale_y_log10(breaks = scales::trans_breaks("log10", function(x) 10^x),
                labels = scales::trans_format("log10", scales::math_format(10^.x))) +
  ggtitle('Number of Ratings in Each Release Year') +
  xlab('Release Year') +
  ylab('Ratings: Count') 

# Number of ratings vs number of years since movie release (movie maturity)
edx_clean %>% ggplot(aes(movieMaturity)) +
  geom_bar(color = I("black")) +
  scale_y_log10(breaks = scales::trans_breaks("log10", function(x) 10^x),
                labels = scales::trans_format("log10", scales::math_format(10^.x))) +
  ggtitle('Number of Years Between Movie Release and Rating Date') +
  xlab('Years Between Movie Release & Rating Date') +
  ylab('Ratings: Count')

# Number of ratings vs number of years since user first rated (user maturity)
edx_clean %>% ggplot(aes(userMaturity)) +
  geom_bar(color = I("black")) +
  scale_y_log10(breaks = scales::trans_breaks("log10", function(x) 10^x),
                labels = scales::trans_format("log10", scales::math_format(10^.x))) +
  ggtitle("Number of Years Between User's First and Current Rating") +
  xlab("Years Between User's First and Current Rating") +
  ylab('Ratings: Count')


# 2.4 Effect of temporal predictor on ratings
##########################################################

# Release year vs rating 
edx_clean %>% group_by(releaseDate) %>%
  summarise(mean_rating = mean(rating)) %>%
  ggplot(aes(releaseDate, mean_rating)) +
  geom_point() +
  ggtitle('Effect of Release Year on Average Rating') +
  xlab('Release Year') +
  ylab('Average Rating') 

# Date of review after release vs rating
edx_clean %>% group_by(movieMaturity) %>%
  summarise(mean_rating = mean(rating)) %>%
  ggplot(aes(movieMaturity, mean_rating)) +
  geom_point() +
  ggtitle('Effect of Movie Maturity on Average Rating') +
  xlab('Number of Years Between Movie Release and Rating Submitted') +
  ylab('Average Rating') 

# Date of review since first user rating vs rating
edx_clean %>% group_by(userMaturity) %>%
  summarise(mean_rating = mean(rating)) %>%
  ggplot(aes(userMaturity, mean_rating)) +
  geom_point() +
  ggtitle('Effect of User Maturity on Average Rating') +
  xlab("Number of Years Between User's 1st Rating and Rating Submitted") +
  ylab('Average Rating') 


##########################################################
# 3 Split Data & Create RMSE Function
##########################################################

# 3.1 Split Edx Into Test and Train Sets
##########################################################

# Make test and train sets, where test set is 20% of the data

# Create index where the proportion of ratings is the same in both sets
# Set the seed for reproducibility
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = edx_clean$rating, 
                                  times = 1, p = 0.2, list = FALSE)

# Create the initial train and test sets
train_set <- edx_clean[-test_index,]
test_set0 <- edx_clean[test_index,]

# Take the user-movie combination absent in train set out of the test test
test_set <- test_set0 %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

# Add rows removed from test set (train_set0) back into train set
train_set0 <- anti_join(test_set0, test_set)
train_set <- rbind(train_set, train_set0)

# 3.2 Function to Evaluate the Models
##########################################################

# Create function for RMSE as the loss function 
# Argument true_ratings are the rating values from the test set
# Argument predicted_ratings are the rating values predicted by the model
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
  }


##########################################################
# 4. Models 
##########################################################

# 4.1 Model 1: Average the ratings 
##########################################################

# Calculate overall mean
mu <- mean(train_set$rating)

# Calculate RMSE
naive_rmse <- RMSE(test_set$rating, mu)

# Place RMSE result in summary table 
rmse_results <- tibble(method = "Just the average", RMSE = naive_rmse)

# Show RMSE results 
rmse_results %>% knitr::kable()


# 4.2 Model 2: Include movie effects b_i
##########################################################

# Calculate movie effects
movie_avgs <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))

# Predict rating with movie effects
predicted_ratings <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  mutate(pred = mu + b_i) %>%
  pull(pred)

# Calculate RMSE
model_2_rmse <- RMSE(predicted_ratings, test_set$rating)

# Place RMSE result in summary table 
rmse_results <- bind_rows(rmse_results,
                          tibble(method="Movie Effect Model",
                                     RMSE = model_2_rmse ))

# Show RMSE results 
rmse_results %>% knitr::kable()

# Plot of b_i values 
qplot(b_i, data = movie_avgs, bins = 30, color = I("black"))
summary(movie_avgs)


# 4.3 Model 3: Include user effects b_u
##########################################################

# Calculate user effects
user_avgs <- train_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

# Predict rating with movie & user effects
predicted_ratings <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

# Calculate RMSE
model_3_rmse <- RMSE(predicted_ratings, test_set$rating)

# Place RMSE result in summary table 
rmse_results <- bind_rows(rmse_results,
                          tibble(method="Movie + User Effects Model",  
                                     RMSE = model_3_rmse ))

# Show RMSE results 
rmse_results %>% knitr::kable()

# Plot of b_u values 
qplot(b_u, data = user_avgs, bins = 30, color = I("black"))
summary(user_avgs)


# 4.4 Model 4: Include genre effects b_g
##########################################################

# Calculate genre effects
genre_avgs <- train_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  group_by(genres) %>%
  summarize(b_g = mean(rating - mu - b_i - b_u))

# Predict rating with movie & user & genre effects
predicted_ratings <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genre_avgs, by='genres') %>%
  mutate(pred = mu + b_i + b_u + b_g) %>%
  pull(pred)

# Calculate RMSE
model_4_rmse <- RMSE(predicted_ratings, test_set$rating)

# Place RMSE result in summary table 
rmse_results <- bind_rows(rmse_results,
                          tibble(method="Movie + User + Genre Effects Model",  
                                     RMSE = model_4_rmse ))

# Show RMSE results 
rmse_results %>% knitr::kable()

# Plot of b_g values 
qplot(b_g, data = genre_avgs, bins = 30, color = I("black"))
summary(genre_avgs)


# 4.5 Model 5: Include movie release time effects b_r
##########################################################

# Calculate release date effects
release_avgs <- train_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genre_avgs, by='genres') %>%
  group_by(releaseDate) %>%
  summarize(b_r = mean(rating - mu - b_i - b_u - b_g))

# Predict rating with movie & user & genre & release date effects
predicted_ratings <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genre_avgs, by='genres') %>%
  left_join(release_avgs, by='releaseDate') %>%
  mutate(pred = mu + b_i + b_u + b_g + b_r) %>%
  pull(pred)

# Calculate RMSE
model_5_rmse <- RMSE(predicted_ratings, test_set$rating)

# Place RMSE result in summary table 
rmse_results <- bind_rows(rmse_results,
                          tibble(method="Movie + User + Genre + Release Effects Model",  
                                     RMSE = model_5_rmse ))

# Show RMSE results 
rmse_results %>% knitr::kable()

# Plot of b_r values 
qplot(b_r, data = release_avgs, bins = 20, color = I("black"))
summary(release_avgs)


# 4.6 Model 6: Include movie review time effects b_ti
##########################################################

# Calculate movie maturity effects
movie_maturity_avgs <- train_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genre_avgs, by='genres') %>%
  left_join(release_avgs, by='releaseDate') %>%
  group_by(movieMaturity) %>%
  summarize(b_ti = mean(rating - mu - b_i - b_u - b_g - b_r))

# Predict rating with movie & user & genre & release date & movie maturity effects
predicted_ratings <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genre_avgs, by='genres') %>%
  left_join(release_avgs, by='releaseDate') %>%
  left_join(movie_maturity_avgs, by='movieMaturity') %>%
  mutate(pred = mu + b_i + b_u + b_g + b_r + b_ti) %>%
  pull(pred)

# Calculate RMSE
model_6_rmse <- RMSE(predicted_ratings, test_set$rating)

# Place RMSE result in summary table 
rmse_results <- bind_rows(rmse_results,
                          tibble(method="Movie + User + Genre + Release + MovieM Effects Model",  
                                     RMSE = model_6_rmse ))

# Show RMSE results 
rmse_results %>% knitr::kable()

# Plot of b_ti values 
qplot(b_ti, data = movie_maturity_avgs, bins = 20, color = I("black"))
summary(movie_maturity_avgs)


# 4.7 Model 7: Include user review time effects b_tu
##########################################################

# Calculate user maturity effects
user_maturity_avgs <- train_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genre_avgs, by='genres') %>%
  left_join(release_avgs, by='releaseDate') %>%
  left_join(movie_maturity_avgs, by='movieMaturity') %>%
  group_by(userMaturity) %>%
  summarize(b_tu = mean(rating - mu - b_i - b_u - b_g - b_r - b_ti))

# Predict rating with movie & user & genre & release date & movie maturity & user maturity effects
predicted_ratings <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genre_avgs, by='genres') %>%
  left_join(release_avgs, by='releaseDate') %>%
  left_join(movie_maturity_avgs, by='movieMaturity') %>%
  left_join(user_maturity_avgs, by='userMaturity') %>%
  mutate(pred = mu + b_i + b_u + b_g + b_r + b_ti + b_tu) %>%
  pull(pred)

# Calculate RMSE
model_7_rmse <- RMSE(predicted_ratings, test_set$rating)

# Place RMSE result in summary table 
rmse_results <- bind_rows(rmse_results,
                          tibble(method="Movie + User + Genre + Release + MovieM + userM Effects Model",  
                                     RMSE = model_7_rmse ))

# Show RMSE results 
rmse_results %>% knitr::kable()

# Plot of b_tu values 
qplot(b_tu, data = user_maturity_avgs, bins = 10, color = I("black"))
summary(user_maturity_avgs)


# 4.8 Model 8: Regularized model
##########################################################

# Create function for regularized model & calculate its RMSE
# Argument l is lambda, tr is training set, te is test set
# Returned value is RMSE
REGM <- function(l, tr, te){

  # Calculate mean from training set
  mu <- mean(tr$rating)
  
  # Calculate movie effect from training set
  b_i <- tr %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  # Calculate user effect from training set
  b_u <- tr %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))

  # Calculate genre effect from training set
  b_g <- tr %>% 
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by="userId") %>%
    group_by(genres) %>%
    summarize(b_g = sum(rating - b_u - b_i - mu)/(n()+l))
  
  # Calculate release date effect from training set
  b_r <- tr %>% 
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by="userId") %>%
    left_join(b_g, by="genres") %>%
    group_by(releaseDate) %>%
    summarize(b_r = sum(rating - b_g - b_u - b_i - mu)/(n()+l))
  
  # Calculate movie maturity effect from training set
  b_ti <- tr %>% 
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by="userId") %>%
    left_join(b_g, by="genres") %>%
    left_join(b_r, by="releaseDate") %>%
    group_by(movieMaturity) %>%
    summarize(b_ti = sum(rating - b_r - b_g - b_u - b_i - mu)/(n()+l))
  
  # Calculate user maturity effect from training set
  b_tu <- tr %>% 
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by="userId") %>%
    left_join(b_g, by="genres") %>%
    left_join(b_r, by="releaseDate") %>%
    left_join(b_ti, by="movieMaturity") %>%
    group_by(userMaturity) %>%
    summarize(b_tu = sum(rating - b_ti - b_r - b_g - b_u - b_i - mu)/(n()+l))

  # Make prediction on test set
  predicted_ratings <- 
    te %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_g, by="genres") %>%
    left_join(b_r, by="releaseDate") %>%
    left_join(b_ti, by="movieMaturity") %>%
    left_join(b_tu, by="userMaturity") %>%
    mutate(pred = mu + b_i + b_u + b_g + b_r + b_ti + b_tu) %>%
    pull(pred)

  # Calculate the RMSE
  return(RMSE(predicted_ratings, te$rating))
}


# 4.9 Model 8: Optimize lambda in regularized model
##########################################################

##### takes a long time ###############

# Choose lambda values (coarse intervals)
lambdas <- seq(0, 10, 1)

# Calculate regularized RMSE values from different lambda (coarse intervals)
rmses <- sapply(lambdas, REGM, train_set, test_set)

# Graph lambda values vs rmse
qplot(lambdas, rmses) 

# Choose lambda values (fine intervals)
lambdas <- seq(4.5, 5.5, 0.1)

# Calculate regularized RMSE values from different lambda (fine intervals)
rmses <- sapply(lambdas, REGM, train_set, test_set)

# Graph lambda values vs rmse
qplot(lambdas, rmses) 

# Find optimal lambda and save its value
lambda <- lambdas[which.min(rmses)]
lambda


# 4.9 Model 8: Visualize effect of regularization
##########################################################

# Calculate regularized movie effects
movie_reg_avgs <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = sum(rating - mu)/(n()+lambda), n_i = n())
summary(movie_reg_avgs[,2:3])

# Plot effect of regularization on bi
tibble(Original = movie_avgs$b_i, 
       Regularlized = movie_reg_avgs$b_i, 
       n = movie_reg_avgs$n_i) %>%
  ggplot(aes(Original, Regularlized, size=sqrt(n))) + 
  geom_point(shape=1, alpha=0.5) +
  ggtitle('Effect of Regularization on Movie Bias: b_i')

# Calculate regularized user effects
user_reg_avgs <- train_set %>% 
  left_join(movie_reg_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - mu - b_i)/(n()+lambda), n_i = n())
summary(user_reg_avgs[,2:3])

# Plot effect of regularization on bu
tibble(Original = user_avgs$b_u, 
       Regularlized = user_reg_avgs$b_u, 
       n = user_reg_avgs$n_i) %>%
  ggplot(aes(Original, Regularlized, size=sqrt(n))) + 
  geom_point(shape=1, alpha=0.5) +
  ggtitle('Effect of Regularization on User Bias: b_u')

# Calculate regularized genre effects
genre_reg_avgs <- train_set %>% 
  left_join(movie_reg_avgs, by='movieId') %>%
  left_join(user_reg_avgs, by='userId') %>%
  group_by(genres) %>%
  summarize(b_g = sum(rating - mu - b_i - b_u)/(n()+lambda), n_i = n())
summary(genre_reg_avgs[,2:3])

# Plot effect of regularization on bg
tibble(Original = genre_avgs$b_g, 
       Regularlized = genre_reg_avgs$b_g, 
       n = genre_reg_avgs$n_i) %>%
  ggplot(aes(Original, Regularlized, size=sqrt(n))) + 
  geom_point(shape=1, alpha=0.5) +
  ggtitle('Effect of Regularization on Genre Bias: b_g')

# Calculate regularized release date effects
release_reg_avgs <- train_set %>% 
  left_join(movie_reg_avgs, by='movieId') %>%
  left_join(user_reg_avgs, by='userId') %>%
  left_join(genre_reg_avgs, by='genres') %>%
  group_by(releaseDate) %>%
  summarize(b_r = sum(rating - mu - b_i - b_u - b_g)/(n()+lambda), n_i = n())
summary(release_reg_avgs[,2:3])

# Plot effect of regularization on br
tibble(Original = release_avgs$b_r, 
       Regularlized = release_reg_avgs$b_r, 
       n = release_reg_avgs$n_i) %>%
  ggplot(aes(Original, Regularlized, size=sqrt(n))) + 
  geom_point(shape=1, alpha=0.5) +
  ggtitle('Effect of Regularization on Release Time Bias: b_r')

# Calculate regularized movie maturity effects
movie_maturity_reg_avgs <- train_set %>% 
  left_join(movie_reg_avgs, by='movieId') %>%
  left_join(user_reg_avgs, by='userId') %>%
  left_join(genre_reg_avgs, by='genres') %>%
  left_join(release_reg_avgs, by='releaseDate') %>%
  group_by(movieMaturity) %>%
  summarize(b_ti = sum(rating - mu - b_i - b_u - b_g - b_r)/(n()+lambda), n_i = n())
summary(movie_maturity_reg_avgs[,2:3])

# Plot effect of regularization on bti
tibble(Original = movie_maturity_avgs$b_ti, 
       Regularlized = movie_maturity_reg_avgs$b_ti, 
       n = movie_maturity_reg_avgs$n_i) %>%
  ggplot(aes(Original, Regularlized, size=sqrt(n))) + 
  geom_point(shape=1, alpha=0.5) +
  ggtitle('Effect of Regularization on Movie Review Time Bias: b_ti')

# Calculate regularized user maturity effects
user_maturity_reg_avgs <- train_set %>% 
  left_join(movie_reg_avgs, by='movieId') %>%
  left_join(user_reg_avgs, by='userId') %>%
  left_join(genre_reg_avgs, by='genres') %>%
  left_join(release_reg_avgs, by='releaseDate') %>%
  left_join(movie_maturity_reg_avgs, by='movieMaturity') %>%
  group_by(userMaturity) %>%
  summarize(b_tu = sum(rating - mu - b_i - b_u - b_g - b_r - b_ti)/(n()+lambda), n_i = n())
summary(user_maturity_reg_avgs[,2:3])

# Plot effect of regularization on btu
tibble(Original = user_maturity_avgs$b_tu, 
       Regularlized = user_maturity_reg_avgs$b_tu, 
       n = user_maturity_reg_avgs$n_i) %>%
  ggplot(aes(Original, Regularlized, size=sqrt(n))) + 
  geom_point(shape=1, alpha=0.5) +
  ggtitle('Effect of Regularization on User Review Time Bias: b_tu')


# 4.10 Model 8: Calculate RMSE of regularized model
##########################################################

# RMSE result in summary table 
rmse_results <- bind_rows(rmse_results,
                          tibble(method="Regularized Model",  
                                     RMSE = min(rmses)))

# Show RMSE results 
rmse_results %>% knitr::kable()


##########################################################
# 5. Data Wrangling (Validation Set)
##########################################################

# Clean the validation data set
validation_clean <- cleaning_data(validation)


##########################################################
# 6. Use Model on Validation Set
##########################################################

# Calculate RMSE of regularized model on validation set
rmse_validation <- sapply(lambda, REGM, edx_clean, validation_clean)

# Show the RMSE of regularized model on validation set
rmse_validation




