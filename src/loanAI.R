###############################################################################################
# loanAI V1.0
# A data mining algorithim for generic loan default model creation.
# Created by Korhan Akcura
###############################################################################################

# Use 6 digits precission.
options(digits=6)
# Set seed to obtain reproducible results.
set.seed(1)
# Using a Data Tree to store the generated decision tree.
install.packages("data.tree")
# Using Caret to obtain confusionMatrix.
library(caret)
library(caTools)
library(data.tree)
library(rstudioapi)

library(datasets)
data(iris)

###############################################################################################
# Data Initilizations. 
###############################################################################################
# Create a strictly typed loan dataset for generic loan data.
# These are generic properties of a loan.
# All of the loan related Datasets will have them.
# Column1:  ID
# Column2:  Last Credit Amount
# Column3:  Last Used Credit
# Column4:  Last Paid
# Column5:  Last Unused Credit : Column2 - Column3
# Column6:  Last Unpaid Credit : Column2 - Column4
# Column7:  Average Credit Amount (integral divided by length)
# Column8:  Average Credit Amount (integral divided by length)
# Column9:  Average of Used Credit (integral divided by length)
# Column10: Average Paid
# Column11: Average Unused Credit : Column7 - Column8
# Column12: Average Unpaid Credit : Column8 - Column9
# Column13: Will Person Default on Loan

# Create an empty data frame with the strict types.
#generic_loan_df <- data.frame(
#  id             = integer(),
#  last_credit    = numeric(),
#  last_used      = numeric(),
#  last_paid      = numeric(),
#  last_unused    = numeric(),
#  last_unpaid    = numeric(),
#  average_credit = numeric(),
#  average_used   = numeric(),
#  average_paid   = numeric(),
#  average_unused = numeric(),
#  average_unpaid = numeric(),
#  default        = logical()
#)

###############################################################################################
# Utility functions. 
###############################################################################################

# Check if a value is numeric
is_numeric <- function(value){
  is_num = !(is.factor(value)|is.logical(value)|is.character(value))
  return(is_num)
}

# Convert a string vector to a number one.
string_to_number <- function(string_vector){
  return(unclass(factor(string_vector)))
}

# Normalize dataset.
normalize_dataset <- function(data_set) {
  data_set <- data_set
  for(i in 1:ncol(data_set)) {
    data_column <- data_set[,i]
    if(is_numeric(data_column[1])) {
      if(min(data_column) < 0 || max(data_column) > 1) {
        # Normalize to be between 1 and 0.
        data_set[,i] <- (data_column-min(data_column))/(max(data_column)-min(data_column))
      }
    } else if(is.logical(data_set[1,i])) {
      data_set[,i] <- as.integer(as.logical(data_column))
    }
  }
  # Now create label columns.
  for(i in 1:ncol(data_set)) {
    if (is.factor(data_set[1,i])) {
      item_list <- strsplit(as.character(data_set[,i]),"-")
      unique_items <- unique(unlist(item_list))
      item_columns <- do.call(rbind,lapply(item_list, function(x) table(factor(x, levels=unique_items))))
      data_set <- cbind(data_set[,-i],item_columns)
    }
  }
  return(data_set)
}

# Combine ensemble models.
combine_ensemble_models <- function(ensemble_list_1, ensemble_list_2){
  output_weight <- append(ensemble_list_1$output_weight, ensemble_list_2$output_weight)
  all_trees <- append(ensemble_list_1$all_trees, ensemble_list_2$all_trees)
  return(list(output_weight = output_weight, all_trees = all_trees))
}

# Get current directory.
#get_current_dir <- function() {
#  current_file_dir <- rstudioapi::getActiveDocumentContext()$path
#  return(gsub("loanAI.R", "", current_file_dir))
#}

# Save a model.
save_model <- function(model) {
  current_time <- format(Sys.time(), "%Y%m%d%H%M%S") 
  file_name <- paste("./","models/","loanAI_",current_time,".rds",sep="")
  saveRDS(model, file = file_name)
}

# Load models.
load_models <- function() {
  models_list <- list()
  models_dir <- paste("./","models/",sep="")
  files <- list.files(models_dir, pattern=NULL, all.files=FALSE, full.names=TRUE)
  files_length <- length(files)
  if(files_length > 0) {
    for(i in 1:length(files)){
      models_list[[i]] <- readRDS(files[[i]])
    }
  }
  return(models_list)
}

###############################################################################################
# Decision Tree implementation. 
###############################################################################################

# Expected information needed.
entropy <- function(class_column)
{
  n<-length(class_column)
  if(n == 0) {
    # If empty return 0.
    return (0)
  }
  # Find frequency.
  freq <- table(class_column)/n
  # Calculate entropy.
  # Add "0.000000001" (1e-9) to handle 0 cases.
  ent <- -sum(freq * log((freq+1e-9), base = 2))
  return(ent)
}

# Information needed.
information_need <- function(class_column, sub_feature_column){
  s1 <- sum(sub_feature_column)
  s2 <- length(sub_feature_column)-s1
  if(s1 ==0 | s2 == 0) {
    return(0)
  }
  info_need = (s1/(s1+s2)*entropy(class_column[sub_feature_column]))+(s2/(s1+s2)*entropy(class_column[!sub_feature_column]))
  return(info_need)
}

# Information gained.
information_gain <- function(class_column,sub_feature_column){
  info_gain = entropy(class_column)-information_need(class_column,sub_feature_column)
  return(info_gain)
}

# Maximumum information gain for a feature.
information_gain_split <- function(class_column,feature_column){
  biggest_change <- NA
  split_value <- NA
  minimum_sub_feature_size <- 5
  is_num <- is_numeric(feature_column)
  for( val in sort(unique(feature_column))){
    if (is_num) {
      # If numeric set sub-feature column as range.
      sub_feature_column <- feature_column < val
    } else {
      sub_feature_column <- feature_column == val
    }
    change <- information_gain(class_column,sub_feature_column)
    
    if((is.na(biggest_change) | change > biggest_change)
        && sum(sub_feature_column) >= minimum_sub_feature_size
        && sum(!sub_feature_column) >= minimum_sub_feature_size
      ){
      biggest_change = change
      split_value = val
    }
  }
  return(list(
    "biggest_change"=biggest_change,
    "split_value"=split_value,
    "is_numeric"=is_num
  ))
}

# Best feauture to split.
best_feature_split <- function(feature_columns,class_column){
  # Traverse over feature_columns.
  results <- sapply(feature_columns,function(x) information_gain_split(class_column,x))
  # Get the feature with the biggest information gain.
  feature <- names(which.max(results['biggest_change',]))
  best_feature <- results[,feature]
  # Store feature's name for future reference.
  best_feature["feature"] <- feature
  return(best_feature)
}

# Get the rows to include in left_node as true/false.
left_node_include_column <- function(feature_columns,best_feature_split){
  if(best_feature_split$is_numeric){
    left_node_includes <- feature_columns[,best_feature_split$feature] < best_feature_split$split_value
  } else {
    left_node_includes <- feature_columns[,best_feature_split$feature] == best_feature_split$split_value
  }
  return(left_node_includes)
}

# Recursive function to create tree nodes.
create_nodes <- function(feature_columns, class_column, parent_node, maximum_depth = 10) {
  best_feature <- best_feature_split(feature_columns,class_column)
  # If there are no more features, make parent_node a leaf.
  if (length(best_feature) > 0) {
    left_node_includes <- left_node_include_column(feature_columns,best_feature)
    
    current_depth <- parent_node$depth + 1
    
    # Create the left node.
    left_node_name <- paste(best_feature$feature, " < ", best_feature$split_value)
    left_node <- parent_node$AddChild(left_node_name)
    left_node$depth <- current_depth
    left_node$is_numeric <- best_feature$is_numeric
    left_node$feature <- best_feature$feature
    left_node$split_value <- best_feature$split_value
    left_node$prediction <- names(which.max(table(class_column[left_node_includes])))
    # Create the right node.
    right_node_name <- paste(best_feature$feature, " >= ", best_feature$split_value)
    right_node <- parent_node$AddChild(right_node_name)
    right_node$depth <- current_depth
    right_node$is_numeric <- best_feature$is_numeric
    right_node$feature <- best_feature$feature
    right_node$split_value <- best_feature$split_value
    right_node$prediction <- names(which.max(table(class_column[!left_node_includes])))
  
    if (current_depth < maximum_depth){
      left_node$is_leaf <- FALSE
      # Create the child node.
      create_nodes(feature_columns[left_node_includes,], class_column, left_node, maximum_depth)
      right_node$is_leaf <- FALSE
      # Create the child node.
      create_nodes(feature_columns[!left_node_includes,], class_column, right_node, maximum_depth)
    } else {
      left_node$is_leaf <- TRUE
      right_node$is_leaf <- TRUE
    }
  } else {
    parent_node$is_leaf <- TRUE
  }
}

# Start the tree.
create_tree <- function(feature_columns, class_column, maximum_depth = 10) {
  tree <- Node$new("ROOT")
  tree$depth <- 0
  tree$is_leaf <- FALSE
  tree$prediction <- NA
  
  create_nodes(feature_columns, class_column, tree, maximum_depth)
  
  return(tree)
}

# Recursive funtion to predict a given feature_row based on a tree_model.
predict_feature_row <- function(feature_row, tree_model){
  # Check if we reached to left.
  if(!tree_model$is_leaf) {
    # Check if the row belowngs to left node.
    left_node <- tree_model$children[[1]]
    if(left_node$is_numeric){
      is_left <- feature_row[,left_node$feature] < left_node$split_value
    } else {
      is_left <- feature_row[,left_node$feature] == left_node$split_value
    }
    if(is_left){
      predict_feature_row(feature_row, left_node)
    } else {
      # If not it belongs to right node.
      right_node <- tree_model$children[[2]]
      predict_feature_row(feature_row, right_node)
    }
  } else {
    return(tree_model$prediction)    
  }
}

# Tree predict function.
predict = function(feature_columns, tree_model){
  prediction <- character(length=dim(feature_columns)[1])
  for(i in 1:dim(feature_columns)[1]){
    prediction[i] <- predict_feature_row(feature_columns[i,], tree_model)
  }
  # Do not add quotations.
  return(noquote(prediction))
}

###############################################################################################
# Boosting implementation 
###############################################################################################

# Create a sample bag from the training set based on weights. 
ensemble_sample <- function(training_columns, class_column, weights, size) {
  all_columns <-  cbind(class_column,training_columns)
  sample_columns_bag <- all_columns[sample(nrow(all_columns), size=size, prob=weights), ]
  return(list(training_columns = sample_columns_bag[,-1], class_column = sample_columns_bag[,1]))
}

# A custom adaboost implementation.
adaptive_boost <- function(training_columns, class_column, num_rounds, max_tree_dept = 10, sample_size = nrow(training_columns)) {
  # A list of trained models.
  all_trees <- list()
  # The weight for different learners.
  output_weight <- c()
  num_training_cols <- nrow(training_columns)

  # Vector with objects weights.
  weights <- rep(1/num_training_cols, num_training_cols)
  # Add an epsilon value in case error_rate is 0.
  epsilon <- 1e-9
  # Keep the full sample size at first run.
  # Reduce sample size to 80 percent.
  sample_size = as.integer(sample_size * (4/5))
  training_sample <- ensemble_sample(training_columns, class_column, weights, sample_size)

  for(i in 1:num_rounds) {
    str(paste(i,"th round of training...",sep=""))
    decison_tree <- create_tree(training_sample$training_columns, training_sample$class_column, max_tree_dept)
    
    train_predict <- predict(training_columns, decison_tree)

    error_rate = sum(weights*(class_column != train_predict))/sum(weights)
    output_weight_i  <- (0.5) * log(( 1 - error_rate + epsilon) / (error_rate + epsilon))
    weights <- weights * exp(output_weight_i * (class_column != train_predict))
    # Create a new sample bag based on calculated weights.
    training_sample <- ensemble_sample(training_columns, class_column, weights, sample_size)
    all_trees[[i]] <- decison_tree
    output_weight[[i]] <- output_weight_i
    # Overwrite trees that have more than 70 percent errror rate.
    if(error_rate > 0.7){
      i <- i -1
      num_rounds <- num_rounds -1
    }
    str(paste(i,"th round of training complate.",sep=""))
  }
  return(list(output_weight = output_weight, all_trees = all_trees))
}

# Combination scheme for numerical predictions.
weighted_mean <- function(vector, weight=rep(1, length(vector))) {
  return(sum(weight*vector)/sum(weight))
}

# Weighted frequency table.
weighted_table <- function(vector, weight=rep(1, length(vector))) {
  frequency_table <- as.table(replace(wt <- tapply(weight, list(vector),
    function(w1) sum(w1)), is.na(wt), 0))
  return(frequency_table)
}

# Combination scheme for non-numerical predictions.
weighted_modal <- function(vector, weight=rep(1, length(vector))) {
  modal <- which.max(weighted_table(vector, weight=weight))
  if (is.factor(vector)) {
    return(factor(levels(vector)[modal], levels=levels(vector)))
  } else {
    return(sort(unique(vector))[modal])
  }
}

# Combine base models by weighted voting/averaging.
predict_ensemble <- function(predict_columns, weights, all_trees) {
  predictions <- data.frame(lapply(all_trees,
    function(tree) predict(predict_columns, tree)))
  # Select combination scheme by prediction data type.
  if (is_numeric(predictions[,1])){
    combination_method <- weighted_mean
  } else {
    combination_method <- weighted_modal
  }
  # Combine predictions.
  combined_prediction <- sapply(1:nrow(predictions),
    function(i) combination_method(as.vector(as.matrix(predictions[i,])), weights))

  # If factor encode the vector as so.
  if (is.factor(predictions)) {
    return(factor(combined_prediction, levels=levels(predictions)))
  } else {
    return(noquote(combined_prediction))
  }
}

###############################################################################################
# Main loanAI subfunctions.
###############################################################################################



###############################################################################################
# Main loanAI function.
###############################################################################################
loanAI <- function(training_columns, class_column, predict_columns, num_rounds = 10, max_tree_dept = 10, num_reduce_size = 5) {
  
  # Normalize Data
  str("Normalizing data...")
  training_columns <- normalize_dataset(training_columns)
  predict_columns <- normalize_dataset(predict_columns)

  # Keep 10 percent of the trainning data to predict performance and store potentially good ensamble models.
  #all_train_columns <-  cbind(class_column,training_columns)
  #all_train_columns$spl = sample.split(all_train_columns, SplitRatio = 0.10)
  #new_test <- subset(subset(all_train_columns, all_train_columns$spl == FALSE), select = -c(spl))
  #test_columns <- new_test[,-1]
  #test_predicts <- new_test[,1]
  #new_train <- subset(subset(all_train_columns, all_train_columns$spl == TRUE), select = -c(spl))
  # Final data to be used for training.
  #training_columns <- new_train[,-1]
  #class_column <- new_train[,1]
  #training_columns <- normalized_train

  # Train big models.
  str("Starting to train...")
  str("###Training big ensemble models###")
  ensemble_models <- adaptive_boost(training_columns, class_column, num_rounds)
  
  # Train smaller models.
  if(num_reduce_size > 0) {
    sample_size = nrow(training_columns)
    for(i in 1:num_reduce_size){
      sample_size = as.integer(sample_size / (i + 1))
      # Break if smaller than 100.
      if(sample_size < 100) {
        break
      }
      str("###Training reduced ensemble models###")
      reduced_ensemble_models <- adaptive_boost(training_columns, class_column, num_rounds, sample_size)
      # Append to ensamble_models.
      ensemble_models <-  combine_ensemble_models(ensemble_models, reduced_ensemble_models)
    }
  }
  
  # Predict performance of the current ensemble.
  #performance_predict <- predict_ensemble(test_columns, ensemble_models$output_weight, ensemble_models$all_trees)
  #cfm <- confusionMatrix(table(performance_predict, test_predicts))
  #str(cfm$overall['Accuracy'])
  
  # Load past generic models.
  past_models <- load_models()

  # Save the current generic model to use in the future.
  save_model(ensemble_models)
  
  # Append past generic models to the ensemble.
  past_models_length <- length(past_models)
  if(past_models_length > 0) {
    for(i in 1:length(past_models_length)){
      ensemble_models <-  combine_ensemble_models(past_models[[i]], ensemble_models)
    }
    str("###Old models are combined.###")
  }
  
  # Return the actual prediction.
  return(predict_ensemble(predict_columns, ensemble_models$output_weight, ensemble_models$all_trees))
}
