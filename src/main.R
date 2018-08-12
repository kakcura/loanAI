###############################################################################################
# loanAI V1.0
# Execution file for generic loan default model creation.
# Created by Korhan Akcura
###############################################################################################
# Get current directory.
get_current_dir <- function() {
  current_file_dir <- rstudioapi::getActiveDocumentContext()$path
  return(gsub("main.R", "", current_file_dir))
}
setwd(get_current_dir())

# Use the loanAI algorithim for data mining.
source("loanAI.R")
cvs_data <- read.csv('../data/default_of_credit_card_clients.csv')

# Get only the complate cases.
loan_data = cvs_data[complete.cases(cvs_data), ]

#loan_data <- loan_data[sample(nrow(loan_data), size=200, replace=FALSE), ]

# Populate the generic loan data.
generic_loan_df <- data.frame(
  last_credit    = loan_data$LIMIT_BAL,
  last_used      = loan_data$BILL_AMT6,
  last_paid      = loan_data$PAY_AMT6,
  last_unused    = loan_data$LIMIT_BAL - loan_data$BILL_AMT6,
  last_unpaid    = loan_data$LIMIT_BAL - loan_data$PAY_AMT6
)
generic_loan_df$average_used   = rowMeans(loan_data[,13:18], na.rm = FALSE, dims = 1)
generic_loan_df$average_paid   = rowMeans(loan_data[,19:24], na.rm = FALSE, dims = 1)
generic_loan_df$average_unused = generic_loan_df$last_credit - generic_loan_df$average_used
generic_loan_df$average_unpaid = generic_loan_df$average_used - generic_loan_df$average_paid
generic_loan_df$past_payments  = rowMeans(loan_data[,7:12], na.rm = FALSE, dims = 1)
generic_loan_df$sex            = loan_data$SEX
generic_loan_df$education      = loan_data$EDUCATION
generic_loan_df$marriage       = loan_data$MARRIAGE
generic_loan_df$age            = loan_data$AGE
generic_loan_df$default        = loan_data$default

#loan_data <- generic_loan_df

loan_data <- generic_loan_df[sample(nrow(generic_loan_df), size=600, replace=FALSE), ]

# Divide 70% of data for training and 30% for test.
loan_data$training_split = sample.split(loan_data, SplitRatio = 0.80)
train <- subset(subset(loan_data, loan_data$training_split == TRUE), select = -c(training_split))
test <- subset(subset(loan_data, loan_data$training_split == FALSE), select = -c(training_split))

# The last column is the classification column.
class_column_index <- ncol(train)

# loanAI does the necessary pre-processing and post-processing.
# We can pass the dataset paramaters directly withouth pre-processing.
loanAI_prediction <- loanAI(train[,-class_column_index], train[,class_column_index], test[,-class_column_index], 50, 20, 0)

loanAI_prediction

cfm <- confusionMatrix(table(loanAI_prediction, test[,class_column_index]))

cfm
