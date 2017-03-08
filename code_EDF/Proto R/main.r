library("readr")

output <- read_delim("/Volumes/Data/Cours/Centrale/3A/Monaco/lqmj/data_EDF/challenge_output_data_training_file_predict_which_clients_reduced_their_consumption.csv", ";", escape_double = FALSE, trim_ws = TRUE)
conso <- read_delim("/Volumes/Data/Cours/Centrale/3A/Monaco/lqmj/data_EDF/consommation-electrique-par-secteurs-dactivite.csv", ";", escape_double = FALSE, trim_ws = TRUE)
training <- read_delim("/Volumes/Data/Cours/Centrale/3A/Monaco/lqmj/data_EDF/training_inputs.csv", ";", escape_double = FALSE, trim_ws = TRUE)
testing <- read_delim("/Volumes/Data/Cours/Centrale/3A/Monaco/lqmj/data_EDF/testing_inputs.csv", ";", escape_double = FALSE, trim_ws = TRUE)

testing$COD_IRIS[!testing$COD_IRIS %in% conso$"INSEE IRIS"]
testing$COD_IRIS[testing$COD_IRIS %in% conso$"INSEE IRIS"]
