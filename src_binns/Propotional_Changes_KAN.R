## Packages
library(R.matlab)
library(ggplot2)
library(cowplot)
library(viridis)
library(scales)
library(sf)
library(sp)
library(GGally)
library(raster)
library(proj4)
library(lme4)
library(ncdf4)
library(readxl)
library(dplyr)
library(tidyr)
library(colorspace)
library(farver) 



##
rm(list = ls())

setwd('D:/BINN/OUTPUT_DATA/')

date_stamp = '20250601-165700_KAN_ONELAYER_GRIDMARGIN_COMPAS2_9700234_lr=1e-02_fold=0_seed=111'

data_path = paste0('D:/Research/BINN/BINN_output/neural_network/',  date_stamp, '/')
################################################
# Define a color theme for each parameter
################################################
para_names = c('diffus', 'cryo', 'q10', 'efolding', 'taucwd', 'taul1', 'taul2', 'tau4doc', 'tau4mic', 'tau4poc', 'tau4maom',
'fl1_MIC', 'fl2_MIC', 'fMIC_DOC', 'fMIC_POC', 'fDOC_MAOM', 'CUEl1', 'CUEl2', 'CUEDOC', 'w-scaling', 'beta')
length(para_names)

para_groups = c('V', 'V', 'Xi', 'Xi', 'K_litter', 'K_litter', 'K_litter', 'K_soil', 'K_soil', 'K_soil', 
'K_soil', 'A', 'A', 'A', 'A', 'A', 'CUE', 'CUE', 'CUE','Xi','I')
length(para_groups)

# Define colors for each parameters
para_colors <- c(
  # ── V (teal) ─────────────────────────────
  diffus      = "#006D77",   # darkest
  cryo        = "#83C5BE",

  # ── Xi (violet) ──────────────────────────
  q10         = "#4f159a",   # darkest
  efolding    = "#6659b3",
  `w-scaling` = "#adaace",   # lightest

  # ── K_litter (green) ─────────────────────
  taucwd      = "#00441B",   # darkest
  taul1       = "#238B45",
  taul2       = "#66C2A4",   # lightest

  # ── K_soil (blue) ─────────────────
  tau4doc     = "#08306B",   # darkest blue
  tau4mic     = "#1a6fb8",
  tau4poc     = "#59a3cf",
  tau4maom    = "#c8dff5",   # lightest blue

  # ── A (red) ──────────────────────────────
  fl1_MIC     = "#a80207",   # darkest
  fl2_MIC     = "#fd514b",
  fMIC_DOC    = "#ea6749",
  fMIC_POC    = "#eaaf9a",
  fDOC_MAOM   = "#fad2c0",   # lightest

  # ── CUE (orange) ─────────────────────────
  CUEl1       = "#5a2713",   # darkest
  CUEl2       = "#FD8D3C",
  CUEDOC      = "#FDAE6B",   # lightest

  # ── I (neutral gray) ─────────────────────
  beta        = "#4c4c4c"
)
length(para_colors)

#############################################
# figure: Propotional Changes of Total Pools
#############################################
categories = c('soc', 'POM', 'MAOM', 'DOC', 'MIC')
# For each pool, calculate the proportional change in carbon stocks due to parameter perturbations
# Load the percent change data
percent_changed_file <- read.table(paste0(data_path, '/KAN_Propotional_Change/proportional_change.txt'), header = FALSE, sep = '\t')
# percent_changed <- as.numeric(percent_changed_file[length(para_names)+1:nrow(percent_changed_file), ])
percent_changed <- as.numeric(percent_changed_file$V1)
percent_changed <- percent_changed[!is.na(percent_changed)]

baseline_file <- read.table(paste0(data_path, '/KAN_Propotional_Change/baseline_values.txt'), header = FALSE, sep = ':')
# Save the last row into baseline_para and remove it from the baseline_file
baseline_para_ori <- baseline_file[nrow(baseline_file), ]
baseline_carbon_ori <- baseline_file[-nrow(baseline_file), ]

###########################################
# figure: Changes of Env_Info vs Parameter
###########################################

# Loop through each environmental info and plot the proportional changes
# Get the environmental info data from folder name within KAN_Propotional_Change
env_info_files <- list.files(paste0(data_path, '/KAN_Propotional_Change/'), full.names = TRUE)
# Get only the file names (after the last '/')
env_info_files <- gsub(paste0(data_path, '/KAN_Propotional_Change/'), '', env_info_files)
# remove all name end with .txt
env_info_names <- env_info_files[!grepl('\\.txt$', env_info_files)]

# Create an output directory for proportional change plots
if (!dir.exists(paste0(data_path, '/Env_Info_Change_Plots'))) {
  dir.create(paste0(data_path, '/Env_Info_Change_Plots'))
}

# i <- 1
for (i in seq_along(env_info_names)) {
  # Get the name of the environmental info
  env_info_name <- env_info_names[i]
  # Use minerals_info_mean as baseline data
  baseline_data <- baseline_carbon_ori
  baseline_para <- baseline_para_ori[2]
  baseline_para <- gsub('\\[|\\]', '', baseline_para)
  baseline_para <- as.numeric(unlist(strsplit(baseline_para, ' ')))
  baseline_para <- baseline_para[!is.na(baseline_para)]

  # Create a data frame to store the proportional change data and actual values
  para_prop_df <- data.frame(baseline_para = baseline_para)
  para_prop_df$baseline_para <- 0
  percent_changed_actual_values <- c(0, percent_changed)

  # Load the parameter data from file
  prop_para_file <- read.table(paste0(data_path, '/KAN_Propotional_Change/', env_info_name, '/para_change.txt'), header = FALSE, sep = ' ')
  # Convert the data to data frame by separating by space
  prop_para_file <- t(as.data.frame(prop_para_file))
  
  # j <- 1
  for (j in seq_along(percent_changed)) {
    # Calculate the proportional change between the baseline and the perturbed parameter
    para_prop <- (prop_para_file[,j] - baseline_para) / baseline_para * 100
    # Add the proportional change to the data frame
    para_prop_df <- cbind(para_prop_df, para_prop) 
  }

  colnames(para_prop_df) <- c('baseline_para', paste0('percent_changed_', percent_changed))
  
  # Reverse the dataframe so that the column are parameters and the rows are percent changes
  para_prop_df <- t(para_prop_df)
  para_prop_df_plot <- para_prop_df
  colnames(para_prop_df_plot) <- para_names
  x_axis <- c(0, percent_changed*100)
  para_prop_df_plot <- data.frame(x_axis, para_prop_df_plot)
  long_df <- para_prop_df_plot %>% 
    pivot_longer(                
      cols      = -x_axis,              
      names_to  = "parameter",
      values_to = "prop_change"
    )
  # Plot the proportional changes as dots into a line plot with predefined colors for each parameter
  p <- ggplot(data = long_df, aes(x = x_axis, y = prop_change, color = parameter)) +
    geom_line(size = 1) +
    geom_point(size = 2) +
    scale_color_manual(values = para_colors) +
    scale_x_continuous(name = "Percent Change (%)", limits = c(-10, 10), breaks = seq(-10, 10, 1)) +
    scale_y_continuous(name = paste0("Proportional Change in ", env_info_name, " (%)"), limits = c(-50, 50), breaks = seq(-50, 50, 10)) +
    theme_minimal() +
    theme(
      axis.title.x = element_text(size = 14),
      axis.title.y = element_text(size = 14),
      legend.position = "none",
      panel.grid.major = element_line(color = "grey80"),
      panel.grid.minor = element_blank()
    ) +
    labs(title = paste0("Proportional Changes in ", env_info_name)) +
    theme(plot.title = element_text(hjust = 0.1, size = 16, face = "bold")) + 
    theme(legend.position = "right")
  
  # Save the plot as a JPEG file
  jpeg(paste0(data_path, '/Env_Info_Change_Plots/', env_info_name, '_para.jpg'), width = 8, height = 6, units = "in", res = 300)
  print(p)
  dev.off()

  # Plot the proportional changes of environmental info vs carbon pools
  # m <- 1
  for (m in 1:length(categories)) {
    # For each parameter, load the proportional change data
    # Use the data and the percent changed to fit a linear model
    # Store the linear model coefficients in a data frame for further plotting
    para_propotional_change_func <- data.frame()
    # Load the baseline data
    baseline_carbon_data <- baseline_data$V2[m]
    baseline_carbon_data <- as.numeric(unlist(strsplit(baseline_carbon_data, ' ')))
    baseline_carbon_data <- baseline_carbon_data[!is.na(baseline_carbon_data)]

    # Load the data by categories
    category_file <- read.table(paste0(data_path, '/KAN_Propotional_Change/', env_info_name, '/', categories





    # Assign the long_df data frame to the para_prop_df data frame
    para_prop_df_plot <- long_df
    para_prop_df_plot_actual_values <- long_df_actual_values
    for (j in 1:length(para_names)) {
      # Load the data
      para_prop_file <- read.table(paste0(data_path, '/Propotional_Change/', para_names[j], '/', categories[i], '_prop_change.txt'), header = FALSE, sep = ',')
      # Calculate the proportional change
      para_prop <- (para_prop_file - baseline_data) / baseline_data * 100
      # Calculate the mean and standard deviation of the proportional change accross each column
      para_prop_mean <- apply(para_prop, 2, mean, na.rm = TRUE)
      para_prop_sd <- apply(para_prop, 2, sd, na.rm = TRUE)
      # Create a data frame with the proportional change data
      para_prop_df <- data.frame(para_prop_mean, para_prop_sd)
      # Add the percent changed data to the data frame
      para_prop_df$percent_changed <- percent_changed
      # Fit a linear model with 0 intercept
      lm_model <- lm(para_prop_mean ~ percent_changed + 0, data = para_prop_df)
      # Get the coefficients of the linear model
      lm_coef <- coef(lm_model)
      # Get the standard error of the coefficients
      lm_se <- summary(lm_model)$coefficients[, 2]
      # Get the R-squared value of the linear model
      lm_r2 <- summary(lm_model)$r.squared
      # Store the coefficients in a data frame with the parameter name
      para_propotional_change_func <- rbind(para_propotional_change_func, data.frame(parameter = para_names[j], coef = lm_coef, se = lm_se, r2 = lm_r2))
      # Multiply the coefficients by the proportional change of the environmental info based on the parameter
      para_prop_df_plot[para_prop_df_plot$parameter == para_names[j], 'prop_change'] <- para_prop_df_plot[para_prop_df_plot$parameter == para_names[j], 'prop_change'] * para_propotional_change_func$coef[para_propotional_change_func$parameter == para_names[j]]
      para_prop_df_plot_actual_values[para_prop_df_plot_actual_values$parameter == para_names[j], 'prop_change'] <- para_prop_df_plot_actual_values[para_prop_df_plot_actual_values$parameter == para_names[j], 'prop_change'] * para_propotional_change_func$coef[para_propotional_change_func$parameter == para_names[j]]
    }
    # Plot the proportional changes as dots into a line plot with predefined colors for each parameter
    p <- ggplot(data = para_prop_df_plot, aes(x = x_axis, y = prop_change, color = parameter)) +
      geom_line(size = 1) +
      geom_point(size = 2) +
      scale_color_manual(values = para_colors) +
      scale_x_continuous(name = "Percent Change (%)", limits = c(-4, 4), breaks = seq(-4, 4, 1)) +
      scale_y_continuous(name = paste0("Proportional Change in ", categories[i], " (%)"), limits = c(-5, 5), breaks = seq(-10, 10, 1)) +
      theme_minimal() +
      theme(
        axis.title.x = element_text(size = 14),
        axis.title.y = element_text(size = 14),
        legend.position = "none",
        panel.grid.major = element_line(color = "grey80"),
        panel.grid.minor = element_blank()
      ) +
      labs(title = paste0("Proportional Changes in ", env_info_name, " vs ", categories[i])) +
      theme(plot.title = element_text(hjust = 0.1, size = 16, face = "bold")) + 
      theme(legend.position = "right")
    
    jpeg(paste0(data_path, '/Propotional_Change/Env_Info_Change/', env_info_name, '/Proportional_Change_', env_info_name, '_', categories[i], '.jpg'), width = 8, height = 6, units = "in", res = 300)
    print(p)
    dev.off()

    # Plot the actual values of the proportional changes
    p <- ggplot(data = para_prop_df_plot_actual_values, aes(x = x_axis_actual_values, y = prop_change, color = parameter)) +
      geom_line(size = 1) +
      geom_point(size = 2) +
      scale_color_manual(values = para_colors) +
      scale_x_continuous(name = env_info_name, limits = c(min(x_axis_actual_values), max(x_axis_actual_values)), breaks = seq(min(x_axis_actual_values), max(x_axis_actual_values), ((max(x_axis_actual_values) - min(x_axis_actual_values)) / 10))) +
      scale_y_continuous(name = paste0("Proportional Change in ", categories[i], " (%)"), limits = c(-5, 5), breaks = seq(-10, 10, 1)) +
      theme_minimal() +
      theme(
        axis.title.x = element_text(size = 14),
        axis.title.y = element_text(size = 14),
        legend.position = "none",
        panel.grid.major = element_line(color = "grey80"),
        panel.grid.minor = element_blank()
      ) +
      labs(title = paste0("Proportional Changes in ", env_info_name, " vs ", categories[i])) +
      theme(plot.title = element_text(hjust = 0.1, size = 16, face = "bold")) + 
      theme(legend.position = "right")

    jpeg(paste0(data_path, '/Propotional_Change/Env_Info_Change/', env_info_name, '/Proportional_Change_', env_info_name, '_', categories[i], '_actual_values.jpg'), width = 8, height = 6, units = "in", res = 300)
    print(p)
    dev.off()
  }

    # Plot the proportional changes of environmental info vs carbon pools with SOC
    for (i in 1:length(categories)) {
      # Skip the first iteration for 'soc'
      if (categories[i] == 'soc') {
        next
      }
      # For each parameter, load the proportional change data
      # Use the data and the percent changed to fit a linear model
      # Store the linear model coefficients in a data frame for further plotting
      para_propotional_change_func <- data.frame()
      para_prop_df_plot <- long_df
      para_prop_df_plot_actual_values <- long_df_actual_values
      # Load the baseline data
      baseline_file <- read.table(paste0(data_path, '/Propotional_Change/baseline_', categories[i], '.txt'), header = FALSE, sep = ',')
      baseline_soc_file <- read.table(paste0(data_path, '/Propotional_Change/baseline_soc.txt'), header = FALSE, sep = ',')
      baseline_file$V1 <- baseline_file$V1 / baseline_soc_file$V1 * 100
      # Expand the baseline data to the same columns as the parameter data
      baseline_data <- matrix(rep(baseline_file$V1, length(percent_changed)), ncol = length(percent_changed))
      for (j in 1:length(para_names)) {
        # Load the data
        para_prop_file <- read.table(paste0(data_path, '/Propotional_Change/', para_names[j], '/', categories[i], '_prop_change.txt'), header = FALSE, sep = ',')
        para_soc_file <- read.table(paste0(data_path, '/Propotional_Change/', para_names[j], '/soc_prop_change.txt'), header = FALSE, sep = ',')
        para_prop_file <- para_prop_file / para_soc_file * 100
        # Calculate the proportional change
        para_prop <- (para_prop_file - baseline_data) / baseline_data * 100
        # Calculate the mean and standard deviation of the proportional change accross each column
        para_prop_mean <- apply(para_prop, 2, mean, na.rm = TRUE)
        para_prop_sd <- apply(para_prop, 2, sd, na.rm = TRUE)
        # Create a data frame with the proportional change data
        para_prop_df <- data.frame(para_prop_mean, para_prop_sd)
        # Add the percent changed data to the data frame
        para_prop_df$percent_changed <- percent_changed
        # Fit a linear model with 0 intercept
        lm_model <- lm(para_prop_mean ~ percent_changed + 0, data = para_prop_df)
        # Get the coefficients of the linear model
        lm_coef <- coef(lm_model)
        # Get the standard error of the coefficients
        lm_se <- summary(lm_model)$coefficients[, 2]
        # Get the R-squared value of the linear model
        lm_r2 <- summary(lm_model)$r.squared
        # Store the coefficients in a data frame with the parameter name
        para_propotional_change_func <- rbind(para_propotional_change_func, data.frame(parameter = para_names[j], coef = lm_coef, se = lm_se, r2 = lm_r2))
        # Multiply the coefficients by the proportional change of the environmental info based on the parameter
        para_prop_df_plot[para_prop_df_plot$parameter == para_names[j], 'prop_change'] <- para_prop_df_plot[para_prop_df_plot$parameter == para_names[j], 'prop_change'] * para_propotional_change_func$coef[para_propotional_change_func$parameter == para_names[j]]
        para_prop_df_plot_actual_values[para_prop_df_plot_actual_values$parameter == para_names[j], 'prop_change'] <- para_prop_df_plot_actual_values[para_prop_df_plot_actual_values$parameter == para_names[j], 'prop_change'] * para_propotional_change_func$coef[para_propotional_change_func$parameter == para_names[j]]
        }
      # Plot the proportional changes as dots into a line plot with predefined colors for each parameter
      p <- ggplot(data = para_prop_df_plot, aes(x = x_axis, y = prop_change, color = parameter)) +
        geom_line(size = 1) +
        geom_point(size = 2) +
        scale_color_manual(values = para_colors) +
        scale_x_continuous(name = "Percent Change (%)", limits = c(-4, 4), breaks = seq(-4, 4, 1)) +
        scale_y_continuous(name = paste0("Proportional Change in ", categories[i], " / SOC (%)"), limits = c(-5, 5), breaks = seq(-10, 10, 1)) +
        theme_minimal() +
        theme(
          axis.title.x = element_text(size = 14),
          axis.title.y = element_text(size = 14),
          legend.position = "none",
          panel.grid.major = element_line(color = "grey80"),
          panel.grid.minor = element_blank()
        ) +
        labs(title = paste0("Proportional Changes in ", env_info_name, " vs ", categories[i], " / SOC")) +
        theme(plot.title = element_text(hjust = 0.1, size = 16, face = "bold")) + 
        theme(legend.position = "right")

      # Save the plot as a JPEG file
      jpeg(paste0(data_path, '/Propotional_Change/Env_Info_Change/', env_info_name, '/Proportional_Change_', env_info_name, '_', categories[i], '_SOC.jpg'), width = 8, height = 6, units = "in", res = 300)
      print(p)
      dev.off()

      # Plot the actual values of the proportional changes
      p <- ggplot(data = para_prop_df_plot_actual_values, aes(x = x_axis_actual_values, y = prop_change, color = parameter)) +
        geom_line(size = 1) +
        geom_point(size = 2) +
        scale_color_manual(values = para_colors) +
        scale_x_continuous(name = env_info_name, limits = c(min(x_axis_actual_values), max(x_axis_actual_values)), breaks = seq(min(x_axis_actual_values), max(x_axis_actual_values), ((max(x_axis_actual_values) - min(x_axis_actual_values)) / 10))) +
        scale_y_continuous(name = paste0("Proportional Change in ", categories[i], " / SOC (%)"), limits = c(-5, 5), breaks = seq(-10, 10, 1)) +
        theme_minimal() +
        theme(
          axis.title.x = element_text(size = 14),
          axis.title.y = element_text(size = 14),
          legend.position = "none",
          panel.grid.major = element_line(color = "grey80"),
          panel.grid.minor = element_blank()
        ) +
        labs(title = paste0("Proportional Changes in ", env_info_name, " vs ", categories[i], " / SOC")) +
        theme(plot.title = element_text(hjust = 0.1, size = 16, face = "bold")) + 
        theme(legend.position = "right")

      # Save the plot as a JPEG file
      jpeg(paste0(data_path, '/Propotional_Change/Env_Info_Change/', env_info_name, '/Proportional_Change_', env_info_name, '_', categories[i], '_SOC_actual_values.jpg'), width = 8, height = 6, units = "in", res = 300)
      print(p)
      dev.off()

}
}
