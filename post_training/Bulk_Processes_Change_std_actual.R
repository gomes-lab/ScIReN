# R script to create SOC difference map

## Packages
library(R.matlab)
library(ggplot2)
library(cowplot)
# library(jcolors)
library(gridExtra)
library(viridis)
library(sf)
library(sp)
library(GGally)
library(raster)
library(proj4)
library(tidyr)
library(dplyr)

##
rm(list = ls())

setwd('D:/Research/BINN/BINN_output/plot/')
# Sys.setenv(PROJ_LIB = "C:/Users/hx293/AppData/Local/R/win-library/4.3/sf/proj")
# Sys.setenv(PROJ_LIB = "C:/Program Files/R/R-4.3.3/library/sf/proj")
Sys.setenv(PROJ_LIB = "C:/Program Files/R/R-4.3.3/library/sf/proj")


## Jet colorbar function
jet.colors <- colorRampPalette(c("#00007F", "blue", "#007FFF", "cyan", "#7FFF7F", "yellow", "#FF7F00", "red", "#7F0000"))
diff.colors <- colorRampPalette(c("#2166AC", "#4393C3", "#92C5DE", "#D1E5F0", "#f6f6f6", "#FDDBC7", "#F4A582", "#D6604D", "#B2182B"))

#############################################################################
# 1 std or 2 std
#############################################################################
# std_num = readline(prompt = "Enter the number of standard deviations (1 or 2): ")
std_num = '2'  

if (std_num != '1' && std_num != '2') {
    stop("Invalid input. Please enter either 1 or 2.")
} else {
    std_num = as.numeric(std_num)
}
input_std_num = paste0('/KAN_Propotional_Change_', std_num, '_std/')
output_std_num = paste0('/Output_plots_', std_num, '_std/')
if (std_num == 1) {
    percentage_change = seq(-1, 1, length.out = 11) # from -1 to 1 standard deviations
} else if (std_num == 2) {
    percentage_change = seq(-2, 2, length.out = 9) # from -2 to 2 standard deviations
}

#############################################################################
# Data Path
#############################################################################
date_stamp = '20250704-213024_KAN_ONELAYER_GRIDMARGIN_COMPAS2_998996_lr=1e-02_fold=0_seed=111'

# job_id = '20250630-003325_KAN_ONELAYER_GRIDMARGIN_COMPAS2_9917068_lr=1e-02_fold=0_seed=111'
# job_id = '20250630-220008_KAN_ONELAYER_GRIDMARGIN_COMPAS2_9926268_lr=1e-02_fold=0_seed=111'
# job_id = '20250704-212930_KAN_ONELAYER_GRIDMARGIN_COMPAS2_998751_lr=1e-02_fold=0_seed=111'
# job_id = '20250704-213024_KAN_ONELAYER_GRIDMARGIN_COMPAS2_998996_lr=1e-02_fold=0_seed=111'
# job_id = '20250704-213005_KAN_ONELAYER_GRIDMARGIN_COMPAS2_998897_lr=1e-02_fold=0_seed=111'

# input and output data path
data_dir_input = paste0('D:/Research/BINN/BINN_output/neural_network/',  date_stamp, '/')
data_dir_output = paste0('D:/Research/BINN/BINN_output/neural_network/',  date_stamp, output_std_num, '_actual_values')
# PRODA data path
data_dir_PRODA = 'D:/Research/BINN/Research_Data/BINN/Server_Script/post_training/soc_component_proda/soc_component_proda/'
data_dir_loc = 'D:/Research/BINN/Research_Data/BINN/Server_Script/post_training/component_calculation/'
# Grid env_info data path
data_dir_grid_env = 'D:/Nutstore/Research_Data/BINN/ENSEMBLE/INPUT_DATA/wosis_2019_snap_shot/env_info_grid_with_Al_Fe.csv'

# create output folder if not exist
if (!dir.exists(data_dir_output)) {
  dir.create(data_dir_output)
}

#############################################################################
# function to increase vertical spacing between legend keys
#############################################################################
# @clauswilke
draw_key_polygon3 <- function(data, params, size) {
  lwd <- min(data$size, min(size) / 4)
  
  grid::rectGrob(
    width = grid::unit(0.6, "npc"),
    height = grid::unit(0.6, "npc"),
    gp = grid::gpar(
      col = data$colour,
      fill = alpha(data$fill, data$alpha),
      lty = data$linetype,
      lwd = lwd * .pt,
      linejoin = "mitre"
    ))
}

# register new key drawing function, 
# the effect is global & persistent throughout the R session
GeomBar$draw_key = draw_key_polygon3

#############################################################################
# Model Information
#############################################################################
# width between two interfaces
dz = c(2e-2, 4e-2, 6e-2, 8e-2, 0.12, 0.16, 
        0.20, 0.24, 0.28, 0.32, 0.36, 0.40, 
        0.44, 0.54, 0.64, 0.74, 0.84, 0.94,
        1.04, 1.14, 2.39, 4.67553390593274,
        7.63519052838329, 11.14, 15.1154248593737)

month_num = 12
kelvin_to_celsius = 273.15
m_to_cm = 100
n_soil_layer = 20
npool = 7
npool_vr = 140
days_per_year = 365

#############################################################################
# Percentage Change Plot
#############################################################################
# Define the percentage change 
# percentage_change = seq(-0.5, 0.5, length.out = 11) # from -50% to +50%


# Define the environmental variables to be changed
var4nn = c("BIO1", "BIO12", "BIO3", "BIO15", "Clay_Silt_avg", "Bulk_Density_avg", "SWC_v_Wilting_Point_avg", "pH_Water_avg", "CEC_avg", "Coarse_Fragments_avg")
var_name = c("Mean Annual Temperature", "Mean Annual Precipitation", "Isothermality (Mean Diurnal Range/Temperature Annual Range)", "Precipitation Seasonality (Coefficient of Variation)", "Clay and Silt Content", "Bulk Density", "Soil Water Content at Wilting Point", "Soil pH", "Cation Exchange Capacity", "Coarse Fragments")
var4nn_max_min = read.table(paste(data_dir_input, input_std_num, 'grid_env_info_max_min.txt', sep = ''), sep = ',', header = FALSE, skip = 1)
var4nn_mean = read.table(paste(data_dir_input, input_std_num, 'grid_env_info_mean.txt', sep = ''), sep = ',', header = FALSE, skip = 1)
var4nn_std = read.table(paste(data_dir_input, input_std_num, 'grid_env_info_std.txt', sep = ''), sep = ',', header = FALSE, skip = 1)

# Matrix: bulk_A_doc, bulk_E_mic, bulk_A_mic, bulk_A_POM, bulk_A_MAOM, bulk_I, bulk_K, bulk_V, bulk_xi, carbon_input, litter_fraction
input_matrix = c('bulk_A_doc', 'bulk_E_mic', 'bulk_A_mic', 'bulk_A_POM', 'bulk_A_MAOM', 'bulk_I', 'bulk_K', 'bulk_V', 'bulk_xi', 'carbon_input', 'litter_fraction')
input_matrix_color = c('#00593e', '#d95f02', '#7570b3', '#e7298a', '#66a61e', '#e6ab02', '#a6761d', '#666666', '#1b9e77', '#d95f02', '#7570b3')
process_name <- c(
  bulk_A_doc        = "Carbon Transfer Efficiency of DOC",
  bulk_E_mic        = "Non-microbial Carbon Transfer Efficiency of MIC",
  bulk_A_mic        = "Microbial Carbon Transfer Efficiency of MIC",
  bulk_A_POM        = "Carbon Transfer Efficiency of POM",
  bulk_A_MAOM       = "Carbon Transfer Efficiency of MAOM",
  bulk_I            = "Carbon Input Allocation",
  bulk_K            = "Baseline Decomposition",
  bulk_V            = "Vertical Transport Rate",
  bulk_xi           = "Environmental Modifier",
  carbon_input      = "Plant Carbon Inputs",
  litter_fraction   = "Litter to Mineral Soil Fraction"
)

# Para names
para_names = c('diffus', 'cryo', 'q10', 'efolding', 'taucwd', 'taul1', 'taul2', 'tau4doc', 'tau4mic', 'tau4poc', 'tau4maom',
'fl1_MIC', 'fl2_POC', 'fMIC_MAOM', 'fMIC_POC', 'fDOC_MAOM', 'CUEl1', 'CUEl2', 'CUEDOC', 'w-scaling', 'beta')
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
  fl1_MIC     = "#520103",   # darkest
  fl2_POC     = "#980500",
  fMIC_MAOM   = "#fc5630",
  fMIC_POC    = "#ff9e7a",
  fDOC_MAOM   = "#ffd7c5",   # lightest

  # ── CUE (orange) ─────────────────────────
  CUEl1       = "#5a2713",   # darkest
  CUEl2       = "#ff6a00",
  CUEDOC      = "#b97b45",   # lightest

  # ── I (neutral gray) ─────────────────────
  beta        = "#4c4c4c"
)
length(para_colors)

# SOC Category names
soc_category_names = c('soc', 'DOC', 'MIC', 'POM', 'MAOM')

# Get the longitude and latitude for plotting the map
grid_env_info = read.csv(data_dir_grid_env, header = TRUE, stringsAsFactors = FALSE)
grid_lon = grid_env_info$Lon
grid_lat = grid_env_info$Lat
grid_data_baseline = cbind(grid_lon, grid_lat)
colnames(grid_data_baseline) = c('lon', 'lat')

# Load the SOC data to get the outliers rows
soc_data_baseline = array(NA, dim = c(nrow(grid_data_baseline), length(soc_category_names)))
i = 1
for (i in 1:length(soc_category_names)) {
    temp_soc_baseline = read.table(paste(data_dir_input, input_std_num, '/baseline', '/', soc_category_names[i], '.txt', sep = ''), sep = ',', header = FALSE)
    # Multiply by the width of the interface and sum each row up
    temp_soc_baseline = sweep(temp_soc_baseline, 2, dz[1:ncol(temp_soc_baseline)], FUN = '*') # gC/m2
    temp_nan_index = which(is.na(temp_soc_baseline[ , 1]))
    temp_soc_baseline_sum = rowSums(temp_soc_baseline, na.rm = TRUE) #gC/m2
    temp_soc_baseline_sum = temp_soc_baseline_sum/1000 # Convert to kgC/m2
    temp_soc_baseline_sum[temp_nan_index] = NA # Set the NA rows to NA
    soc_data_baseline[ , i] = temp_soc_baseline_sum
}
soc_data_baseline = cbind(grid_data_baseline, soc_data_baseline)
dim(soc_data_baseline)
outliers_rows = which(soc_data_baseline[ , 3] > 2000)


## Plot 1: Percentage Change of Bulk Processes
if (!dir.exists(paste0(data_dir_output, '/Percentage_Change_Bulk_Processes'))) {
    dir.create(paste0(data_dir_output, '/Percentage_Change_Bulk_Processes'))
}

for (var_idx in 1:length(var4nn)) {
    change_start_time = Sys.time()
    var_name_temp = var4nn[var_idx]
    print(paste0("Processing variable: ", var_name_temp))

    # Initialize an array to store plots
    bulk_process_prop_change <- matrix(NA_real_,
                                     nrow = length(percentage_change),
                                     ncol = length(input_matrix),
                                     dimnames = list(NULL, input_matrix))

    for (imatrix in 1:length(input_matrix)) {
      bulk_process_prop_change_temp = array(NA, dim = c(nrow(grid_data_baseline), length(percentage_change)))
      # Loop through each percentage change
      for (ichange in 1:length(percentage_change)) {
          # Load the data for the current variable and percentage change
          if (input_matrix[imatrix] == 'bulk_I') {
              bulk_process_file = paste0(data_dir_input, input_std_num, var_name_temp, '/prop_change_', sprintf("%.2f", percentage_change[ichange]), '_', 'para', '.txt')
              bulk_process_temp = read.table(bulk_process_file, sep = ',', header = FALSE)
              bulk_process_temp = as.data.frame(bulk_process_temp[ , 21])
          } else {
              bulk_process_file = paste0(data_dir_input, input_std_num, var_name_temp, '/prop_change_', sprintf("%.2f", percentage_change[ichange]), '_', input_matrix[imatrix], '.txt')
              bulk_process_temp = read.table(bulk_process_file, sep = ',', header = FALSE)
          }

          if (dim(bulk_process_temp)[2] != 1) {
              stop(paste0("Error: The file ", bulk_process_file, " does not have exactly one column."))
          }
          # Assign the data to the corresponding column in the array
          bulk_process_prop_change_temp[ , ichange] = as.numeric(bulk_process_temp[ , 1])
      }
      # Drop the rows with NAs
      bulk_process_na_rows = which(!complete.cases(bulk_process_prop_change_temp))
      # Combine the rows with outliers rows
      bulk_process_drop_rows = unique(c(bulk_process_na_rows, outliers_rows))
      bulk_process_prop_change_temp = bulk_process_prop_change_temp[-bulk_process_drop_rows, ]
      # Average the bulk processes across grid points
      bulk_process_prop_change_avg_temp = apply(bulk_process_prop_change_temp, 2, mean, na.rm = TRUE)
      # Assign the averaged data to the corresponding column in the array
      bulk_process_prop_change[ , imatrix] = bulk_process_prop_change_avg_temp
    }

    # # Calculate the proportional change
    # bulk_process_prop_change_baseline <- bulk_process_prop_change[(length(percentage_change)+1)/2, ]
    # bulk_process_prop_change_baseline <- rep(bulk_process_prop_change_baseline, each = length(percentage_change))
    # bulk_process_prop_change <- (bulk_process_prop_change - bulk_process_prop_change_baseline) / bulk_process_prop_change_baseline * 100

    # Update the percentage change when 2*std is exceeding max and min values
    # Change the percentage_change that is closest to i, where i*std = max or min value
    percentage_change_plot = percentage_change
    if (var4nn_mean[var_idx, 1] + std_num * var4nn_std[var_idx, 1] > 1) {
        percentage_change_plot[which.min(abs(percentage_change_plot - std_num))] = (1 - var4nn_mean[var_idx, 1]) / var4nn_std[var_idx, 1]
    }
    if (var4nn_mean[var_idx, 1] - std_num * var4nn_std[var_idx, 1] < 0) {
        percentage_change_plot[which.min(abs(percentage_change_plot + std_num))] = (0- var4nn_mean[var_idx, 1]) / var4nn_std[var_idx, 1]
    }

    # tidy the data for plotting
    plot_df <- as_tibble(bulk_process_prop_change) |>
            mutate(percentage = percentage_change_plot) |>
            pivot_longer(cols = all_of(input_matrix),
                        names_to  = "process",
                        values_to = "prop_change") |>
            filter(!process %in% c("carbon_input", "litter_fraction"))

    plot_df$process <- factor(plot_df$process, levels = input_matrix)
    plot_cols <- input_matrix_color[ names(process_name) %in% unique(plot_df$process) ]
    # Create a plot of percentage changes of each bulk process by dots and lines connecting them
    y_limit <- max(max(plot_df$prop_change, na.rm = TRUE), 100)
    # Plot each bulk process with different colors from input_matrix_color
    p_bulk_process <- ggplot(plot_df, aes(x = percentage, y = prop_change, colour = process)) +
       geom_line(size = 3) +
       geom_point(size = 6) +
       # scale_color_manual(values = input_matrix_color, name   = "Bulk process") +
       scale_color_manual(values = plot_cols, breaks = names(process_name), labels = process_name, name = NULL) +
       scale_x_continuous(breaks = percentage_change,
                          labels = function(x) sprintf("%.1f std", x)) +
       scale_y_continuous(breaks = seq(-100, y_limit, (y_limit - (-100))/10),
                          labels = function(x) sprintf("%+.0f%%", x)) +
        coord_cartesian(xlim = c(-std_num, std_num), ylim = c(-100, y_limit)) +
       labs(
         title = paste0(var_name[var_idx]),
         x     = "Change in environmental variable (std)",
         y     = "Bulk Processes"
       ) +
       theme_minimal(base_family = "Helvetica") +
       theme(
         plot.title      = element_text(size = 35, face = "bold", hjust = 0.5),
          axis.title.x    = element_text(size = 30, face = "bold"),
          axis.title.y    = element_text(size = 30, face = "bold"),
          axis.text.x     = element_text(size = 25, face = "bold"),
          axis.text.y     = element_text(size = 25, face = "bold"),
          legend.title    = element_text(size = 30, face = "bold"),
          legend.text     = element_text(size = 25, face = "bold", margin = margin(b = 12, unit = "pt")),
          legend.key.size = unit(2, "cm"),
          legend.key.spacing.y = unit(0.25, "cm"),
         legend.position = "right"
       )
    # Save the plot
    jpeg(filename = paste0(data_dir_output, '/Percentage_Change_Bulk_Processes/', var_name_temp, '_percentage_change_bulk_processes.jpg'), width = 25, height = 15, units = "in", res = 300)
    print(p_bulk_process)
    dev.off()

    # Update the plot with the actual values of the environmental variable by applying the max and min values
    env_info_update_value = read.table(paste(data_dir_input, input_std_num, var_name_temp, '/grid_env_info_update.txt', sep = ''), sep = ',', header = FALSE)
    env_info_update_value = cbind(percentage_change_plot, env_info_update_value)
    colnames(env_info_update_value) = c('percentage_change', 'env_info_update_value')
    env_info_update_value$env_info_actual_value = env_info_update_value$env_info_update_value * (var4nn_max_min[var_idx, 2] - var4nn_max_min[var_idx, 1]) + var4nn_max_min[var_idx, 1]

    # Replace the x axis with the updated values of the environmental variable
    p_bulk_process_updated <- p_bulk_process +
        scale_x_continuous(breaks = percentage_change,
                           labels = sprintf("%.2f", env_info_update_value$env_info_update_value)) +
        coord_cartesian(xlim = c(-std_num, std_num), ylim = c(-100, y_limit)) +               
        labs(
          title = paste0(var_name[var_idx], " (Normal Value)"),
          x     = "Normal value of environmental variable",
          y     = "Bulk Processes"
        )
    # Save the plot with updated values
    jpeg(filename = paste0(data_dir_output, '/Percentage_Change_Bulk_Processes/', var_name_temp, '_percentage_change_bulk_processes_norm.jpg'), width = 25, height = 15, units = "in", res = 300)
    print(p_bulk_process_updated)
    dev.off()

    # Replace the x axis with the actual values of the environmental variable
    p_bulk_process_actual <- p_bulk_process +
        scale_x_continuous(breaks = percentage_change,
                           labels = sprintf("%.2f", env_info_update_value$env_info_actual_value)) +
        # coord_cartesian(xlim = c(min(env_info_update_value$env_info_actual_value), max(env_info_update_value$env_info_actual_value))) +
        labs(
          title = paste0(var_name[var_idx], " (Actual Value)"),
          x     = "Actual value of environmental variable",
          y     = "Bulk Processes"
        )
    # Save the plot with actual values
    jpeg(filename = paste0(data_dir_output, '/Percentage_Change_Bulk_Processes/', var_name_temp, '_percentage_change_bulk_processes_actual.jpg'), width = 25, height = 15, units = "in", res = 300)
    print(p_bulk_process_actual)
    dev.off()

    print(paste0("Proportional change plot for variable ", var_name_temp, " saved successfully with time taken: ", Sys.time() - change_start_time, " seconds"))
}


## Plot 2: Percentage Change of Parameters
if (!dir.exists(paste0(data_dir_output, '/Percentage_Change_Parameters'))) {
    dir.create(paste0(data_dir_output, '/Percentage_Change_Parameters'))
}


for (var_idx in 1:length(var4nn)) {
    change_start_time = Sys.time()
    var_name_temp = var4nn[var_idx]
    print(paste0("Processing variable: ", var_name_temp))

    # Initialize an array to store plots
    para_prop_change <- matrix(NA_real_,
                               nrow = length(percentage_change),
                               ncol = length(para_names),
                               dimnames = list(NULL, para_names))

    # Loop through each percentage change
    for (ichange in 1:length(percentage_change)) {
        # Load the data for the current variable and percentage change
        para_file = paste0(data_dir_input, input_std_num, var_name_temp, '/prop_change_', sprintf("%.2f", percentage_change[ichange]), '_', 'para.txt')
        para_temp = read.table(para_file, sep = ',', header = FALSE)
        if (dim(para_temp)[2] != length(para_names)) {
            stop(paste0("Error: The file ", para_file, " does not have exactly ", length(para_names), " columns."))
        }

    # Drop the rows with NAs
    para_nan_rows = which(!complete.cases(para_temp))
    # Combine the rows with outliers rows
    para_drop_rows = unique(c(para_nan_rows, outliers_rows))
    # Drop the rows with NAs
    para_temp = para_temp[-para_drop_rows,]
    # Average the parameters across grid points
    para_prop_change_avg = apply(para_temp, 2, mean, na.rm = TRUE)
    # Assign the averaged data to the corresponding column in the array
    para_prop_change[ichange, ] = para_prop_change_avg
    }
    
    # # Calculate the proportional change
    # para_prop_change_baseline <- para_prop_change[(length(percentage_change)+1)/2, ]
    # para_prop_change_baseline <- rep(para_prop_change_baseline, each = length(percentage_change))
    # para_prop_change <- (para_prop_change - para_prop_change_baseline) / para_prop_change_baseline * 100

    # Update the percentage change when 2*std is exceeding max and min values
    percentage_change_plot = percentage_change
    if (var4nn_mean[var_idx, 1] + std_num * var4nn_std[var_idx, 1] > 1) {
        percentage_change_plot[which.min(abs(percentage_change_plot - std_num))] = (1 - var4nn_mean[var_idx, 1]) / var4nn_std[var_idx, 1]
    }
    if (var4nn_mean[var_idx, 1] - std_num * var4nn_std[var_idx, 1] < 0) {
        percentage_change_plot[which.min(abs(percentage_change_plot + std_num))] = (0 - var4nn_mean[var_idx, 1]) / var4nn_std[var_idx, 1]
    }

    # tidy the data for plotting
    plot_df <- as_tibble(para_prop_change) |>
            mutate(percentage = percentage_change_plot) |>
            pivot_longer(cols = all_of(para_names),
                        names_to  = "parameter",
                        values_to = "prop_change")

    plot_df$parameter <- factor(plot_df$parameter, levels = para_names)
    
    # Create a plot of percentage changes of each parameter by dots and lines connecting them
    ylimit <- max(max(plot_df$prop_change, na.rm = TRUE), 100)
    p_para <- ggplot(plot_df, aes(x = percentage, y = prop_change, colour = parameter)) +
       geom_line(size = 3) +
       geom_point(size = 6) +
       scale_color_manual(values = para_colors, name   = "Parameter") +
       scale_x_continuous(breaks = percentage_change,
                          labels = function(x) sprintf("%.1f std", x)) +
        scale_y_continuous(breaks = seq(-100, ylimit, (ylimit - (-100))/10),
                          labels = function(x) sprintf("%+.0f%%", x)) +
        coord_cartesian(ylim = c(-100, ylimit), xlim = c(-std_num, std_num)) +
        labs(
          title = paste0(var_name[var_idx]),
          x     = "Change in environmental variable (std)",
          y     = "Parameter Norm Value"
        ) +
        theme_minimal(base_family = "Helvetica") +
        theme(
          plot.title      = element_text(size = 35, face = "bold", hjust = 0.5),
          axis.title.x    = element_text(size = 30, face = "bold"),
          axis.title.y    = element_text(size = 30, face = "bold"),
          axis.text.x     = element_text(size = 25, face = "bold"),
          axis.text.y     = element_text(size = 25, face = "bold"),
          legend.title    = element_text(size = 30, face = "bold"),
          legend.text     = element_text(size = 25, face = "bold", margin = margin(b = 12, unit = "pt")),
          legend.key.size = unit(2, "cm"),
          legend.key.spacing.y = unit(0.25, "cm"),
          legend.position = "right"
        )
    # Save the plot
    jpeg(filename = paste0(data_dir_output, '/Percentage_Change_Parameters/', var_name_temp, '_percentage_change_parameters.jpg'), width = 20, height = 15, units = "in", res = 300)
    print(p_para)
    dev.off()

    # Update the plot with the actual values of the environmental variable by applying the max and min values
    env_info_update_value = read.table(paste(data_dir_input, input_std_num, var_name_temp, '/grid_env_info_update.txt', sep = ''), header = FALSE, sep = ',')
    env_info_update_value = cbind(percentage_change_plot, env_info_update_value)
    colnames(env_info_update_value) = c('percentage_change', 'env_info_update_value')
    env_info_update_value$env_info_actual_value = env_info_update_value$env_info_update_value * (var4nn_max_min[var_idx, 2] - var4nn_max_min[var_idx, 1]) + var4nn_max_min[var_idx, 1]
    # Replace the x axis with the updated values of the environmental variable
    p_para_update <- p_para +
        scale_x_continuous(breaks = percentage_change,
                           labels = sprintf("%.2f", env_info_update_value$env_info_update_value)) +
        labs(
          title = paste0(var_name[var_idx], " (Normal Value)"),
          x     = "Normal value of environmental variable",
          y     = "Parameter Norm Value"
        )
    # Save the plot with updated values
    jpeg(filename = paste0(data_dir_output, '/Percentage_Change_Parameters/', var_name_temp, '_percentage_change_parameters_norm.jpg'), width = 20, height = 15, units = "in", res = 300)
    print(p_para_update)
    dev.off()

    # Replace the x axis with the actual values of the environmental variable
    p_para_actual <- p_para +
        scale_x_continuous(breaks = percentage_change,
                           labels = sprintf("%.2f", env_info_update_value$env_info_actual_value)) +
        labs(
          title = paste0(var_name[var_idx], " (Actual Value)"),
          x     = "Actual value of environmental variable",
          y     = "Parameter Norm Value"
        )
    # Save the plot with actual values
    jpeg(filename = paste0(data_dir_output, '/Percentage_Change_Parameters/', var_name_temp, '_percentage_change_parameters_actual.jpg'), width = 20, height = 15, units = "in", res = 300)
    print(p_para_actual)
    dev.off()

    print(paste0("Proportional change plot for variable ", var_name_temp, " saved successfully with time taken: ", Sys.time() - change_start_time, " seconds"))
}


## Plot 3: Percentage Change of SOC Components
if (!dir.exists(paste0(data_dir_output, '/Percentage_Change_SOC_Components'))) {
    dir.create(paste0(data_dir_output, '/Percentage_Change_SOC_Components'))
}

for (var_idx in 1:length(var4nn)) {
    change_start_time = Sys.time()
    var_name_temp = var4nn[var_idx]
    print(paste0("Processing variable: ", var_name_temp))

    # Initialize an array to store plots
    soc_prop_change <- matrix(NA_real_,
                              nrow = length(percentage_change),
                              ncol = length(soc_category_names),
                              dimnames = list(NULL, soc_category_names))

    # Loop through each category and percentage change
    for (icategory in 1:length(soc_category_names)) {
        soc_prop_change_temp = array(NA, dim = c(nrow(grid_data_baseline), length(percentage_change)))
        for (ichange in 1:length(percentage_change)) {
            # Load the data for the current variable, category and percentage change
            soc_file = paste0(data_dir_input, input_std_num, var_name_temp, '/prop_change_', sprintf("%.2f", percentage_change[ichange]), '_', soc_category_names[icategory], '.txt')
            soc_temp = read.table(soc_file, sep = ',', header = FALSE)
            soc_temp = sweep(soc_temp, 1, dz[1:ncol(soc_temp)], FUN = '*') # gC/m2
            temp_nan_index = which(is.na(soc_temp[ , 1]) | soc_temp[ , 1] == 0)
            soc_temp_sum = rowSums(soc_temp, na.rm = TRUE) #gC/m2
            soc_temp_sum = soc_temp_sum/1000 # Convert to kgC/m2
            soc_temp_sum[temp_nan_index] = NA # Set the NA rows to NA
            # Assign the data to the corresponding column in the array
            soc_prop_change_temp[ , ichange] = as.numeric(soc_temp_sum)
        }
        # Get the rows with NAs
        soc_prop_change_na_rows = which(!complete.cases(soc_prop_change_temp))
        # Combine the rows with outliers rows
        soc_prop_change_drop_rows = unique(c(soc_prop_change_na_rows, outliers_rows))
        # Drop the rows with NAs
        soc_prop_change_temp = soc_prop_change_temp[-soc_prop_change_drop_rows,]
        # Average the SOC components across grid points
        soc_prop_change_avg_temp = apply(soc_prop_change_temp, 2, mean, na.rm = TRUE)
        # Assign the averaged data to the corresponding column in the array
        soc_prop_change[ , icategory] = soc_prop_change_avg_temp
    }
    # # Calculate the proportional change
    # soc_prop_change_baseline <- soc_prop_change[(length(percentage_change)+1)/2, ]
    # soc_prop_change_baseline <- rep(soc_prop_change_baseline, each = length(percentage_change))
    # soc_prop_change <- (soc_prop_change - soc_prop_change_baseline) / soc_prop_change_baseline * 100

    # Update the percentage change when 2*std is exceeding max and min values
    percentage_change_plot = percentage_change
    if (var4nn_mean[var_idx, 1] + std_num * var4nn_std[var_idx, 1] > 1) {
        percentage_change_plot[which.min(abs(percentage_change_plot - std_num))] = (1 - var4nn_mean[var_idx, 1]) / var4nn_std[var_idx, 1]
    }
    if (var4nn_mean[var_idx, 1] - std_num * var4nn_std[var_idx, 1] < 0) {
        percentage_change_plot[which.min(abs(percentage_change_plot + std_num))] = (0 - var4nn_mean[var_idx, 1]) / var4nn_std[var_idx, 1]
    }
    # tidy the data for plotting
    plot_df <- as_tibble(soc_prop_change) |>
            mutate(percentage = percentage_change_plot) |>
            pivot_longer(cols = all_of(soc_category_names),
                        names_to  = "soc_component",
                        values_to = "prop_change")
    plot_df$soc_component <- factor(plot_df$soc_component, levels = soc_category_names)
  

    # Select the maximum number between soc_prop_change and 100
    y_limit <- max(max(plot_df$prop_change, na.rm = TRUE), 100)

    # Create a plot of percentage changes of each SOC component by dots and lines connecting them
    p_soc <- ggplot(plot_df, aes(x = percentage, y = prop_change, colour = soc_component)) +
       geom_line(size = 3) +
       geom_point(size = 6) +
       scale_color_manual(values = c("#976500", "#0884d1", "#df1414", "#00be82", "#4e0096"), name   = "SOC Component") +
       scale_x_continuous(breaks = percentage_change,
                          labels = function(x) sprintf("%.1f std", x)) +
       scale_y_continuous(breaks = seq(-100, y_limit, (y_limit - (-100))/10),
                          labels = function(x) sprintf("%+.0f%%", x)) +
        coord_cartesian(ylim = c(-100, y_limit), xlim = c(-std_num, std_num)) +
        labs(
          title = paste0(var_name[var_idx]),
          x     = "Change in environmental variable (std)",
          y     = "Soil Carbon (kgC/m2)"
        ) +
        theme_minimal(base_family = "Helvetica") +
        theme(
          plot.title      = element_text(size = 35, face = "bold", hjust = 0.5),
          axis.title.x    = element_text(size = 30, face = "bold"),
          axis.title.y    = element_text(size = 30, face = "bold"),
          axis.text.x     = element_text(size = 25, face = "bold"),
          axis.text.y     = element_text(size = 25, face = "bold"),
          legend.title    = element_text(size = 30, face = "bold"),
          legend.text     = element_text(size = 25, face = "bold", margin = margin(b = 12, unit = "pt")),
          legend.key.size = unit(2, "cm"),
          legend.key.spacing.y = unit(0.25, "cm"),
          legend.position = "right"
        )
    # Save the plot
    jpeg(filename = paste0(data_dir_output, '/Percentage_Change_SOC_Components/', var_name_temp, '_percentage_change_soc_components.jpg'), width = 20, height = 15, units = "in", res = 300)
    print(p_soc)
    dev.off()

    # Update the plot with the actual values of the environmental variable by applying the max and min values
    env_info_update_value = read.table(paste(data_dir_input, input_std_num, var_name_temp, '/grid_env_info_update.txt', sep = ''), header = FALSE, sep = ',')
    env_info_update_value = cbind(percentage_change_plot, env_info_update_value)
    colnames(env_info_update_value) = c('percentage_change', 'env_info_update_value')
    env_info_update_value$env_info_actual_value = env_info_update_value$env_info_update_value * (var4nn_max_min[var_idx, 2] - var4nn_max_min[var_idx, 1]) + var4nn_max_min[var_idx, 1]
    # Replace the x axis with the updated values of the environmental variable
    p_soc_update <- p_soc +
        scale_x_continuous(breaks = percentage_change,
                           labels = sprintf("%.2f", env_info_update_value$env_info_update_value)) +
        labs(
          title = paste0(var_name[var_idx], " (Normal Value)"),
          x     = "Normal value of environmental variable",
          y     = "Soil Carbon (kgC/m2)"
        )
    # Save the plot with updated values
    jpeg(filename = paste0(data_dir_output, '/Percentage_Change_SOC_Components/', var_name_temp, '_percentage_change_soc_components_norm.jpg'), width = 20, height = 15, units = "in", res = 300)
    print(p_soc_update)
    dev.off()

    # Replace the x axis with the actual values of the environmental variable
    p_soc_actual <- p_soc +
        scale_x_continuous(breaks = percentage_change,
                           labels = sprintf("%.2f", env_info_update_value$env_info_actual_value)) +
        labs(
          title = paste0(var_name[var_idx], " (Actual Value)"),
          x     = "Actual value of environmental variable",
          y     = "Soil Carbon (kgC/m2)"
        )
    # Save the plot with actual values
    jpeg(filename = paste0(data_dir_output, '/Percentage_Change_SOC_Components/', var_name_temp, '_percentage_change_soc_components_actual.jpg'), width = 20, height = 15, units = "in", res = 300)
    print(p_soc_actual)
    dev.off()

    print(paste0("Proportional change plot for variable ", var_name_temp, " saved successfully with time taken: ", Sys.time() - change_start_time, " seconds"))
}

## Plot 4: Percentage Change of SOC Fractions
if (!dir.exists(paste0(data_dir_output, '/Percentage_Change_SOC_Fractions'))) {
    dir.create(paste0(data_dir_output, '/Percentage_Change_SOC_Fractions'))
}

for (var_idx in 1:length(var4nn)) {
    change_start_time = Sys.time()
    var_name_temp = var4nn[var_idx]
    print(paste0("Processing variable: ", var_name_temp))

    # Initialize an array to store plots
    soc_prop_change <- matrix(NA_real_,
                              nrow = length(percentage_change),
                              ncol = length(soc_category_names),
                              dimnames = list(NULL, soc_category_names))

    # Load the baseline SOC data first
    soc_category_name = 'soc'
    soc_category_idx = which(soc_category_names == soc_category_name)

    # Loop through each category and percentage change
    for (icategory in 1:length(soc_category_names)) {
        soc_prop_change_temp = array(NA, dim = c(nrow(grid_data_baseline), length(percentage_change)))
        for (ichange in 1:length(percentage_change)) {
            # Load the data for the current variable, category and percentage change
            soil_frac_file = paste0(data_dir_input, input_std_num, var_name_temp, '/prop_change_', sprintf("%.2f", percentage_change[ichange]), '_', soc_category_names[icategory], '.txt')
            soc_file = paste0(data_dir_input, input_std_num, var_name_temp, '/prop_change_', sprintf("%.2f", percentage_change[ichange]), '_', soc_category_name, '.txt')
            soil_frac_temp = read.table(soil_frac_file, sep = ',', header = FALSE)
            soc_temp = read.table(soc_file, sep = ',', header = FALSE)
            soil_frac_temp = sweep(soil_frac_temp, 1, dz[1:ncol(soil_frac_temp)], FUN = '*') # gC/m2
            soc_temp = sweep(soc_temp, 1, dz[1:ncol(soc_temp)], FUN = '*') # gC/m2
            temp_nan_index = which(is.na(soil_frac_temp[ , 1]) | soil_frac_temp[ , 1] == 0 | is.na(soc_temp[ , 1]) | soc_temp[ , 1] == 0)
            soil_frac_temp_sum = rowSums(soil_frac_temp, na.rm = TRUE) #gC/m2
            soc_temp_sum = rowSums(soc_temp, na.rm = TRUE) #gC/m2
            soil_frac_temp_sum = soil_frac_temp_sum/1000 # Convert to kgC/m2
            soc_temp_sum = soc_temp_sum/1000 # Convert to kgC/m2
            soil_frac_temp_sum[temp_nan_index] = NA # Set the NA rows to NA
            soc_temp_sum[temp_nan_index] = NA # Set the NA rows to NA
            # Assign the data to the corresponding column in the array
            soc_prop_change_temp[ , ichange] = as.numeric(soil_frac_temp_sum / soc_temp_sum) # Calculate the fraction of SOC component
        }
        # Get the rows with NAs
        soc_prop_change_na_rows = which(!complete.cases(soc_prop_change_temp))
        # Combine the rows with outliers rows
        soc_prop_change_drop_rows = unique(c(soc_prop_change_na_rows, outliers_rows))
        # Drop the rows with NAs
        soc_prop_change_temp = soc_prop_change_temp[-soc_prop_change_drop_rows,]
        # Average the SOC components across grid points
        soc_prop_change_avg_temp = apply(soc_prop_change_temp, 2, mean, na.rm = TRUE)
        # Assign the averaged data to the corresponding column in the array
        soc_prop_change[ , icategory] = soc_prop_change_avg_temp
    }

    # Calculate the proportional change
    soc_prop_change_baseline <- soc_prop_change[(length(percentage_change)+1)/2, ]
    soc_prop_change_baseline <- rep(soc_prop_change_baseline, each = length(percentage_change))
    soc_prop_change <- (soc_prop_change - soc_prop_change_baseline) / soc_prop_change_baseline * 100

    # Update the percentage change when 2*std is exceeding max and min values
    percentage_change_plot = percentage_change
    if (var4nn_mean[var_idx, 1] + std_num * var4nn_std[var_idx, 1] > 1) {
        percentage_change_plot[which.min(abs(percentage_change_plot - std_num))] = (1 - var4nn_mean[var_idx, 1]) / var4nn_std[var_idx, 1]
    }
    if (var4nn_mean[var_idx, 1] - std_num * var4nn_std[var_idx, 1] < 0) {
        percentage_change_plot[which.min(abs(percentage_change_plot + std_num))] = (0 - var4nn_mean[var_idx, 1]) / var4nn_std[var_idx, 1]
    }
    # tidy the data for plotting
    plot_df <- as_tibble(soc_prop_change) |>
            mutate(percentage = percentage_change_plot) |>
            pivot_longer(cols = all_of(soc_category_names),
                        names_to  = "soc_component",
                        values_to = "prop_change")
    plot_df$soc_component <- factor(plot_df$soc_component, levels = soc_category_names)
  

    # Select the maximum number between soc_prop_change and 100
    y_limit <- max(max(plot_df$prop_change, na.rm = TRUE), 100)

    # Create a plot of percentage changes of each SOC component by dots and lines connecting them
    p_soc <- ggplot(plot_df, aes(x = percentage, y = prop_change, colour = soc_component)) +
       geom_line(size = 3) +
       geom_point(size = 6) +
       scale_color_manual(values = c("#976500", "#0884d1", "#df1414", "#00be82", "#4e0096"), name   = "SOC Component") +
       scale_x_continuous(breaks = percentage_change,
                          labels = function(x) sprintf("%.1f std", x)) +
       scale_y_continuous(breaks = seq(-100, y_limit, (y_limit - (-100))/10),
                          labels = function(x) sprintf("%+.0f%%", x)) +
        coord_cartesian(ylim = c(-100, y_limit), xlim = c(-std_num, std_num)) +
        labs(
          title = paste0(var_name[var_idx]),
          x     = "Change in environmental variable (std)",
          y     = "Carbon Fraction (%)"
        ) +
        theme_minimal(base_family = "Helvetica") +
        theme(
          plot.title      = element_text(size = 35, face = "bold", hjust = 0.5),
          axis.title.x    = element_text(size = 30, face = "bold"),
          axis.title.y    = element_text(size = 30, face = "bold"),
          axis.text.x     = element_text(size = 25, face = "bold"),
          axis.text.y     = element_text(size = 25, face = "bold"),
          legend.title    = element_text(size = 30, face = "bold"),
          legend.text     = element_text(size = 25, face = "bold", margin = margin(b = 12, unit = "pt")),
          legend.key.size = unit(2, "cm"),
          legend.key.spacing.y = unit(0.25, "cm"),
          legend.position = "right"
        )
    # Save the plot
    jpeg(filename = paste0(data_dir_output, '/Percentage_Change_SOC_Fractions/', var_name_temp, '_percentage_change_soil_fractions.jpg'), width = 20, height = 15, units = "in", res = 300)
    print(p_soc)
    dev.off()

    # Update the plot with the actual values of the environmental variable by applying the max and min values
    env_info_update_value = read.table(paste(data_dir_input, input_std_num, var_name_temp, '/grid_env_info_update.txt', sep = ''), header = FALSE, sep = ',')
    env_info_update_value = cbind(percentage_change_plot, env_info_update_value)
    colnames(env_info_update_value) = c('percentage_change', 'env_info_update_value')
    env_info_update_value$env_info_actual_value = env_info_update_value$env_info_update_value * (var4nn_max_min[var_idx, 2] - var4nn_max_min[var_idx, 1]) + var4nn_max_min[var_idx, 1]
    # Replace the x axis with the updated values of the environmental variable
    p_soc_update <- p_soc +
        scale_x_continuous(breaks = percentage_change,
                           labels = sprintf("%.2f", env_info_update_value$env_info_update_value)) +
        labs(
          title = paste0(var_name[var_idx], " (Normal Value)"),
          x     = "Normal value of environmental variable",
          y     = "Carbon Fraction (%)"
        )
    # Save the plot with updated values
    jpeg(filename = paste0(data_dir_output, '/Percentage_Change_SOC_Fractions/', var_name_temp, '_percentage_change_soil_fractions_norm.jpg'), width = 20, height = 15, units = "in", res = 300)
    print(p_soc_update)
    dev.off()

    # Replace the x axis with the actual values of the environmental variable
    p_soc_actual <- p_soc +
        scale_x_continuous(breaks = percentage_change,
                           labels = sprintf("%.2f", env_info_update_value$env_info_actual_value)) +
        labs(
          title = paste0(var_name[var_idx], " (Actual Value)"),
          x     = "Actual value of environmental variable",
          y     = "Carbon Fraction (%)"
        )
    # Save the plot with actual values
    jpeg(filename = paste0(data_dir_output, '/Percentage_Change_SOC_Fractions/', var_name_temp, '_percentage_change_soil_fractions_actual.jpg'), width = 20, height = 15, units = "in", res = 300)
    print(p_soc_actual)
    dev.off()

    print(paste0("Proportional change plot for variable ", var_name_temp, " saved successfully with time taken: ", Sys.time() - change_start_time, " seconds"))
}
