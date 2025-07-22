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

##
rm(list = ls())

setwd('D:/Research/BINN/BINN_output/plot/')
# Sys.setenv(PROJ_LIB = "C:/Users/hx293/AppData/Local/R/win-library/4.3/sf/proj")
Sys.setenv(PROJ_LIB = "C:/Program Files/R/R-4.3.3/library/sf/proj")


## Jet colorbar function
jet.colors <- colorRampPalette(c("#00007F", "blue", "#007FFF", "cyan", "#7FFF7F", "yellow", "#FF7F00", "red", "#7F0000"))
diff.colors <- colorRampPalette(c("#2166AC", "#4393C3", "#92C5DE", "#D1E5F0", "#f6f6f6", "#FDDBC7", "#F4A582", "#D6604D", "#B2182B"))

#############################################################################
# Data Path
#############################################################################
date_stamp = '20250704-213024_KAN_ONELAYER_GRIDMARGIN_COMPAS2_998996_lr=1e-02_fold=0_seed=111'

# input and output data path
data_dir_input = paste0('D:/Research/BINN/BINN_output/neural_network/',  date_stamp, '/')
data_dir_output = paste0('D:/Research/BINN/BINN_output/neural_network/',  date_stamp, '/Output_plots/')
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


# Matrix: bulk_A_doc, bulk_E_mic, bulk_A_mic, bulk_A_POM, bulk_A_MAOM, bulk_I, bulk_K, bulk_V, bulk_xi, carbon_input, litter_fraction
input_matrix = c('bulk_A_doc', 'bulk_E_mic', 'bulk_A_mic', 'bulk_A_POM', 'bulk_A_MAOM', 'bulk_I', 'bulk_K', 'bulk_V', 'bulk_xi', 'carbon_input', 'litter_fraction')

# Para names
para_names = c('diffus', 'cryo', 'q10', 'efolding', 'taucwd', 'taul1', 'taul2', 'tau4doc', 'tau4mic', 'tau4poc', 'tau4maom',
'fl1_MIC', 'fl2_POC', 'fMIC_MAOM', 'fMIC_POC', 'fDOC_MAOM', 'CUEl1', 'CUEl2', 'CUEDOC', 'w-scaling', 'beta')

# SOC Category names
soc_category_names = c('soc', 'DOC', 'MIC', 'POM', 'MAOM')

# Get the longitude and latitude for plotting the map
grid_env_info = read.csv(data_dir_grid_env, header = TRUE, stringsAsFactors = FALSE)
grid_lon = grid_env_info$Lon
grid_lat = grid_env_info$Lat
grid_data_baseline = cbind(grid_lon, grid_lat)
colnames(grid_data_baseline) = c('lon', 'lat')

# Load the bulk processes data
bulk_process_baseline = array(NA, dim = c(nrow(grid_data_baseline), length(input_matrix)))
for (i in 1:length(input_matrix)) {
    if (input_matrix[i] == 'bulk_I') {
      temp_bulk_baseline = read.table(paste(data_dir_input, 'KAN_Propotional_Change_2_std/', '/baseline', '/para', '.txt', sep = ''), sep = ',', header = FALSE)
      temp_bulk_baseline = as.data.frame(temp_bulk_baseline[ , 21])
      bulk_process_baseline[ , i] = temp_bulk_baseline[ , 1]
    } else {
        temp_bulk_baseline = read.table(paste(data_dir_input, 'KAN_Propotional_Change_2_std/', '/baseline', '/', input_matrix[i], '.txt', sep = ''), sep = ',', header = FALSE)
        bulk_process_baseline[ , i] = temp_bulk_baseline[ , 1]
    }
}

bulk_process_baseline = cbind(grid_data_baseline, bulk_process_baseline)
dim(bulk_process_baseline)

# Load the para data
para_data_baseline = read.table(paste(data_dir_input, 'KAN_Propotional_Change_2_std/', '/baseline', '/para.txt', sep = ''), sep = ',', header = FALSE)
para_data_baseline = cbind(grid_data_baseline, para_data_baseline)
dim(para_data_baseline)

# Load the SOC data
soc_data_baseline = array(NA, dim = c(nrow(grid_data_baseline), length(soc_category_names)))
i = 1
for (i in 1:length(soc_category_names)) {
    temp_soc_baseline = read.table(paste(data_dir_input, 'KAN_Propotional_Change_2_std/', '/baseline', '/', soc_category_names[i], '.txt', sep = ''), sep = ',', header = FALSE)
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

# Remove the NA rows
# bulk_process_baseline = bulk_process_baseline[complete.cases(bulk_process_baseline), ]
# para_data_baseline = para_data_baseline[complete.cases(para_data_baseline), ]
# soc_data_baseline = soc_data_baseline[complete.cases(soc_data_baseline), ]
# soc_data_baseline = soc_data_baseline[(soc_data_baseline[ , 3] != 0), ] # Remove rows with all zeros
bulk_process_nan_rows = which(!complete.cases(bulk_process_baseline))
# Get the rows larger than 2000 in the first column of soc_data_baseline
soc_data_outliers_rows = which(soc_data_baseline[ , 3] > 2000) 
# Combine the two rows
bulk_process_nan_rows = unique(c(bulk_process_nan_rows, soc_data_outliers_rows))
# para_data_nan_rows = which(!complete.cases(para_data_baseline))
bulk_process_baseline = bulk_process_baseline[-bulk_process_nan_rows, ]
para_data_baseline = para_data_baseline[-bulk_process_nan_rows, ]
soc_data_baseline = soc_data_baseline[-bulk_process_nan_rows, ]

dim(bulk_process_baseline)
colnames(bulk_process_baseline) = c('lon', 'lat', input_matrix)
colnames(para_data_baseline) = c('lon', 'lat', para_names)
colnames(soc_data_baseline) = c('lon', 'lat', soc_category_names)

# Convert to data frame
bulk_process_baseline = as.data.frame(bulk_process_baseline)
head(bulk_process_baseline)
class(bulk_process_baseline)
para_data_baseline = as.data.frame(para_data_baseline)
soc_data_baseline = as.data.frame(soc_data_baseline)
dim(bulk_process_baseline)
dim(soc_data_baseline)

#################################################################################
# plot figures
#################################################################################
#-------------------------------------soc stock and Residence Time map
world_coastline = st_read('D:/Nutstore/Research_Data/Map_Plot/ne_110m_land/ne_110m_land.shp', layer = 'ne_110m_land')
coord_info = '+proj=robin'
world_coastline <- st_transform(world_coastline, CRS(coord_info))

ocean_left = cbind(rep(-180, 100), seq(from = 80, to = -56, by = -(80 + 56)/(100 -1)))
ocean_right = cbind(rep(180, 100), seq(from = -56, to = 80, by = (80 + 56)/(100 -1)))
ocean_top = cbind(seq(from = 180, to = -180, by = -(360)/(100 -1)), rep(80, 100))
ocean_bottom = cbind(seq(from = -180, to = 180, by = (360)/(100 -1)), rep(-56, 100))

# # Try to plot only the mainland US
# US_left = cbind(rep(-180, 100), seq(from = 24, to = 50, by = -(24 - 50)/(100 -1)))
# US_right = cbind(rep(180, 100), seq(from = 50, to = 24, by = (24 - 50)/(100 -1)))
# US_top = cbind(seq(from = 180, to = -180, by = -(360)/(100 -1)), rep(50, 100))
# US_bottom = cbind(seq(from = -180, to = 180, by = (360)/(100 -1)), rep(24, 100))

world_ocean = rbind(ocean_left, ocean_bottom, ocean_right, ocean_top)
# world_ocean = rbind(US_left, US_bottom, US_right, US_top)
world_ocean = as.matrix(world_ocean)

world_ocean <- project(xy = world_ocean, proj = coord_info)

world_ocean = data.frame(world_ocean)
colnames(world_ocean) = c('lon', 'lat')

# lat_limits = rbind(c(-62, 24.5), c(-140, 50))
# lat_limits = rbind(c(0, -56), c(0, 80))
# Try global map
lat_limits = rbind(c(-180, -56), c(180, 80))
lat_limits_robin = project(xy = as.matrix(lat_limits), proj = coord_info) 

# transfer lon and lat to robinson projection 
lon_lat_transfer = project(xy = as.matrix(bulk_process_baseline[ , c('lon', 'lat')]), proj = coord_info) 
bulk_process_baseline[ , c('lon', 'lat')] = lon_lat_transfer
lon_lat_transfer = project(xy = as.matrix(para_data_baseline[ , c('lon', 'lat')]), proj = coord_info)
para_data_baseline[ , c('lon', 'lat')] = lon_lat_transfer
lon_lat_transfer = project(xy = as.matrix(soc_data_baseline[ , c('lon', 'lat')]), proj = coord_info)
soc_data_baseline[ , c('lon', 'lat')] = lon_lat_transfer

# plot data only within the shapefile constraint
bulk_process_baseline_world <- st_as_sf(bulk_process_baseline, coords = c('lon', 'lat'), crs = st_crs(world_coastline))
bulk_process_baseline_world <- st_intersection(bulk_process_baseline_world, world_coastline)
para_data_baseline_world <- st_as_sf(para_data_baseline, coords = c('lon', 'lat'), crs = st_crs(world_coastline))
para_data_baseline_world <- st_intersection(para_data_baseline_world, world_coastline)
soc_data_baseline_world <- st_as_sf(soc_data_baseline, coords = c('lon', 'lat'), crs = st_crs(world_coastline))
soc_data_baseline_world <- st_intersection(soc_data_baseline_world, world_coastline)

# Extract the coordinates from the geometry column
coords_bulk <- st_coordinates(bulk_process_baseline_world$geometry)
coords_para <- st_coordinates(para_data_baseline_world$geometry)
coords_soc <- st_coordinates(soc_data_baseline_world$geometry)
# Add lon and lat back to the data frame to the first two columns
bulk_process_baseline_world$lat <- coords_bulk[, 2]
bulk_process_baseline_world$lon <- coords_bulk[, 1]
para_data_baseline_world$lat <- coords_para[, 2]
para_data_baseline_world$lon <- coords_para[, 1]
soc_data_baseline_world$lat <- coords_soc[, 2]
soc_data_baseline_world$lon <- coords_soc[, 1]

# move the longtitude to the first column
bulk_process_baseline_world <- bulk_process_baseline_world[c("lon", "lat", setdiff(names(bulk_process_baseline_world), c("lon", "lat")))]
para_data_baseline_world <- para_data_baseline_world[c("lon", "lat", setdiff(names(para_data_baseline_world), c("lon", "lat")))]
soc_data_baseline_world <- soc_data_baseline_world[c("lon", "lat", setdiff(names(soc_data_baseline_world), c("lon", "lat")))]
# remove the geometry column
bulk_process_baseline_world <- st_drop_geometry(bulk_process_baseline_world)
para_data_baseline_world <- st_drop_geometry(para_data_baseline_world)
soc_data_baseline_world <- st_drop_geometry(soc_data_baseline_world)
# remove all column after the 13th column
bulk_process_baseline_world <- bulk_process_baseline_world[ , 1:13]
head(bulk_process_baseline_world)
# remove all column after the length of para_names + 2
para_data_baseline_world <- para_data_baseline_world[ , 1:(length(para_names) + 2)]
head(para_data_baseline_world)
# remove all column after the length of soc_category_names + 2
soc_data_baseline_world <- soc_data_baseline_world[ , 1:(length(soc_category_names) + 2)]
head(soc_data_baseline_world)

# Combine lon and lat
bulk_process_baseline_world[ , c('lon_lat')] = paste(bulk_process_baseline_world$lon, bulk_process_baseline_world$lat, sep = '_')
para_data_baseline_world[ , c('lon_lat')] = paste(para_data_baseline_world$lon, para_data_baseline_world$lat, sep = '_')
soc_data_baseline_world[ , c('lon_lat')] = paste(soc_data_baseline_world$lon, soc_data_baseline_world$lat, sep = '_')

#####################################################################################
# Plot the bulk processes
#####################################################################################
process_scale_option = c('identity', 'identity', 'identity', 'identity', 'identity', 'identity', 'identity', 'identity', 'identity', 'identity', 'identity')

process_name =  c('Carbon Transfer Efficiency of DOC', 
                  'Non-microbial Carbon Transfer Efficiency of MIC', 
                  'Microbial Carbon Transfer Efficiency of MIC',
                  'Carbon Transfer Efficiency of POM', 
                  'Carbon Transfer Efficiency of MAOM',
                  'Carbon Input Allocation', 
                  'Baseline Decomposition', 
                  'Vertical Transport Rate',
                  'Environmental Modifier', 
                  'Plant Carbon Inputs', 
                  'Litter to Mineral Soil Fraction')

process_unit = c('unitless',
                 'unitless',
                 'unitless',
                 'unitless',
                 'unitless',
                 'unitless', 
                 expression(paste('yr'^'-1', sep = '')),
                 expression(paste('yr'^'-1', sep = '')),
                 'unitless', 
                 expression(paste('gCm'^'-2', ' yr'^'-1', sep = '')), 
                 'unitless')

# Loop through all bulk processes
# category = 1

for (category in 1:length(input_matrix)) {
  middle_data_baseline_world = bulk_process_baseline_world[ , c(1:2, 2 + category)]
  colnames(middle_data_baseline_world) = c('lon', 'lat', 'process')

  legend_lower_baseline = apply(bulk_process_baseline_world[ , c(3:(ncol(bulk_process_baseline_world)-1))], 2, quantile, prob = 0.05, na.rm = TRUE)
  legend_upper_baseline = apply(bulk_process_baseline_world[ , c(3:(ncol(bulk_process_baseline_world)-1))], 2, quantile, prob = 0.95, na.rm = TRUE)

  # plot figure
  p_baseline =
    ggplot() +
    geom_tile(data = middle_data_baseline_world, aes(x = lon, y = lat, fill = process), height = 60000, width = 60000, na.rm = TRUE) +
    scale_fill_gradientn(name = process_unit[category], colours = rev(viridis(15)), na.value="transparent", limits = c(legend_lower_baseline[category], legend_upper_baseline[category]), trans = process_scale_option[category], oob = scales::squish) +
    geom_sf(data = world_coastline, fill = NA, color = 'black', linewidth = 1) + 
	# coord_sf(xlim = lat_limits_robin[ , 1], ylim = lat_limits_robin[ , 2], datum = NA) +
    coord_sf(ylim = lat_limits_robin[ , 2], datum = NA) +
    # geom_polygon(data = world_ocean, aes(x = lon, y = lat), fill = NA, color = 'black', size = 2) +
    # theme(legend.position = 'none') +
    theme(legend.justification = c(0, 0), legend.position = c(-0.03, 0.02), legend.background = element_rect(fill = NA), legend.text.align = 0) +
    # theme(legend.justification = c(0.5, 0), legend.position = c(0.5, 0), legend.background = element_rect(fill = NA), legend.direction = 'horizontal') +
    # change the size of colorbar
    guides(fill = guide_colorbar(direction = 'vertical', barwidth = 2, barheight = 10, title.position = 'top', title.hjust = 0, label.hjust = 0, frame.linewidth = 0), reverse = FALSE) +
    theme(legend.text = element_text(size = 30, ), legend.title = element_text(size = 35)) +
    # add title
    labs(title = paste(process_name[category], sep = ''), x = '', y = '') + 
    # modify the position of title
    theme(plot.title = element_text(hjust = 0.5, vjust = -1, size = 40)) + 
    # modify the font size
    theme(axis.title = element_text(size = 20)) + 
    theme(panel.background = element_rect(fill = NA, colour = NA)) +
    # modify the margin
    theme(axis.text.x = element_blank(), axis.ticks.x = element_blank(), axis.text.y = element_blank(), axis.ticks.y = element_blank()) + 
    theme(plot.margin = unit(c(0, 0, 0, 0), 'inch')) +
    theme(axis.text=element_text(size = 35, color = 'black')) 

  eval(parse(text = paste('p_baseline', category, ' = p_baseline', sep = '')))

}

jpeg(paste(data_dir_output, 'bulk_process_baseline_map.jpg', sep = ''), width = 40, height = 20, units = 'in', res = 300)
plot_grid(p_baseline1, p_baseline2, p_baseline3, 
          p_baseline4, p_baseline5, p_baseline6,
          p_baseline7, p_baseline8, p_baseline9,
        p_baseline10, p_baseline11, ncol = 3, nrow = 4, align = 'hv', axis = 'tblr', 
        labels = c('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K'),
        label_size = 70,
        label_x = 0.05, label_y = 0.25,
        label_fontface = 'bold',
        label_fontfamily = 'Arial'
)
dev.off()

#####################################################################################
# Plot the parameter data
#####################################################################################
# ipara = 1
for (ipara in 1:length(para_names)) {
  middle_data_baseline_world = para_data_baseline_world[ , c(1:2, 2 + ipara)]
  colnames(middle_data_baseline_world) = c('lon', 'lat', 'para')

  legend_lower_baseline = apply(para_data_baseline_world[ , c(3:(ncol(para_data_baseline_world)-1))], 2, quantile, prob = 0.05, na.rm = TRUE)
  legend_upper_baseline = apply(para_data_baseline_world[ , c(3:(ncol(para_data_baseline_world)-1))], 2, quantile, prob = 0.95, na.rm = TRUE)

  # plot figure
  para_baseline =
    ggplot() +
    geom_tile(data = middle_data_baseline_world, aes(x = lon, y = lat, fill = para), height = 60000, width = 60000, na.rm = TRUE) +
    scale_fill_gradientn(name = 'unitless', colours = rev(viridis(15)), na.value="transparent", limits = c(legend_lower_baseline[ipara], legend_upper_baseline[ipara]), trans = 'identity', oob = scales::squish) +
    geom_sf(data = world_coastline, fill = NA, color = 'black', linewidth = 1) + 
	# coord_sf(xlim = lat_limits_robin[ , 1], ylim = lat_limits_robin[ , 2], datum = NA) +
    coord_sf(ylim = lat_limits_robin[ , 2], datum = NA) +
    # geom_polygon(data = world_ocean, aes(x = lon, y = lat), fill = NA, color = 'black', size = 2) +
    # theme(legend.position = 'none') +
    theme(legend.justification = c(0, 0), legend.position = c(-0.03, 0.02), legend.background = element_rect(fill = NA), legend.text.align = 0) +
    # theme(legend.justification = c(0.5, 0), legend.position = c(0.5, 0), legend.background = element_rect(fill = NA), legend.direction = 'horizontal') +
    # change the size of colorbar
    guides(fill = guide_colorbar(direction = 'vertical', barwidth = 2, barheight = 10, title.position = 'top', title.hjust = 0, label.hjust = 0, frame.linewidth = 0), reverse = FALSE) +
    theme(legend.text = element_text(size = 30, ), legend.title = element_text(size = 35)) +
    # add title
    labs(title = paste(para_names[ipara], sep = ''), x = '', y = '') + 
    # modify the position of title
    theme(plot.title = element_text(hjust = 0.5, vjust = -1, size = 40)) + 
    # modify the font size
    theme(axis.title = element_text(size = 20)) + 
    theme(panel.background = element_rect(fill = NA, colour = NA)) +
    # modify the margin
    theme(axis.text.x = element_blank(), axis.ticks.x = element_blank(), axis.text.y = element_blank(), axis.ticks.y = element_blank()) + 
    theme(plot.margin = unit(c(0, 0, 0, 0), 'inch')) +
    theme(axis.text=element_text(size = 35, color = 'black')) 

  eval(parse(text = paste('para_baseline', ipara, ' = para_baseline', sep = '')))

}

jpeg(paste(data_dir_output, 'para_baseline_map.jpg', sep = ''), width = 45, height = 50, units = 'in', res = 300)
plot_grid(para_baseline1, para_baseline2, para_baseline3, 
          para_baseline4, para_baseline5, para_baseline6,
          para_baseline7, para_baseline8, para_baseline9,
        para_baseline10, para_baseline11, para_baseline12,
        para_baseline13, para_baseline14, para_baseline15,
        para_baseline16, para_baseline17, para_baseline18,
        para_baseline19, para_baseline20, para_baseline21,
        ncol = 3, nrow = 7, align = 'hv', axis = 'tblr', 
        labels = c('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U'),
        label_size = 70,
        label_x = 0.05, label_y = 1.05,
        label_fontface = 'bold',
        label_fontfamily = 'Arial'
)
dev.off()

#####################################################################################
# Plot the SOC data
#####################################################################################
# isoc = 1
# # Drop the row of SOC (soc_data_baseline_world[, 3]) if > 2000
# soc_data_baseline_world = soc_data_baseline_world[soc_data_baseline_world$soc <= 2000, ] # Drop the row of SOC if > 2000 kgC/m2
max(soc_data_baseline_world$soc)

for (isoc in 1:length(soc_category_names)) {
  middle_data_baseline_world = soc_data_baseline_world[ , c(1:2, 2 + isoc)]
  colnames(middle_data_baseline_world) = c('lon', 'lat', 'soc')

  legend_lower_baseline = apply(soc_data_baseline_world[ , c(3:(ncol(soc_data_baseline_world)-1))], 2, quantile, prob = 0.05, na.rm = TRUE)
  legend_upper_baseline = apply(soc_data_baseline_world[ , c(3:(ncol(soc_data_baseline_world)-1))], 2, quantile, prob = 0.95, na.rm = TRUE)

  # plot figure
  soc_baseline =
    ggplot() +
    geom_tile(data = middle_data_baseline_world, aes(x = lon, y = lat, fill = soc), height = 60000, width = 60000, na.rm = TRUE) +
    scale_fill_gradientn(name = 'kgC/m2', colours = rev(viridis(15)), na.value="transparent", limits = c(legend_lower_baseline[isoc], legend_upper_baseline[isoc]), trans = 'identity', oob = scales::squish) +
    geom_sf(data = world_coastline, fill = NA, color = 'black', linewidth = 1) + 
	# coord_sf(xlim = lat_limits_robin[ , 1], ylim = lat_limits_robin[ , 2], datum = NA) +
    coord_sf(ylim = lat_limits_robin[ , 2], datum = NA) +
    # geom_polygon(data = world_ocean, aes(x = lon, y = lat), fill = NA, color = 'black', size = 2) +
    # theme(legend.position = 'none') +
    theme(legend.justification = c(0, 0), legend.position = c(-0.03, 0.02), legend.background = element_rect(fill = NA), legend.text.align = 0) +
    # theme(legend.justification = c(0.5, 0), legend.position = c(0.5, 0), legend.background = element_rect(fill = NA), legend.direction = 'horizontal') +
    # change the size of colorbar
    guides(fill = guide_colorbar(direction = 'vertical', barwidth = 2, barheight = 10, title.position = 'top', title.hjust = 0, label.hjust = 0, frame.linewidth = 0), reverse = FALSE) +
    theme(legend.text = element_text(size = 30, ), legend.title = element_text(size = 35)) +
    # add title
    labs(title = paste(soc_category_names[isoc], sep = ''), x = '', y = '') + 
    # modify the position of title
    theme(plot.title = element_text(hjust = 0.5, vjust = -1, size = 40)) + 
    # modify the font size
    theme(axis.title = element_text(size = 20)) + 
    theme(panel.background = element_rect(fill = NA, colour = NA)) +
    # modify the margin
    theme(axis.text.x = element_blank(), axis.ticks.x = element_blank(), axis.text.y = element_blank(), axis.ticks.y = element_blank()) + 
    theme(plot.margin = unit(c(0, 0, 0, 0), 'inch')) +
    theme(axis.text=element_text(size = 35, color = 'black')) 

  eval(parse(text = paste('soc_baseline', isoc, ' = soc_baseline', sep = '')))

}

jpeg(paste(data_dir_output, 'soc_baseline_map.jpg', sep = ''), width = 30, height = 10, units = 'in', res = 300)
plot_grid(soc_baseline1, soc_baseline2, soc_baseline3, 
          soc_baseline4, soc_baseline5, 
        ncol = 3, nrow = 2, align = 'hv', axis = 'tblr', 
        labels = c('A', 'B', 'C', 'D', 'E'),
        label_size = 70,
        label_x = 0.05, label_y = 1.05,
        label_fontface = 'bold',
        label_fontfamily = 'Arial'
)
dev.off()