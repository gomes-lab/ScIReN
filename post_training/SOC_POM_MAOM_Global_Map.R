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
Sys.setenv(PROJ_LIB = "C:/Users/hx293/AppData/Local/R/win-library/4.3/sf/proj")


## Jet colorbar function
jet.colors <- colorRampPalette(c("#00007F", "blue", "#007FFF", "cyan", "#7FFF7F", "yellow", "#FF7F00", "red", "#7F0000"))
diff.colors <- colorRampPalette(c("#2166AC", "#4393C3", "#92C5DE", "#D1E5F0", "#f6f6f6", "#FDDBC7", "#F4A582", "#D6604D", "#B2182B"))

#############################################################################
# Data Path
#############################################################################
date_stamp = '20241118-153400_Global_SPECTRALREG_With_POM_MAOM_Weight_1_6688488'

# input and output data path
data_dir_input = paste0('D:/Research/BINN/BINN_output/neural_network/',  date_stamp, '/')
data_dir_output = paste0('D:/Research/BINN/BINN_output/neural_network/',  date_stamp, '/Result_Map/')
# PRODA data path
data_dir_PRODA = 'D:/Research/BINN/Research_Data/BINN/Server_Script/post_training/soc_component_proda/soc_component_proda/'
data_dir_loc = 'D:/Research/BINN/Research_Data/BINN/Server_Script/post_training/component_calculation/'
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

# Loop through three categories: SOC, POM, MAOM
# category= 'pom'
categories = c('soc', 'pom', 'maom')
for (category in categories) {

    #############################################################################
    # read in data
    #############################################################################
    ## SOC Data
    if (category == 'soc') {
        # BINN Prediction
        binn_pred = read.csv(paste(data_dir_input, '/Test', '/nn_test_', 'best_simu_soc_', date_stamp, '.csv', sep = ''), header = FALSE)
        binn_pred = data.matrix(binn_pred)
        # Observation
        obs = read.csv(paste(data_dir_input, '/nn_obs_soc_', date_stamp, '.csv', sep = ''), header = FALSE)
        obs = data.matrix(obs)
    } else if (category == 'pom') {
        # BINN Prediction
        binn_pred = read.csv(paste(data_dir_input, '/Test', '/nn_test_', 'best_simu_pom_', date_stamp, '.csv', sep = ''), header = FALSE)
        binn_pred = data.matrix(binn_pred)
        # Observation
        obs = read.csv(paste(data_dir_input, '/nn_obs_pom_', date_stamp, '.csv', sep = ''), header = FALSE)
        obs = data.matrix(obs)
    } else if (category == 'maom') {
        # BINN Prediction
        binn_pred = read.csv(paste(data_dir_input, '/Test', '/nn_test_', 'best_simu_maom_', date_stamp, '.csv', sep = ''), header = FALSE)
        binn_pred = data.matrix(binn_pred)
        # Observation
        obs = read.csv(paste(data_dir_input, '/nn_obs_maom_', date_stamp, '.csv', sep = ''), header = FALSE)
        obs = data.matrix(obs)
    }

   

    ## Depth Data for each soc observation
    upper_depth = read.csv(paste(data_dir_input, '/Test', '/nn_test_', 'upper_depth_', date_stamp, '.csv', sep = ''), header = FALSE)
    upper_depth = data.matrix(upper_depth)
    lower_depth = read.csv(paste(data_dir_input, '/Test', '/nn_test_', 'lower_depth_', date_stamp, '.csv', sep = ''), header = FALSE)
    lower_depth = data.matrix(lower_depth)

    # Select rows where there's at least one valid soc prediction (among 200 columns)
    valid_profile_row = which(rowSums(!is.na(binn_pred)) > 0 & rowSums(!is.na(obs)) > 0)




    # For each row, if valid_profile_loc is valid in that row, calculate the difference between the observation and prediction and times the difference between the upper and lower depth
    # If the valid_profile_loc is not valid in that row, set the difference to be NA
    # Store the sum of the difference in a new vector
    diff = rep(NA, nrow(binn_pred))
    for (i in valid_profile_row) {
        temp_diff_sum = 0
        for (j in 1:200) {
            if (!is.na(binn_pred[i, j])) {
                temp_diff_sum = temp_diff_sum + (obs[i] - binn_pred[i, j]) * (lower_depth[i, j] - upper_depth[i, j])
            }
        }
        diff[i] = temp_diff_sum
    }


    # Get the lon and lat for plotting
    binn_lon = read.csv(paste(data_dir_input, '/Test', '/nn_test_', 'lons_', date_stamp, '.csv', sep = ''), header = FALSE)
    binn_lat = read.csv(paste(data_dir_input, '/Test', '/nn_test_', 'lats_', date_stamp, '.csv', sep = ''), header = FALSE)
    binn_loc = cbind(binn_lon, binn_lat)
    colnames(binn_loc) = c('lon', 'lat')

    # Bind the lon and lat with the soc_diff
    current_data_binn = cbind(binn_loc, diff)
    colnames(current_data_binn) = c('lon', 'lat', 'diff')


    # exclude the data with nan value for all input variables
    current_data_binn = current_data_binn[valid_profile_row, ]


    #################################################################################
    # plot figures
    #################################################################################
    #-------------------------------------soc stock and Residence Time map
    world_coastline = st_read('D:/Nutstore/Research_Data/Map_Plot/ne_110m_land/ne_110m_land.shp', layer = 'ne_110m_land')
    world_coastline <- st_transform(world_coastline, CRS('+proj=robin'))
    # world_coastline = st_read('D:/Nutstore/Research_Data/Map_Plot/cb_2018_us_state_500k/cb_2018_us_state_500k.shp', layer = 'cb_2018_us_state_500k')
    # Define the bounding box for the mainland U.S. (excluding Alaska)
    # bbox <- st_bbox(c(xmin = -125, xmax = -66, ymin = 20, ymax = 50), crs = st_crs(world_coastline))
    # world_coastline <- st_crop(world_coastline, bbox)
    # coord_info = '+proj=aea +lat_1=20 +lat_2=49.38 +lon_0=-96 +x_0=0 +y_0=0 +datum=NAD83'
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
    lon_lat_transfer = project(xy = as.matrix(current_data_binn[ , c('lon', 'lat')]), proj = coord_info) 
    current_data_binn[ , c('lon', 'lat')] = lon_lat_transfer
    # plot data only within the shapefile constraint
    current_data_binn <- st_as_sf(current_data_binn, coords = c('lon', 'lat'), crs = st_crs(world_coastline))
    current_data_binn <- st_intersection(current_data_binn, world_coastline)

    # Extract the coordinates from the geometry column
    coords <- st_coordinates(current_data_binn$geometry)
    # Add lon and lat back to the data frame to the first two columns
    current_data_binn$lat <- coords[, 2]
    current_data_binn$lon <- coords[, 1]
    # move the longtitude to the first column
    current_data_binn <- current_data_binn[c("lon", "lat", setdiff(names(current_data_binn), c("lon", "lat")))]
    # remove the geometry column
    current_data_binn <- st_drop_geometry(current_data_binn)
    # remove all column after the 3rd column
    current_data_binn <- current_data_binn[ , 1:3]

    # Normalized the difference between -1 and 1
    # For positive difference, normalized to 0 to 1
    for (i in 1:nrow(current_data_binn)) {
        if (current_data_binn$diff[i] > 0) {
            current_data_binn$normalized_diff[i] = current_data_binn$diff[i] / max(current_data_binn$diff, na.rm = TRUE)
        } else {
            current_data_binn$normalized_diff[i] = -1 * current_data_binn$diff[i] / min(current_data_binn$diff, na.rm = TRUE)
        }
    }


    # Set the lower and upper limit for the legend
    legend_lower_diff = min(current_data_binn$normalized_diff, na.rm = TRUE)
    legend_upper_diff = max(current_data_binn$normalized_diff, na.rm = TRUE)

    # Name based on the category
    if (category == 'soc') {
        plot_name = 'SOC Difference'
        plot_title = 'SOC Difference Map'
    } else if (category == 'pom') {
        plot_name = 'POM Difference'
        plot_title = 'POM Difference Map'
    } else if (category == 'maom') {
        plot_name = 'MAOM Difference'
        plot_title = 'MAOM Difference Map'
    }

    # Plot the difference map for SOC
    map_diff_soc = ggplot() +
        # geom_tile(data = current_data_binn_us, aes(x = lon, y = lat, fill = normalized_soc_diff), height = 60000, width = 60000, na.rm = TRUE) +
        geom_point(data = current_data_binn, aes(x = lon, y = lat, color = normalized_diff), size = 3) +
        # scale_fill_gradientn(name = 'SOC Difference', colours = rev(viridis(15)), na.value="transparent", limits = c(legend_lower_diff_soc, legend_upper_diff_soc), trans = 'identity', oob = scales::squish) +
        # Use diff.colors for the colorbar
        scale_color_gradientn(name = plot_name, colours = diff.colors(15), na.value="transparent", limits = c(legend_lower_diff, legend_upper_diff), trans = 'identity', oob = scales::squish) +
        geom_sf(data = world_coastline, fill = NA, color = 'black', linewidth = 0.6) + 
        # geom_polygon(data = world_ocean, aes(x = lon, y = lat), fill = NA, color = 'black', size = 2) +
        coord_sf(xlim = lat_limits_robin[ , 1], ylim = lat_limits_robin[ , 2], datum = NA) +
        # change the background to black and white
        # coord_equal() +
        # theme_map() +
        ylim(lat_limits_robin[ , 2]) +
        # change the legend properties
        # theme(legend.position = 'none') +
        theme(legend.justification = c(0, 0), legend.position = c(0, 0), legend.background = element_rect(fill = NA), legend.text.align = 0, legend.key.height = unit(1.2, 'cm'), legend.key.width = unit(1, 'cm')) +
        # theme(legend.justification = c(0.5, 0), legend.position = c(0.5, 0), legend.background = element_rect(fill = NA), legend.direction = 'horizontal') +
        # change the size of colorbar
        guides(fill = guide_colorbar(direction = 'vertical', barwidth = 1, barheight = 3, title.position = 'top', title.hjust = 0, label.hjust = 0, frame.linewidth = 0), reverse = FALSE) +
        theme(legend.text = element_text(size = 12, ), legend.title = element_text(size = 14)) +
        # add title
        labs(title = plot_title) +
        # modify the position of title
        theme(plot.title = element_text(hjust = 0.5, vjust = -1, size = 30)) + 
        # modify the font size
        # theme(axis.title = element_text(size = 30)) + 
        theme(axis.title = element_blank()) +
        theme(panel.background = element_rect(fill = NA, colour = NA)) +
        # modify the margin
        theme(axis.text.x = element_blank(), axis.ticks.x = element_blank(), axis.text.y = element_blank(), axis.ticks.y = element_blank()) + 
        theme(plot.margin = unit(c(0, 0, 0, 0), 'inch'))
        # theme(axis.text=element_text(size = 15, color = 'black'))

    # Save the map
    jpeg(paste0(data_dir_output, plot_title, '_', date_stamp, '.jpeg'), width = 16, height = 8, units = 'in', res = 300)
    print(map_diff_soc)
    dev.off()

    print(paste0('Finish plotting ', plot_title))
}