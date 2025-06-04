## Packages
library(R.matlab)
library(ggplot2)
library(cowplot)
library(viridis)
library(scales)
library(ncdf4)

##
rm(list = ls())

setwd('D:/Research/Binn/BINN_output/plot/')

############################
# load data
############################
date_stamp = '20250404-114716_BINN_Global_COMPAS'

data_path = paste0('D:/Research/BINN/BINN_output/neural_network/',  date_stamp, '/')
# data_path = paste0('C:/Users/Hardy/Models/BINN/',  date_stamp, '/')

binn_para = read.csv(paste(data_path, 'Test/nn_test_best_pred_para_', date_stamp, '.csv', sep = ''), header = FALSE, sep = ',')
binn_para = data.matrix(binn_para)

binn_simu_soc = read.csv(paste(data_path, 'Test/nn_test_best_simu_soc_', date_stamp, '.csv', sep = ''), header = FALSE, sep = ',')
binn_simu_soc = data.matrix(binn_simu_soc)
binn_simu_pom = read.csv(paste(data_path, 'Test/nn_test_best_simu_pom_', date_stamp, '.csv', sep = ''), header = FALSE, sep = ',')
binn_simu_pom = data.matrix(binn_simu_pom)
binn_simu_maom = read.csv(paste(data_path, 'Test/nn_test_best_simu_maom_', date_stamp, '.csv', sep = ''), header = FALSE, sep = ',')
binn_simu_maom = data.matrix(binn_simu_maom)

# binn_simu_soc = read.csv(paste(data_path, 'model_training_history/nn_val_pred_soc_', date_stamp, '_520', '.csv', sep = ''), header = FALSE, sep = ',')
# binn_simu_soc = data.matrix(binn_simu_soc)

binn_obs_soc = read.csv(paste(data_path, '/nn_obs_soc_', date_stamp, '.csv', sep = ''), header = FALSE, sep = ',')
binn_obs_soc = data.matrix(binn_obs_soc)
binn_obs_pom = read.csv(paste(data_path, '/nn_obs_pom_', date_stamp, '.csv', sep = ''), header = FALSE, sep = ',')
binn_obs_pom = data.matrix(binn_obs_pom)
binn_obs_maom = read.csv(paste(data_path, '/nn_obs_maom_', date_stamp, '.csv', sep = ''), header = FALSE, sep = ',')
binn_obs_maom = data.matrix(binn_obs_maom)


ncfname = 'D:/Nutstore/Research_Data/BINN/ENSEMBLE/INPUT_DATA/wosis_2019_snap_shot/soc_profile_wosis_2019_snapshot_hugelius_mishra.nc'
# ncfname = 'C:/Research_Data/BINN/ENSEMBLE/INPUT_DATA/wosis_2019_snap_shot/soc_profile_wosis_2019_snapshot_hugelius_mishra.nc'
profile_info = nc_open(ncfname)
profile_info = ncvar_get(profile_info)


para_names = c('diffus', 'cryo', 'q10', 'efolding', 'taucwd', 'taul1', 'taul2', 'tau4doc', 'tau4mic', 'tau4poc', 'tau4maom','fl1_DOC', 
              'fl1_MIC', 'fl2_MIC', 'fl2_MAOM', 'fMIC_DOC', 'fMIC_POC',  'fDOC_MIC', 'CUEl1', 'CUEl2', 'CUEDOC', 'w-scaling', 'beta')


############################
# figure: soc binn vs. obs
############################
categories = c('soc', 'pom', 'maom')
# category= 'pom'
for (category in categories) {
  if (category == 'soc') {
    binn_simu = binn_simu_soc
    binn_obs = binn_obs_soc
    file_name = 'soc_obs_all_vs_binn.jpeg'
    plot_title =  'BINN SOC simulation vs. observation'
  } else if (category == 'pom') {
    binn_simu = binn_simu_pom
    binn_obs = binn_obs_pom
    file_name = 'pom_obs_all_vs_binn.jpeg'
    plot_title =  'BINN POM simulation vs. observation'
  } else if (category == 'maom') {
    binn_simu = binn_simu_maom
    binn_obs = binn_obs_maom
    file_name = 'maom_obs_all_vs_binn.jpeg'
    plot_title =  'BINN MAOM simulation vs. observation'
  }
  # Check nan value
  valid_profile_loc = which(rowSums(!is.na(binn_simu)) > 0 & rowSums(!is.na(binn_obs)) > 0)

  current_data = cbind(as.vector(binn_simu[valid_profile_loc, ]),
                      as.vector(binn_obs[valid_profile_loc, ])
  )/1000
  current_data = data.frame(current_data[which(is.na(current_data[ , 1]) == 0), ])
  colnames(current_data) = c('binn', 'obs')
  # Drop the 0 value
  # dim(current_data)
  # current_data = current_data[which(current_data$obs > 0), ]
  # # current_data = current_data[which(current_data$binn > 0), ]
  # # Check the dimension
  # dim(current_data)


  # NSE
  explain_var = 1 - sum((current_data$binn - current_data$obs)^2)/
    sum((mean(current_data$obs) - current_data$obs)^2)
  print(explain_var)

  # Correlation
  # cor_value = cor(current_data$binn, current_data$obs, use = 'complete.obs')
  cor_value = cor.test(current_data$binn, current_data$obs)
  print(cor_value)
  # # Linear regression
  # # lm_value = lm(current_data$binn ~ current_data$obs)
  # lm_value = lm(log10(current_data$binn) ~ log10(current_data$obs))
  # # Check the min and max value of current_data$obs and current_data$binn
  # print(paste('The min and max value of obs is: ', min(current_data$obs), ' ', max(current_data$obs), sep = ''))
  # print(paste('The min and max value of binn is: ', min(current_data$binn), ' ', max(current_data$binn), sep = ''))

  # print the current_data$obs if it is negative
  print(current_data[which(current_data$obs < 0), ])


  jpeg(paste(data_path, file_name, sep = ''), width = 10, height = 10, units = 'in', res = 300)
  obs_vs_binn = ggplot(data = current_data) + 
    stat_bin_hex(aes(x = obs, y = binn), bins = 100) +
    scale_fill_gradientn(name = 'Count', colors = viridis(7), trans = 'identity', limits = c(1, 20), oob = scales::squish) +
    scale_y_continuous(limits = c(0.1, 1000), trans = 'log10', labels = trans_format('log10', math_format(10^.x))) + 
    scale_x_continuous(limits = c(0.1, 1000), trans = 'log10', labels = trans_format('log10', math_format(10^.x))) + 
    # scale_y_continuous(limits = c(0.1, 250)) +
    # scale_x_continuous(limits = c(0.1, 250)) +
    theme_classic() + 
    # add title
    labs(title = plot_title, x = expression(paste('Observation (kg C m'^'-3', ')', sep = '')), y = expression(paste('BINN simulation (kg C m'^'-3', ')', sep = ''))) +
    # change the legend properties
    guides(fill = guide_colorbar(direction = 'horizontal', barwidth = 15, barheight = 2.5, title.position = 'right', title.hjust = 0, title.vjust = 0.8, label.hjust = 0.5, frame.linewidth = 0), reverse = FALSE) +
    theme(legend.text = element_text(size = 25), legend.title = element_text(size = 25))  +
    theme(legend.justification = c(0, 1), legend.position = c(0, 1), legend.background = element_rect(fill = NA)) + 
    # modify the position of title
    theme(plot.title = element_text(hjust = 0.5, size = 30)) + 
    # add a line to show the correlation and NSE
    geom_abline(slope = 1, intercept = 0, size = 1, color = 'black') +
    # geom_abline(intercept = 0, slope = cor_value$estimate, color = 'red', size = 1) +
    # geom_abline(intercept = lm_value$coefficients[1], slope = lm_value$coefficients[2], color = 'red', size = 1) +
    # modify the font size
    # add the NSE value
    annotate("text", x = 0.3, y = 100, label = paste('NSE: ', round(explain_var, 4), sep = ''), size = 10, color = 'black') +
    # add the correlation value
    # annotate("text", x = 0.55, y = 200, label = paste('Correlation: ', round(cor_value$estimate, 4), sep = ''), size = 10, color = 'black') +
    # modify the margin
    # theme(axis.text.x = element_blank(), axis.ticks.x = element_blank(), axis.text.y = element_blank(), axis.ticks.y = element_blank()) + 
    theme(plot.margin = unit(c(0., 0.2, 0.2, 0.2), 'inch')) +
    theme(axis.text=element_text(size = 30, color = 'black'), axis.title = element_text(size = 35), axis.line = element_line(size = 1), axis.ticks = element_line(size = 1, color = 'black'), axis.ticks.length = unit(0.12, 'inch')) 
  print(obs_vs_binn)
  dev.off()
  print(paste('The figure ', file_name, ' has been saved.', sep = ''))
}


############################
# figure: Distribution of parameters
############################
valid_profile_loc = which(is.na(binn_para[ , 1]) == 0)


ipara = 3
for (ipara in 1:length(para_names)) {
  # correlation
  binn_para_plot = binn_para[valid_profile_loc, ipara]

  # set the limit for both x and y axis
  limit_binn = c(0, 1)
  
  p_corr =
    ggplot(data = data.frame(value = binn_para_plot), aes(x = value)) + 
    geom_histogram(aes(y = ..density..), binwidth = 0.01, fill = "grey70", color = "black", alpha = 0.5) +
    geom_density(color = "blue", size = 1) +
    scale_x_continuous(limits = limit_binn) +
    labs(x = "Parameter Value", y = "Density", title = para_names[ipara]) +
    # change the legend properties
    guides(fill = guide_colorbar(direction = 'horizontal', barwidth = 15, barheight = 2.5, title.position = 'right', title.hjust = 0, title.vjust = 0.8, label.hjust = 0.5, frame.linewidth = 0), reverse = FALSE) +
    theme(legend.text = element_text(size = 25), legend.title = element_text(size = 25))  +
    theme(legend.justification = c(1, 0), legend.position = 'None', legend.background = element_rect(fill = NA)) + 
    # modify the position of title
    theme(plot.title = element_text(hjust = 0.5, size = 50)) + 
    # modify the font size
    # modify the margin
    # theme(axis.text.x = element_blank(), axis.ticks.x = element_blank(), axis.text.y = element_blank(), axis.ticks.y = element_blank()) + 
    theme(plot.margin = unit(c(0., 0.2, 0.2, 0.2), 'inch')) +
    theme(axis.text=element_text(size = 30, color = 'black'), axis.title = element_text(size = 35), axis.line = element_line(size = 1), axis.ticks = element_line(size = 1, color = 'black'), axis.ticks.length = unit(0.12, 'inch')) 
  
  eval(parse(text = paste('p_corr', ipara, ' = p_corr', sep = '')))
  
}

jpeg(paste(data_path, 'para_binn_dist.jpeg', sep = ''), width = 42, height = 28, units = 'in', res = 300)
plot_grid(p_corr1, p_corr2, p_corr3, p_corr4, p_corr5, p_corr6,
          p_corr7, p_corr8, p_corr9, p_corr10, p_corr11, p_corr12, 
          p_corr13, p_corr14, p_corr15, p_corr16, p_corr17, p_corr18,
          p_corr19, p_corr20, p_corr21, p_corr22, p_corr23, NULL,
          nrow = 4, ncol = 6 ,
          rel_widths = c(1, 1, 1, 1, 1, 1)
)
dev.off()




############################
# Loss history
############################

# load .txt file with , as delimiter
# avg_loss_history = read.table(paste(data_path, 'avg_NSE_', date_stamp, '_cesm2_clm5_cen_vr_v2.txt', sep = ''), sep = ',', header = FALSE)
avg_loss_history = read.table(paste(data_path, 'avg_NSE_', date_stamp, '.txt', sep = ''), sep = ',', header = FALSE)
train_loss_history = avg_loss_history[ , 2]
val_loss_history = avg_loss_history[ , 5]

# Plot figure
epoch_num = length(train_loss_history)

current_data = rbind(cbind(c(1:epoch_num), train_loss_history, 1),
                     cbind(c(1:epoch_num), val_loss_history, 2)
                     )
colnames(current_data) = c('epoch', 'loss', 'set')
current_data = data.frame(current_data)

color_scheme = c('#005AB5', '#DC3220')
line_label = c('Training', 'Validation')

jpeg(paste(data_path, "train_loss.jpg"), width = 10, height = 10, units = 'in', res = 300)

ggplot(data = current_data) +
  # geom_ribbon(aes(x = epoch, ymin = loss - sd, ymax = loss + sd, fill = as.factor(set)), alpha = 0.15) +
  geom_line(aes(x = epoch, y = loss, color = as.factor(set)), alpha = 1, size = 2) +
  scale_y_continuous(trans = 'identity', n.breaks = 7) +
  scale_x_continuous(trans = 'identity', n.breaks = 7) +
  coord_cartesian(ylim = c(0, 1), xlim = c(0, epoch_num)) +
  scale_color_manual(name = '', labels = line_label, values = color_scheme) +
  scale_fill_manual(name = '', labels = line_label, values = color_scheme) +
  # change the background to black and white
  theme_classic() +
  # theme(legend.position = 'None') +
  # theme(panel.background = element_rect(fill = 'grey98'), plot.background = element_rect(fill = 'grey98'))+
  theme(legend.justification = c(1, 1), legend.position = c(1, 1), legend.background = element_rect(fill = NA), legend.text.align = 0) +
  theme(legend.text = element_text(size = 30), legend.title = element_text(size = 30))  +
  theme(legend.key = element_rect(color = NA, fill = NA), legend.key.size = unit(0.8, 'inch')) +
  # add title
  labs(x = 'Epoch', y = paste('Loss')) +
  # modify the position of title
  # modify the font sizea
  theme(plot.margin = unit(c(0.2, 0.2, 0.2, 0.2), 'inch'), plot.background = element_rect(fill = 'transparent', color = NULL)) +
  theme(axis.text=element_text(size = 30, color = 'black'), axis.title = element_text(size = 35), axis.line = element_line(size = 1), axis.ticks = element_line(size = 1, color = 'black'), axis.ticks.length = unit(0.12, 'inch'))

dev.off()

