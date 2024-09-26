## Packages
library(R.matlab)
library(ggplot2)
library(cowplot)
library(jcolors)
library(viridis)
library(scales)
library(ncdf4)

##
rm(list = ls())

setwd('/Users/phoenix/Google_Drive/Tsinghua_Luo/Projects/BINNS')

############################
# load data
############################
data_path = '/Users/phoenix/Google_Drive/Tsinghua_Luo/Projects/DATAHUB/BINNS/OUTPUT_DATA/'

date_stamp = '2023-10-02'

binn_para = read.csv(paste(data_path, 'neural_network/nn_best_pred_para_', date_stamp, '.csv', sep = ''), header = FALSE, sep = ',')
binn_para = data.matrix(binn_para)

mcmc_para = readMat('/Users/phoenix/Google_Drive/Tsinghua_Luo/Projects/DATAHUB/ENSEMBLE/OUTPUT_DATA/mcmc_summary_cesm2_clm5_cen_vr_v2/cesm2_clm5_cen_vr_v2_para_mean.mat')
mcmc_para = mcmc_para$para.mean

binn_simu_soc = read.csv(paste(data_path, 'neural_network/nn_best_simu_soc_', date_stamp, '.csv', sep = ''), header = FALSE, sep = ',')
binn_simu_soc = data.matrix(binn_simu_soc)

binn_obs_soc = read.csv(paste(data_path, 'neural_network/nn_obs_soc_', date_stamp, '.csv', sep = ''), header = FALSE, sep = ',')
binn_obs_soc = data.matrix(binn_obs_soc)


ncfname = '/Users/phoenix/Google_Drive/Tsinghua_Luo/Projects/DATAHUB/ENSEMBLE/INPUT_DATA/wosis_2019_snap_shot/soc_profile_wosis_2019_snapshot_hugelius_mishra.nc'
profile_info = nc_open(ncfname)
profile_info = ncvar_get(profile_info)


para_names = c('diffus', 'cryo', 'q10', 'efolding', 
               'taucwd', 'taul1', 'taul2', 'tau4s1', 'tau4s2', 'tau4s3', 
               'fl1s1', 'fl2s1', 'fl3s2', 'fs1s2', 'fs1s3', 'fs2s1', 'fs2s3', 'fs3s1', 'fcwdl2', 
               'w-scaling', 'beta')

############################
# figure: para mcmc vs. binn
############################
valid_profile_loc = which(is.na(binn_para[ , 1]) == 0)

corr_process_summary = array(NA, dim = c(length(para_names), 2))

ipara = 3
for (ipara in 1:length(para_names)) {
  # correlation
  middle_data_corr = cbind(mcmc_para[valid_profile_loc, ipara], binn_para[valid_profile_loc, ipara])
  middle_data_corr = data.frame(middle_data_corr)
  colnames(middle_data_corr) = c('mcmc', 'binn')
  
  limit_mcmc = quantile(mcmc_para[valid_profile_loc, ipara], probs = c(0, 1), na.rm = TRUE)
  limit_binn = quantile(binn_para[valid_profile_loc, ipara], probs = c(0, 1), na.rm = TRUE)
  
  corr_process_middle = cor.test(middle_data_corr$mcmc, middle_data_corr$binn, na.rm = TRUE)
  corr_process_summary[ipara, ] = c(corr_process_middle$estimate, corr_process_middle$p.value)
  
  p_corr =
    ggplot() + 
    stat_bin_hex(data = middle_data_corr, aes(x = mcmc, y = binn), bins = 30) +
    scale_fill_gradientn(name = 'Count', colors = viridis(7), limits = c(0, 20), trans = 'identity', oob = scales::squish) +
    geom_abline(slope = 1, intercept = 0, size = 2, color = 'black') +
    scale_x_continuous(trans = 'identity', limits = limit_mcmc) +
    scale_y_continuous(trans = 'identity', limits = limit_binn) +
    theme_classic() + 
    # add title
    labs(title = para_names[ipara], x = 'MCMC', y = 'BINN') + 
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

jpeg(paste('./figures/para_binn_vs_mcmc.jpeg', sep = ''), width = 42, height = 28, units = 'in', res = 300)
plot_grid(p_corr1, p_corr2, p_corr3, p_corr4, p_corr5, p_corr6,
          p_corr7, p_corr8, p_corr9, p_corr10, p_corr11, p_corr12, 
          p_corr13, p_corr14, p_corr15, p_corr16, p_corr17, p_corr18,
          p_corr19, p_corr20, p_corr21, NULL, NULL, NULL,
          nrow = 4, ncol = 6 ,
          rel_widths = c(1, 1, 1, 1, 1, 1)
)
dev.off()

############################
# figure: soc binn vs. obs
############################
current_data = cbind(as.vector(binn_simu_soc[valid_profile_loc, ]),
                     as.vector(binn_obs_soc[valid_profile_loc, ])
)/1000
current_data = data.frame(current_data[which(is.na(current_data[ , 1]) == 0), ])
colnames(current_data) = c('binn', 'obs')


explain_var = 1 - sum((current_data$binn - current_data$obs)^2)/
  sum((mean(current_data$obs) - current_data$obs)^2)

jpeg(paste('./figures/soc_obs_vs_binn.jpeg', sep = ''), width = 10, height = 10, units = 'in', res = 300)
ggplot(data = current_data) + 
  stat_bin_hex(aes(x = obs, y = binn), bins = 100) +
  scale_fill_gradientn(name = 'Count', colors = viridis(7), trans = 'identity', limits = c(1, 20), oob = scales::squish) +
  scale_y_continuous(limits = c(0.1, 1000), trans = 'log10', labels = trans_format('log10', math_format(10^.x))) + 
  scale_x_continuous(limits = c(0.1, 1000), trans = 'log10', labels = trans_format('log10', math_format(10^.x))) + 
  geom_abline(slope = 1, intercept = 0, size = 1, color = 'black') +
  theme_classic() + 
  # add title
  labs(title = '', x = expression(paste('Observation (kg C m'^'-3', ')', sep = '')), y = expression(paste('BINN simulation (kg C m'^'-3', ')', sep = ''))) +
  # change the legend properties
  guides(fill = guide_colorbar(direction = 'horizontal', barwidth = 15, barheight = 2.5, title.position = 'right', title.hjust = 0, title.vjust = 0.8, label.hjust = 0.5, frame.linewidth = 0), reverse = FALSE) +
  theme(legend.text = element_text(size = 25), legend.title = element_text(size = 25))  +
  theme(legend.justification = c(0, 1), legend.position = c(0, 1), legend.background = element_rect(fill = NA)) + 
  # modify the position of title
  theme(plot.title = element_text(hjust = 0.5, size = 50)) + 
  # modify the font size
  # modify the margin
  # theme(axis.text.x = element_blank(), axis.ticks.x = element_blank(), axis.text.y = element_blank(), axis.ticks.y = element_blank()) + 
  theme(plot.margin = unit(c(0., 0.2, 0.2, 0.2), 'inch')) +
  theme(axis.text=element_text(size = 30, color = 'black'), axis.title = element_text(size = 35), axis.line = element_line(size = 1), axis.ticks = element_line(size = 1, color = 'black'), axis.ticks.length = unit(0.12, 'inch')) 

dev.off()


########################################
# profile distribution
#########################################
# can be changed to state or world to have US and world map
world_coastline = rgdal::readOGR(dsn='/Users/phoenix/Google_Drive/Tsinghua_Luo/World_Vector_Shape/ne110m/ne_110m_land.shp',layer = 'ne_110m_land')
world_coastline <- fortify(world_coastline)
Map.Using = world_coastline


current_data = cbind(profile_info[valid_profile_loc, 4], profile_info[valid_profile_loc, 5])
current_data = data.frame(current_data)
colnames(current_data) = c('lon', 'lat')

jpeg(paste('./figures/profile_distribution.jpeg', sep = ''), width = 12, height = 6, units = 'in', res = 300)

ggplot(data = current_data) +
  geom_point(aes(x = lon, y = lat), color = 'black', shape = 16, size = 1, alpha = 1) + 
  geom_polygon(data = Map.Using, aes(x = long, y = lat, group = group), fill = NA, color = 'black', size = 0.3) +
  ylim(c(-56, 80)) +
  # change the background to black and white
  theme_bw() +
  # change the legend properties
  # theme(legend.position = 'none') +
  theme(legend.justification = c(0, 0), legend.position = c(0, 0), legend.background = element_rect(fill = NA), legend.text.align = 0) +
  theme(legend.text = element_text(size = 35), legend.title = element_text(size = 35))  +
  guides(colour = guide_legend(override.aes = list(size = 5))) +
  theme(legend.text = element_text(size = 15), legend.title = element_text(size = 20)) +
  # add title
  labs(x = '', y = '') + 
  theme(axis.text.x = element_blank(), axis.ticks.x = element_blank(), axis.text.y = element_blank(), axis.ticks.y = element_blank()) + 
  # modify the position of title
  theme(plot.title = element_text(hjust = 0.5, size = 40)) + 
  # modify the font size
  theme(axis.title = element_text(size = 20)) + 
  # modify the margin
  theme(plot.margin = unit(c(0.1, 0.1, 0.1, 0.1), 'inch')) +
  theme(axis.text=element_text(size = 30))

dev.off()





