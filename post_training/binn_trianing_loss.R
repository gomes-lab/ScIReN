## Packages
library(R.matlab)
library(ggplot2)
library(cowplot)
library(jcolors)
library(viridis)

##
rm(list = ls())

setwd('/Users/phoenix/Google_Drive/Tsinghua_Luo/Projects/BINNS')

############################
# load loss history
############################
data_path = '/Users/phoenix/Google_Drive/Tsinghua_Luo/Projects/DATAHUB/BINNS/OUTPUT_DATA/'

date_stamp = '2023-10-02'

train_loss_history = read.csv(paste(data_path, 'neural_network/train_loss_history_', date_stamp, '.csv', sep = ''), header = FALSE, sep = ',')
val_loss_history = read.csv(paste(data_path, 'neural_network/val_loss_history_', date_stamp, '.csv', sep = ''), header = FALSE, sep = ',')

############################
# plot figure
############################
epoch_num = length(train_loss_history[ , 1])

current_data = rbind(cbind(c(1:epoch_num), apply(train_loss_history, 1, mean, na.rm = TRUE), apply(train_loss_history, 1, sd, na.rm = TRUE), 1),
                     cbind(c(1:epoch_num), apply(val_loss_history, 1, mean, na.rm = TRUE), apply(val_loss_history, 1, sd, na.rm = TRUE), 2)
                     )
colnames(current_data) = c('epoch', 'loss', 'sd', 'set')
current_data = data.frame(current_data)

color_scheme = c('#005AB5', '#DC3220')
line_label = c('Training', 'Validation')

jpeg(paste('./figures/training_loss.jpeg', sep = ''), width = 10, height = 10, units = 'in', res = 300)

ggplot(data = current_data) +
  geom_ribbon(aes(x = epoch, ymin = loss - sd, ymax = loss + sd, fill = as.factor(set)), alpha = 0.15) +
  geom_line(aes(x = epoch, y = loss, color = as.factor(set)), alpha = 1, size = 2) +
  scale_y_continuous(trans = 'log10', n.breaks = 7) +
  scale_x_continuous(trans = 'identity', n.breaks = 7) +
  coord_cartesian(ylim = c(0.3, 10), xlim = c(0, epoch_num)) +
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






