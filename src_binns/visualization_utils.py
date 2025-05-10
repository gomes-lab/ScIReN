import math
import matplotlib.pyplot as plt
import os
import pandas as pd
import geopandas as gpd
import numpy as np
import torch
from matplotlib.colors import Normalize 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, euclidean_distances
from scipy.interpolate import interpn
import subprocess
import matplotlib.patches as patches

# Get the hash of the latest Git commit.
# TODO - this is not a visualization method, but temporarily putting it here for convenience
def get_git_revision_hash():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()


def plot_multiple_losses(filename, losses, labels, splits, y_ranges=None):
    """
    Creates multiple plots (one for each loss type).
    Each plot contains a curve for each split (train/validation).

    losses should be a list of list of lists. The outermost list
    contains one element for each plot. Then, the middle list
    contains one element for each curve. For each curve, we have
    one element for each epoch.

    labels should be a list of titles for each plot.
    splits should be a list of labels for each curve within the plot (same for all plots).
    """
    n_plots = len(losses)
    n_cols = 3
    n_rows = math.ceil(n_plots / n_cols)
    fig, axeslist = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 3*n_rows))
    for plot_idx in range(len(losses)):
        ax = axeslist.ravel()[plot_idx]
        plot_losses = losses[plot_idx]
        loss_type = labels[plot_idx]
        for curve_idx, loss_curve in enumerate(plot_losses):
            ax.plot(np.arange(len(loss_curve)), loss_curve, label=splits[curve_idx])
        ax.set_xlabel("Epoch #")
        ax.set_ylabel(loss_type)
        if y_ranges[plot_idx] is not None:
            ax.set_ylim(y_ranges[plot_idx])
        ax.set_title(loss_type)
        ax.legend()

    # Remove unused plots
    for plot_idx in range(len(losses), n_cols*n_rows):
        fig.delaxes(axeslist.ravel()[plot_idx])

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()



def plot_losses(filename, losses, labels, min_val=None, max_val=None):
    """
    Plots all loss curves to the given filename.
    'losses' should be a list of lists, where each inner list represents
    a particular loss (at each epoch).
    'labels' should contain a string for each loss type.
    """
    for loss_idx, loss_curve in enumerate(losses):
        plt.plot(np.arange(len(loss_curve)), loss_curve, label=labels[loss_idx])
    plt.xlabel('Epoch #')
    plt.ylabel('Value')
    if min_val is not None and max_val is not None:
        plt.ylim([min_val, max_val])
    plt.legend()
    plt.savefig(filename)
    plt.close()


def density_scatter(x, y, ax=None, sort=True, bins=20, **kwargs):
    """Plot a scatter between x/y with density coloring (with 2d histogram).
    
    Code from https://stackoverflow.com/questions/20105364/how-can-i-make-a-scatter-plot-colored-by-density-in-matplotlib
    """

    if ax is None :
        fig, ax = plt.subplots()
    data , x_e, y_e = np.histogram2d( x, y, bins = bins, density = True )
    z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y]).T , method = "splinef2d", bounds_error = False)

    # To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0

    # Sort the points by density, so that the densest points are plotted last
    if sort :
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

    ax.scatter( x, y, c=z, **kwargs )
    norm = Normalize(vmin = np.min(z), vmax = np.max(z))

    return ax


def plot_single_scatter(ax, x, y, x_label, y_label, title, should_align=True):
    # Convert PyTorch Tensor or list into Numpy array
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    elif type(x) == list:
        x = np.array(x)
    if torch.is_tensor(y):
        y = y.detach().cpu().numpy()
    elif type(y) == list:
        y = np.array(y)
    not_nan = ~np.isnan(x) & ~np.isnan(y)
    y = y[not_nan]
    x = x[not_nan]

    # If there are at least 2 datapoints, fit linear regression,
    # plot trendline, compute stats
    if x.size >= 2:
        x = x.reshape(-1, 1)
        regression = LinearRegression(fit_intercept=True).fit(x, y)
        slope = regression.coef_[0]
        intercept = regression.intercept_
        regression_line = slope * x + intercept
        regression_equation = 'y={:.2f}x+{:.2f}'.format(slope, intercept)
        identity_line = x
        y_pred = regression.predict(x)

        # Compute statistics
        y = y.ravel()
        x = x.ravel()
        if should_align:
            r2 = r2_score(y, x)
        else:
            r2 = r2_score(y, y_pred)
        corr = np.corrcoef(x, y)[0, 1]
        mae = mean_absolute_error(x, y)
        mse = mean_squared_error(x, y)
        # rmse =  math.sqrt(mean_squared_error(x, y))

        # Plot stats, regression line, labels
        if should_align:
            stats_string = '(R^2={:.3f}, MSE={:.3f}, MAE={:.3f}, Corr={:.3f})'.format(r2, mse, mae, corr)
        else:
            stats_string = '(R^2={:.3f}, Corr={:.3f})'.format(r2, corr)
        ax.plot(x, regression_line, 'r', label=regression_equation + ' ' + stats_string) # ' (R^2={:.2f}, Corr={:.2f}, MAPE={:.2f})'.format(r2, corr, mape))
        if should_align:
            ax.plot(x, identity_line, 'g', label='Identity function')
        ax.legend(fontsize=10)

    # Plot scatterplot for this crop type
    if x.size > 500:
        density_scatter(x, y, ax=ax, s=8) #, s=2, color='black')
    elif x.size > 100:
        ax.scatter(x, y, color="k", s=30)
    else:
        ax.scatter(x, y, color="k", s=50)

    ax.tick_params(labelsize=12)
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    if should_align and x.size >= 1:
        min_value = min(np.min(x), np.min(y))
        max_value = max(np.max(x), np.max(y))
        margin = (max_value - min_value) * 0.02
        ax.set_xlim(min_value - margin, max_value + margin)
        ax.set_ylim(min_value - margin, max_value + margin)

    ax.set_title(title + " (num datapoints: " + str(len(x)) + ")", fontsize=12)


def plot_true_vs_predicted(filename, y_hat, y):
    """
    Plot a scatter of y vs. y_hat to filename.
    Assumes y and y_hat are single-dimensional and have no nans"""
    assert y.shape == y_hat.shape
    plot_single_scatter(plt.gca(), y_hat, y, "Predicted", "True", "True vs predicted SOC")
    plt.savefig(filename)
    plt.close()


def plot_true_vs_predicted_multiple(filename, y_hats, ys, titles, cols=None):
    """
    Plots multiple scatters to a single filename.
    Scatter i plots y_hats[i] as prediction and ys[i] as true values,
    and will have title titles[i].

    y_hats is a list of 1D numpy arrays or Tensors
    ys is a list of 1D numpy arrays or Tensors
    titles is a list of strings
    All lists must have same length
    """
    n_plots = len(y_hats)
    assert len(y_hats) == len(ys)
    assert len(y_hats) == len(titles)

    if cols is None:
        cols = min(4, n_plots)
    rows = math.ceil(n_plots / cols)
    fig, axeslist = plt.subplots(rows, cols, figsize=(7*cols, 7*rows), squeeze=False)
    fig.suptitle('True vs predicted', fontsize=13)
    for i in range(n_plots):
        ax = axeslist.ravel()[i]
        plot_single_scatter(ax, y_hats[i], ys[i], "Predicted", "True", titles[i])
    plt.tight_layout()
    fig.subplots_adjust(top=0.90)
    plt.savefig(filename)
    plt.close()


def plot_observations_world_map(lons, lats, values, plot_dir, var_name, title=None, us_only=False, ax=None, min_val=None, max_val=None, world=None, states=None):
    if title is None:
        title = var_name
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 5))

    # If data were tensors, first convert to numpy
    if torch.is_tensor(lons):
        lons = lons.detach().cpu().numpy()
    if torch.is_tensor(lats):
        lats = lats.detach().cpu().numpy()
    if torch.is_tensor(values):
        values = values.detach().cpu().numpy()
    if min_val is None:
        min_val = values.min()
    if max_val is None:
        max_val = values.max()
    if world is None:
        world = gpd.read_file("../ENSEMBLE/INPUT_DATA/maps/ne_110m_admin_0_countries_lakes.shp")  # Formerly gpd.datasets.get_path('naturalearth_lowres'))
    if states is None and us_only:
        states = gpd.read_file("../ENSEMBLE/INPUT_DATA/maps/ne_110m_admin_1_states_provinces_lakes.shp")

    # Exclude NaNs and Infs
    df = pd.DataFrame({"lon": lons,
                       "lat": lats,
                        var_name: values})
    df = df[~np.isnan(df[var_name])]
    df = df[~np.isinf(df[var_name])]

    # Plot world map
    if us_only:
        ax.set_xlim(-124.8, -66.9)
        ax.set_ylim(24.5, 49.4)
    world.boundary.plot(ax=ax, color='gray')
    if us_only:
        states.boundary.plot(ax=ax, color='gray')

    # Plot points
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat))
    gdf.plot(column=var_name, ax=ax, marker='o', markersize=8, legend=True, zorder=10, vmin=min_val, vmax=max_val, legend_kwds={'label': f'Change to {var_name}'})  # legend_kwds={'shrink': 0.7},
    ax.set_title(title)

    if plot_dir is not None:
        plt.savefig(os.path.join(plot_dir, "map_{}.png".format(var_name)), bbox_inches='tight')
        plt.close()

    # # Plot histograms of the raw values
    # # filtered_values = values[~np.isnan(values)]
    # # filtered_values = filtered_values[~np.isinf(filtered_values)]
    # values = values[~np.isnan(values)]
    # values = values[~np.isinf(values)]
    # plt.hist(values, bins=30)
    # plt.title(title)
    # plt.savefig(os.path.join(plot_dir, "histogram_{}.png".format(var_name)))
    # plt.close()


def plot_map_grid(filename, lons_list, lats_list, values_list, vars_list, us_only=False, cols=None):
    n_plots = len(lons_list)
    assert len(lons_list) == len(lats_list)
    assert len(lons_list) == len(values_list)
    assert len(lons_list) == len(vars_list)
    world = gpd.read_file("../ENSEMBLE/INPUT_DATA/maps/ne_110m_admin_0_countries_lakes.shp")  # Formerly gpd.datasets.get_path('naturalearth_lowres'))
    states = gpd.read_file("../ENSEMBLE/INPUT_DATA/maps/ne_110m_admin_1_states_provinces_lakes.shp")

    if cols is None:
        cols = min(4, n_plots)
    rows = math.ceil(n_plots / cols)
    fig, axeslist = plt.subplots(rows, cols, figsize=(12*cols, 6*rows), squeeze=False)

    for row_idx in range(rows):
        # For each row, ensure consistent colorbar range
        all_vals = torch.cat(values_list[cols*row_idx:cols*(row_idx+1)])
        all_vals = all_vals[~torch.isnan(all_vals)]
        min_val, max_val = all_vals.min().item(), all_vals.max().item()
        for col_idx in range(cols):
            i = cols*row_idx + col_idx
            if i < n_plots:
                ax = axeslist.ravel()[i]
                plot_observations_world_map(lons_list[i], lats_list[i], values_list[i], plot_dir=None, 
                                            var_name=vars_list[i], title=None, us_only=us_only, ax=ax,
                                            min_val=min_val, max_val=max_val, world=world, states=states)

    plt.tight_layout()
    fig.subplots_adjust(top=0.90)
    plt.savefig(filename)
    plt.close()


def plot_matrix(matrix, row_labels, col_labels, filename, title):
    """
    Source: https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
    """
    assert len(row_labels) == matrix.shape[0]
    assert len(col_labels) == matrix.shape[1]
    if torch.is_tensor(matrix):
        matrix = matrix.detach().cpu().numpy()
    vmax = np.max(np.abs(matrix))
    fig, ax = plt.subplots(figsize=(0.5*len(col_labels)+3, 0.5*len(row_labels)+3))
    ax.imshow(matrix, cmap="RdBu", vmin=-vmax, vmax=vmax)

    # Show all ticks and label them with the respective list entries
    ax.tick_params(axis="x", labelsize=20)
    ax.tick_params(axis="y", labelsize=20)
    ax.set_xticks(range(len(col_labels)), labels=col_labels,
                rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(row_labels)), labels=row_labels)

    # Loop over data dimensions and create text annotations.
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            text = ax.text(j, i, round(matrix[i, j], 2),
                        ha="center", va="center", color="w")

    ax.set_title(title)  # "Harvest of local farmers (in tons/year)")
    fig.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_shape_function(feat_nn, feat_vals, ax, title, xlabel, ylabel, divide_by=1.0, scaling_x_min=None, scaling_x_max=None, scaling_y_min=None, scaling_y_max=None):
    """
    divide_by is an optional factor to divide the y value by
    scaling_min, scaling_max were the max/min used to scale the feature before passing to BINN.
    If they are set, plot the shape function in the original units.
    """
    # If no scaling, set scaling_x/y_min to be 0, scaling_x/y_max to be 1
    if scaling_x_min is None:
        scaling_x_min = 0
    if scaling_x_max is None:
        scaling_x_max = 1
    if scaling_y_min is None:
        scaling_y_min = 0
    if scaling_y_max is None:
        scaling_y_max = 1

    # min, max and intermediate feature values, AFTER being normalized to [0, 1]
    min_x = feat_vals.min().item() - 0.05
    max_x = feat_vals.max().item() + 0.05
    xs = torch.linspace(min_x, max_x, 100)

    # Original feature values BEFORE min/max normalization
    original_min_x = min_x * (scaling_x_max - scaling_x_min) + scaling_x_min
    original_max_x = max_x * (scaling_x_max - scaling_x_min) + scaling_x_min
    original_xs = xs * (scaling_x_max - scaling_x_min) + scaling_x_min  # Original x before min/max normalization
    ys = feat_nn(xs).squeeze() / divide_by
    ys = ys * (scaling_y_max - scaling_y_min)   # Only consider relative changes, so don't add y_min
    min_y = ys.min().item() - 0.05
    max_y = ys.max().item() + 0.05
    ax.plot(original_xs, ys)
    ax.set_xlim(original_min_x, original_max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    def shade_by_density_blocks(color: list = [0.9, 0.5, 0.5]):
        """
        Source: https://github.com/AmrMKayid/nam/blob/main/nam/utils/graphing.py
        """
        single_feature_data = feat_vals  # single_features[name]
        x_n_blocks = 20  # min(n_blocks, len(unique_feat_data))

        segments = (original_max_x - original_min_x) / x_n_blocks
        density = np.histogram(single_feature_data, bins=x_n_blocks)
        normed_density = density[0] / np.max(density[0])
        rect_params = []

        for p in range(x_n_blocks):
            start_x = original_min_x + segments * p
            end_x = original_min_x + segments * (p + 1)
            d = min(1.0, 0.01 + normed_density[p])
            rect_params.append((d, start_x, end_x))

        for param in rect_params:
            alpha, start_x, end_x = param
            rect = patches.Rectangle(
                (start_x, min_y - 1),
                end_x - start_x,
                max_y - min_y + 1,
                linewidth=0.01,
                edgecolor=color,
                facecolor=color,
                alpha=alpha,
            )
            ax.add_patch(rect)

    shade_by_density_blocks()


def plot_histogram(ax, values, title):
    """
    Plots a histogram of the given 'values' to 'ax'. The title contains the 'title' string
    (along with mean/std/min/max)
    """
    ax.hist(values, bins=30)
    ax.set_title("{} (n={}):\nmean={:.2f}, std={:.2f}\nmin={:.2f}, max={:.2f}\np1={:.2f}, p99={:.2f}".format(
                 title, len(values), np.mean(values), np.std(values), np.min(values), np.max(values),
                 np.quantile(values, 0.01), np.quantile(values, 0.99)))


def plot_histogram_multiple(list_of_values, names, filename, n_rows=None):
    """
    list_of_values is a list that contains Numpy arrays. Plots a histogram for each flattened array.
    """
    assert len(list_of_values) == len(names)
    channel_idx = 0
    if n_rows is None:
        n_rows = 3
    n_cols = math.ceil(len(list_of_values) / n_rows)
    fig, axeslist = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 3*n_rows))

    # Loop through each band/percentile combination
    for idx, values in enumerate(list_of_values):
        values_flat = values.flatten()
        plot_histogram(axeslist.ravel()[idx], values_flat, names[idx])
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()