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


def plot_losses(filename, train_losses, val_losses):
    assert len(train_losses) == len(val_losses)
    epochs = list(range(len(train_losses)))
    plt.plot(epochs, train_losses, color='blue', label='Train loss')
    plt.plot(epochs, val_losses, color='red', label='Validation loss')
    plt.xlabel('Epoch #')
    plt.ylabel('Loss')
    plt.title('Losses')
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
        rmse =  math.sqrt(mean_squared_error(x, y))

        # Plot stats, regression line, labels
        if should_align:
            stats_string = '(R^2={:.3f}, RMSE={:.3f}, Corr={:.3f})'.format(r2, rmse, corr)
        else:
            stats_string = '(R^2={:.3f}, Corr={:.3f})'.format(r2, corr)
        ax.plot(x, regression_line, 'r', label=regression_equation + ' ' + stats_string) # ' (R^2={:.2f}, Corr={:.2f}, MAPE={:.2f})'.format(r2, corr, mape))
        if should_align:
            ax.plot(x, identity_line, 'g', label='Identity function')
        ax.legend(fontsize=13)

    # Plot scatterplot for this crop type
    if x.size > 500:
        density_scatter(x, y, ax=ax, s=8) #, s=2, color='black')
    elif x.size > 100:
        ax.scatter(x, y, color="k", s=30)
    else:
        ax.scatter(x, y, color="k", s=50)

    ax.tick_params(labelsize=13)
    ax.set_xlabel(x_label, fontsize=13)
    ax.set_ylabel(y_label, fontsize=13)
    if should_align and x.size >= 1:
        min_value = min(np.min(x), np.min(y))-0.1
        max_value = max(np.max(x), np.max(y))+0.1
        ax.set_xlim(min_value, max_value)
        ax.set_ylim(min_value, max_value)

    ax.set_title(title + " (num datapoints: " + str(len(x)) + ")", fontsize=13)


def plot_true_vs_predicted(filename, y_hat, y):
    assert y.shape == y_hat.shape
    num_outputs = y.shape[1]
    rows = math.ceil(num_outputs / 4)
    cols = 4
    fig, axeslist = plt.subplots(rows, cols, figsize=(9*cols, 9*rows), squeeze=False)
    fig.suptitle('True vs predicted soil carbon', fontsize=13)
    for i in range(num_outputs):
        ax = axeslist.ravel()[i]
        plot_single_scatter(ax, y_hat[:, i], y[:, i], "Predicted", "True", f"Layer {i+1}")
    plt.tight_layout()
    fig.subplots_adjust(top=0.90)
    plt.savefig(filename)
    plt.close()


def plot_observations_world_map(lons, lats, values, plot_dir, var_name, title=None):
    if title is None:
        title = var_name
    df = pd.DataFrame({"lon": lons,
                       "lat": lats,
                        var_name: values})

    # Plot world map
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat))
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    gdf.plot(column=var_name, ax=world.boundary.plot(color='gray', figsize=(25, 18)), marker='o', markersize=10, legend=True, legend_kwds={'shrink': 0.7}, zorder=10)
    plt.title(title)
    plt.savefig(os.path.join(plot_dir, "map_{}.png".format(var_name)), bbox_inches='tight')
    plt.close()

    # Plot histograms of the raw values
    plt.hist(values[~np.isnan(values)], bins=30)
    plt.title(title)
    plt.savefig(os.path.join(plot_dir, "histogram_{}.png".format(var_name)))
    plt.close()