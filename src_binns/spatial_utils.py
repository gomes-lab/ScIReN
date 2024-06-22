from scipy.stats import wasserstein_distance
from math import radians, cos, sin, asin, sqrt
import math
import torch
import numpy as np
import torch.nn.parallel
from torch.utils.data.sampler import BatchSampler
import visualization_utils

def deg_to_rad(x):
  return x * math.pi / 180

def latlon_to_cart(lat,lon):
  x = np.cos(lat) * np.cos(lon)
  y = np.cos(lat) * np.sin(lon)
  z = np.sin(lat)
  cart_coord = np.column_stack((x, y, z))
  return cart_coord

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371
    return c * r * 1000

# Helper function for 2+d distance
def newDistance(a, b, nd_dist="great_circle"):
    # Distance options are ["great_circle" (2D only), "euclidean", "wasserstein" (for higher-dimensional coordinate embeddings)]
    if a.shape[0]==2:
      x1, y1 = a[0], a[1]
      x2, y2 = b[0], b[1]
      if nd_dist=="euclidean":
        d = math.sqrt( ((x1-x2)**2)+((y1-y2)**2))
      else:
        d = haversine(x1,y1,x2,y2)
    if a.shape[0]==3:
      x1, y1, z1 = a[0], a[1], a[2]
      x2, y2, z2 = b[0], b[1], b[2]
      d = math.sqrt(math.pow(x2 - x1, 2) +
                  math.pow(y2 - y1, 2) +
                  math.pow(z2 - z1, 2)* 1.0)
    if a.shape[0]>3:
      if nd_dist=="wasserstein":
        d = wasserstein_distance(a.reshape(-1).detach(),b.reshape(-1).detach())
        #d = sgw_cpu(a.reshape(1,-1).detach(),b.reshape(1,-1).detach())
      else:
        d = torch.pow(a.reshape(1,1,-1) - b.reshape(1,1,-1), 2).sum(2)
    return d

# Helper function for edge weights
def makeEdgeWeight(x, edge_index):
  to = edge_index[0]
  fro = edge_index[1]
  edge_weight = []
  for i in range(len(to)):
    edge_weight.append(newDistance(x[to[i]],x[fro[i]]) / 200000) # probably want to do inverse distance eventually. newDistance returns distance in meters, convert to 200-km
  # max_val = max(edge_weight)
  # rng = max_val - min(edge_weight)
  # edge_weight = [(max_val - elem) / rng for elem in edge_weight]
  # min_val = min(edge_weight)
  # rng = max(edge_weight) - min_val
  # edge_weight = [(elem - min_val) / rng for elem in edge_weight]
  # edge_weight = [np.exp(-5*elem) for elem in edge_weight]
  return torch.Tensor(edge_weight)


# For each node, sum the weights of incoming edges. Note that if edges are listed as [start, end] in edge_index,
# knn_graph means that there are exactly k edges going IN to each node.
# edge_index: [2, num_edges]
# edge_weight: [num_edges]
def sumIncomingWeights(n_nodes, edge_index, edge_weight):
  assert edge_index.shape[0] == 2
  assert edge_index.shape[1] == edge_weight.shape[0]
  in_sums = torch.zeros((n_nodes), device=edge_weight.device)
  out_sums = torch.zeros((n_nodes), device=edge_weight.device)
  in_counts = torch.zeros((n_nodes), device=edge_weight.device)
  out_counts = torch.zeros((n_nodes), device=edge_weight.device)
  for e_idx in range(edge_index.shape[1]):
    start_node = edge_index[0, e_idx]
    end_node = edge_index[1, e_idx]
    out_sums[start_node] += edge_weight[e_idx]
    out_counts[start_node] += 1
    in_sums[end_node] += edge_weight[e_idx]
    in_counts[end_node] += 1
  # print("In", in_counts, in_sums, "Out", out_counts, out_sums)
  return in_sums.unsqueeze(1)


# # Helper function for edge weights, using a learned exponential decay
# def makeEdgeWeight(x, edge_index):
#   to = edge_index[0]
#   fro = edge_index[1]
#   edge_weight = []
#   for i in range(len(to)):
#     edge_weight.append(newDistance(x[to[i]],x[fro[i]])) # probably want to do inverse distance eventually
#   max_val = max(edge_weight)
#   rng = max_val - min(edge_weight)
#   edge_weight = [(max_val - elem) / rng for elem in edge_weight]
#   return torch.Tensor(edge_weight)


# knn graph to adjacency matrix (probably already built)
def knn_to_adj(knn, n):
  adj_matrix = torch.zeros(n, n, dtype=float) #lil_matrix((n, n), dtype=float)
  for i in range(len(knn[0])):
    tow = knn[0][i]
    fro = knn[1][i]
    adj_matrix[tow,fro] = 1 # should be bidectional?
  return adj_matrix.T

def normal_torch(tensor,min_val=0):
  t_min = torch.min(tensor)
  t_max = torch.max(tensor)
  if t_min == 0 and t_max == 0:
    return torch.tensor(tensor)
  if min_val == -1:
    tensor_norm = 2 * ((tensor - t_min) / (t_max - t_min)) - 1
  if min_val== 0:
    tensor_norm = ((tensor - t_min) / (t_max - t_min))
  return torch.tensor(tensor_norm)

def lw_tensor_local_moran(y,w_sparse,na_to_zero=True,norm=True,norm_min_val=0):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  y = y.reshape(-1)
  n = len(y)
  n_1 = n - 1
  z = y - y.mean()
  sy = y.std()
  z /= sy
  den = (z * z).sum()
  zl = torch.tensor(w_sparse * z).to(device)
  mi = n_1 * z * zl / den
  if na_to_zero==True:
    mi[torch.isnan(mi)] = 0
  if norm==True:
    mi = normal_torch(mi,min_val=norm_min_val)
  return torch.tensor(mi)


class SpatialBlockBatchSampler(BatchSampler):
    """
    Sample batches by cutting the domain into rectangles. Note that
    the number of examples in each batch may vary significantly.

    Code is inspired by: https://stackoverflow.com/questions/74252067/efficiently-sample-batches-from-only-one-class-at-each-iteration-with-pytorch
    https://stackoverflow.com/questions/66065272/customizing-the-batch-with-specific-elements
    """
    def __init__(self, batch_size, all_coords, plot_dir=None):
        # "batch_size" is the avg batch size (roughly).
        # "all_coords" should be of shape [num_examples, 2]: column 0 is lon and column 1 is lat.
        self.batch_size = batch_size
        self.all_coords = all_coords
        n_datapoints = all_coords.shape[0]
        sorted_lons = np.sort(all_coords[:, 0])
        sorted_lats = np.sort(all_coords[:, 1])

        # Calculate number of columns and rows first
        width = sorted_lons[-1] - sorted_lons[0]
        height = sorted_lats[-1] - sorted_lats[0]
        n_batches = int(n_datapoints / batch_size)
        n_cols = int(np.sqrt(n_batches*(width/height)))
        n_rows = int(n_batches / n_cols)
        print("Lons", sorted_lons[0], "to", sorted_lons[-1], "Lats", sorted_lats[0], "to", sorted_lats[-1], "Datapoints", n_datapoints)
        print("Width", width, "Height", height, "Rows", n_rows, "Cols", n_cols)

        # Cut into equally-sized columns first
        all_block_indices = []
        for i in range(n_cols):
           lower_lon = sorted_lons[int(n_datapoints*i/n_cols)]
           upper_lon = sorted_lons[int(n_datapoints*(i+1)/n_cols)-1]
           col_indices = np.flatnonzero((self.all_coords[:, 0] >= lower_lon) & (self.all_coords[:, 0] <= upper_lon))
           print("Column", i, len(col_indices))

           # Cut this column into equal-sized rows
           col_coords = self.all_coords[col_indices]
           col_sorted_lats = np.sort(col_coords[:, 1])
           for j in range(n_cols):
              lower_lat = col_sorted_lats[int(len(col_sorted_lats)*i/n_rows)]
              upper_lat = col_sorted_lats[int(len(col_sorted_lats)*(i+1)/n_rows)-1]
              block_indices = col_indices[(col_coords[:, 0] >= lower_lat) & (col_coords[:, 0] <= upper_lat)]
              all_block_indices.append(block_indices)

        self.batches = all_block_indices

        # Visualization: create array with batch assignment of each example
        if plot_dir is not None:
          batch_assignments = np.ones((n_datapoints))*-9999
          for block_num, block_indices in enumerate(all_block_indices):
             batch_assignments[block_indices] = block_num
          visualization_utils.plot_observations_world_map(all_coords[:, 0].detach().cpu().numpy(),
                                                          all_coords[:, 1].detach().cpu().numpy(),
                                                          batch_assignments,
                                                          plot_dir, "spatial_batches", us_only=True)


    def __iter__(self):
        batches = np.shuffle(self.batches)
        # for j in range(self.n_batches):
        #     # # OLD: First randomly choose a region (region)
        #     # if j < len(self.region_list):
        #     #     batch_region = self.region_list[j]
        #     # else:
        #     #     batch_region = random.choice(self.region_list)

        #     # Randomly choose "regions_per_batch" regions
        #     batch_regions = np.random.choice(self.region_list, size=self.regions_per_batch, replace=False)

        #     # Select indices from the selected regions, without replacement
        #     batch_indices = np.array([], dtype=int)
        #     for batch_region in batch_regions:
        #         n_indices = min(self.batch_size // self.regions_per_batch, len(self.regions[batch_region]))
        #         batch_indices = np.concatenate((batch_indices, np.random.choice(self.regions[batch_region], n_indices, replace=False)))

        #     # Edge case: if the batch is not "full" yet, choose another region and
        #     # draw randomly from it. If it's still not enough, keep drawing more regions
        #     remaining_regions = [c for c in self.region_list if (c not in batch_regions)]
        #     while len(batch_indices) < self.batch_size:
        #         new_region = random.choice(remaining_regions)
        #         n_indices = min(self.batch_size-len(batch_indices), len(self.regions[new_region]))
        #         batch_indices = np.concatenate((batch_indices, np.random.choice(self.regions[new_region], n_indices, replace=False)))
        #         remaining_regions = [c for c in remaining_regions if c != new_region]

        #     # Store the list of indices for this batch
        #     batches.append(batch_indices)
        return iter(batches)

    def __len__(self):
        return self.n_batches
