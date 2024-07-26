# Ball-Mapper
## Analyzing Air Pollutants Using Ball Mapper Algorithm
### Import Lib
`import numpy as np
import seaborn as sns
import networkx as nx
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
%matplotlib inline
from pyballmapper import BallMapper
from pyballmapper.plotting import graph_GUI
from bokeh.plotting import figure, show
from scipy.spatial.distance import cosine
from matplotlib import colormaps as cm
import warnings
warnings.filterwarnings("ignore")`
### Ball Mapper 
`def Ballmapper(data, region_col, eps, metric='euclidean'):
    
    # Extracting numerical data and region information
    numerical_data = data.drop(columns=['Ticker', region_col])
    regions = data[region_col].values
    
    # Computing pairwise distances
    dist_matrix = squareform(pdist(numerical_data, metric))
    n_samples = dist_matrix.shape[0]
    
    # Initializing graph and landmark selection
    bm_graph = nx.Graph()
    landmarks = []
    points_covered = set()
    
    # Landmark selection and cluster formation
    while len(points_covered) < n_samples:
        for i in range(n_samples):
            if i not in points_covered:
                landmarks.append(i)
                points_covered.add(i)
                # Create a node for each landmark
                bm_graph.add_node(i, region=regions[i], points_covered=[i])
                break
           # Expanding clusters around landmarks
        for j in range(n_samples):
            if dist_matrix[landmarks[-1], j] <= eps:
                points_covered.add(j)
                bm_graph.nodes[landmarks[-1]]['points_covered'].append(j)
    
    # Connecting clusters with overlapping points
    for i in range(len(landmarks)):
        for j in range(i + 1, len(landmarks)):
            if set(bm_graph.nodes[landmarks[i]]['points_covered']) & set(bm_graph.nodes[landmarks[j]]['points_covered']):
                bm_graph.add_edge(landmarks[i], landmarks[j])
    
    return bm_graph, regions`
`data = pd.read_csv('PM2.5_BM1.csv')`
`data.head()`
![M1](https://github.com/user-attachments/assets/84a325da-a3f4-4fe2-bcae-4e811b4fceca)
## Visualization of PM2.5
`import numpy as np
from scipy.spatial.distance import pdist, squareform
import networkx as nx
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")`

`data = pd.read_csv('PM2.5_BM1.csv')`

`eps = 0.065`

#### Generating the Ball Mapper graph with specified parameters
`bm_graph, regions = Ballmapper(data, 'Region', eps, metric='cosine')`

#### Define unique regions and assign a base color map
`unique_regions = np.unique(regions)
color_map = plt.cm.get_cmap('hsv', len(unique_regions))`

#### Customization of colors for specific regions
`region_to_color = {region: color_map(i) for i, region in enumerate(unique_regions)}
region_to_color['Central'] = '#03c04a'
region_to_color['Southern'] = '#fcd12a'
region_to_color['Northern'] = '#006837'
region_to_color['Eastern'] = '#a6d96a'
region_to_color['Sabah'] = '#a50026'
region_to_color['Sarawak'] = '#f46d43'`
#### Determine node sizes based on the number of points in each cluster
`node_sizes = [len(bm_graph.nodes[i]['points_covered']) * 70 for i in bm_graph.nodes]  # Adjust size factor as needed`

#### Node colors based on the region-to-color mapping
`node_colors = [region_to_color[bm_graph.nodes[i]['region']] for i in bm_graph.nodes]`

#### Visualization
`plt.figure(figsize=(7, 4))
pos = nx.spring_layout(bm_graph, k=0.3, iterations=15, seed=30)
nx.draw_networkx_edges(bm_graph, pos, alpha=0.6)
nx.draw_networkx_nodes(bm_graph, pos, node_size=node_sizes, node_color=node_colors, alpha=0.8)
nx.draw_networkx_labels(bm_graph, pos, font_size=8, horizontalalignment='center')`

#### Legend to reflect colors
`legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=region) for region, color in region_to_color.items()]
plt.legend(handles=legend_handles, title="Regions", bbox_to_anchor=(1, 1))
#plt.title("PM2.5 Ball Mapper Graph with Cosine Metric, eps=0.065")
plt.axis('off')
plt.tight_layout()
plt.savefig('PM2.5_BM.png', dpi=300)
plt.show()`

![image](https://github.com/user-attachments/assets/e4669998-a76e-42d4-8b8a-e355e3d3584c)
## Save Node Details
`import pandas as pd`

`def node_details(G, data):`
#### Using list comprehension to gather node details directly
   ` node_details_list = [{
        'Node ID': node,
        'Region': G.nodes[node]['region'],
        'Points Covered': len(G.nodes[node]['points_covered']),
        'Stations': ", ".join(data.iloc[G.nodes[node]['points_covered']]['Station'].astype(str).tolist())
    } for node in G.nodes()]`
    
#### Converting the list of dictionaries to a DataFrame
  `  node_details_df = pd.DataFrame(node_details_list)
    
    return node_details_df`

#### function with the graph object and data, and store the result in a variable
`node_details_df = node_details(bm_graph, data)`

#### Now you can directly work with the DataFrame in Python
`print(node_details_df.head())  # For example, print the first few rows`

#### save the DataFrame to a CSV file
`node_details_df.to_csv('PM2.5_cosine_Nodes.csv', index=False)`
## Cardinality of Cycle Basis and Estrada Index
`import numpy as np
import networkx as nx`

#### bm_graph is Ball Mapper graph
`A = nx.adjacency_matrix(bm_graph).todense()
eigenvalues = np.linalg.eigvals(A)
estrada_index = np.sum(np.exp(eigenvalues))`

#### bm_graph is Ball Mapper graph
`cycles = nx.cycle_basis(bm_graph)
cardinality_of_cycle_basis = len(cycles)
print("Cycle Basis:", cardinality_of_cycle_basis, "Estrada Index:", estrada_index)
print()`
## BallMapper Graph Summary Statistics
`import pandas as pd`

####  bm_graph is already created and defined

#### 1. Number of Clusters
`num_clusters = bm_graph.number_of_nodes()`

#### 2. Cluster Sizes
`cluster_sizes = [len(bm_graph.nodes[node]['points_covered']) for node in bm_graph.nodes()]`

#### 3. Cluster Connectivity (Degree)
`cluster_degrees = [bm_graph.degree(node) for node in bm_graph.nodes()]`

#### Creating a DataFrame for clearer analysis
`cluster_data = pd.DataFrame({
    'Cluster Size': cluster_sizes,
    'Cluster Degree': cluster_degrees
})`

#### 4. Distribution of Cluster Sizes - Summary Statistics
`cluster_size_distribution = cluster_data['Cluster Size'].describe()`

#### Print summary statistics
`print(f"Number of Clusters: {num_clusters}")
print("Cluster Size Distribution:\n", cluster_size_distribution)
print("\nCluster Data (First 5 Clusters):\n", cluster_data.head())`

#### Save the detailed cluster data to a CSV file
`cluster_data.to_csv('cluster_data.csv', index=False)`

#### save Summary Statistics
`cluster_size_distribution.to_frame().to_csv('PM2.5_cluster_size_distributions.csv')`
![image](https://github.com/user-attachments/assets/c6da93c9-ab0e-401c-94b9-bf5abac7e32b)
## Density of PM2.5 Ballmapper graph
`import numpy as np
import networkx as nx
Density = nx.density(bm_graph)
print('Density of PM2.5 Ballmapper graph:', Density)`
## Fragmentation
`import networkx as nx`
`FI = sum([len(component) >= k for component in nx.connected_components(bm_graph)])
print("Number of components with size >= k:", FI)`
## Time Series of Stations Clustered in Node 20
`import pandas as pd
import matplotlib.pyplot as plt`

#### Load the data
`dfb = pd.read_csv('PM2.5_NODE20.csv')`

#### Generate a date range assuming sequential daily measurements from January 1, 2018
`date_range = pd.date_range(start='2018-01-01', periods=len(dfb), freq='D')
dfb['Date'] = date_range
dfb.set_index('Date', inplace=True)`

#### Calculate the 30-day rolling average
`rolling_dfb = dfb.rolling(window=30).mean()`

#### Plotting
`fig, ax = plt.subplots(figsize=(7, 4))
rolling_dfb.plot(ax=ax)
ax.set_xlabel('Date [2018-2020]')
ax.set_ylabel('PM2.5 Concentration [ppm]')
plt.tight_layout()
plt.savefig('PM2.5_BM_TS_NODE_20.png', dpi=400)`
![image](https://github.com/user-attachments/assets/d9f01ac7-6a56-4f33-b2ee-81cc1615a284)




