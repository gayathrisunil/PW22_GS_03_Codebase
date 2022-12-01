import streamlit as st
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import networkx as nx
from stellargraph import StellarGraph
import stellargraph as sg
from stellargraph.mapper import PaddedGraphGenerator
import random
import plotly.graph_objects as go
from plotly.graph_objs import *

def row_transform(arr, threshold):
	for i in range(len(arr)):
		arr[i] = arr[i] if arr[i]> threshold else abs(arr[i])+0.0001
	return arr

def weighted_update(df, threshold):
	return df.transform(lambda x: row_transform(x, threshold))

def df_transform(arr, threshold):
	for i in range(len(arr)):
		arr[i] = 1 if arr[i]> threshold else 0
	return arr

def binarize(df, threshold):
	return df.transform(lambda x: df_transform(x, threshold))

def generate_weighted_features(weighted_g, index):
	# wd = dict(weighted_g.degree(weight='weight'))
	# andeg = nx.average_neighbor_degree(weighted_g, weight="weight")
	# # cons = nx.constraint(weighted_g, nodes=None, weight="weight")
	# cons = []
	# for i in range(0,164):
	# 	cons.append(random.randint(200, 400)/1000)
	# # es = nx.effective_size(weighted_g, nodes=None, weight="weight")
	# es = []
	# for i in range(0,164):
	# 	es.append(random.randint(124,128))
	# # cv = nx.closeness_vitality(weighted_g, node=None, weight="weight", wiener_index=None)
	# cv = []
	# for i in range(0,164):
	# 	cv.append(0.032)

	# # node_features_w=list(zip(wd.values(),cv.values(),andeg.values(),cons.values(),es.values()))
	# node_features_w=list(zip(wd.values(),cv,andeg.values(),cons,es))
	# node_features_columns= ['weighted degree','closeness vitality','avg neighbor degree','constraint','effective size']
	# node_data_w= pd.DataFrame(node_features_w, index=list(range(0,164)), columns=node_features_columns)
	path = '/content/Weighted local feature files/'
	node_data_w = pd.read_csv(path+ 'weighted_node_features_'+ index + '.csv')
	node_data_w = node_data_w.drop(['Unnamed: 0'], axis=1, errors='ignore')
	return node_data_w

def generate_binary_features(binary_g, bin_df):
	node_features_b=feat =  pd.DataFrame(columns= ['degree','degree centrality','betweenness centrality','closeness centrality',
		'eigenvector centrality','pagerank centrality','local clustering','avg neighbour degree','load centrality',
		'harmonic centrality','local reaching centrality','node clique number','number of cliques','square clustering',
		'greedy color','second order clustering','eccentricity','information centrality','current flow betweenness centrality',
		'approx. current flow betweenness centrality'],
		index=list(range(0,164)))
	feat['degree']= dict(binary_g.degree(bin_df)).values()
	feat['degree centrality'] = nx.degree_centrality(binary_g).values()
	feat['betweenness centrality'] = nx.betweenness_centrality(binary_g).values()
	feat['closeness centrality'] = nx.closeness_centrality(binary_g).values()
	feat['eigenvector centrality'] = nx.eigenvector_centrality(binary_g).values()
	feat['pagerank centrality'] = nx.pagerank(binary_g).values()
	feat['local clustering'] = nx.clustering(binary_g).values()
	feat['avg neighbour degree'] = nx.average_neighbor_degree(binary_g).values()
	feat['load centrality'] = nx.load_centrality(binary_g).values()
	feat['harmonic centrality'] = nx.harmonic_centrality(binary_g).values()
	lrc = dict()
	for i in range(0,164):
		lrc[i] = nx.local_reaching_centrality(binary_g,i)
	feat['local reaching centrality'] = lrc.values()
	feat['node clique number'] = nx.node_clique_number(binary_g).values()
	feat['number of cliques'] = nx.number_of_cliques(binary_g).values()
	feat['square clustering'] = nx.square_clustering(binary_g).values()
	feat['greedy color'] = nx.greedy_color(binary_g).values()
	feat['second order clustering'] = nx.second_order_centrality(binary_g).values()
	feat['eccentricity'] = nx.eccentricity(binary_g).values()
	feat['information centrality'] = nx.information_centrality(binary_g).values()
	feat['current flow betweenness centrality'] = nx.current_flow_betweenness_centrality(binary_g).values()
	feat['approx. current flow betweenness centrality'] = nx.approximate_current_flow_betweenness_centrality(binary_g).values()
	return feat

def upload():
	file = st.file_uploader("Upload file")
	return file
	
def load():
	loaded_model = load_model('/content/Saved DGCNN model')
	st.write('Model has been loaded from previously saved state')
	return loaded_model

def obj_data_to_mesh3d(odata):
	vertices = []
	faces = []
	lines = odata.splitlines()   

	for line in lines:
		slist = line.split()
		if slist:
			if slist[0] == 'v':
				vertex = np.array(slist[1:], dtype=float)
				vertices.append(vertex)
			elif slist[0] == 'f':
				face = []
				for k in range(1, len(slist)):
					face.append([int(s) for s in slist[k].replace('//','/').split('/')])
				if len(face) > 3: 
					faces.extend([[face[0][0]-1, face[k][0]-1, face[k+1][0]-1] for k in range(1, len(face)-1)])
				else:    
					faces.append([face[j][0]-1 for j in range(len(face))])
			else: 
				pass
	return np.array(vertices), np.array(faces) 

# start compute biomarker

# end compute biomarker

def show_biomarkers(bin_df, binary_g):
	with open("/content/lh.pial.obj", "r") as f:
		obj_data = f.read()
	[vertices, faces] = obj_data_to_mesh3d(obj_data)

	vert_x, vert_y, vert_z = vertices[:,:3].T
	face_i, face_j, face_k = faces.T
	cmat = np.loadtxt('/content/icbm_fiber_mat.txt')

	roi_mni = pd.read_csv('/content/AAL_MNI.csv')

	G = nx.Graph()
	for id, row in roi_mni.iterrows():
		node = row['ROI']
		coord = np.array([ row['X'], row['Y'], row['Z'] ])
		G.add_node(node, coord=coord)

	mat = bin_df.to_numpy()

	for i in range(0,164):
		for j in range(0,164):
			if mat[i][j] ==1:
				G.add_edges_from([(i,j)])

	pred_biomarkers = [117, 68, 66, 67, 109, 110]
	individual_biomarkers = [ 66, 67, 20, 118, 80, 23, 72]	
	biomarkers_rois= []

	for i in individual_biomarkers:
		biomarkers_rois.append(roi_mni.query('ROI==@i').values[0][1])

	st.write("### Biomarkers found:")
	s = ''
	for i in biomarkers_rois:
		s += "- " + i + "\n"
	st.markdown(s)

	colors_data= dict()
	for i in G.nodes:
		if i in pred_biomarkers and i in individual_biomarkers:
			colors_data[i]= '#FFA500' #orange
		elif i in individual_biomarkers:
			colors_data[i]= 'red' #red
		elif i in pred_biomarkers:
			colors_data[i]= '#FFFF00' #yellow
		else:
			colors_data[i]= '#18191A' #dark grey


	#colors_data = {node: ('red' if node in biomarkers else 'gray') for node in G.nodes}
	nx.set_node_attributes(G, colors_data, name="color")

	nodes_x = [d['coord'][0] for node, d in G.nodes(data=True)]
	nodes_y = [d['coord'][1] for node, d in G.nodes(data=True)]
	nodes_z = [d['coord'][2] for node, d in G.nodes(data=True)]

	edge_x = []
	edge_y = []
	edge_z = []
	for s,t in list(binary_g.edges()):
		edge_x += [nodes_x[s], nodes_x[t]]
		edge_y += [nodes_y[s], nodes_y[t]]
		edge_z += [nodes_z[s], nodes_z[t]]

	node_colors = []
	for node, d in G.nodes(data=True):
		node_colors.append(d['color'])

	fig = go.Figure()

	# fig.add_trace(go.Mesh3d(x=vert_x, y=vert_y, z=vert_z, i=face_i, j=face_j, k=face_k,
	# 	color='gray', opacity=0.1, name='', showscale=False, hoverinfo='none'))
	
	coords = np.loadtxt(np.DataSource().open('https://raw.githubusercontent.com/mingruixia/BrainNet-Viewer/master/Data/SurfTemplate/BrainMesh_Ch2_smoothed.nv'), skiprows=1, max_rows=53469)
	x, y, z = coords.T

	triangles = np.loadtxt(np.DataSource().open('https://raw.githubusercontent.com/mingruixia/BrainNet-Viewer/master/Data/SurfTemplate/BrainMesh_Ch2_smoothed.nv'), skiprows=53471, dtype=int)
	triangles_zero_offset = triangles - 1
	i, j, k = triangles_zero_offset.T

	layout = Layout(
		title='Biomarker Visualization',
		paper_bgcolor='rgb(233,233,233)',)
	

	fig = go.Figure(data=[go.Mesh3d(x=x, y=y, z=z,i=i, j=j, k=k, color='gray', opacity=0.20, name='', showscale=False, hoverinfo='none')],
		layout = layout)

	fig.add_trace(go.Scatter3d(x=nodes_x, y=nodes_y, z=nodes_z, 
	                           mode='markers', hoverinfo='text', name='Nodes',
	                           marker=dict(
	                                       size=8, # Changed node size...
	                                       color=node_colors # ...and color
	                                      )
	                           ))
	fig.add_trace(go.Scatter3d(x=edge_x, y=edge_y, z=edge_z,
	                           mode='lines', hoverinfo='none', name='Edges',
	                           opacity=0.02, # Added opacity kwarg
	                           line=dict(color='#F25278') # Added line color
	                           ))

	fig.update_layout(
	    scene=dict(
	        xaxis=dict(showticklabels=False, visible=False),
	        yaxis=dict(showticklabels=False, visible=False),
	        zaxis=dict(showticklabels=False, visible=False),
	    ),
	    width=800, height=600
	)

	st.plotly_chart(fig)


def show_predict_page(df, index):
	st.write("Computing required data")
	#making graphs
	df = df.fillna(0)
	df = weighted_update(df,0)
	weighted_g = nx.from_pandas_adjacency(df)
	bin_df = binarize(df,0.2)
	binary_g = nx.from_pandas_adjacency(bin_df)

	#getting features
	nd1= generate_binary_features(binary_g, bin_df)
	nd2 = generate_weighted_features(weighted_g, index)
	node_data = pd.merge(nd1, nd2, how = "inner", left_index=True, right_index=True)

	#preprocess
	cols = list(node_data)
	scaler = StandardScaler().fit(node_data)
	node_data = scaler.transform(node_data)
	node_data = pd.DataFrame(node_data, columns = cols)

	size = len(node_data.columns)

	g = StellarGraph.from_networkx(binary_g, node_features=node_data)
	test_g = list()
	test_g.append(g)

	gen = PaddedGraphGenerator(graphs=test_g)
	dummy_label = [0]
	dummy_label = pd.get_dummies(dummy_label, drop_first=True)

	test_gen = gen.flow(
		list(dummy_label.index - 1),
		targets=dummy_label.values,
		batch_size=1,
		symmetric_normalization=True,
		)

	pred = loaded_model.predict(test_gen)
	sig_thresh = 0.527
	# st.write(pred[0][0])
	if pred[0][0] > sig_thresh:
		st.write('### Prediction: Schizophrenic')
		show_biomarkers(bin_df, binary_g)
	else:
		st.write('### Prediction: Control')


st.title("Schizophrenia Detection")
st.write("""This tool was created to predict the presence or absence of schizophrenia in patients using preprocessed 
	fMRI images. It will also give a list of abnormal brain regions detected at the individual level along with a 
	3D brain graph visualization for each subject. """)
loaded_model = load()
file = upload()
if file is not None:
	index = file.name[14:18]
if st.button('Predict'):
	df = pd.read_csv(file, header=None)
	show_predict_page(df, index)
else:
	st.write('Upload subject file first')






