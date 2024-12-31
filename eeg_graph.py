import eegraph
G = eegraph.Graph()
G.load_data(path = 'eeg_sample_1.edf', exclude = ['EEG TAntI1-TAntI', 'EEG TAntD1-TAntD', 'EEG EKG1-EKG2'])

G = eegraph.Graph()
G.load_data(path = 'eeg_sample_1.edf', electrode_montage_path = 'electrodemontage.set.ced')

graphs, connectivity_matrix = G.modelate(window_size = 2, connectivity = 'pearson_correlation', threshold = 0.8)

G.visualize(graphs[0], 'graph_1')


