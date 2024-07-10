from flask import Flask, render_template, request, session
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)
app.secret_key = 'supersecretkey'

df = pd.read_csv('data_diabetes.csv')

def divisive_clustering(data, distance_matrix, max_clusters):
    clusters = [data]
    while len(clusters) < max_clusters:
        largest_cluster_idx = np.argmax([len(cluster) for cluster in clusters])
        largest_cluster = clusters.pop(largest_cluster_idx)

        if len(largest_cluster) <= 1:
            clusters.append(largest_cluster)
            break

        sub_clusters = split_cluster(largest_cluster, distance_matrix)
        clusters.extend(sub_clusters)

    return clusters

def split_cluster(cluster, distance_matrix):
    n = len(cluster)
    if n == 1:
        return [cluster]

    center1 = cluster[0]
    center2 = cluster[1]

    cluster1 = []
    cluster2 = []

    for point in cluster:
        if np.linalg.norm(point - center1) < np.linalg.norm(point - center2):
            cluster1.append(point)
        else:
            cluster2.append(point)

    return [np.array(cluster1), np.array(cluster2)]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        max_clusters = int(request.form['max_clusters'])
        
        # Pisahkan kolom nama kota dan koordinat
        cities = df.iloc[:, 0]
        coordinates = df.iloc[:, -2:]  # Latitude and Longitude
        data = df.iloc[:, 1:-2].values  # Exclude City and Coordinates
        
        distance_matrix = euclidean_distances(data, data)
        clusters = divisive_clustering(data, distance_matrix, max_clusters)

        cluster_labels = np.zeros(len(df), dtype=int)
        for cluster_idx, cluster in enumerate(clusters):
            for point in cluster:
                idx = np.where(np.all(data == point, axis=1))[0][0]
                cluster_labels[idx] = cluster_idx

        df['Cluster'] = cluster_labels
        df['City'] = cities  # Tambahkan kembali kolom nama kota
        df['Latitude'] = coordinates['Latitude']
        df['Longitude'] = coordinates['Longitude']

        session['clusters'] = [cluster.tolist() for cluster in clusters]
        session['df'] = df.to_dict(orient='records')

        return render_template('results.html', tables=[df.to_html(classes='data')], titles=df.columns.values, max_clusters=max_clusters)

    return render_template('index.html')

@app.route('/subplot')
def subplot():
    clusters = [np.array(cluster) for cluster in session.get('clusters', [])]
    plt.figure()
    colors = ['r', 'g', 'b', 'y', 'c', 'm', 'k']
    for i, cluster in enumerate(clusters):
        plt.scatter(cluster[:, 0], cluster[:, 1], c=colors[i % len(colors)], label=f'Cluster {i+1}')
    plt.legend()
    plt.title('Divisive Clustering')
    plt.xlabel('Fitur 1')
    plt.ylabel('Fitur 2')

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    return render_template('subplot.html', plot_url=plot_url)

@app.route('/dendrogram')
def dendrogram_view():
    import pandas as pd
    from scipy.cluster.hierarchy import dendrogram, linkage
    import matplotlib.pyplot as plt
    import io
    import base64
    from flask import render_template, session
    
    # Ambil DataFrame dari sesi
    df = pd.DataFrame(session.get('df'))
    
    # Pastikan hanya kolom numerik yang diambil
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    
    # Lakukan klasterisasi hirarkis
    linked = linkage(numeric_df, 'ward')
    
    # Buat plot dendrogram
    plt.figure(figsize=(10, 7))
    dendrogram(linked, labels=df['City'].tolist(), orientation='top', distance_sort='descending', show_leaf_counts=True)
    plt.title('Dendrogram')
    
    # Simpan plot sebagai gambar dalam memori
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    
    # Render template dengan gambar yang dihasilkan
    return render_template('dendrogram.html', plot_url=plot_url)

@app.route('/map')
def map_view():
    clusters = session.get('clusters', [])
    df = pd.DataFrame(session.get('df'))
    return render_template('map.html', clusters=clusters, df=df.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)