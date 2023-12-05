import pandas as pd
import numpy as np
import joblib
import sklearn
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from sklearn.pipeline import make_pipeline
import json

df = pd.read_csv('spotifyAtt_2023.csv')
c_dados = df.to_json()

tipos_de_dados = df.dtypes
print(tipos_de_dados)

print(sklearn.__version__)
# Selecione as colunas relevantes para o agrupamento
selected_columns = ['streams', 'danceability_%', 'energy_%']

# Crie uma matriz de características
X = df[selected_columns]

# Normalização dos dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Escolha o número de clusters (k)
k = 3

# Crie o modelo K-Means
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)

# Treine o modelo
kmeans.fit(X_scaled)

# Adicione rótulos aos dados originais
df['cluster_label'] = kmeans.labels_

# Adicione os centroides ao DataFrame
df_centroids = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=selected_columns)
df_centroids['cluster_label'] = range(1, k+1)

# Visualize os resultados
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot dos dados
ax.scatter(df['streams'], df['danceability_%'], df['energy_%'], c=df['cluster_label'], cmap='viridis', label='Dados')

# Scatter plot dos centroides
ax.scatter(df_centroids['streams'], df_centroids['danceability_%'], df_centroids['energy_%'], c='red', marker='X', s=200, label='Centroides')

ax.set_title('K-Means Clustering com Centroides')
ax.set_xlabel('Streams')
ax.set_ylabel('Danceability %')
ax.set_zlabel('Energy %')

# Ajuste da escala dos eixos para garantir precisão na visualização
ax.set_xlim(df['streams'].min(), df['streams'].max())
ax.set_ylim(df['danceability_%'].min(), df['danceability_%'].max())
ax.set_zlim(df['energy_%'].min(), df['energy_%'].max())

def recomended(track):
    curr_track = df[df['track_name'] == track]

    if not curr_track.empty:

        # Extraia os valores das características
        curr_track_features = curr_track[selected_columns].values

        # Normalização dos dados usando o mesmo scaler do K-Means
        curr_track_features_scaled = scaler.transform(curr_track_features)

        # Atribuição ao cluster usando o modelo K-Means
        user_cluster = kmeans.predict(curr_track_features_scaled)[0]

        # Recupere outras músicas do mesmo cluster
        suggested_playlist = df[df['cluster_label'] == user_cluster]

        # Calcule a distância euclidiana entre a música do usuário e as outras músicas no cluster
        distances = np.linalg.norm(suggested_playlist[selected_columns].values - curr_track_features_scaled, axis=1)

        # Ordene as músicas pelo valor da distância (quanto menor, mais parecida)
        suggested_playlist = suggested_playlist.copy()  # Crie uma cópia explícita
        suggested_playlist['distance'] = distances
        suggested_playlist = suggested_playlist.sort_values(by='distance').head(15)

        # Crie uma playlist sugerida
        suggested_playlist_names = suggested_playlist['track_name'].tolist()

        return json.dumps(suggested_playlist_names)
    else:
        return(f'Música "{track}" não encontrada no banco de dados.')
