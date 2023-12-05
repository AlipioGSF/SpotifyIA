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

df = pd.read_csv('spotifyAtt_2023.csv')

tipos_de_dados = df.dtypes
print(tipos_de_dados)

print(sklearn.__version__)