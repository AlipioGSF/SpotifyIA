from flask import Flask, jsonify, request

app = Flask(__name__)

tasks = [
    {
        'id': 1,
        'title': 'Estudar Flask',
        'done': False
    },
    {
        'id': 2,
        'title': 'Construir uma API',
        'done': False
    }
]

@app.route('/tasks', methods=['GET'])
def get_tasks():
    return jsonify({'tasks': tasks})

@app.route('/tasks/<int:task_id>', methods=['GET'])
def get_task(task_id):
    task = next((task for task in tasks if task['id'] == task_id), None)
    if task is None:
        return jsonify({'error': 'Tarefa não encontrada'}), 404
    return jsonify({'task': task})

@app.route('/tasks', methods=['POST'])
def create_task():
    if not request.json or 'title' not in request.json:
        return jsonify({'error': 'A tarefa deve ter um título'}), 400
    task = {
        'id': len(tasks) + 1,
        'title': request.json['title'],
        'done': False
    }
    tasks.append(task)
    return jsonify({'task': task}), 201

if __name__ == '__main__':
    app.run(debug=True)

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

ax.legend()
plt.show()