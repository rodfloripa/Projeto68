
# 1. Descoberta de Papéis Estruturais em Grafos com Zachary Karate Club

<div align="justify">

Este projeto explora um dos conceitos mais interessantes de Graph Machine Learning:

a diferença entre detecção de comunidades e descoberta de papéis estruturais.

Normalmente, algoritmos de grafos tentam encontrar grupos densamente conectados, conhecidos como comunidades. Porém, em muitos sistemas reais, nós podem exercer funções semelhantes mesmo estando em regiões completamente diferentes da rede.

Por exemplo:

- líderes
- hubs
- pontes
- nós periféricos

Todos esses representam papéis estruturais.

Neste projeto utilizamos o clássico grafo Zachary Karate Club para comparar:

- Community Detection
- Role Discovery

Utilizando:

- Node2Vec
- Clustering
- Features Estruturais
- RolX simplificado
- PCA

</div>

---

# 2. O Dataset Zachary Karate Club

<div align="justify">

O Zachary Karate Club é um dos datasets mais clássicos da teoria de grafos.

Ele representa relações sociais entre membros de um clube de karatê estudado por Wayne Zachary nos anos 1970.

A rede eventualmente se dividiu em dois grupos principais devido a conflitos internos, tornando esse dataset extremamente importante para:

- community detection
- graph embeddings
- graph clustering
- role discovery

</div>

---

# 3. Importação das Bibliotecas

```python
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from node2vec import Node2Vec
from networkx.algorithms.community import greedy_modularity_communities
```

<div align="justify">

As bibliotecas utilizadas possuem funções específicas:

- `networkx` → manipulação de grafos
- `node2vec` → embeddings estruturais
- `KMeans` → clusterização
- `PCA` → redução de dimensionalidade
- `matplotlib` → visualização

</div>

---

# 4. Carregamento do Grafo

```python
G = nx.karate_club_graph()
```

<div align="justify">

O NetworkX já fornece o Zachary Karate Club pronto.

Cada nó representa um membro do clube.

Cada aresta representa interação social entre membros.

</div>

---

# 5. Detecção de Comunidades

```python
communities = list(
    greedy_modularity_communities(G)
)

community_labels = {}

for idx, community in enumerate(communities):

    for node in community:

        community_labels[node] = idx
```

<div align="justify">

Aqui utilizamos modularidade gulosa para detectar comunidades.

O algoritmo tenta maximizar conexões internas e minimizar conexões externas.

O objetivo é responder:

“quais nós pertencem ao mesmo grupo?”

Mas isso ainda não identifica funções estruturais semelhantes.

</div>

---

# 6. Geração de Embeddings com Node2Vec

```python
node2vec = Node2Vec(
    G,
    dimensions=16,
    walk_length=20,
    num_walks=100,
    workers=1,
    seed=42
)

model = node2vec.fit(
    window=5,
    min_count=1
)
```

<div align="justify">

O Node2Vec transforma nós em vetores densos.

Ele realiza random walks no grafo e aprende representações vetoriais similares ao Word2Vec.

Nós estruturalmente semelhantes tendem a ficar próximos no espaço vetorial.

Isso permite aplicar algoritmos tradicionais de Machine Learning sobre grafos.

</div>

---

# 7. Extração dos Embeddings

```python
embeddings = []

for node in nodes:

    embeddings.append(
        model.wv[str(node)]
    )

embeddings = np.array(embeddings)
```

<div align="justify">

Cada nó passa a possuir um embedding numérico.

Esses embeddings representam padrões topológicos aprendidos pelo Node2Vec.

Agora o grafo pode ser analisado em espaço vetorial.

</div>

---

# 8. Role Discovery com Node2Vec

```python
kmeans_embed = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

role_labels_node2vec = (
    kmeans_embed.fit_predict(embeddings)
)
```

<div align="justify">

Após gerar embeddings, utilizamos KMeans para encontrar agrupamentos estruturais.

Esses clusters representam possíveis papéis estruturais:

- hubs
- intermediários
- periféricos

Mesmo que os nós estejam em comunidades diferentes.

</div>

---

# 9. RolX Simplificado

```python
betweenness = nx.betweenness_centrality(G)

closeness = nx.closeness_centrality(G)

pagerank = nx.pagerank(G)

features.append([
    degree,
    clustering,
    betweenness[node],
    closeness[node],
    pagerank[node]
])
```

<div align="justify">

RolX é uma técnica clássica para role discovery.

A ideia principal é:

extrair características estruturais dos nós e clusterizar essas assinaturas.

Neste projeto utilizamos:

- Degree
- Clustering Coefficient
- Betweenness
- Closeness
- PageRank

Essas métricas descrevem o comportamento estrutural de cada nó.

</div>

---

# 10. Clusterização Estrutural

```python
kmeans_roles = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

role_labels_structural = (
    kmeans_roles.fit_predict(features_scaled)
)
```

<div align="justify">

Agora os papéis são descobertos diretamente pelas métricas topológicas.

Nós com comportamentos semelhantes acabam pertencendo ao mesmo role.

Isso é diferente de comunidades.

Comunidades analisam proximidade.

Papéis analisam função estrutural.

</div>

---

# 11. Redução de Dimensionalidade

```python
pca = PCA(n_components=2)

embeddings_2d = pca.fit_transform(
    embeddings
)
```

<div align="justify">

Os embeddings possuem múltiplas dimensões.

O PCA reduz os vetores para 2 dimensões, permitindo visualização gráfica.

Assim conseguimos visualizar:

- comunidades
- papéis estruturais
- separações topológicas

</div>

---

# 12. Visualização dos Embeddings

```python
axes[0].scatter(
    embeddings_2d[:, 0],
    embeddings_2d[:, 1],
    c=[community_labels[n] for n in nodes],
    s=250
)
```

<div align="justify">

Os gráficos mostram como os nós são agrupados segundo critérios diferentes.

Primeiro:

- agrupamento por comunidades

Depois:

- agrupamento por papéis

Isso evidencia que:

nós estruturalmente semelhantes podem não estar próximos no grafo.

</div>

---

# 13. Visualização do Grafo

```python
nx.draw_networkx_nodes(
    G,
    pos,
    node_color=role_labels_structural,
    node_size=700,
    cmap=plt.cm.Set3
)
```

<div align="justify">

O grafo é desenhado colorindo os nós segundo seus papéis estruturais.

Isso permite identificar visualmente:

- hubs
- regiões periféricas
- nós de intermediação

</div>

---

# 14. Comparação Comunidades vs Papéis

```python
matrix[community, role] += 1
```

<div align="justify">

A matriz final compara:

- comunidade
- papel estrutural

Essa análise mostra que:

um mesmo papel pode existir em várias comunidades.

Isso comprova que:

community detection e role discovery resolvem problemas diferentes.

</div>

---

# 15. Resultados Obtidos

<div align="justify">

O projeto normalmente evidencia padrões interessantes:

- hubs agrupados no mesmo role
- nós periféricos agrupados
- comunidades diferentes compartilhando funções semelhantes

Isso demonstra que papéis estruturais capturam padrões globais da topologia.

</div>

---

# 16. Possíveis Extensões

<div align="justify">

Este projeto pode evoluir para temas extremamente avançados:

- Struc2Vec
- GraphSAGE
- Temporal Role Discovery
- Dynamic Graph Embeddings
- Fraud Detection
- Social Network Analysis
- Cybersecurity
- Recommendation Systems

Também é possível substituir:

- PCA por UMAP
- KMeans por GMM
- Node2Vec por DeepWalk
- RolX por métodos hierárquicos

</div>

---

# 17. Conclusão

<div align="justify">

Este projeto demonstra uma diferença fundamental na análise de grafos:

comunidades representam proximidade local.

Papéis estruturais representam função topológica.

Dois nós podem exercer funções semelhantes mesmo estando em regiões completamente diferentes da rede.

Essa distinção é extremamente importante em aplicações modernas como:

- redes sociais
- sistemas financeiros
- telecomunicações
- cibersegurança
- detecção de fraude

A principal contribuição deste projeto é mostrar visualmente e quantitativamente que análise estrutural vai muito além de simples clusterização por comunidades.

</div>
