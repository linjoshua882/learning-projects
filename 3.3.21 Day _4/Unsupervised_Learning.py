from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.cluster.hierarchy import fcluster
import matplotlib.pyplot as plt
import pandas as pd


model = KMeans(n_clusters=3)
model.fit(points)

labels = model.predict(new_points)

print(labels)

"""
"""

xs = new_points[:,0]
ys = new_points[:,1]

plt.scatter(xs, ys, c=labels, alpha=0.5)

centroids = model.cluster_centers_

centroids_x = centroids[:,0]
centroids_y = centroids[:,1]

plt.scatter(centroids_x, centroids_y, marker='D', s=50)
plt.show()

"""
"""

ks = range(1, 6)
inertias = []

for k in ks:

    model = KMeans(n_clusters=k)
    model.fit(samples)
    
    inertias.append(model.inertia_)

plt.plot(ks, inertias, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()

"""
"""

model = KMeans(n_clusters=3)

labels = model.fit_predict(samples)

df = pd.DataFrame({'labels': labels, 'varieties': varieties})

ct = pd.crosstab(df['labels'], df['varieties'])

print(ct)

"""
"""

scaler = StandardScaler()
kmeans = KMeans(n_clusters = 4)
pipeline = make_pipeline(scaler, kmeans)

"""
"""

pipeline.fit(samples)
labels = pipeline.predict(samples)

df = pd.DataFrame({'labels': labels, 'species': species})

ct = pd.crosstab(df['labels'], df['species'])

print(ct)

"""
"""

normalizer = Normalizer()

kmeans = KMeans(n_clusters = 10)

pipeline = make_pipeline(normalizer, kmeans)
pipeline.fit(movements)

"""
"""

labels = pipeline.predict(movements)

df = pd.DataFrame({'labels': labels, 'companies': companies})

print(df.sort_values('labels'))

"""
"""

mergings = linkage(samples, method='complete')

dendrogram(mergings,
           labels=varieties,
           leaf_rotation=90,
           leaf_font_size=6,
)
plt.show()

"""
"""

normalized_movements = normalize(movements)

mergings = linkage(normalized_movements, 'complete')

dendrogram(mergings,
        labels=companies,
        leaf_rotation=90,
        leaf_font_size=6,
)
plt.show()

"""
"""

mergings = linkage(samples, method='single')

dendrogram(mergings,
            labels=country_names,
            leaf_rotation=90,
            leaf_font_size=6,
)
plt.show()

"""
"""

labels = fcluster(mergings, 6, criterion='distance')

df = pd.DataFrame({'labels': labels, 'varieties': varieties})

ct = pd.crosstab(df['labels'], df['varieties'])

print(ct)

"""
"""

model = TSNE(learning_rate = 200)

tsne_features = TSNE.fit_transform(model, samples)

xs = tsne_features[:,0]

ys = tsne_features[:,1]

plt.scatter(xs, ys, c=variety_numbers)
plt.show()

"""
"""

model = TSNE(learning_rate = 50)

tsne_features = TSNE.fit_transform(model, normalized_movements)

xs = tsne_features[:,0]
ys = tsne_features[:,1]

plt.scatter(xs, ys, alpha=0.5)

for x, y, company in zip(xs, ys, companies):
    plt.annotate(company, (x, y), fontsize=5, alpha=0.75)
plt.show()

"""
"""

width = grains[:,0]
length = grains[:,1]

plt.scatter(width, length)
plt.axis('equal')
plt.show()

correlation, pvalue = pearsonr(width, length)

print(correlation)

"""
"""

model = PCA()

pca_features = model.fit_transform(grains)

xs = pca_features[:,0]
ys = pca_features[:,1]

plt.scatter(xs, ys)
plt.axis('equal')
plt.show()

correlation, pvalue = pearsonr(xs, ys)

print(correlation)

"""
"""

plt.scatter(grains[:,0], grains[:,1])

model = PCA()
model.fit(grains)
mean = model.mean_

first_pc = model.components_[0,:]
plt.arrow(mean[0], mean[1], first_pc[0], first_pc[1], color='red', width=0.01)

plt.axis('equal')
plt.show()

"""
"""

scaler = StandardScaler()

pca = PCA()

pipeline = make_pipeline(scaler, pca)

pipeline.fit(samples)

features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_)
plt.xlabel('PCA feature')
plt.ylabel('variance')
plt.xticks(features)
plt.show()

"""
"""

pca = PCA(n_components=2)
pca.fit(scaled_samples)
pca_features = pca.transform(scaled_samples)

print(pca_features.shape)

"""
"""

tfidf = TfidfVectorizer() 
csr_mat = tfidf.fit_transform(documents)

print(csr_mat.toarray())

words = tfidf.get_feature_names()

print(words)

"""
"""

svd = TruncatedSVD(n_components=50)
kmeans = KMeans(n_clusters=6)
pipeline = make_pipeline(svd, kmeans)

"""
"""

pipeline.fit(articles)
labels = pipeline.predict(articles)

df = pd.DataFrame({'label': labels, 'article': titles})

print(df.sort_values('label'))

"""
"""

model = NMF(n_components = 6)
model.fit(articles)

nmf_features = model.transform(articles)

print(nmf_features.round(2))

"""
"""

df = pd.DataFrame(nmf_features, index=titles)

print(df.loc['Anne Hathaway'])
print(df.loc['Denzel Washington'])

"""
"""

