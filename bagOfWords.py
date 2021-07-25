from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import euclidean_distances
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale

df = pd.read_excel('first_300_des_res.xlsx', engine='openpyxl')
cities = df['city'].tolist()[:12]
data = df['description'].tolist()[:12]
similarity_heatmap_data = pd.DataFrame()

vectorizer = CountVectorizer()
features = vectorizer.fit_transform(data).todense()
print(vectorizer.vocabulary_)

res = []
for i in range(len(cities)):
    for j in range(len(cities)):
        if i == j:
            similarity = 30
        else:
            similarity = euclidean_distances(features[i], features[j])[0][0]
        res.append(similarity)


index = 0
normalized_res = minmax_scale(res)
for i in range(len(cities)):
    for j in range(len(cities)):
        similarity_heatmap_data = similarity_heatmap_data.append(
            {
                'similarity': normalized_res[index],
                'city1': cities[i],
                'city2': cities[j]
            },
            ignore_index=True
        )
        index = index + 1
# plot sims/ set results
similarity_heatmap = similarity_heatmap_data.pivot(index="city1", columns="city2", values="similarity")
ax = sns.heatmap(similarity_heatmap, cmap="YlGnBu", vmin=0, vmax=1, annot=True, annot_kws={'size': 8})
plt.title("text Similarity - bag of words")
for label1 in ax.get_yticklabels():
    label1.set_weight('bold')
for label2 in ax.get_xticklabels():
    label2.set_weight('bold')
plt.show()
