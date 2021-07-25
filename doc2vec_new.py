import gensim
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale

df = pd.read_excel('first_300_des_res.xlsx', engine='openpyxl')
docLabels = df['city'].tolist()[:12]
data = df['description'].tolist()[:12]


class LabeledLineSentence(object):
    def __init__(self, doc_list, labels_list):
        self.labels_list = labels_list
        self.doc_list = doc_list

    def __iter__(self):
        for idx, doc in enumerate(self.doc_list):
            yield gensim.models.doc2vec.TaggedDocument(doc,
                                                       [self.labels_list[idx]])


# iterator returned over all documents
it = LabeledLineSentence(data, docLabels)
model = gensim.models.Doc2Vec(vector_size=12, min_count=0, alpha=0.025, min_alpha=0.025)
model.build_vocab(it)
# training of model
for epoch in range(100):
    print("iteration" + str(epoch + 1))
    model.train(it, total_examples=model.corpus_count, epochs=model.epochs)
    model.alpha -= 0.002
    model.min_alpha = model.alpha

# saving the created model
model.save('doc2vec.model')
print("model saved")

# loading the model
d2v_model = gensim.models.doc2vec.Doc2Vec.load("doc2vec.model")

# heatmap
similarity_heatmap_data = pd.DataFrame()
res = []
for i in range(len(docLabels)):
    similar_doc = d2v_model.docvecs.most_similar(docLabels[i], topn=12)
    similar_doc = dict(similar_doc)
    for j in range(len(docLabels)):
        if i == j:
            similarity = 1
        else:
            similarity = similar_doc[docLabels[j]]
        res.append(similarity)

# normalize
index = 0
normalized_res = minmax_scale(res)
for i in range(len(docLabels)):
    for j in range(len(docLabels)):
        similarity_heatmap_data = similarity_heatmap_data.append(
            {
                'similarity': normalized_res[index],
                'city1': docLabels[i],
                'city2': docLabels[j]
            },
            ignore_index=True
        )
        index = index + 1

# plot sims/ set results
similarity_heatmap = similarity_heatmap_data.pivot(index="city1", columns="city2", values="similarity")
ax = sns.heatmap(similarity_heatmap, cmap="YlGnBu", vmin=0, vmax=1, annot=True, annot_kws={'size': 8})
plt.title("text Similarity - Doc2vec")
for label1 in ax.get_yticklabels():
    label1.set_weight('bold')
for label2 in ax.get_xticklabels():
    label2.set_weight('bold')
plt.show()
