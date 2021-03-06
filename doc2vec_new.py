import gensim
import pandas as pd

df = pd.read_excel('first_300_des_res.xlsx', engine='openpyxl')
docLabels = df['city'].tolist()
data= df['description'].tolist()

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
model = gensim.models.Doc2Vec(vector_size=500, min_count=0, alpha=0.025, min_alpha=0.025)
model.build_vocab(it)
#training of model
for epoch in range(100):
 print("iteration"+str(epoch+1))
 model.train(it,total_examples=model.corpus_count,epochs=model.epochs)
 model.alpha -= 0.002
 model.min_alpha = model.alpha

#saving the created model
model.save('doc2vec.model')
print("model saved")

#loading the model
d2v_model = gensim.models.doc2vec.Doc2Vec.load("doc2vec.model")

# write the result to txt
f = open("10most_similarities.txt", "w+", errors="ignore")
for city in range(len(docLabels)):
    similar_doc = d2v_model.docvecs.most_similar(docLabels[city])
    # for x in similar_doc:
    sentence = ', '.join('{} {}'.format(x[0], x[1]) for x in similar_doc)
    out = docLabels[city] + "= " + sentence
    f.write(out + '\n')
f.close()