import tensorflow as tf
#tf.compat.v1.placeholder()
#tf.compat.v1.disable_v2_behavior()
import tensorflow_hub as hub
from sklearn.preprocessing import minmax_scale
'''
Versions of packages 
tensorflow==2.0.0
tensorflow-estimator==2.0.1
tensorflow-hub==0.7.0
'''

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
'''
Versions of packages
numpy==1.17.2
seaborn==0.9.0
matplotlib==3.1.1
pandas==0.25.1
'''
# Function to sort hte list by second item of tuple
def Sort_Tuple(tup):
    # reverse = None (Sorts in Ascending order)
    # key is set to sort using second element of
    # sublist lambda has been used
    tup.sort(key=lambda x: (-x[1],x[0]))
    return tup

# load universal sentence encoder module
def load_USE_encoder(module):
    with tf.Graph().as_default():
        sentences = tf.compat.v1.placeholder(tf.string)
        embed = hub.Module(module)
        embeddings = embed(sentences)
        session = tf.compat.v1.train.MonitoredSession()
    return lambda x: session.run(embeddings, {sentences: x})

# load the encoder module
encoder = load_USE_encoder('./USE')
df = pd.read_excel('first_300_des_res.xlsx',engine='openpyxl')
docLabels = df['city'].tolist()
messages= df['description'].tolist()[:12]

# encode the messages
encoded_messages = encoder(messages)
# print("encoded",encoded_messages)

# cosine similarities
num_messages = len(messages)
similarities_df = pd.DataFrame()
similarity_heatmap_data = pd.DataFrame()
res = []
for i in range(num_messages):
    for j in range(num_messages): 
        # cos(theta) = x * y / (mag_x * mag_y)
        dot_product = np.dot(encoded_messages[i], encoded_messages[j])
        mag_i = np.sqrt(np.dot(encoded_messages[i], encoded_messages[i]))
        mag_j = np.sqrt(np.dot(encoded_messages[j], encoded_messages[j]))

        cos_theta = dot_product / (mag_i * mag_j)
        res.append(cos_theta)
        similarities_df = similarities_df.append(
            {
                'similarity': cos_theta,
                'message1': messages[i],
                'message2': messages[j]
            },
            ignore_index=True
        )
index = 0
normalized_res = minmax_scale(res)
for i in range(num_messages):
    for j in range(num_messages):
        # for heatmap
        similarity_heatmap_data = similarity_heatmap_data.append(
            {
                'similarity': normalized_res[index],
                'city1': docLabels[i],
                'city2': docLabels[j]
            },
            ignore_index=True
        )
        index = index+1

# convert similarity matrix into dataframe
similarity_heatmap = similarity_heatmap_data.pivot("city1", "city2", "similarity")

# filter data frame get 10 biggest similarity for each city
# similarity heatmap = dataframe 12*12

for i in range(len(messages)):
    result = []
    for j in range(len(messages)):
        if i != j :
            result.append((j, similarity_heatmap.iat[i, j]))
    # sort the list
    sort_list = Sort_Tuple(result);
    ans = ""
    # 10 most similar cities
    for a_tuple in sort_list[:10]:
        index = a_tuple[0]
        ans += " (" + str(docLabels[index]) +", " + str(a_tuple[1]) +") "
    df.loc[i, '10 most similarities'] = ans
#df.to_excel('10 most similar cities.xlsx')
df.to_excel('10mostSimilarCities.xlsx')

# visualize the results
ax = sns.heatmap(similarity_heatmap, cmap="YlGnBu", vmin=0, vmax=1, annot=True, annot_kws={'size': 8})
plt.title("Text Similarity")
for label1 in ax.get_yticklabels():
    label1.set_weight('bold')
for label2 in ax.get_xticklabels():
    label2.set_weight('bold')
plt.show()