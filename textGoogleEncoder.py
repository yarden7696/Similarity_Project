import tensorflow as tf
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

# load universal sentence encoder module
def load_USE_encoder(module):
    with tf.Graph().as_default():
        sentences = tf.compat.v1.placeholder(tf.string)
        embed = hub.Module(module)
        embeddings = embed(sentences)
        session = tf.compat.v1.train.MonitoredSession()
    return lambda x: session.run(embeddings, {sentences: x})


# load the encoder module
def text_similarity_result():
    encoder = load_USE_encoder('./USE')
    df = pd.read_excel('first_300_des_res.xlsx', engine='openpyxl')
    docLabels = df['city'].tolist()
    messages = df['description'].tolist()[:12]

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

    # normalize
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
            index = index + 1

    # convert similarity matrix into dataframe
    similarity_heatmap = similarity_heatmap_data.pivot("city1", "city2", "similarity")

    # visualize the results
    ax = sns.heatmap(similarity_heatmap, cmap="YlGnBu", vmin=0, vmax=1, annot=True, annot_kws={'size': 8})
    plt.title("Text Similarity")
    for label1 in ax.get_yticklabels():
        label1.set_weight('bold')
    for label2 in ax.get_xticklabels():
        label2.set_weight('bold')
    plt.show()
    return similarity_heatmap
