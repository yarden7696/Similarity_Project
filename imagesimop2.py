import os
import numpy as np
import seaborn as sns
from img2vec_pytorch import Img2Vec
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import minmax_scale
from PIL import Image
import statistics
# Initialize Img2Vec with GPU
img2vec = Img2Vec()
import glob
import matplotlib.pyplot as plt
NumPics=20
import pandas as pd
from numpy import dot
from numpy.linalg import norm

def image_option2_similarity():
    vector_list= []
    cities = []
    similarity_heatmap_data = pd.DataFrame()
    citiesfinal= []
    # go through all cities pictures directories
    '''change to directory of images in your computer'''
    for filepath in glob.iglob(r'C:\Users\shova\Desktop\finalProject\im\*', recursive=True):
        city=str(filepath)
        city= city[39:]
        cities.append(city)
        image_list = []
        # Go through all pictures in directory
        for f in os.listdir(filepath):
            img = Image.open(os.path.join(filepath, f)).convert('RGB')
            image_list.append(img)
        vectors = img2vec.get_vec(image_list)
        vector_list.append(vectors)
    # Make cosine similarity between vectors

    arr = np.zeros(400*len(cities)*len(cities))
    s = 0
    for i in range(len(cities)):
        for j in range(len(cities)):
            for k in range(NumPics):
                for l in range(NumPics):
                    cos_sim = cosine_similarity(vector_list[i][k].reshape((1, -1)), vector_list[j][l].reshape((1, -1)))[0][0]
                    arr[s] = cos_sim
                    s = s+1

    standartDev= np.std(arr)
    mean_l = sum(arr)/len(arr)
    print("standart dev:", standartDev)
    print("mean:", mean_l)
    res = []
    for i in range(len(cities)):
        for j in range(len(cities)):
            s = 0
            for k in range(NumPics):
                for l in range(NumPics):
                    cos_sim = cosine_similarity(vector_list[i][k].reshape((1, -1)), vector_list[j][l].reshape((1, -1)))[0][0]
                    if cos_sim >= mean_l+standartDev: s = s+1
            print("similarity between ", cities[i], "to", cities[j], "is:", float(s/400))
            res.append(float(s/400))

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
            index = index+1

    # plot sims/ set results
    similarity_heatmap = similarity_heatmap_data.pivot(index="city1", columns="city2", values="similarity")
    ax = sns.heatmap(similarity_heatmap, cmap="YlGnBu", vmin=0, vmax=1, annot=True, annot_kws={'size': 8})
    plt.title("Image Similarity - Option 2")
    for label1 in ax.get_yticklabels():
        label1.set_weight('bold')
    for label2 in ax.get_xticklabels():
        label2.set_weight('bold')
    plt.show()
    return  similarity_heatmap