import os
import numpy as np
import seaborn as sns
from img2vec_pytorch import Img2Vec
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
from sklearn.preprocessing import minmax_scale

# Initialize Img2Vec with GPU
img2vec = Img2Vec()
import glob
import matplotlib.pyplot as plt

NumPics = 20
import pandas as pd
from numpy import dot
from numpy.linalg import norm

def image_option3_similarity():
    vector_list = []
    cities = []
    similarity_heatmap_data = pd.DataFrame()
    citiesfinal = []
    # go through all cities pictures directories
    '''change to directory of images in your computer'''
    for filepath in glob.iglob(r'C:\Users\shova\Desktop\finalProject\im\*', recursive=True):
        city = str(filepath)
        city = city[39:]
        cities.append(city)
        image_list = []
        # Go through all pictures in directory
        for f in os.listdir(filepath):
            img = Image.open(os.path.join(filepath, f)).convert('RGB')
            image_list.append(img)
        vectors = img2vec.get_vec(image_list)
        vector_list.append(vectors)
    # Make cosine similarity between vectors
    pic2pics = []
    pic2city = []
    city2city = []
    for i in range(len(cities)):
        pic2city.clear()
        # for every pic in city i - make cosine similarity with all other pics from all cities
        for j in range(NumPics):
            pic2pics.clear()
            for k in range(len(cities)):
                if cities[i] != cities[k]:
                    for l in range(NumPics):
                        cos_sim = \
                        cosine_similarity(vector_list[i][j].reshape((1, -1)), vector_list[k][l].reshape((1, -1)))[0][0]
                        pic2pics.append((cities[k], l, cos_sim))
            # add similarity between pic j from city i to all other pics
            pic2city.append(pic2pics.copy())
        # city2city - list of cities -
        # for every city we have list of 20 lists that every list contain similarity between one pic to all other pic
        city2city.append(pic2city.copy())

    # option3 - for every x in  pictures of city A for every cities
    # choose to add 5 to city that her picture is the most similar to x
    max2pic = []
    max2city = []
    final_score = []
    for city, i in zip(city2city, range(len(cities))):
        max2pic.clear()
        temp_score = [0] * 12
        for pic, j in zip(city, range(20)):
            (city_name, pic_num, sim) = max(pic, key=lambda x: x[2])
            print(cities[i], "pic", j, "- the most similar pic is:", city_name, "pic", pic_num, "sim=", sim)
            temp_score[cities.index(city_name)] += 5
            max2pic.append((city_name, pic_num, sim))
        final_score.append(temp_score.copy())
        max2city.append(max2pic.copy())

    for i in range(len(cities)):
        score = final_score[i]
        index = 0
        normalized_res = minmax_scale(score)
        for j in range(len(cities)):
            if cities[i] == cities[j]:
                normalized_res[index] = 1
            similarity_heatmap_data = similarity_heatmap_data.append(
                {
                    'similarity': normalized_res[index],
                    'city1': cities[i],
                    'city2': cities[j]
                },
                ignore_index=True
            )
            index = index + 1

    similarity_heatmap = similarity_heatmap_data.pivot(index="city1", columns="city2", values="similarity")
    ax = sns.heatmap(similarity_heatmap, cmap="YlGnBu", vmin=0, vmax=1, annot=True, annot_kws={'size': 8})
    plt.title("Image Similarity - Option 3")
    for label1 in ax.get_yticklabels():
        label1.set_weight('bold')
    for label2 in ax.get_xticklabels():
        label2.set_weight('bold')
    plt.show()
    return similarity_heatmap
