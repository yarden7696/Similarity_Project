import os
import numpy as np
import seaborn as sns
from img2vec_pytorch import Img2Vec
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
# Initialize Img2Vec with GPU
img2vec = Img2Vec()
import glob
import matplotlib.pyplot as plt
NumPics=20
import pandas as pd
from numpy import dot
from numpy.linalg import norm

vector_list= []
cities = []
similarity_heatmap_data = pd.DataFrame()
citiesfinal= []
# go through all cities pictures directories
'''change to directory of images in your computer'''
for filepath in glob.iglob(r'C:\Users\user\Desktop\final_project\im\*', recursive=True):
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
pic2pics=[]
pic2city=[]
city2city=[]
total=[]
for i in range(len(cities)):
    for j in range(len(cities)):
        cos_sim = cosine_similarity(vector_list[i].reshape((1, -1)), vector_list[j].reshape((1, -1)))[0][0]
        # make heatmap of my option 1:
        print("similarity between ", cities[i] ,"to", cities[j], "is:", cos_sim)
        similarity_heatmap_data = similarity_heatmap_data.append(
                        {
                            'similarity': cos_sim,
                            'city1': cities[i],
                            'city2': cities[j]
                        },
                        ignore_index=True
                    )



# plot sims/ set results
similarity_heatmap = similarity_heatmap_data.pivot(index="city1", columns="city2", values="similarity")
ax = sns.heatmap(similarity_heatmap, cmap="YlGnBu")
plt.show()




#city2city[i][k].max()  for i in range(len(city2city)):
# def normalize(lst):
#     s = sum(lst)
#     return map(lambda x: float(x)/s, lst)