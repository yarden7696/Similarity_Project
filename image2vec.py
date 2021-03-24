import os
import numpy as np
from img2vec_pytorch import Img2Vec
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
# Initialize Img2Vec with GPU
img2vec = Img2Vec()
import glob
import pandas as pd
from numpy import dot
from numpy.linalg import norm

vector_list= []
cities = []
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
for i in range(len(vector_list)):
    for j in range(len(vector_list)):
       # cos_sim = np.inner(vector_list[i],vector_list[j]) / (norm(vector_list[i]) * norm(vector_list[i]))
        #print("cosine between",i,j,"is:",cos_sim)
        cos_sim= cosine_similarity(vector_list[i].reshape((1, -1)), vector_list[j].reshape((1, -1)))[0][0]
        print("similarity betweeen ",cities[i],"and",cities[j], "is:", cos_sim)
        #sims[i].append(cos_sim)
# plot sims/ set results
