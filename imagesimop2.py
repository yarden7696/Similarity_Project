import os
import numpy as np
import seaborn as sns
from img2vec_pytorch import Img2Vec
from sklearn.metrics.pairwise import cosine_similarity
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

arr = np.zeros(400*len(cities)*len(cities))
s=0
for i in range(len(cities)):
    for j in range(len(cities)):
        for k in range(NumPics):
            for l in range(NumPics):
                cos_sim = cosine_similarity(vector_list[i][k].reshape((1, -1)), vector_list[j][l].reshape((1, -1)))[0][0]
                arr[s]= cos_sim
                s=s+1

standartDev= np.std(arr)
mean_l = sum(arr)/len(arr)
print("standart dev:", standartDev)
print("mean:", mean_l)
for i in range(len(cities)):
    for j in range(len(cities)):
        s=0
        for k in range(NumPics):
            for l in range(NumPics):
                cos_sim = cosine_similarity(vector_list[i][k].reshape((1, -1)), vector_list[j][l].reshape((1, -1)))[0][0]
                if cos_sim>= mean_l+standartDev: s=s+1
        print("similarity between ", cities[i] ,"to", cities[j], "is:", float(s/400))
        similarity_heatmap_data = similarity_heatmap_data.append(
                        {
                            'similarity': float(s/400),
                            'city1': cities[i],
                            'city2': cities[j]
                        },
                        ignore_index=True
                    )



# for i in range(len(cities)):
#     for j in range(len(cities)):
#         for k in range(NumPics):
#             for l in range(NumPics):
#                         cos_sim = \
#                         cosine_similarity(vector_list[i][k].reshape((1, -1)), vector_list[j][l].reshape((1, -1)))[0][0]
#                         arr[s] = cos_sim
#                         s = s + 1
#                 print("arr is:", arr)
#                 # mean and standart deviation
#                 standartDev = np.std(arr)
#                 mean_l = sum(arr) / len(arr)
#                 print("standart dev:", standartDev)
#                 print("mean:", mean_l)
#                 ans = np.where(arr >= standartDev + mean_l, 1, 0)
#                 print("length anss", len(ans))
#                 print(ans)
#                 print("sim:", sum(ans))
        #print("similarity between", cities[i], "to", cities[j], "is:", similarity)
        # c= np.zeros(400)
        # flatList.clear()
        # similarity_heatmap_data = similarity_heatmap_data.append(
        #                 {
        #                     'similarity': similarity,
        #                     'city1': cities[i],
        #                     'city2': cities[j]
        #                 },
        #                 ignore_index=True
        #             )
        # for every city we have list of 20 lists that every list contain similarity between one pic to all other pic
       # city2city.append(pic2city.copy())
# plot sims/ set results
similarity_heatmap = similarity_heatmap_data.pivot(index="city1", columns="city2", values="similarity")
ax = sns.heatmap(similarity_heatmap, cmap="YlGnBu")
plt.show()










#city2city[i][k].max()  for i in range(len(city2city)):
# def normalize(lst):
#     s = sum(lst)
#     return map(lambda x: float(x)/s, lst)