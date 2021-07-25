import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from flatbuffers.builder import np

import finalScore
import historySearch
import image2vec
import imagesimop2
import imagesimop3
import imgsimop1
import use

cities = ['Antalya', 'Bangkok', 'Beijing', 'Crete',
          'Delhi', 'Hong-Kong', 'Los-Angeles', 'Mumbai-Bombay',
          'New-York-City', 'Paris', 'Rhodes', 'Rome']

text_similarity = use.text_similarity_result()
AMT_result = finalScore.AMT_similarity()
image_option1 = imgsimop1.image_option1_similarity()
image_option2 = imagesimop2.image_option2_similarity()
image_option3 = imagesimop3.image_option3_similarity()
image_our_option = image2vec.image_our_option_similarity()
history_search = historySearch.history_search_similarity()
combination_mat = (text_similarity.copy() + image_option1.copy() + history_search.copy()) / 3
similarity_heatmap_data = pd.DataFrame()
for i in (range(len(cities))):
    for j in (range(len(cities))):
        similarity_heatmap_data = similarity_heatmap_data.append(
                {
                    'similarity': combination_mat.iloc[i,j],
                    'city1': cities[i],
                    'city2': cities[j]
                },
                ignore_index=True
        )
similarity_heatmap = similarity_heatmap_data.pivot(index="city1", columns="city2", values="similarity")
ax = sns.heatmap(similarity_heatmap, cmap="YlGnBu", vmin=0, vmax=1, annot=True, annot_kws={'size': 8})
plt.title("Combination Matrix Similarity ")
for label1 in ax.get_yticklabels():
    label1.set_weight('bold')
for label2 in ax.get_xticklabels():
    label2.set_weight('bold')
plt.show()

# ----------------- distance between matrices ----------------- #

text_dif = 0
image_option1_dif = 0
image_option2_dif = 0
image_option3_dif = 0
image_our_option_dif = 0
history_search_dif = 0
combination_mat_dif = 0

for i in cities:
    for j in cities:
        text_dif += abs(text_similarity.loc[i, j] - AMT_result.loc[i, j])
        image_option1_dif += abs(image_option1.loc[i, j] - AMT_result.loc[i, j])
        image_option2_dif += abs(image_option2.loc[i, j] - AMT_result.loc[i, j])
        image_option3_dif += abs(image_option3.loc[i, j] - AMT_result.loc[i, j])
        image_our_option_dif += abs(image_our_option.loc[i, j] - AMT_result.loc[i, j])
        history_search_dif += abs(history_search.loc[i, j] - AMT_result.loc[i, j])
        combination_mat_dif += abs(combination_mat.loc[i,j] - AMT_result.loc[i, j])

print("")
print("text =", text_dif)
print("image option 1 =", image_option1_dif)
print("image option 2 =", image_option2_dif)
print("image option 3 =", image_option3_dif)
print("image our option =", image_our_option_dif)
print("history search =", history_search_dif)
print("combination mat =", combination_mat_dif)

min = min(text_dif, image_option1_dif, image_option2_dif, image_option3_dif, image_our_option_dif, history_search_dif,combination_mat_dif)
res = [('text similarity',text_dif),('image option 1',image_option1_dif) ,('image option 2',image_option2_dif) ,
       ('image option 3',image_option3_dif) ,('image our option',image_our_option_dif) ,('history search',history_search_dif) ,('combination mat',combination_mat_dif)]

print("in distance between matrices: the must represent method is :",end= " " )
for i in res:
    if i[1] == min:
        print(i[0], ",", end= " ")

print("")
print("")

# # ----------------- bigger the median ----------------- #
print("Bigger than the median :")

print("Option 1 - The treshold is the median for each matrix")
text_mat = text_similarity.copy()
AMT_mat = AMT_result.copy()
img_op1_mat = image_option1.copy()
img_op2_mat = image_option2.copy()
img_op3_mat = image_option3.copy()
img_our_mat = image_our_option.copy()
history_search_mat = history_search.copy()
combination = combination_mat.copy()

text_dif = 0
image_option1_dif = 0
image_option2_dif = 0
image_option3_dif = 0
image_our_option_dif = 0
history_search_dif = 0
combination_mat_dif = 0

for i in cities:
    for j in cities:
        if text_similarity.loc[i, j] >= np.median(text_similarity):
            text_mat.loc[i,j] = 1
        else:
            text_mat.loc[i,j] = 0
        if image_option1.loc[i, j] >= np.median(image_option1):
           img_op1_mat.loc[i,j] = 1
        else:
            img_op1_mat.loc[i,j] = 0
        if image_option2.loc[i, j] >= np.median(image_option2):
            img_op2_mat.loc[i,j] = 1
        else:
            img_op2_mat.loc[i,j] = 0
        if image_option3.loc[i, j] >= np.median(image_option3):
            img_op3_mat.loc[i,j] = 1
        else:
            img_op3_mat.loc[i,j] = 0
        if image_our_option.loc[i, j] >= np.median(image_our_option):
            img_our_mat.loc[i,j] = 1
        else:
            img_our_mat.loc[i,j] = 0
        if history_search.loc[i, j] >= np.median(history_search):
            history_search_mat.loc[i,j] = 1
        else:
            history_search_mat.loc[i,j] = 0
        if AMT_result.loc[i,j] >= np.median(AMT_result):
            AMT_mat.loc[i,j] = 1
        else:
            AMT_mat.loc[i,j] = 0
        if combination_mat.loc[i, j] >= np.median(combination_mat):
            combination.loc[i, j] = 1
        else:
            combination.loc[i, j] = 0


for i in cities:
    for j in cities:
        if AMT_mat.loc[i,j] == text_mat.loc[i,j]:
            text_dif+= 1
        if AMT_mat.loc[i,j] == img_op1_mat.loc[i,j]:
            image_option1_dif+= 1
        if AMT_mat.loc[i,j] == img_op2_mat.loc[i,j]:
            image_option2_dif+= 1
        if AMT_mat.loc[i,j] == img_op3_mat.loc[i,j]:
            image_option3_dif+= 1
        if AMT_mat.loc[i,j] == img_our_mat.loc[i,j]:
            image_our_option_dif+= 1
        if AMT_mat.loc[i,j] == history_search_mat.loc[i,j]:
            history_search_dif+= 1
        if AMT_mat.loc[i, j] == combination.loc[i, j]:
            combination_mat_dif += 1


print("text =", text_dif)
print("image option 1 =", image_option1_dif)
print("image option 2 =", image_option2_dif)
print("image option 3 =", image_option3_dif)
print("image our option =", image_our_option_dif)
print("history search =", history_search_dif)
print("combination mat =", combination_mat_dif)


max1 = max(text_dif, image_option1_dif, image_option2_dif, image_option3_dif, image_our_option_dif, history_search_dif,combination_mat_dif)
res = [('text similarity',text_dif),('image option 1',image_option1_dif) ,('image option 2',image_option2_dif) ,
       ('image option 3',image_option3_dif),('image our option',image_our_option_dif) ,('history search',history_search_dif) ,('combination mat',combination_mat_dif)]


print("Best result in median method 1 is :", end= " " )
for i in res:
    if i[1] == max1:
        print(i[0],",", end= " ")
print("")
print("")




print("Option 2 - The treshold of all matrixes is the median = np.median(AMT_result)")
text_mat = text_similarity.copy()
AMT_mat = AMT_result.copy()
img_op1_mat = image_option1.copy()
img_op2_mat = image_option2.copy()
img_op3_mat = image_option3.copy()
img_our_mat = image_our_option.copy()
history_search_mat = history_search.copy()
combination = combination_mat.copy()

text_dif = 0
image_option1_dif = 0
image_option2_dif = 0
image_option3_dif = 0
image_our_option_dif = 0
history_search_dif = 0
combination_mat_dif = 0

for i in cities:
    for j in cities:
        if text_similarity.loc[i, j] >= np.median(AMT_result):
            text_mat.loc[i,j] = 1
        else:
            text_mat.loc[i,j] = 0
        if image_option1.loc[i, j] >= np.median(AMT_result):
           img_op1_mat.loc[i,j] = 1
        else:
            img_op1_mat.loc[i,j] = 0
        if image_option2.loc[i, j] >= np.median(AMT_result):
            img_op2_mat.loc[i,j] = 1
        else:
            img_op2_mat.loc[i,j] = 0
        if image_option3.loc[i, j] >= np.median(AMT_result):
            img_op3_mat.loc[i,j] = 1
        else:
            img_op3_mat.loc[i,j] = 0
        if image_our_option.loc[i, j] >= np.median(AMT_result):
            img_our_mat.loc[i,j] = 1
        else:
            img_our_mat.loc[i,j] = 0
        if history_search.loc[i, j] >= np.median(AMT_result):
            history_search_mat.loc[i,j] = 1
        else:
            history_search_mat.loc[i,j] = 0
        if AMT_result.loc[i,j] >= np.median(AMT_result):
            AMT_mat.loc[i,j] = 1
        else:
            AMT_mat.loc[i,j] = 0
        if combination_mat.loc[i, j] >= np.median(AMT_result):
            combination.loc[i, j] = 1
        else:
            combination.loc[i, j] = 0

for i in cities:
    for j in cities:
        if AMT_mat.loc[i,j] == text_mat.loc[i,j]:
            text_dif+= 1
        if AMT_mat.loc[i,j] == img_op1_mat.loc[i,j]:
            image_option1_dif+= 1
        if AMT_mat.loc[i,j] == img_op2_mat.loc[i,j]:
            image_option2_dif+= 1
        if AMT_mat.loc[i,j] == img_op3_mat.loc[i,j]:
            image_option3_dif+= 1
        if AMT_mat.loc[i,j] == img_our_mat.loc[i,j]:
            image_our_option_dif+= 1
        if AMT_mat.loc[i,j] == history_search_mat.loc[i,j]:
            history_search_dif+= 1

        if AMT_mat.loc[i, j] == combination.loc[i, j]:
            combination_mat_dif += 1


print("text =", text_dif)
print("image option 1 =", image_option1_dif)
print("image option 2 =", image_option2_dif)
print("image option 3 =", image_option3_dif)
print("image our option =", image_our_option_dif)
print("history search =", history_search_dif)
print("combination mat =", combination_mat_dif)

max2 = max(text_dif, image_option1_dif, image_option2_dif, image_option3_dif, image_our_option_dif, history_search_dif,combination_mat_dif)
res = [('text similarity',text_dif),('image option 1',image_option1_dif) ,('image option 2',image_option2_dif) ,
       ('image option 3',image_option3_dif) ,('image our option',image_our_option_dif) ,('history search',history_search_dif) ,('combination mat',combination_mat_dif)]

print("Best result in median method 2 is :",end= " ")
for i in res:
    if i[1] == max2:
        print(i[0],",", end= " ")
print("")
print("")


print("Option 3 - The treshold of all matrixes is the mean = np.mean(AMT_result)")
text_mat = text_similarity.copy()
AMT_mat = AMT_result.copy()
img_op1_mat = image_option1.copy()
img_op2_mat = image_option2.copy()
img_op3_mat = image_option3.copy()
img_our_mat = image_our_option.copy()
history_search_mat = history_search.copy()
combination = combination_mat.copy()

text_dif = 0
image_option1_dif = 0
image_option2_dif = 0
image_option3_dif = 0
image_our_option_dif = 0
history_search_dif = 0
combination_mat_dif = 0


for i in cities:
    for j in cities:
        if text_similarity.loc[i, j] >= np.average(AMT_result):
            text_mat.loc[i,j] = 1
        else:
            text_mat.loc[i,j] = 0
        if image_option1.loc[i, j] >= np.average(AMT_result):
           img_op1_mat.loc[i,j] = 1
        else:
            img_op1_mat.loc[i,j] = 0
        if image_option2.loc[i, j] >= np.average(AMT_result):
            img_op2_mat.loc[i,j] = 1
        else:
            img_op2_mat.loc[i,j] = 0
        if image_option3.loc[i, j] >= np.average(AMT_result):
            img_op3_mat.loc[i,j] = 1
        else:
            img_op3_mat.loc[i,j] = 0
        if image_our_option.loc[i, j] >= np.average(AMT_result):
            img_our_mat.loc[i,j] = 1
        else:
            img_our_mat.loc[i,j] = 0
        if history_search.loc[i, j] >= np.average(AMT_result):
            history_search_mat.loc[i,j] = 1
        else:
            history_search_mat.loc[i,j] = 0
        if AMT_result.loc[i,j] >= np.average(AMT_result):
            AMT_mat.loc[i,j] = 1
        else:
            AMT_mat.loc[i,j] = 0
        if combination_mat.loc[i, j] >= np.average(AMT_result):
            combination.loc[i, j] = 1
        else:
            combination.loc[i, j] = 0

for i in cities:
    for j in cities:
        if AMT_mat.loc[i,j] == text_mat.loc[i,j]:
            text_dif+= 1
        if AMT_mat.loc[i,j] == img_op1_mat.loc[i,j]:
            image_option1_dif+= 1
        if AMT_mat.loc[i,j] == img_op2_mat.loc[i,j]:
            image_option2_dif+= 1
        if AMT_mat.loc[i,j] == img_op3_mat.loc[i,j]:
            image_option3_dif+= 1
        if AMT_mat.loc[i,j] == img_our_mat.loc[i,j]:
            image_our_option_dif+= 1
        if AMT_mat.loc[i,j] == history_search_mat.loc[i,j]:
            history_search_dif+= 1
        if AMT_mat.loc[i, j] == combination.loc[i, j]:
            combination_mat_dif += 1


print("text =", text_dif)
print("image option 1 =", image_option1_dif)
print("image option 2 =", image_option2_dif)
print("image option 3 =", image_option3_dif)
print("image our option =", image_our_option_dif)
print("history search =", history_search_dif)
print("combination mat =", combination_mat_dif)

max3 = max(text_dif, image_option1_dif, image_option2_dif, image_option3_dif, image_our_option_dif, history_search_dif,combination_mat_dif)
res = [('text similarity',text_dif),('image option 1',image_option1_dif) ,('image option 2',image_option2_dif) ,
       ('image option 3',image_option3_dif) ,('image our option',image_our_option_dif) ,('history search',history_search_dif) ,('combination mat',combination_mat_dif)]

print("Best result in median method 3 is :", end= " ")
for i in res:
    if i[1] == max3:
        print(i[0],",", end= " ")
print("")
print("")