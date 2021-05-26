import use
import finalScore
import imgsimop1
import imagesimop2
import imagesimop3
import image2vec
import historySearch
cities = ['Antalya', 'Bangkok', 'Beijing', 'Crete',
          'Delhi', 'Hong-Kong', 'Los-Angeles', 'Mumbai-Bombay',
          'New-York-City', 'Paris', 'Rhodes', 'Rome']


# ----------------- distance between matrices ----------------- #
text_similarity = use.text_similarity_result()
AMT_result = finalScore.AMT_similarity()
image_option1 = imgsimop1.image_option1_similarity()
image_option2 = imagesimop2.image_option2_similarity()
image_option3 = imagesimop3.image_option3_similarity()
image_our_option = image2vec.image_our_option_similarity()
history_search = historySearch.history_search_similarity()

text_dif = 0
image_option1_dif = 0
image_option2_dif = 0
image_option3_dif = 0
image_our_option_dif = 0
history_search_dif = 0
for i in cities:
    for j in cities:
        text_dif += abs(text_similarity.loc[i, j] - AMT_result.loc[i, j])
        image_option1_dif += abs(image_option1.loc[i, j] - AMT_result.loc[i, j])
        image_option2_dif += abs(image_option2.loc[i, j] - AMT_result.loc[i, j])
        image_option3_dif += abs(image_option3.loc[i, j] - AMT_result.loc[i, j])
        image_our_option_dif += abs(image_our_option.loc[i, j] - AMT_result.loc[i, j])
        history_search_dif += abs(history_search.loc[i, j] - AMT_result.loc[i, j])

print("text =", text_dif)
print("image option 1 =", image_option1_dif)
print("image option 2 =", image_option2_dif)
print("image option 3 =", image_option3_dif)
print("image our option =", image_our_option_dif)
print("history search =", history_search_dif)
min = min(text_dif, image_option1_dif, image_option2_dif, image_option3_dif, image_our_option_dif, history_search_dif)
winner = ""
if min == text_dif:
    winner = "text similarity"
elif min == image_option1_dif:
    winner = "image option 1"
elif min == image_option2_dif:
    winner = "image option 2"
elif min == image_option3_dif:
    winner = "image option 3"
elif min == image_our_option_dif:
    winner = "image our option"
else:
    winner = "history search"
print("in distance between matrices: the must represent method is :", winner)


# ----------------- bigger the median ----------------- #

