import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.preprocessing import minmax_scale


def history_search_similarity():
    similarity_heatmap_data = pd.DataFrame()
    df = pd.read_excel('DistinctCityDestNet.xlsx', engine='openpyxl')
    cities = ['Antalya', 'Bangkok', 'Beijing', 'Crete',
              'Delhi', 'Hong-Kong', 'Los-Angeles', 'Mumbai-Bombay',
              'New-York-City', 'Paris', 'Rhodes', 'Rome']
    N = df['N'].tolist()
    res = []
    index = 0
    for i in range(len(cities)):
        for j in range(len(cities)):
            res.append(math.log(N[index] + 1))
            index = index + 1

    # normalize
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
            index = index + 1

    # plot sims/ set results
    similarity_heatmap = similarity_heatmap_data.pivot(index="city1", columns="city2", values="similarity")
    ax = sns.heatmap(similarity_heatmap, cmap="YlGnBu", annot=True, annot_kws={'size': 8})
    plt.title("History Search Similarity")
    for label1 in ax.get_yticklabels():
        label1.set_weight('bold')
    for label2 in ax.get_xticklabels():
        label2.set_weight('bold')
    plt.show()
    return similarity_heatmap
