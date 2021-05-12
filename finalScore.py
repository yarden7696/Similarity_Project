from statistics import stdev

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    similarity_heatmap_mean = pd.DataFrame()
    similarity_heatmap_dev = pd.DataFrame()
    df = pd.read_excel('resultsTop12.xlsx', engine='openpyxl')
    cities_dup = df['Input.city'].tolist()
    cities = []
    [cities.append(x) for x in cities_dup if x not in cities]
    city = 0
    for i in range(len(cities)):
        for j in range(len(cities)):
            answerTo = df['Answer.to' + cities[j]].tolist()
            mean = sum(answerTo[city*10 : (city+1)*10]) / 70
            similarity_heatmap_mean = similarity_heatmap_mean.append(
                {
                    'mean': mean,
                    'city1': cities[i],
                    'city2': cities[j]
                },
                ignore_index=True
            )
            std = stdev(answerTo[city * 10: (city + 1) * 10])
            similarity_heatmap_dev = similarity_heatmap_dev.append(
                {
                    'stdev': std,
                    'city1': cities[i],
                    'city2': cities[j]
                },
                ignore_index=True
            )
        city = city + 1

    print(similarity_heatmap_mean)
    print(similarity_heatmap_dev)

    similarity_heatmap = similarity_heatmap_mean.pivot(index="city1", columns="city2", values="mean")
    ax = sns.heatmap(similarity_heatmap, cmap="YlGnBu", vmin=0, vmax=1)
    plt.title("results turkers - mean")
    plt.show()

if __name__ == '__main__':
    main()