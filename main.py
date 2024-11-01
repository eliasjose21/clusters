import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

dataframe = pd.read_csv('df_test.csv')
colunas = ["price","bedrooms","grade","has_basement","living_in_m2","renovated","nice_view","perfect_condition","real_bathrooms","has_lavatory","single_floor","month","quartile_zone"]

valores_x = dataframe[colunas]

inercia = []
rangex = range(1,11)

for index in rangex:
    means = KMeans(n_clusters=index,random_state=42).fit(valores_x)
    inercia.append(means.inertia_)


plt.plot(rangex,inercia)
plt.xlabel("Número de Clusters")
plt.ylabel("Inércia")
plt.title("Método Cotovelo - Definição do N Clusters")
plt.show()



def verifica_dados_cluster():
    n_clusters = 4
    means = KMeans(n_clusters=n_clusters,random_state=42)
    means.fit(valores_x)
    dataframe['cluster'] = means.labels_
    print(dataframe)
    
verifica_dados_cluster()
dataframe.to_csv('dataset_com_clusters.csv', index=False)

print("Arquivo CSV salvo com sucesso!")