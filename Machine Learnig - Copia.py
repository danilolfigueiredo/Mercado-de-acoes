import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

def best_feactures():
    features_list = ('Open','High','Low','Volume','mm21')

    k_best_features = SelectKBest(k='all')
    k_best_features.fit_transform(features, labels-1)  #erro
    k_best_features_scores = k_best_features.scores_
    raw_pairs = zip(features_list[1:], k_best_features_scores)
    ordered_pairs = list(reversed(sorted(raw_pairs, key=lambda x: x[1])))

    k_best_features_final = dict(ordered_pairs[:15])
    best_features = k_best_features_final.keys()
    print ('')
    print ("Melhores features:")
    print (k_best_features_final)


#Lendo CSV com pandas
df = pd.read_csv("EURUSD_M15.csv", delimiter = '\t')
#convertendo coluna para formato tempo
df['Time'] = pd.to_datetime(df['Time'])
df['Volume'] = pd.to_numeric(df['Volume'], downcast='float')

#removendo dados nulos
df.dropna(inplace=True)
#reindexando data frame
df= df.reset_index(drop=True)

#criando campos de Médias móveis
#df['mm5'] = df['Close'].rolling(5).mean() #ruim
df['mm21'] = df['Close'].rolling(21).mean()

#empurrando valor de Close um candle pra trás para prever valor do candle seguinte
df['Close'] = df['Close'].shift(-1)

#definindo treino
qtd_linhas = len(df)

qtd_linhas_treino = round(.70 * qtd_linhas)
qtd_linhas_teste = qtd_linhas - 50
qtd_linhas_validacao = qtd_linhas_treino - qtd_linhas_teste

#reindexando data frame
df= df.reset_index(drop=True)

#removendo dados nulos
df.dropna(inplace=True)

#definindo Features e Labels
features = df.drop(['Close','Time'],1)
labels = df['Close']

#removendo dados nulos
df.dropna(inplace=True)
features.dropna(inplace=True)
labels.dropna(inplace=True)
#Escolhendo melhores features (em função)

best_feactures()

#normalizando dados de entrada
scaler = MinMaxScaler().fit(features)
features_scale = scaler.transform(features)

#separa dados de treino teste e validação
x_train = features_scale[:qtd_linhas_treino]
x_test = features_scale[qtd_linhas_treino:qtd_linhas_teste]

y_train = labels[:qtd_linhas_treino]
y_test = labels[qtd_linhas_treino:qtd_linhas_teste]

#regressão linear
lr = linear_model.LinearRegression()
lr.fit(x_train, y_train)
pred= lr.predict(x_test)
cd =r2_score(y_test, pred)

print(f'Coeficiente de determinação:{cd * 100:.2f}')

#previsão
previsao = features_scale[qtd_linhas_teste:qtd_linhas] ###Finalzinho disso aqui tá dando erro

data_pregao_full = df['Time']
data_pregao = data_pregao_full[qtd_linhas_teste:qtd_linhas]

res_full = df['Close']
res = res_full[qtd_linhas_teste:qtd_linhas]

pred = lr.predict(previsao)

df=pd.DataFrame({'Time':data_pregao, 'real':res, 'previsao':pred})

df['real'] = df['real'].shift(+1)

df.set_index('Time', inplace = True)

print(df)

