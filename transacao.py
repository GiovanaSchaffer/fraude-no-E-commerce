import os
os.environ['SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL']='True'
import numpy as np 
import pandas as pd 
import matplotlib.patches as mpatches
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPClassifier
from imblearn.combine import SMOTEENN

hln = 20
lr = 0.1 # learning rate
P = 0.75
SEED = 923
SEED = np.random.seed(SEED)
valRatio = .33
shuffle = True
maxValFails = 20 # max. number of validation fails
nmax = 500 # max. number of epochs

# Lista das colunas categóricas que precisam ser codificadas
colunas_categoricas = ['Payment Method', 
                       'Product Category', 
                       'Device Used',
                       'Mes',
                       'Periodo do Dia']
def cod(dataframe): 

    # Calculando a frequência de cada categoria em cada coluna categórica
    for coluna in colunas_categoricas:
        frequencia = dataframe[coluna].value_counts()
        mapeamento_frequencia = frequencia.to_dict()
    
        # Substituindo os valores na coluna pelo valor da frequência correspondente
        dataframe[coluna + '_Frequencia'] = dataframe[coluna].map(mapeamento_frequencia)
        
    dataframe['Periodo do Dia_Frequencia'] = dataframe['Periodo do Dia_Frequencia'].astype('int64')
    
    return dataframe


def ajustarDB(dataframe):
    
    '''
    Essas colunas são identificadores únicos e geralmente não são relevantes 
    para determinar a fraude em uma transação. Elas podem ser excluídas da análise. 
    '''

    dataframe = dataframe.drop(columns = ['Transaction ID', 
                                          'Customer ID',
                                          'Customer Location', 
                                          'IP Address', 
                                          'Shipping Address',
                                          'Billing Address',
                                          'Transaction Hour'])
    
    
    # Converter a coluna de data para o tipo datetime
    dataframe['Transaction Date'] = pd.to_datetime(dataframe['Transaction Date'])

    # Extrair o mês da data e colocar em uma nova coluna
    dataframe['Mes'] = dataframe['Transaction Date'].dt.strftime('%B')  # '%B' retorna o nome completo do mês

    # Discretizar a hora do dia em intervalos fixos (manhã, tarde, noite)
    bins = [-1, 6, 12, 18, 24]
    labels = ['Madrugada', 'Manhã', 'Tarde', 'Noite']
    dataframe['Periodo do Dia'] = pd.cut(dataframe['Transaction Date'].dt.hour, bins=bins, labels=labels, right=False)
    
    # Agora podemos excluir essa coluna

    
    dataframe = cod(dataframe)
    
    dataframe = dataframe.drop(columns = colunas_categoricas)
    dataframe = dataframe.drop(columns = ['Transaction Date'])
    
    return dataframe

data = pd.read_csv('Fraudulent_E-Commerce_Transaction_Data_2.csv',header = 0)
data = ajustarDB(data)

y = data['Is Fraudulent']
X = data.drop(['Is Fraudulent'],axis=1)

print(data.head())

X_train_,X_test_,y_train,y_test = train_test_split(X,
                                                   y,
                                                   train_size= P,
                                                   random_state=SEED,
                                                   shuffle=True)

# Aplicar combinação de SMOTE e ENN
smote_enn = SMOTEENN(random_state= SEED)

X_train_, y_train = smote_enn.fit_resample(X_train_, y_train)

net = MLPClassifier(hidden_layer_sizes = hln,
                    activation = 'logistic', 
                    solver ='sgd', 
                    batch_size = 1,                    
                    alpha = 0, 
                    momentum = 0,
                    learning_rate='constant', 
                    learning_rate_init=lr,
                    max_iter=nmax,                     
                    shuffle=shuffle, 
                    random_state = SEED,
                    tol = 1e-10, 
                    n_iter_no_change = maxValFails,
                    early_stopping = True,
                    validation_fraction = valRatio,
                    verbose = False)

net.fit(X_train_,y_train)

plt.figure()
plt.plot(net.loss_curve_,'b',label='Training set')
plt.grid()
plt.plot(net.validation_scores_,'r',label='Validation set')
plt.xlabel('epoch');
plt.ylabel('log loss');
plt.xlim([0,net.n_iter_-1])
L = list(range(0,net.n_iter_,5))
plt.xticks(ticks=L,labels=[str(i+1) for i in L])
plt.legend()

plt.show()