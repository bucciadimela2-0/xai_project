 ########## FIRST DEMO #########
import numpy as np
from pysr import PySRRegressor
           # se hai Julia 1.10 installata


#Creating the dataset
X = 2 * np.random.randn(100, 5)
y = 2.5382 * np.cos(X[:, 3]) + X[:, 0] ** 2 - 0.5


#instantiate the py regressor model
model = PySRRegressor(
    output_directory= "output_demo1",
    maxsize=20, # dimensione massima dell'equazione simbolica
    niterations=40,  #numero di ierazione di ricerca evolutiva - quante volte cerca di migliorare la formula 
    binary_operators=["+", "*"], #il modello può usare solo questi operatori binari
    unary_operators=[
        "cos",
        "exp",
        "sin",
        "inv(x) = 1/x", # sintassi di julia
        #quali funzioni unarie può usare, funzioni con un solo argomento 
    ],
    extra_sympy_mappings={"inv": lambda x: 1 / x},
    # mappa per dire a SymPy come interpretare gli operatori personalizzati.
    elementwise_loss="loss(prediction, target) = (prediction - target)^2",
    # Definisce la loss in sintassi Julia 
)

#fitta il modello 
model.fit(X, y)


