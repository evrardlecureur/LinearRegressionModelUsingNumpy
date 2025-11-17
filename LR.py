
# je vais choisir de importer Numpy pour pouvoir faire les calculs rapidement et m'habituer a la librairie 
import numpy as np 
 


class LinearRegression:
    def __init__(self, fit_intercept=True, n_iterations=1000, learning_rate=0.01): #valeurs par defaut 
        self.coef = None       # attribut d’instance
        self.intercept = fit_intercept  # on l'ajoutera ou non  (sans intercept le model passe par (0,0) , par de terme constant )
        self.intercept_value = None 
        self.n_iterations = n_iterations 
        self.learning_rate = learning_rate

 

    def fit(self , X , y):
        # verfier que les 2 tableaux Numpy on les bonnes dimensions 
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        if X.shape[0] == y.shape[0] :
            #ducoup ils sont la bonne taile , en calcul matriciel , notre model sera de la forme y = X*B avec le produit matriciel usuel , on doit commencer par ajouter des 1 dans la premier colonne de X si on a intercept 
            if self.intercept : 
                constante = np.ones( [ X.shape[0],1 ] )
                X = np.hstack( [constante,X] )  #permet de concatener 
            self.X = X # mettre a jour self 

            # on est pret a commencer les calculs à savoir trouver le bon B un array contenant des coefficients , je vais utiliser la descente de gradient avec comme fonction de cout le MSE 
            # on commence par initialiser B avec des coeffs aleatoires , enfaite ils devraient pas etre aleatoires mais je corrigerais ce probleme a la fin . 
            
            nfeatures = X.shape[1]

            n = y.shape[0]


            B = np.random.randn(X.shape[1],1) # variance de 1 , valeurs de -inf a +inf ( dsitribution normale)
            for iterations in range (self.n_iterations):

                ypred = np.dot(X,B) #fait le produit matriciel

                error = ypred - y

                gradients = ( 2 / n )  * X.T.dot(error)   # met X en transposé et multiplie par l'erreur de forme (l,c) = (nombre de prediction , 1)  : ∇J(B) = 2/n * ​XTransposé(XB−y)
                # on a donc une liste de tout les gradiants pour chaque feature 
                # on ajuste B 
                B = B - self.learning_rate * gradients  

                


            # on doit maitenant remplir self.coef avec la liste des coeffs , et voir si on a intercept_value ou pas

            if self.intercept:
                self.intercept_value = B[0, 0]    # B0 le terme constant de la regression 
                self.coef = B[1:].flatten()       # on flatten pour avoir que une liste forme [ B1 , B2  ,...]
            else:
                self.intercept_value = 0
                self.coef = B.flatten()      
             


        else :
            print("training data and traget must have coherent shapes")




        
    def predict(self, X ):
        if self.intercept : 

            ypred = X.dot(self.coef.reshape(-1,1))
            # on ajoute la constante car le cas ou intercept est True 
            ypred = ypred + self.intercept_value



        else : 
            ypred = X.dot(self.coef.reshape(-1,1))




        return ypred.flatten()
    
    



