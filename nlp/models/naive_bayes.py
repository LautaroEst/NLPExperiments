


from sklearn.naive_bayes import MultinomialNB
from .main_classes import GenericMLModel


class NaiveBayes(GenericMLModel):

    name = "naive_bayes"

    def __init__(self,alpha=1.0,fit_prior=True,class_prior=True):
        params = dict(
            alpha=alpha,
            fit_prior=fit_prior,
            class_prior=class_prior
        )
        super().__init__(**params)
        self.model = MultinomialNB(**params)
        self.sklearn_params = {} # TO DO: COMPLETAR LOS PARÁMETROS

    def fit(self,X,y):
        self.model.fit(X,y)
        self.sklearn_params = {} # TO DO: COMPLETAR LOS PARÁMETROS

    def predict(self,X):
        return self.model.predict(X)