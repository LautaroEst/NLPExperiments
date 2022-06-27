


from sklearn.naive_bayes import MultinomialNB
from .main_classes import GenericMLMainModel


class NaiveBayes(GenericMLMainModel):

    name = "naive_bayes"

    def __init__(self,task,alpha=1.0,fit_prior=True,class_prior=None):
        config_params = dict(
            alpha=alpha,
            fit_prior=fit_prior,
            class_prior=class_prior
        )
        super().__init__(task,**config_params)
        self.model = MultinomialNB(**config_params)
        self._state_dict = {} # TO DO: COMPLETAR LOS PARÁMETROS

    def fit(self,X,y):
        self.model.fit(X,y)
        self._state_dict = {} # TO DO: COMPLETAR LOS PARÁMETROS

    def init_model(self):
        pass

    def forward(self,X):
        return {
            "predictions": self.model.predict(X)
        }