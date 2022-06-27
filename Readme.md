# NLPExperiments: A repository to perform experiments in NLP

This repository contains the source code to perform some standard classification experiments in NLP. 

* `experiments`: root directory for all experiments. Each directory within this folder contains a `configs` directory where configs file for that experiment are saved. Some scripts and bash files are also included here to reproduce results of each specific run. This directory additionally contains an output folder where results are saved and it is not pushed to origin.
* `nlp`: python module to reproduce the experiment pipeline.
* `scripts`: some common scripts that are called by the experiments scripts (inside each experiment directory) that perform some general task like training a model, a tokenizer, evaluate them, etc.
* `data`: data directories where the data used in the experiments are saved.


## TO DO:

* Hacer una clase `BaseNLPFeaturesExtractor` que use internamente un tokenizer y heredar de esa clase todos los extractors de NLP. Esto se quiere hacer para unificar el tokenizer con el extractor. 
* Implementar una clase `DataModule` que guarde los datos procesados/sin procesar que permita manejar el flujo de datos del entrenamineto y la preparación. Es decir, yo quiero hacer dos scripts. El primero crea un extractor, lo entrena y preprocesa los datos para que sean más manejables en el entrenamiento (ej: en NLP, entrena el tokenizador, pasa a índices las palabras, unifica las oraciones, encuentra los token_types_ids, etc. y en audio extrae los mfcc's). Estos datos se guardan con el data_module en disco. Es decir, el primer script es init_extractor + prepare_data + save. El segundo script, levanta los datos procesados, levanta el extractor y le dice que en lugar de usar `prepare_data`, use `transform`. Es decir, es load + transform.

```Python
class BaseFeatureExtractor:

    ...

    def init_extractor():
        ...

    def prepare_data():
        ...

    def transform():
        ...

    def load():
        ...
    
    def save():
        ...

class BaseNLPFeatureExtractor(BaseFeatureExtractor):

    def __init__(self,**config):
        self.tokenizer = init_tokenizer(**config)
        ...
    
```

* Mejorar el config de training (implementar configs para optimizador, loss, etc.). Quizás implementar modelo y training todo en uno
* Implementar un trainer que haga cross-validation
* Implementar un trainer que permita hacer un resume de un entrenamiento

* Features selection de teaming para algunas de las preguntas
* Elegir las preguntas que van para teaming
* Hablar con ramiro para preguntar sobre las preguntas en castellano
