# NLPExperiments: A repository to perform experiments in NLP

This repository contains the source code to perform some standard classification experiments in NLP. 

* `experiments`: root directory for all experiments. Each directory within this folder contains a `configs` directory where configs file for that experiment are saved. Some scripts and bash files are also included here to reproduce results of each specific run. This directory additionally contains an output folder where results are saved and it is not pushed to origin.
* `nlp`: python module to reproduce the experiment pipeline.
* `scripts`: some common scripts that are called by the experiments scripts (inside each experiment directory) that perform some general task like training a model, a tokenizer, evaluate them, etc.
* `data`: data directories where the data used in the experiments are saved.


## TO DO:

* Unificar sklearn model con torch
* Mejorar el config de training (implementar configs para optimizador, loss, etc.). Quizás implementar modelo y training todo en uno
* Implementar cross-validation
* Pensar en alguna solución para hacer un resume de un entrenamiento

* Features selection de teaming para algunas de las preguntas
* Elegir las preguntas que van para teaming
* Hablar con ramiro para preguntar sobre las preguntas en castellano
