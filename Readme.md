# Data Mining - AutoML

> UFABC Data Mining Course project about AutoML 

## How to use it

First [install auto-sklearn](http://automl.github.io/auto-sklearn/stable/installation.html) or use Docker:

```sh
docker build -t automl .
docker run --name=automl -v ~/dev/data-mining-project/shared:/usr/src/shared/ automl
```

This will run the `src/main.py` file and train a new model.

The code for model evaluation is `src/model_evaluation.py`.

# Results

https://mateuszitelli.github.io/data-mining-project/
