To run and evaluate this project, begin by installing pytorch, preferably with CUDA installed.
Then, install the rest of the packages from requirements.txt using pip.
At this point, you should be able to run train.py which will train the text encoder for document retrieval.
The training can either be completed naturally or interrupted early, both will save the trained model locally. 
Afterwards, eval.py should be run to evaluate the efficacy of the different models. 
This will also train and evaluate the BIM and BM25 models.
Edit MAX_EVAL_SIZE in eval.py if memory becomes an issue.