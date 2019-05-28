.PHONY: features trained-model scores evaluation test clean-pyc clean-env 

# Create a virtual environment named pennylane-env
pennylane-env/bin/activate: requirements.txt
	test -d pennylane-env || virtualenv pennylane-env
	. pennylane-env/bin/activate; pip install -r requirements.txt
	touch pennylane-env/bin/activate

venv: pennylane-env/bin/activate

# Below are for reproducing feature generation, modeling, scoring, evaluation and post-process
data/bank_processed.csv: src/generate_features.py 
	python src/generate_features.py --config=config/config.yaml --output=data/bank_processed.csv
features: data/bank_processed.csv


models/bank-prediction.pkl: data/bank_processed.csv src/train_model.py
	python src/train_model.py --config=config/config.yaml --input=data/bank_processed.csv --output=models/bank-prediction.pkl
trained-model: models/bank-prediction.pkl


models/bank_test_scores.csv: src/score_model.py
	python src/score_model.py --config=config/config.yaml
scores: models/bank_test_scores.csv

models/model_evaluation.csv: src/evaluate_model.py
	python src/evaluate_model.py --config=config/config.yaml --output=models/model_evaluation.csv
evaluation: models/model_evaluation.csv


# Create the database
#data/churn.db:
#	python run.py create

#database: data/churn.db

# Pull raw data from github
get_data:
	python src/import_data.py

# Run all tests
test:
	pytest src/test.py

# Clean up things
clean-tests:
	rm -rf .pytest_cache
	rm -r test/model/test/
	mkdir test/model/test
	touch test/model/test/.gitkeep

clean-env:
	rm -r pennylane-env

clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	rm -rf .pytest_cache

clean: clean-env clean-pyc
