install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

test:
	python -m pytest -vv --cov=src src/test_*.py
	python -m pytest --nbval sorc/*.ipynb

format:	
	black src/*.py
	nbqa black src/*.ipynb

lint:
	nbqa ruff src/*.ipynb
	ruff check src/*.py

all: install lint format test 