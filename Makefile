install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

test:
	python -m pytest -vv --cov=sorc sorc/test_main*.py
	python -m pytest --nbval sorc/*.ipynb

format:	
	black src/*.py
	nbqa black src/*.ipynb

lint:
	nbqa ruff sorc/*.ipynb
	ruff check sorc/*.py

all: install lint format test 
