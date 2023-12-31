install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

test:
	python -m pytest -vv
	python -m pytest --nbval sorc/*.ipynb

format:	
	black sorc/*.py
	nbqa black sorc/*.ipynb

lint:
	nbqa ruff sorc/*.ipynb
	ruff check sorc/*.py

all: install lint format test 
