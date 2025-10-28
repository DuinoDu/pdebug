# Makefile used to execute some commands locally

env:
	pip3 install -e ".[all]"
	pre-commit install

lint:
	pre-commit run --all-files

watch-lint:
	ptw --runner "pre-commit run -a"

test:
	pytest -s ./pdebug/tests
	pytest -s ./pdebug/visp/tests
	pytest -s ./pdebug/pdag/tests
	pytest -s ./pdebug/runnb/tests
	pytest -s ./pdebug/piata/tests
	pytest -s ./pdebug/otn/tests
	pytest -s ./pdebug/utils/tests
	pytest -s ./pdebug/debug/tests

watch-test:
	ptw --runner "pytest tests -s pdebug/tests"

isort:
	isort .

black:
	black . -l 79

flake8:
	flake8 .

pydocstyle:
	pydocstyle --match-dir='(?!test|project).*'

wheel:
	python3 setup.py sdist bdist_wheel; \
	ls dist

upload:
	python setup.py bdist_wheel upload -r hobot-local

clean-wheel:
	@rm -rf dist

clean: clean-wheel
	@rm -r build *.egg-info tmp_* tmp*
