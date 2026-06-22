# Makefile used to execute some commands locally

PYTHON := $(shell \
	if [ -x .venv/bin/python ]; then \
		echo .venv/bin/python; \
	else \
		echo python; \
	fi \
)
PYTEST := $(PYTHON) -m pytest
TEST_DIRS := ./pdebug/tests ./pdebug/visp/tests ./pdebug/pdag/tests \
	./pdebug/runnb/tests ./pdebug/piata/tests ./pdebug/otn/tests \
	./pdebug/algo/tests ./pdebug/geometry/tests ./pdebug/utils/tests

env:
	$(PYTHON) -m pip install -e ".[all,dev]"
	# pre-commit install

lint:
	pre-commit run --all-files

watch-lint:
	ptw --runner "pre-commit run -a"

test:
	$(PYTEST) -s ./pdebug/tests
	$(PYTEST) -s ./pdebug/visp/tests
	$(PYTEST) -s ./pdebug/pdag/tests
	$(PYTEST) -s ./pdebug/runnb/tests
	$(PYTEST) -s ./pdebug/piata/tests
	$(PYTEST) -s ./pdebug/otn/tests
	$(PYTEST) -s ./pdebug/algo/tests
	$(PYTEST) -s ./pdebug/geometry/tests
	$(PYTEST) -s ./pdebug/utils/tests
	$(PYTEST) -s ./pdebug/debug/tests || [ $$? -eq 5 ]

watch-test:
	ptw --runner "$(PYTEST) -s $(TEST_DIRS)"

isort:
	isort .

black:
	black . -l 79

flake8:
	flake8 .

pydocstyle:
	pydocstyle --match-dir='(?!test|project).*'

wheel:
	$(PYTHON) -m pip wheel . --no-deps -w dist; \
	ls dist

upload:
	$(PYTHON) -m twine upload dist/*

clean-wheel:
	@rm -rf dist

clean: clean-wheel
	@rm -r build *.egg-info tmp_* tmp*
