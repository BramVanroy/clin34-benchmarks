quality:
	ruff check src/clin34 scripts/
	ruff format --check src/clin34 scripts/

style:
	ruff check src/clin34 scripts/ --fix
	ruff format src/clin34 scripts/
