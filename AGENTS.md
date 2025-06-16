# Contributor Guide

## Dev Environment Tips  
- Use `uv venv` …  
- Run `uv pip install -r requirements.txt` …  
- Format code with `black .` and lint using `ruff check .`.  
- Run all tests using `pytest`, or filter by test name with `pytest -k "<test_name>"`.  
- Generate a coverage report with `pytest --cov=your_package_name --cov-report=term-missing`.  
- Use `mypy` to perform static type checks if configured.  
- Follow the project layout, and mirror test structure in `tests/` for clarity.

## Testing Instructions
- Find the CI plan in the .github/workflows folder.
- Run every check defined for that package.
- The commit should pass all tests before you merge.
- Fix any test or type errors until the whole suite is green.
- Add or update tests for the code you change, even if nobody asked.

## PR instructions
Title format: [<project_name>] <Title>