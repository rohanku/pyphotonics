.PHONY: format

format:
	black .
lint:
	find . -type f -name "*.py" | xargs pylint 
