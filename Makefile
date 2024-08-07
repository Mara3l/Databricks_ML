# Makefile for creating a Python virtual environment

# Default target
.PHONY: all
all: .venv/bin/activate

# Create virtual environment
.venv/bin/activate: requirements.txt
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt

# Install dependencies
.PHONY: install
install: .venv/bin/activate
	.venv/bin/pip install -r requirements.txt

# Clean the virtual environment
.PHONY: clean
clean:
	rm -rf .venv
