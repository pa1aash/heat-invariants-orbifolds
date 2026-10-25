# Reproduce the computations of the paper. `make help` lists the targets and their run times.
# Each target prints PASS, FAIL or SKIP for every step and exits nonzero if any step fails.
PYTHON ?= python3
export PYTHON
STEPS = $(PYTHON) tools/steps.py

.PHONY: help verify full figures tables data-check all clean

help:
	@echo 'make verify      fast exact checks (about 22 min)'
	@echo 'make full        long computations (about 16 min, plus about 60 min with the optional NGSolve/Blender steps)'
	@echo 'make figures     rebuild every figure and compare with the committed files (about 1 min, plus about 1 min with the optional NGSolve/Blender steps)'
	@echo 'make tables      regenerate every table and compare with the committed files (about 1 min)'
	@echo 'make data-check  compare every data file with SHA256SUMS (seconds)'
	@echo 'make all         everything above (about 40 min, plus about 61 min with the optional NGSolve/Blender steps)'
	@echo 'make clean       remove caches and compiled helpers'
	@echo ''
	@echo 'Times measured on an Apple M-series laptop with 8 GB of memory. PYTHON selects the interpreter (default python3).'
	@echo 'Run `python3 tools/steps.py --list` for every step.'

verify:
	@$(STEPS) verify
full:
	@$(STEPS) full
figures:
	@$(STEPS) figures
tables:
	@$(STEPS) tables
data-check:
	@$(STEPS) data-check
all:
	@$(STEPS) all
clean:
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf theory/pte/search_T3_modular
	rm -rf theory/pte/search_T3
