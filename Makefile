
VENV = .venv
UV = uv
GREEN = \033[0;32m
NC = \033[0m

.PHONY: all install run debug clean fclean re lint lint-strict

all: install

install:
	@echo "$(GREEN)Installing dependencies with uv...$(NC)"
	$(UV) sync

run:
	@echo "$(GREEN)Running the main CLI...$(NC)"
	$(UV) run python -m student

debug:
	@echo "$(GREEN)Running in debug mode (pdb)...$(NC)"
	$(UV) run python -m pdb -m student

clean:
	@echo "$(GREEN)Cleaning temporary files and caches...$(NC)"
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@rm -rf .mypy_cache
	@rm -rf .pytest_cache

lint:
	@echo "$(GREEN)Running linters (flake8 and mypy)...$(NC)"
	$(UV) run flake8 src student
	$(UV) run mypy src student --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs


fclean: clean
	@echo "$(GREEN)Borrando el entorno virtual...$(NC)"
	@rm -rf $(VENV)

re: fclean all