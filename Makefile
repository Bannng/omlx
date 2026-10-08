# Developer shortcuts. The server itself builds with plain pip and needs none of these.
PYTHON ?= python3

.PHONY: dev web app

# Editable install of the server and the web UI with dev tools.
dev:
	$(PYTHON) -m pip install -e ".[dev]"

# Rebuild the web UI CSS and normalize its translation files.
web:
	cd apps/omlx-web && $(PYTHON) build_css.py && $(PYTHON) normalize_i18n.py

# Build a runnable macOS app bundle.
app:
	apps/omlx-mac/Scripts/build.sh release
