---
name: run-python
description: >-
  Use this skill whenever running Python scripts, modules, or commands in the
  microvla repository. Instructs how to execute Python modules using uv.
---

# Running Python Modules in microvla

Always execute Python scripts and modules in this repository using `uv run`.

## Usage

- **Run a Python script:**
  ```bash
  uv run python <script.py>
  ```
  or directly:
  ```bash
  uv run <script.py>
  ```

- **Run a Python module:**
  ```bash
  uv run python -m <module_name>
  ```

