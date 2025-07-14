## Prerequisites

- **uv** installed:
  ```bash
  curl -Ls https://astral.sh/uv/install.sh | bash
  export PATH="$HOME/.cargo/bin:$PATH"
  ```
- **Python 3.10** available as `python3.10`

---

## 1. Create & activate the environment

Run:

```bash
cd ~/mlx/week6/bb-finetune/ben_dev
uv venv .venv --python=python3.10
source .venv/bin/activate
```

---

## 2. Install project dependencies

1. Still in the above folder, run:
   ```bash
   # Compile dependencies into a lock file
   uv pip compile --output-file uv.lock pyproject.toml

   # Sync the environment to match the lock file
   uv pip sync uv.lock
   ```
2. Now the `.venv` has all required packages.
3. To check torch and jax installed:
    ```
    python - << EOF
    import torch, jax
    print("Torch:",    torch.__version__)
    print("JAX:",      jax.__version__)
    EOF
    ```
> **Tip**: Whenever you add new dependancies (see 5 below) to the `pyproject.toml`, rerun these commands to refresh `uv.lock` and reinstall.

---

## 3. Register the Jupyter kernel

Run this **once** to expose the env to JupyterLab or VS Code:

```bash
python -m ipykernel install \
  --user \
  --name jax-torch-uv-env \
  --display-name "Python (.venv)"
```


```

## 4. Day-to-day usage

```bash
# Activate the environment
source .venv/bin/activate

# Work as usual (run scripts, notebooks, FastAPI, etc.)

# Deactivate when done
deactivate
```

---

## 5. Adding new dependencies

Whenever you need to add a new package:

1. Add it under `[project].dependencies` in `pyproject.toml`.
2. Re-run:
   ```bash
   uv pip compile pyproject.toml --output-file uv.lock
   uv pip sync uv.lock
   ```

---