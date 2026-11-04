"""Ejecuta el flujo reproducible completo de la práctica 2.

El flujo conserva la separación de responsabilidades de los notebooks:
1. `main.ipynb`: EDA y regresión logística.
2. `main_b.ipynb`: Random Forest y validación anidada.
3. `main_c.ipynb`: integración, comparación y prueba estadística.

Debe ejecutarse desde cualquier directorio. Los notebooks usan rutas relativas
al directorio `code`, por lo que el ejecutor fija ese directorio como cwd.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


NOTEBOOKS = ("main.ipynb", "main_b.ipynb", "main_c.ipynb")


def execute_notebook(notebook: Path, output: Path, cwd: Path) -> None:
    command = [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        "--ExecutePreprocessor.timeout=1800",
        "--ExecutePreprocessor.kernel_name=python3",
        "--output",
        str(output),
        str(notebook),
    ]
    env = os.environ.copy()
    miniconda_jupyter = Path(sys.prefix) / "share" / "jupyter"
    env["JUPYTER_PATH"] = os.pathsep.join(
        [str(miniconda_jupyter), env.get("JUPYTER_PATH", "")]
    ).rstrip(os.pathsep)
    subprocess.run(command, cwd=cwd, check=True, env=env)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ejecuta los notebooks A, B y C de la práctica 2."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("executed"),
        help="Directorio para los notebooks ejecutados (relativo a code).",
    )
    args = parser.parse_args()

    code_dir = Path(__file__).resolve().parent
    output_dir = (code_dir / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for notebook_name in NOTEBOOKS:
        notebook = code_dir / notebook_name
        if not notebook.is_file():
            raise FileNotFoundError(f"No se encontró el notebook: {notebook}")
        execute_notebook(notebook, output_dir / notebook.name, code_dir)
        print(f"OK: {notebook_name}")


if __name__ == "__main__":
    main()
