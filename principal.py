# -*- coding: utf-8 -*-
"""
Runner interactivo para categorización con Ollama.
- Lista archivos .xlsx/.xls en la carpeta IN (junto a este script).
- Permite seleccionar un archivo y llama al script principal `categ_ollama_es.py`.
- Deja la salida en la carpeta OUT con un nombre derivado del archivo de entrada.

Uso básico:
    python run_categ_selector.py

Opciones (todas son opcionales):
    --model bge-m3               Modelo de embeddings de Ollama (default: bge-m3)
    --otro-th 0.70               Umbral de 'Otro' (default: 0.70)
    --kw-weight 0.30             Peso del plus por keywords (0..1, default: 0.30)
    --sheet-db db                Nombre de hoja de datos (default: db)
    --sheet-cats categorias      Nombre de hoja de categorías (default: categorias)
    --include-cols COLS...       Lista de columnas a usar (whitelist). Si se usa, solo se consideran estas.
    --exclude-cols COLS...       Lista de columnas a excluir por nombre exacto (no se usan para clasificar).
    --exclude-re REGEX           Expresión regular para excluir columnas por patrón (insensible a mayúsculas).
    --verbose                    Muestra columnas usadas/enviadas al principal.
    --all                        Procesa TODOS los excels en IN sin preguntar
    --file N                     Procesa el archivo N del listado (1..N) sin preguntar

Requisitos:
    - Tener `categ_ollama_es.py` en el mismo directorio o proveer ruta absoluta en --main
    - Ollama corriendo (o el script principal hará fallback a TF-IDF si no está disponible)
"""
import argparse, os, sys, subprocess
from pathlib import Path

def list_excels(in_dir: Path):
    files = [p for p in in_dir.glob("*") if p.suffix.lower() in (".xlsx",".xls") and not p.name.startswith("~$")]
    files.sort()
    return files

def ensure_out(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir

def next_out_path(out_dir: Path, in_file: Path):
    base = in_file.stem + "_categ_ollama.xlsx"
    out_path = out_dir / base
    i = 2
    while out_path.exists():
        out_path = out_dir / f"{in_file.stem}_categ_ollama_{i}.xlsx"
        i += 1
    return out_path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", default="categ_ollama_es.py",
                    help="Ruta al script principal (default: categ_ollama_es.py en la misma carpeta)")
    ap.add_argument("--model", default="bge-m3")
    # Si prefieres tus valores: cambia a default=0.14 y default=0.7
    ap.add_argument("--otro-th", type=float, default=0.14, help="Umbral de 'Otro' (0..1)")
    ap.add_argument("--kw-weight", type=float, default=0.7, help="Peso de keywords (0..1)")
    ap.add_argument("--sheet-db", default="db")
    ap.add_argument("--sheet-cats", default="categorias")

    # NUEVOS: pasan directo al script principal
    ap.add_argument("--include-cols", nargs="*", default=None,
                    help="Whitelist de columnas a usar para clasificar (exporta todas igualmente).")
    ap.add_argument("--exclude-cols", nargs="*", default=["PROGRAMA"],
                    help="Columnas a excluir de la clasificación (por nombre exacto).")
    ap.add_argument("--exclude-re", default=None,
                    help="Regex para excluir columnas por patrón (case-insensitive).")
    ap.add_argument("--verbose", action="store_true", help="Muestra los flags de columnas que se enviarán.")

    ap.add_argument("--all", action="store_true", help="Procesar todos los archivos en IN")
    ap.add_argument("--file", type=int, default=0, help="Procesar el archivo N del listado (1..N)")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    in_dir = here / "IN"
    out_dir = here / "OUT"
    ensure_out(out_dir)
    in_dir.mkdir(parents=True, exist_ok=True)

    main_script = Path(args.main)
    if not main_script.is_absolute():
        main_script = (here / main_script).resolve()
    if not main_script.exists():
        print(f"[ERROR] No encuentro el script principal en: {main_script}")
        sys.exit(1)

    files = list_excels(in_dir)
    if not files:
        print(f"[INFO] No hay .xlsx/.xls en {in_dir}. Copia tus archivos allí y vuelve a ejecutar.")
        sys.exit(0)

    # Mostrar listado
    print("\nArchivos disponibles en IN/:")
    for i, p in enumerate(files, start=1):
        print(f"  [{i}] {p.name}")
    print("")

    to_process = []
    if args.all:
        to_process = files
    elif args.file:
        idx = args.file
        if idx < 1 or idx > len(files):
            print(f"[ERROR] --file debe estar entre 1 y {len(files)}")
            sys.exit(1)
        to_process = [files[idx-1]]
    else:
        while True:
            sel = input(f"Selecciona un número (1..{len(files)}), 'a' para todos o 'q' para salir: ").strip().lower()
            if sel == 'q':
                print("Cancelado por el usuario.")
                sys.exit(0)
            if sel == 'a':
                to_process = files
                break
            if sel.isdigit():
                n = int(sel)
                if 1 <= n <= len(files):
                    to_process = [files[n-1]]
                    break
            print("Entrada no válida. Intenta de nuevo.")

    # Ejecutar por cada archivo seleccionado
    py = sys.executable or "python"

    # Prepara los flags de columnas (solo si vienen) para mostrar y pasar
    if args.verbose:
        print("\n[runner] Flags de columnas a enviar al principal:")
        print("  --include-cols:", args.include_cols)
        print("  --exclude-cols:", args.exclude_cols)
        print("  --exclude-re  :", args.exclude_re)

    for f in to_process:
        out_path = next_out_path(out_dir, f)
        cmd = [
            py, str(main_script),
            "--in", str(f),
            "--out", str(out_path),
            "--sheet-db", args.sheet_db,
            "--sheet-cats", args.sheet_cats,  # el principal lo ignora, pero lo dejamos por compatibilidad
            "--model", args.model,
            "--otro-th", str(args.otro_th),
            "--kw-weight", str(args.kw_weight),
        ]

        # Passthrough de flags nuevos (nunca vacíos)
        if args.include_cols:
            cmd.append("--include-cols")
            cmd.extend(args.include_cols)
        if args.exclude_cols:
            cmd.append("--exclude-cols")
            cmd.extend(args.exclude_cols)
        if args.exclude_re:
            cmd.extend(["--exclude-re", args.exclude_re])
        if args.verbose:
            cmd.append("--verbose")

        print("\n>>> Ejecutando:", " ".join(cmd))
        try:
            subprocess.run(cmd, check=True)
            print(f"[OK] Resultado: {out_path}")
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Falló el procesamiento de {f.name}: {e}")
        except Exception as e:
            print(f"[ERROR] {e}")

if __name__ == "__main__":
    main()
