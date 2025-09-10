# backend/scripts/generate_data.py

import sys
import os
import csv
from copy import deepcopy
from datetime import datetime
from time import sleep

# Rich imports
from rich.console import Console
from rich.progress import track

# Add the backend directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from backend.klotski.utils import Block
from backend.klotski.solver import Solver
from backend.scripts.generate_random_klotski_board import generate_random_klotski_board
from backend.scripts.board_generator import render_klotski_board  

console = Console()


def ensure_data_dir(output_path):
    dir_name = os.path.dirname(output_path)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)


def generate_klotski_data(num_samples=2, output=None, randomize=True, show_steps=False, show_initial=False):
    if output is None:
        now_str = datetime.now().strftime("%Y%m%d-%H%M%S")
        output = f"backend/data/generated_states_{now_str}.csv"
    ensure_data_dir(output)
    winning_row, winning_col = 3, 1

    with open(output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample", "step", "block_states"])

        sample_range = range(num_samples)
        iterator = sample_range if show_steps else track(sample_range, description="Solving puzzles...")

        for sample_idx in iterator:
            blocks = generate_random_klotski_board()

            # Defensive flattening if nested list returned
            if isinstance(blocks[0], list):
                blocks = blocks[0]

            if show_initial:
                console.print(f"\n[bold magenta]Sample {sample_idx + 1} initial board:[/bold magenta]")
                render_klotski_board(blocks)  

            solver = Solver(deepcopy(blocks), winning_row, winning_col)
            solver.solve()
            solution = solver.get_solution_boards()

            if solution is None or len(solution) == 0:
                if show_steps:
                    console.print(f"\n[bold red]Sample {sample_idx + 1} has no solution.[/bold red]")
                flat = []
                for b in blocks:
                    flat.extend([b.num_rows, b.num_cols, b.row_pos, b.col_pos])
                writer.writerow([sample_idx, 0, flat])
                continue

            if show_steps:
                console.print(f"\n[bold blue]Sample {sample_idx + 1}/{num_samples}[/bold blue]")
                console.print(f"[bold green]Solution found in {len(solution) - 1} moves[/bold green]")

            for idx, state in enumerate(solution):
                flat = []
                if show_steps:
                    console.print(f"[bold yellow]Step {idx}[/bold yellow]")
                    render_klotski_board(state)   
                    sleep(0.3)
                for b in state:
                    flat.extend([b.num_rows, b.num_cols, b.row_pos, b.col_pos])
                writer.writerow([sample_idx, idx, flat])


if __name__ == "__main__":
    try:
        num_samples = int(input("Enter number of iterations: "))
    except ValueError:
        print("Invalid input. Defaulting to 1.")
        num_samples = 1

    show_steps = input("Show each solving step? (y/n): ").strip().lower() in ("y", "yes")
    show_initial = False
    if not show_steps:
        show_initial = input("Show initial board even if not showing steps? (y/n): ").strip().lower() in ("y", "yes")

    generate_klotski_data(num_samples=num_samples, randomize=True, show_steps=show_steps, show_initial=show_initial)
