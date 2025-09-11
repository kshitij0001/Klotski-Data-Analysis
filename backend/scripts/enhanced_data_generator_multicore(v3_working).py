# backend/scripts/enhanced_data_generator_multicore.py

import sys
import os
import csv
import json
from copy import deepcopy
from datetime import datetime
from time import time
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

# Rich imports
from rich.console import Console
from rich.progress import Progress, BarColumn, TimeElapsedColumn, TimeRemainingColumn, TextColumn

# Add the backend directory to sys.path (so worker subprocesses can import relative modules when they import inside worker)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

console = Console()


def ensure_data_dir(output_path):
    """Create directory if it doesn't exist"""
    dir_name = os.path.dirname(output_path)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)


def _worker_generate_and_solve(args):
    """
    Worker function executed in subprocess. Returns a list of row-dicts (one per step if save_intermediate_steps, else single).
    args: tuple(sample_idx, save_intermediate_steps, is_json, rows, cols, solver_timeout)
    """
    # Import inside worker to avoid pickling issues
    try:
        from backend.scripts.generate_random_klotski_board import generate_random_klotski_board
        from backend.scripts.board_feature_extractor import (
            calculate_board_features, count_blocking_pieces, find_goal_block
        )
        from backend.scripts.board_generator import render_klotski_board
        from backend.klotski.solver import Solver
    except Exception as e:
        # Can't import — return an error-like row so main process can log it
        return {'error': f'ImportError in worker: {e}', 'sample_idx': args[0]}

    sample_idx, save_intermediate_steps, is_json, rows, cols, solver_timeout = args

    start_time = time()

    try:
        blocks, block_counts = generate_random_klotski_board()

        # Handle nested list if returned
        if isinstance(blocks, tuple):
            blocks = blocks[0] if isinstance(blocks[0], list) else blocks
        elif isinstance(blocks[0], list):
            blocks = blocks[0]

        # Save initial block states as compact tuples
        initial_block_states = [
            (b.num_rows, b.num_cols, b.row_pos, b.col_pos)
            for b in blocks
        ]

        # Calculate board features
        board_features = calculate_board_features(blocks)
        try:
            board_features['blocks_between_goal_target'] = count_blocking_pieces(
                find_goal_block(blocks), blocks
            )
        except Exception:
            board_features['blocks_between_goal_target'] = None

        # Attempt to solve with ThreadPoolExecutor timeout (per-worker)
        def solve_puzzle(blocks_data):
            """Helper function to run solver in thread inside worker process"""
            try:
                solver = Solver(list(blocks_data), 3, 1)
                solver.solve()
                return solver.get_solution_boards()
            except Exception:
                return None

        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(solve_puzzle, blocks)
                solution_path = future.result(timeout=solver_timeout)
            solve_time = time() - start_time
        except FutureTimeoutError:
            solution_path = None
            solve_time = float(solver_timeout)
        except Exception:
            solution_path = None
            solve_time = time() - start_time

        # Solution features
        if solution_path and len(solution_path) > 0:
            solution_features = {
                'is_solvable': True,
                'solution_length': max(0, len(solution_path) - 1),
                'solve_time_seconds': solve_time,
            }
        else:
            solution_features = {
                'is_solvable': False,
                'solution_length': 0,
                'solve_time_seconds': solve_time,
            }

        # Board visual for initial state
        try:
            board_visual = render_klotski_board(blocks, rows=rows, cols=cols, return_string=True)
        except Exception:
            board_visual = None

        # Base puzzle_data dict
        puzzle_data_base = {
            'puzzle_id': f"puzzle_{sample_idx:06d}",
            'timestamp': datetime.now().isoformat(),
            **board_features,
            'board_visual': board_visual,
            **solution_features,
            'initial_block_states': initial_block_states if is_json else str(initial_block_states)
        }

        rows_out = []

        if save_intermediate_steps and solution_path and len(solution_path) > 1:
            for step_idx, step_blocks in enumerate(solution_path):
                step_visual = None
                try:
                    step_visual = render_klotski_board(step_blocks, rows=rows, cols=cols, return_string=True)
                except Exception:
                    step_visual = None

                step_data = dict(puzzle_data_base)
                step_data['step_number'] = step_idx
                step_data['board_visual'] = step_visual
                rows_out.append(step_data)
        else:
            # For consistency add step_number field if save_intermediate_steps True (but no solution)
            if save_intermediate_steps:
                puzzle_data_base['step_number'] = 0
            rows_out.append(puzzle_data_base)

        return rows_out

    except Exception as e:
        # Return a single dict describing error so main can continue
        return [{'error': f'Exception for sample {sample_idx}: {e}', 'sample_idx': sample_idx}]


def generate_enhanced_klotski_data(
    num_samples=10,
    output=None,
    show_progress=True,
    show_details=False,
    save_intermediate_steps=False,
    num_workers=None,
    rows=5,
    cols=4,
    solver_timeout=30
):
    """
    Generate comprehensive Klotski puzzle dataset with multicore processing.

    Args:
        num_samples: Number of puzzles to generate
        output: Output file path (CSV or JSON)
        show_progress: Show progress bar (rich)
        show_details: Show detailed output for each puzzle (main process prints)
        save_intermediate_steps: Save each step of solution path
        num_workers: Number of worker processes (defaults to cpu_count())
        rows, cols: board dimensions for rendering (defaults to 5x4)
        solver_timeout: per-puzzle solver timeout in seconds (default 30)
    """

    if output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
        output = os.path.join(data_dir, f"enhanced_dataset_{timestamp}.csv")

    ensure_data_dir(output)

    console.print(f"\n[bold green]Generating Enhanced Klotski Dataset[/bold green]")
    console.print(f"Samples: {num_samples}")
    console.print(f"Output: {output}")
    console.print(f"Features: block counts, goal position, adjacent blocks, solvability, solution length\n")

    is_json = output.lower().endswith('.json')

    # CSV headers (kept same as original)
    csv_headers = [
        'puzzle_id', 'timestamp', 'is_solvable',
        # Board features
        'total_blocks', 'block_count_1x1', 'block_count_1x2', 'block_count_2x1', 'block_count_2x2',
        # Goal block features
        'goal_initial_row', 'goal_initial_col', 'goal_distance_to_target', 'goal_manhattan_distance',
        'blocks_between_goal_target',
        # Adjacent block analysis
        'adjacent_1x1_count', 'adjacent_1x2_count', 'adjacent_2x1_count',
        'wall_adjacent_sides',
        # Visual representation
        'board_visual',
        # Solution features
        'solution_length', 'solve_time_seconds',
        # Step tracking (when save_intermediate_steps=True)
        'step_number',
        # New compact block states
        'initial_block_states'
    ]

    # Prepare output
    puzzles_data_json = [] 
    output_file = None
    csv_writer = None

    if is_json:
        puzzles_data_json = []
    else:
        output_file = open(output, 'w', newline='', encoding='utf-8')
        csv_writer = csv.DictWriter(output_file, fieldnames=csv_headers)
        csv_writer.writeheader()

    # Determine worker count
    if num_workers is None:
        try:
            num_workers = max(1, cpu_count())
        except Exception:
            num_workers = 1

    sample_range = list(range(num_samples))

    console.print(f"Using {num_workers} worker(s) for generation.")

    successful_generations = 0
    solvable_count = 0

    # Create arguments for each sample for the worker
    worker_args = [
        (sample_idx, save_intermediate_steps, is_json, rows, cols, solver_timeout)
        for sample_idx in sample_range
    ]

    # Use multiprocessing pool to parallelize
    try:
        with Pool(processes=num_workers) as pool:
            # imap_unordered returns results as they complete
            results_iter = pool.imap_unordered(_worker_generate_and_solve, worker_args)

            # Set up Rich progress bar
            if show_progress:
                progress = Progress(
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    "[progress.percentage]{task.percentage:>3.0f}%",
                    "•",
                    TimeElapsedColumn(),
                    "•",
                    TimeRemainingColumn(),
                    console=console,
                )
                task = progress.add_task("[green]Generating puzzles...", total=num_samples)
                progress.start()
            else:
                progress = None
                task = None

            # Consume results as they arrive
            processed = 0
            for result in results_iter:
                # result is expected to be a list of dicts (rows) OR a single error dict
                # Normalize to list
                rows_list = result if isinstance(result, list) else [result]

                for row in rows_list:
                    # If worker reported an import/exception error
                    if isinstance(row, dict) and 'error' in row:
                        console.print(f"[yellow]Worker error for sample {row.get('sample_idx', 'unknown')}: {row['error']}[/yellow]")
                        continue

                    # Write to JSON accumulation or CSV immediately
                    if is_json:
                        puzzles_data_json.append(row)
                    else:
                        # Ensure all CSV headers exist in row (fill missing)
                        csv_row = {h: row.get(h, None) for h in csv_headers}
                        csv_writer.writerow(csv_row)

                    successful_generations += 1
                    if row.get('is_solvable'):
                        solvable_count += 1

                processed += 1
                if progress:
                    progress.update(task, advance=1)
            if progress:
                progress.stop()

    except KeyboardInterrupt:
        console.print("[red]Generation interrupted by user (KeyboardInterrupt).[/red]")
    except Exception as e:
        console.print(f"[red]Fatal error during multiprocessing: {e}[/red]")
    finally:
        # Finalize output
        if is_json:
            try:
                with open(output, 'w', encoding='utf-8') as f:
                    json.dump(puzzles_data_json, f, indent=2, ensure_ascii=False)
            except Exception as e:
                console.print(f"[red]Error writing JSON output: {e}[/red']")
        else:
            try:
                if output_file:
                    output_file.close()
            except Exception as e:
                console.print(f"[red]Error closing output file: {e}[/red]")

    # Summary
    console.print(f"\n[bold green]Dataset Generation Complete![/bold green]")
    console.print(f"Successfully generated (rows written): {successful_generations}")
    console.print(f"Requested puzzles: {num_samples}")
    console.print(f"Output saved to: {output}")
    console.print(f"Features included: {len(csv_headers)} columns")
    console.print(f"Solvable puzzles (counted in-run): {solvable_count if successful_generations > 0 else 'Check file'}")

    return output


if __name__ == "__main__":
    console.print("[bold cyan]Enhanced Klotski Data Generator (multicore)[/bold cyan]")
    console.print("This will generate puzzles with comprehensive features for data science analysis.\n")

    try:
        num_samples = int(input("Number of puzzles to generate (default 100): ") or "100")
    except ValueError:
        num_samples = 100

    output_format = input("Output format (csv/json, default csv): ").strip().lower() or "csv"
    show_details = input("Show detailed progress? (y/n, default n): ").strip().lower() in ("y", "yes")
    save_steps = input("Save intermediate solution steps? (y/n, default n): ").strip().lower() in ("y", "yes")
    try:
        default_workers = max(1, cpu_count())
    except Exception:
        default_workers = 1
    try:
        workers_input = input(f"Number of worker processes (default {default_workers}): ").strip()
        num_workers = int(workers_input) if workers_input else default_workers
    except ValueError:
        num_workers = default_workers

    # Generate output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    output_file = os.path.join(data_dir, f"enhanced_dataset_{timestamp}.{output_format}")

    # Generate dataset
    generated_file = generate_enhanced_klotski_data(
        num_samples=num_samples,
        output=output_file,
        show_progress=True,
        show_details=show_details,
        save_intermediate_steps=save_steps,
        num_workers=num_workers
    )

    console.print(f"\n[bold green]Dataset ready for ML![/bold green]")
    console.print(f"Clean feature set: {generated_file}")
    console.print("Features: block counts, goal position, adjacent blocks, solvability, solution length")
