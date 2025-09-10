# backend/scripts/enhanced_data_generator.py

import sys
import os
import json
import csv
import math
from copy import deepcopy
from datetime import datetime
from time import time, sleep
from collections import Counter

# Rich imports
from rich.console import Console
from rich.progress import track
from rich.table import Table

# Add the backend directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from backend.klotski.utils import Block
from backend.klotski.solver import Solver
from backend.scripts.generate_random_klotski_board import generate_random_klotski_board
from backend.scripts.board_generator import render_klotski_board
from backend.scripts.board_feature_extractor import (
    calculate_board_features, count_blocking_pieces, find_goal_block
)

console = Console()

def ensure_data_dir(output_path):
    """Create directory if it doesn't exist"""
    dir_name = os.path.dirname(output_path)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)




def generate_enhanced_klotski_data(num_samples=10, output=None, save_intermediate_steps=False, show_progress=True, show_details=False):
    """
    Generate comprehensive Klotski puzzle dataset with rich features.
    
    Args:
        num_samples: Number of puzzles to generate
        output: Output file path (CSV or JSON)
        show_progress: Show progress bar
        show_details: Show detailed output for each puzzle
    """
    
    if output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = f"backend/data/enhanced_dataset_{timestamp}.csv"
    
    ensure_data_dir(output)
    
    console.print(f"\n[bold green]Generating Enhanced Klotski Dataset[/bold green]")
    console.print(f"Samples: {num_samples}")
    console.print(f"Output: {output}")
    console.print(f"Features: Board complexity, solution metrics, adjacent block analysis\n")
    
    # Determine output format
    is_json = output.lower().endswith('.json')
    
    # Storage for all puzzle data
    all_puzzles = []
    
    # CSV headers
    csv_headers = [
        'puzzle_id', 'timestamp', 'is_solvable',
        # Board features
        'total_blocks', 'block_count_1x1', 'block_count_1x2', 'block_count_2x1', 'block_count_2x2',
        'board_density', 'empty_spaces', 'blocks_touching_walls', 'corner_blocks',
        'center_density', 'edge_density',
        # Goal block features
        'goal_initial_row', 'goal_initial_col', 'goal_distance_to_target', 'goal_manhattan_distance',
        'blocks_between_goal_target',
        # Adjacent block analysis
        'adjacent_1x1_count', 'adjacent_1x2_count', 'adjacent_2x1_count', 'adjacent_2x2_count',
        'total_adjacent_blocks', 'adjacent_1x1_ratio', 'adjacent_1x2_ratio', 'adjacent_2x1_ratio',
        'wall_adjacent_sides',
        # Visual representation
        'board_visual',
        # Solution features
        'solution_length', 'solve_time_seconds'
    ]
    
    # Add step column if saving intermediate steps
    if save_intermediate_steps:
        csv_headers.insert(2, 'step_number')  # Insert after timestamp
    
    # Open output file
    if is_json:
        puzzles_data = []
    else:
        output_file = open(output, 'w', newline='')
        csv_writer = csv.DictWriter(output_file, fieldnames=csv_headers)
        csv_writer.writeheader()
    
    # Generate puzzles
    sample_range = range(num_samples)
    iterator = sample_range if not show_progress else track(sample_range, description="Generating puzzles...")
    
    successful_generations = 0
    
    for sample_idx in iterator:
        try:
            # Generate puzzle
            start_time = time()
            blocks, block_counts = generate_random_klotski_board()
            
            # Handle nested list if returned
            if isinstance(blocks, tuple):
                blocks = blocks[0] if isinstance(blocks[0], list) else blocks
            elif isinstance(blocks[0], list):
                blocks = blocks[0]
            
            # Calculate board features
            board_features = calculate_board_features(blocks)
            board_features['blocks_between_goal_target'] = count_blocking_pieces(
                find_goal_block(blocks), blocks
            )
            
            # Attempt to solve
            solver = Solver(deepcopy(blocks), 3, 1)  # Target position (3,1)
            solver.solve()
            solution_path = solver.get_solution_boards()
            solve_time = time() - start_time
            
            # Calculate solution features
            if solution_path and len(solution_path) > 0:
                solution_features = {
                    'is_solvable': True,
                    'solution_length': len(solution_path) - 1,
                    'solve_time_seconds': solve_time,
                }
            else:
                solution_features = {
                    'is_solvable': False,
                    'solution_length': 0,
                    'solve_time_seconds': solve_time,
                }
            
            # Capture board visual using the shared board generator
            board_visual = render_klotski_board(blocks, rows=5, cols=4, return_string=True)
            
            # Combine all features
            puzzle_data = {
                'puzzle_id': f"puzzle_{sample_idx:06d}",
                'timestamp': datetime.now().isoformat(),
                **board_features,
                'board_visual': board_visual,
                **solution_features
            }
            
            # Handle intermediate steps if requested
            if save_intermediate_steps and solution_path and len(solution_path) > 1:
                # Save each step of the solution
                for step_idx, step_blocks in enumerate(solution_path):
                    step_data = puzzle_data.copy()
                    step_data['step_number'] = step_idx
                    step_data['board_visual'] = render_klotski_board(step_blocks, rows=5, cols=4, return_string=True)
                    
                    # Save step data
                    if is_json:
                        puzzles_data.append(step_data)
                    else:
                        csv_writer.writerow(step_data)
            else:
                # Save just initial state
                if save_intermediate_steps:
                    puzzle_data['step_number'] = 0
                
                # Save data (only when not saving intermediate steps, or for unsolvable puzzles)
                if is_json:
                    puzzles_data.append(puzzle_data)
                else:
                    csv_writer.writerow(puzzle_data)
            
            successful_generations += 1
            
            # Show details if requested
            if show_details:
                console.print(f"\n[bold blue]Puzzle {sample_idx + 1}[/bold blue]")
                console.print(f"Solvable: {puzzle_data['is_solvable']}")
                console.print(f"Adjacent blocks to goal: {puzzle_data['total_adjacent_blocks']}")
                if puzzle_data['is_solvable']:
                    console.print(f"Solution length: {puzzle_data['solution_length']} moves")
                render_klotski_board(blocks)
                sleep(0.5)
        
        except Exception as e:
            console.print(f"[red]Error generating puzzle {sample_idx}: {e}[/red]")
            continue
    
    # Finalize output
    if is_json:
        with open(output, 'w') as f:
            json.dump(puzzles_data, f, indent=2)
    else:
        output_file.close()
    
    # Summary
    console.print(f"\n[bold green]Dataset Generation Complete![/bold green]")
    console.print(f"Successfully generated: {successful_generations}/{num_samples} puzzles")
    console.print(f"Output saved to: {output}")
    console.print(f"Features included: {len(csv_headers)} columns")
    
    # Quick stats
    if successful_generations > 0:
        solvable_count = sum(1 for p in (puzzles_data if is_json else []) if p.get('is_solvable', False))
        console.print(f"Solvable puzzles: {solvable_count if is_json else 'Check file'}")
    
    return output

if __name__ == "__main__":
    # Interactive setup
    console.print("[bold cyan]Enhanced Klotski Data Generator[/bold cyan]")
    console.print("This will generate puzzles with comprehensive features for data science analysis.\n")
    
    try:
        num_samples = int(input("Number of puzzles to generate (default 100): ") or "100")
    except ValueError:
        num_samples = 100
    
    output_format = input("Output format (csv/json, default csv): ").strip().lower() or "csv"
    save_steps = input("Save intermediate solution steps? (y/n, default n): ").strip().lower() in ("y", "yes")
    show_details = input("Show detailed progress? (y/n, default n): ").strip().lower() in ("y", "yes")
    
    # Generate output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"backend/data/enhanced_dataset_{timestamp}.{output_format}"
    
    # Generate dataset
    generated_file = generate_enhanced_klotski_data(
        num_samples=num_samples,
        output=output_file,
        save_intermediate_steps=save_steps,
        show_progress=True,
        show_details=show_details
    )



    console.print(f"\n[bold green]Ready for EDA![/bold green]")
    console.print(f"Enhanced dataset is ready at: {generated_file}")
    console.print("Key features include:")
    console.print("• Board complexity metrics")
    console.print("• Goal block positioning analysis") 
    console.print("• [bold yellow]Adjacent block types to red block[/bold yellow]")
    console.print("• Solution difficulty scoring")
    console.print("• Move pattern analysis")