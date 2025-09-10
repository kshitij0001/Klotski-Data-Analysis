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

console = Console()

def ensure_data_dir(output_path):
    """Create directory if it doesn't exist"""
    dir_name = os.path.dirname(output_path)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)

def capture_board_visual(blocks, rows=5, cols=4):
    """
    Generate ASCII board representation as string (similar to render_klotski_board but returns string).
    Uses 'X' for empty spaces as requested.
    """
    # Build board index grid (None for empty cells)
    board = [[None for _ in range(cols)] for _ in range(rows)]
    for idx, block in enumerate(blocks):
        for r in range(block.row_pos, block.row_pos + block.num_rows):
            for c in range(block.col_pos, block.col_pos + block.num_cols):
                board[r][c] = idx

    # Helper to check same block
    def same_block(r1, c1, r2, c2):
        if not (0 <= r1 < rows and 0 <= c1 < cols and 0 <= r2 < rows and 0 <= c2 < cols):
            return False
        return board[r1][c1] == board[r2][c2] and board[r1][c1] is not None

    H, V = "─", "│"

    # Horizontal boundaries
    h = [[False for _ in range(cols)] for _ in range(rows + 1)]
    for c in range(cols):
        h[0][c] = True
        h[rows][c] = True
    for r_edge in range(1, rows):
        for c in range(cols):
            h[r_edge][c] = board[r_edge - 1][c] != board[r_edge][c]

    # Vertical boundaries
    v = [[False for _ in range(cols + 1)] for _ in range(rows)]
    for r in range(rows):
        v[r][0] = True
        v[r][cols] = True
        for c_edge in range(1, cols):
            v[r][c_edge] = board[r][c_edge - 1] != board[r][c_edge]

    # Junction char chooser
    def junction_char(up, down, left, right):
        if up and down and left and right: return "┼"
        if up and down and right and not left: return "├"
        if up and down and left and not right: return "┤"
        if left and right and down and not up: return "┬"
        if left and right and up and not down: return "┴"
        if down and right and not up and not left: return "┌"
        if down and left and not up and not right: return "┐"
        if up and right and not down and not left: return "└"
        if up and left and not down and not right: return "┘"
        if left and right and not up and not down: return H
        if up and down and not left and not right: return V
        if right: return H
        if left: return H
        if up: return V
        if down: return V
        return " "

    # Cell content chooser (simplified, no colors)
    def cell_symbol(block_idx):
        if block_idx is None:
            return " X "
        # Just return spaces for occupied cells since we're focusing on structure
        return "   "

    # Build the visual string
    output_lines = []
    
    for r_node in range(rows + 1):
        # Junction line
        line = ""
        for c_node in range(cols + 1):
            hor_left = (c_node > 0 and h[r_node][c_node - 1])
            hor_right = (c_node < cols and h[r_node][c_node])
            ver_up = (r_node > 0 and v[r_node - 1][c_node]) if r_node > 0 else False
            ver_down = (r_node < rows and v[r_node][c_node]) if r_node < rows else False
            j = junction_char(ver_up, ver_down, hor_left, hor_right)
            line += j
            if c_node < cols:
                line += (H * 3) if h[r_node][c_node] else "   "
        output_lines.append(line)

        # Content line
        if r_node < rows:
            content = ""
            for c in range(cols):
                content += (V if v[r_node][c] else " ")
                content += cell_symbol(board[r_node][c])
            content += (V if v[r_node][cols] else " ")
            output_lines.append(content)

    return "\n".join(output_lines)

def calculate_distance(pos1, pos2):
    """Calculate Euclidean distance between two positions"""
    return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

def calculate_manhattan_distance(pos1, pos2):
    """Calculate Manhattan distance between two positions"""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def get_block_type_string(block):
    """Get block type as string (e.g., '2x2', '1x1')"""
    return f"{block.num_rows}x{block.num_cols}"

def find_goal_block(blocks):
    """Find the 2x2 goal block"""
    for block in blocks:
        if block.num_rows == 2 and block.num_cols == 2:
            return block
    return None

def get_adjacent_block_types(goal_block, all_blocks, rows=5, cols=4):
    """
    Analyze what types of blocks are adjacent to the goal block.
    Returns counts and percentages of each block type touching the goal block.
    """
    if not goal_block:
        return {}
    
    # Get all positions occupied by goal block
    goal_positions = set()
    for r in range(goal_block.row_pos, goal_block.row_pos + goal_block.num_rows):
        for c in range(goal_block.col_pos, goal_block.col_pos + goal_block.num_cols):
            goal_positions.add((r, c))
    
    # Get all adjacent positions (including diagonals)
    adjacent_positions = set()
    for r, c in goal_positions:
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:  # Skip the goal block itself
                    continue
                adj_r, adj_c = r + dr, c + dc
                if 0 <= adj_r < rows and 0 <= adj_c < cols:
                    adjacent_positions.add((adj_r, adj_c))
    
    # Remove goal block positions from adjacent
    adjacent_positions -= goal_positions
    
    # Find which blocks occupy adjacent positions
    adjacent_block_types = []
    for block in all_blocks:
        if block == goal_block:
            continue
        
        # Check if this block occupies any adjacent position
        block_positions = set()
        for r in range(block.row_pos, block.row_pos + block.num_rows):
            for c in range(block.col_pos, block.col_pos + block.num_cols):
                block_positions.add((r, c))
        
        if block_positions.intersection(adjacent_positions):
            adjacent_block_types.append(get_block_type_string(block))
    
    # Count adjacent block types
    type_counts = Counter(adjacent_block_types)
    total_adjacent = len(adjacent_block_types)
    
    result = {
        'adjacent_1x1_count': type_counts.get('1x1', 0),
        'adjacent_1x2_count': type_counts.get('1x2', 0),
        'adjacent_2x1_count': type_counts.get('2x1', 0),
        'adjacent_2x2_count': type_counts.get('2x2', 0),
        'total_adjacent_blocks': total_adjacent,
        'adjacent_1x1_ratio': type_counts.get('1x1', 0) / max(total_adjacent, 1),
        'adjacent_1x2_ratio': type_counts.get('1x2', 0) / max(total_adjacent, 1),
        'adjacent_2x1_ratio': type_counts.get('2x1', 0) / max(total_adjacent, 1),
        'wall_adjacent_sides': calculate_wall_adjacency(goal_block, rows, cols)
    }
    
    return result

def calculate_wall_adjacency(goal_block, rows=5, cols=4):
    """Calculate how many sides of the goal block are touching walls"""
    if not goal_block:
        return 0
    
    wall_sides = 0
    
    # Check top wall
    if goal_block.row_pos == 0:
        wall_sides += 1
    
    # Check bottom wall
    if goal_block.row_pos + goal_block.num_rows == rows:
        wall_sides += 1
    
    # Check left wall
    if goal_block.col_pos == 0:
        wall_sides += 1
    
    # Check right wall
    if goal_block.col_pos + goal_block.num_cols == cols:
        wall_sides += 1
    
    return wall_sides

def calculate_board_features(blocks, rows=5, cols=4):
    """Calculate comprehensive board complexity features"""
    
    # Block type counts
    block_counts = Counter()
    for block in blocks:
        block_type = get_block_type_string(block)
        block_counts[block_type] += 1
    
    # Calculate occupied cells
    occupied_cells = sum(block.num_rows * block.num_cols for block in blocks)
    total_cells = rows * cols
    
    # Find goal block
    goal_block = find_goal_block(blocks)
    
    # Calculate features
    features = {
        # Basic counts
        'total_blocks': len(blocks),
        'block_count_1x1': block_counts.get('1x1', 0),
        'block_count_1x2': block_counts.get('1x2', 0),
        'block_count_2x1': block_counts.get('2x1', 0),
        'block_count_2x2': block_counts.get('2x2', 0),
        
        # Density metrics
        'board_density': occupied_cells / total_cells,
        'empty_spaces': total_cells - occupied_cells,
        
        # Goal block analysis
        'goal_initial_row': goal_block.row_pos if goal_block else -1,
        'goal_initial_col': goal_block.col_pos if goal_block else -1,
        'goal_distance_to_target': calculate_distance(
            (goal_block.row_pos, goal_block.col_pos), (3, 1)
        ) if goal_block else -1,
        'goal_manhattan_distance': calculate_manhattan_distance(
            (goal_block.row_pos, goal_block.col_pos), (3, 1)
        ) if goal_block else -1,
        
        # Board geometry
        'blocks_touching_walls': sum(1 for block in blocks if is_touching_wall(block, rows, cols)),
        'corner_blocks': sum(1 for block in blocks if is_in_corner(block, rows, cols)),
        'center_density': calculate_center_density(blocks, rows, cols),
        'edge_density': calculate_edge_density(blocks, rows, cols),
    }
    
    # Add adjacent block analysis
    if goal_block:
        adjacent_features = get_adjacent_block_types(goal_block, blocks, rows, cols)
        features.update(adjacent_features)
    else:
        # Default values if no goal block
        features.update({
            'adjacent_1x1_count': 0,
            'adjacent_1x2_count': 0,
            'adjacent_2x1_count': 0,
            'adjacent_2x2_count': 0,
            'total_adjacent_blocks': 0,
            'adjacent_1x1_ratio': 0,
            'adjacent_1x2_ratio': 0,
            'adjacent_2x1_ratio': 0,
            'wall_adjacent_sides': 0
        })
    
    return features

def is_touching_wall(block, rows, cols):
    """Check if block is touching any wall"""
    return (block.row_pos == 0 or 
            block.row_pos + block.num_rows == rows or
            block.col_pos == 0 or 
            block.col_pos + block.num_cols == cols)

def is_in_corner(block, rows, cols):
    """Check if block is in a corner"""
    corners = [
        (block.row_pos == 0 and block.col_pos == 0),
        (block.row_pos == 0 and block.col_pos + block.num_cols == cols),
        (block.row_pos + block.num_rows == rows and block.col_pos == 0),
        (block.row_pos + block.num_rows == rows and block.col_pos + block.num_cols == cols)
    ]
    return any(corners)

def calculate_center_density(blocks, rows, cols):
    """Calculate density in center area of board"""
    center_cells = 0
    occupied_center = 0
    
    # Define center as middle 2x2 area
    center_start_row = (rows - 2) // 2
    center_start_col = (cols - 2) // 2
    
    for r in range(center_start_row, center_start_row + 2):
        for c in range(center_start_col, center_start_col + 2):
            center_cells += 1
            # Check if any block occupies this position
            for block in blocks:
                if (block.row_pos <= r < block.row_pos + block.num_rows and
                    block.col_pos <= c < block.col_pos + block.num_cols):
                    occupied_center += 1
                    break
    
    return occupied_center / center_cells if center_cells > 0 else 0

def calculate_edge_density(blocks, rows, cols):
    """Calculate density along board edges"""
    edge_cells = 0
    occupied_edge = 0
    
    for r in range(rows):
        for c in range(cols):
            # Check if this is an edge cell
            if r == 0 or r == rows - 1 or c == 0 or c == cols - 1:
                edge_cells += 1
                # Check if any block occupies this position
                for block in blocks:
                    if (block.row_pos <= r < block.row_pos + block.num_rows and
                        block.col_pos <= c < block.col_pos + block.num_cols):
                        occupied_edge += 1
                        break
    
    return occupied_edge / edge_cells if edge_cells > 0 else 0

def count_blocking_pieces(goal_block, all_blocks, target_row=3, target_col=1):
    """Count how many pieces are between goal block and target position"""
    if not goal_block:
        return 0
    
    # Simple heuristic: count blocks in the path rectangle
    min_row = min(goal_block.row_pos, target_row)
    max_row = max(goal_block.row_pos + goal_block.num_rows - 1, target_row + 1)
    min_col = min(goal_block.col_pos, target_col)
    max_col = max(goal_block.col_pos + goal_block.num_cols - 1, target_col + 1)
    
    blocking_count = 0
    for block in all_blocks:
        if block == goal_block:
            continue
        
        # Check if block intersects with the path rectangle
        if (block.row_pos <= max_row and block.row_pos + block.num_rows > min_row and
            block.col_pos <= max_col and block.col_pos + block.num_cols > min_col):
            blocking_count += 1
    
    return blocking_count

def analyze_solution_patterns(solution_path):
    """Analyze patterns in the solution path"""
    if not solution_path or len(solution_path) < 2:
        return {
            'horizontal_moves': 0,
            'vertical_moves': 0,
            'goal_piece_moves': 0,
            'move_sequence_entropy': 0,
            'backtrack_count': 0,
            'unique_positions': 0
        }
    
    horizontal_moves = 0
    vertical_moves = 0
    goal_piece_moves = 0
    position_hashes = set()
    
    for i in range(1, len(solution_path)):
        prev_state = solution_path[i-1]
        curr_state = solution_path[i]
        
        # Create position hash for uniqueness
        pos_hash = tuple(tuple((b.row_pos, b.col_pos, b.num_rows, b.num_cols)) for b in curr_state)
        position_hashes.add(pos_hash)
        
        # Find which block moved
        moved_block = None
        for j, (prev_block, curr_block) in enumerate(zip(prev_state, curr_state)):
            if (prev_block.row_pos != curr_block.row_pos or 
                prev_block.col_pos != curr_block.col_pos):
                moved_block = curr_block
                break
        
        if moved_block:
            # Determine move direction
            prev_block = prev_state[j]
            if prev_block.row_pos != moved_block.row_pos:
                vertical_moves += 1
            if prev_block.col_pos != moved_block.col_pos:
                horizontal_moves += 1
            
            # Check if goal piece moved
            if moved_block.num_rows == 2 and moved_block.num_cols == 2:
                goal_piece_moves += 1
    
    # Calculate entropy (simplified)
    total_moves = horizontal_moves + vertical_moves
    if total_moves > 0:
        h_ratio = horizontal_moves / total_moves
        v_ratio = vertical_moves / total_moves
        entropy = -(h_ratio * math.log2(h_ratio + 1e-10) + v_ratio * math.log2(v_ratio + 1e-10))
    else:
        entropy = 0
    
    return {
        'horizontal_moves': horizontal_moves,
        'vertical_moves': vertical_moves,
        'goal_piece_moves': goal_piece_moves,
        'move_sequence_entropy': entropy,
        'backtrack_count': len(solution_path) - len(position_hashes),  # Approximation
        'unique_positions': len(position_hashes)
    }

def calculate_difficulty_score(board_features, solution_features):
    """Calculate a composite difficulty score (0-10)"""
    if not solution_features.get('is_solvable', False):
        return 10.0
    
    # Weighted factors
    length_factor = min(solution_features.get('solution_length', 0) / 50.0, 1.0) * 3
    density_factor = board_features.get('board_density', 0) * 2
    blocking_factor = min(board_features.get('blocks_between_goal_target', 0) / 5.0, 1.0) * 2
    goal_distance_factor = min(board_features.get('goal_distance_to_target', 0) / 6.0, 1.0) * 2
    adjacent_factor = board_features.get('total_adjacent_blocks', 0) / 8.0 * 1
    
    difficulty = length_factor + density_factor + blocking_factor + goal_distance_factor + adjacent_factor
    return min(difficulty, 10.0)

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
        'solution_length', 'solve_time_seconds', 'horizontal_moves', 'vertical_moves',
        'goal_piece_moves', 'move_sequence_entropy', 'backtrack_count', 'unique_positions',
        # Difficulty metrics
        'difficulty_score', 'cognitive_load_estimate', 'algorithmic_complexity'
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
                
                # Analyze solution patterns
                pattern_features = analyze_solution_patterns(solution_path)
                solution_features.update(pattern_features)
            else:
                solution_features = {
                    'is_solvable': False,
                    'solution_length': 0,
                    'solve_time_seconds': solve_time,
                    'horizontal_moves': 0,
                    'vertical_moves': 0,
                    'goal_piece_moves': 0,
                    'move_sequence_entropy': 0,
                    'backtrack_count': 0,
                    'unique_positions': 0
                }
            
            # Calculate difficulty scores
            difficulty_score = calculate_difficulty_score(board_features, solution_features)
            
            # Capture board visual
            board_visual = capture_board_visual(blocks)
            
            # Combine all features
            puzzle_data = {
                'puzzle_id': f"puzzle_{sample_idx:06d}",
                'timestamp': datetime.now().isoformat(),
                **board_features,
                'board_visual': board_visual,
                **solution_features,
                'difficulty_score': round(difficulty_score, 2),
                'cognitive_load_estimate': round(difficulty_score * 1.1 + 0.5, 2),
                'algorithmic_complexity': round(difficulty_score * 0.9 + 0.3, 2)
            }
            
            # Handle intermediate steps if requested
            if save_intermediate_steps and solution_path and len(solution_path) > 1:
                # Save each step of the solution
                for step_idx, step_blocks in enumerate(solution_path):
                    step_data = puzzle_data.copy()
                    step_data['step_number'] = step_idx
                    step_data['board_visual'] = capture_board_visual(step_blocks)
                    
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
                console.print(f"Difficulty: {puzzle_data['difficulty_score']}/10")
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