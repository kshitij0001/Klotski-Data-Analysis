# backend/scripts/board_feature_extractor.py

import math
from collections import Counter
from backend.scripts.functions import (
    get_block_positions, get_adjacent_positions, calculate_area_density,
    get_center_positions, get_edge_positions
)

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
    
    # Get all adjacent positions using utility function
    adjacent_positions = get_adjacent_positions(goal_block, rows, cols, include_diagonals=True)
    
    # Find which blocks occupy adjacent positions
    adjacent_block_types = []
    for block in all_blocks:
        if block == goal_block:
            continue
        
        # Check if this block occupies any adjacent position
        block_positions = get_block_positions(block)
        
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
    center_positions = get_center_positions(rows, cols)
    return calculate_area_density(blocks, center_positions)

def calculate_edge_density(blocks, rows, cols):
    """Calculate density along board edges"""
    edge_positions = get_edge_positions(rows, cols)
    return calculate_area_density(blocks, edge_positions)

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