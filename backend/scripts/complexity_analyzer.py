# backend/script/complexity_analyzer.py

import math
from collections import Counter
    
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

def calculate_cognitive_load(difficulty_score):
    """Estimate cognitive load based on difficulty score"""
    return round(difficulty_score * 1.1 + 0.5, 2)

def calculate_algorithmic_complexity(difficulty_score):
    """Estimate algorithmic complexity based on difficulty score"""
    return round(difficulty_score * 0.9 + 0.3, 2)

def analyze_puzzle_complexity(board_features, solution_features):
    """
    Comprehensive complexity analysis for Klotski puzzles
    
    Returns:
        dict: Complete complexity metrics including difficulty, cognitive load, and algorithmic complexity
    """
    difficulty_score = calculate_difficulty_score(board_features, solution_features)
    
    return {
        'difficulty_score': round(difficulty_score, 2),
        'cognitive_load_estimate': calculate_cognitive_load(difficulty_score),
        'algorithmic_complexity': calculate_algorithmic_complexity(difficulty_score),
        'complexity_factors': {
            'solution_length_factor': min(solution_features.get('solution_length', 0) / 50.0, 1.0) * 3,
            'density_factor': board_features.get('board_density', 0) * 2,
            'blocking_factor': min(board_features.get('blocks_between_goal_target', 0) / 5.0, 1.0) * 2,
            'goal_distance_factor': min(board_features.get('goal_distance_to_target', 0) / 6.0, 1.0) * 2,
            'adjacent_factor': board_features.get('total_adjacent_blocks', 0) / 8.0 * 1
        }
    }