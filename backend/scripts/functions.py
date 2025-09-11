# backend/scripts/functions.py

def get_block_positions(block):
    """Get all positions occupied by a block"""
    positions = set()
    for r in range(block.row_pos, block.row_pos + block.num_rows):
        for c in range(block.col_pos, block.col_pos + block.num_cols):
            positions.add((r, c))
    return positions


def get_adjacent_positions(block, rows=5, cols=4, include_diagonals=True):
    """Get all positions adjacent to a block"""
    block_positions = get_block_positions(block)
    adjacent_positions = set()
    
    for r, c in block_positions:
        if include_diagonals:
            directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        else:
            directions = [(-1, 0), (0, -1), (0, 1), (1, 0)]
        
        for dr, dc in directions:
            adj_r, adj_c = r + dr, c + dc
            if 0 <= adj_r < rows and 0 <= adj_c < cols:
                adjacent_positions.add((adj_r, adj_c))
    
    # Remove the block's own positions
    adjacent_positions -= block_positions
    return adjacent_positions