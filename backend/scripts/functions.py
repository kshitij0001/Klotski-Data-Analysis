# backend/utils/board_utils.py

def get_block_positions(block):
    """Get all positions occupied by a block"""
    positions = set()
    for r in range(block.row_pos, block.row_pos + block.num_rows):
        for c in range(block.col_pos, block.col_pos + block.num_cols):
            positions.add((r, c))
    return positions

def is_position_occupied_by_any_block(r, c, blocks):
    """Check if a position is occupied by any block"""
    for block in blocks:
        if (block.row_pos <= r < block.row_pos + block.num_rows and
            block.col_pos <= c < block.col_pos + block.num_cols):
            return True
    return False

def calculate_area_density(blocks, area_positions):
    """Calculate density for a specific area defined by positions"""
    if not area_positions:
        return 0
    
    occupied_count = 0
    for r, c in area_positions:
        if is_position_occupied_by_any_block(r, c, blocks):
            occupied_count += 1
    
    return occupied_count / len(area_positions)

def get_center_positions(rows=5, cols=4):
    """Get positions in the center 2x2 area of the board"""
    center_start_row = (rows - 2) // 2
    center_start_col = (cols - 2) // 2
    
    positions = set()
    for r in range(center_start_row, center_start_row + 2):
        for c in range(center_start_col, center_start_col + 2):
            positions.add((r, c))
    return positions

def get_edge_positions(rows=5, cols=4):
    """Get all edge positions of the board"""
    positions = set()
    for r in range(rows):
        for c in range(cols):
            if r == 0 or r == rows - 1 or c == 0 or c == cols - 1:
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