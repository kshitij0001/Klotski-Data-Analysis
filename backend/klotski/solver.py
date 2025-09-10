# # backend/klotski/solver.py

from typing import List, Optional, Set, Callable, Dict, Any
from .board import Board
from .utils import Block, Move
import heapq
import csv
import json

class TreeNode:
    """
    Node in the search tree for Klotski.
    """
    def __init__(self, board: Board, parent: Optional['TreeNode'] = None, move: Optional[Move] = None, g_cost: int = 0):
        self.board: Board = board
        self.parent: Optional['TreeNode'] = parent
        self.move: Optional[Move] = move
        self.g_cost: int = g_cost  # Cost from root to this node

    def get_children(self, seen_hashes: Set[str]) -> List['TreeNode']:
        children = []
        for move in self.board.get_moves():
            try:
                child_board = self._apply_move(move)
                board_hash = child_board.get_hash()
                if board_hash not in seen_hashes:
                    seen_hashes.add(board_hash)
                    children.append(TreeNode(child_board, self, move, self.g_cost + 1))
            except ValueError:
                continue
        return children

    def _apply_move(self, move: Move) -> Board:
        new_blocks = []
        for block in self.board.blocks:
            new_block = Block(
                num_rows=block.num_rows,
                num_cols=block.num_cols,
                row_pos=block.row_pos,
                col_pos=block.col_pos
            )
            if (block.row_pos == move.block.row_pos and
                block.col_pos == move.block.col_pos and
                block.num_rows == move.block.num_rows and
                block.num_cols == move.block.num_cols):
                from .utils import move_block
                move_block(new_block, move.dirs)
            new_blocks.append(new_block)
        return Board(new_blocks, self.board.winning_row, self.board.winning_col)

    def get_path(self) -> List['TreeNode']:
        path = []
        node = self
        while node is not None:
            path.append(node)
            node = node.parent
        return list(reversed(path))

class Solver:
    """
    Klotski solver supporting BFS and A* (with optional heuristic).
    Logs all visited states and transitions for data analysis.
    """
    def __init__(self, blocks: List[Block], winning_row: int, winning_col: int, heuristic: Optional[Callable[[Board], int]] = None):
        self.root = TreeNode(Board(blocks, winning_row, winning_col))
        self.solution_node: Optional[TreeNode] = None
        self.heuristic = heuristic
        self.visited_log: List[Dict[str, Any]] = []  # For data export
        self.nodes_expanded: int = 0

    def solve(self) -> None:
        """
        Runs BFS (if heuristic is None) or A* (if heuristic is provided).
        Logs all visited states and transitions.
        """
        if self.heuristic is None:
            # BFS
            from collections import deque
            seen_hashes: Set[str] = set()
            queue = deque()
            queue.append(self.root)
            seen_hashes.add(self.root.board.get_hash())
            while queue:
                node = queue.popleft()
                self._log_node(node)
                self.nodes_expanded += 1
                if node.board.is_solved():
                    self.solution_node = node
                    return
                for child in node.get_children(seen_hashes):
                    queue.append(child)
        else:
            # A* Search
            seen_hashes: Set[str] = set()
            heap = []
            root_f = self.heuristic(self.root.board)
            heapq.heappush(heap, (root_f, 0, self.root))
            seen_hashes.add(self.root.board.get_hash())
            while heap:
                _, _, node = heapq.heappop(heap)
                self._log_node(node)
                self.nodes_expanded += 1
                if node.board.is_solved():
                    self.solution_node = node
                    return
                for child in node.get_children(seen_hashes):
                    f = child.g_cost + self.heuristic(child.board)
                    heapq.heappush(heap, (f, child.g_cost, child))
        # If no solution found, solution_node remains None

    def _log_node(self, node: TreeNode) -> None:
        """
        Logs the node's state for data analysis.
        """
        entry = {
            "hash": node.board.get_hash(),
            "depth": node.g_cost,
            "parent_hash": node.parent.board.get_hash() if node.parent else None,
            "move": self._move_to_dict(node.move),
            "is_solution": node.board.is_solved()
        }
        self.visited_log.append(entry)

    @staticmethod
    def _move_to_dict(move: Optional[Move]) -> Optional[Dict[str, Any]]:
        if move is None:
            return None
        return {
            "block": {
                "num_rows": move.block.num_rows,
                "num_cols": move.block.num_cols,
                "row_pos": move.block.row_pos,
                "col_pos": move.block.col_pos
            },
            "dirs": [d.name for d in move.dirs]
        }

    def get_solution_boards(self) -> List[List[Block]]:
        """
        Returns the sequence of board block lists from start to solution.
        """
        if self.solution_node is None:
            return []
        path = []
        node = self.solution_node
        while node is not None:
            path.append([
                Block(
                    num_rows=block.num_rows,
                    num_cols=block.num_cols,
                    row_pos=block.row_pos,
                    col_pos=block.col_pos
                ) for block in node.board.blocks
            ])
            node = node.parent
        path.reverse()
        return path

    def export_log_csv(self, filename: str) -> None:
        """
        Exports the visited log as a CSV file.
        """
        if not self.visited_log:
            print("No log to export.")
            return
        keys = self.visited_log[0].keys()
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for row in self.visited_log:
                writer.writerow(row)

    def export_log_jsonl(self, filename: str) -> None:
        """
        Exports the visited log as a JSONL file.
        """
        with open(filename, 'w') as f:
            for row in self.visited_log:
                f.write(json.dumps(row) + '\n')

    def get_statistics(self) -> Dict[str, Any]:
        """
        Returns statistics about the search.
        """
        return {
            "nodes_expanded": self.nodes_expanded,
            "solution_length": len(self.get_solution_boards()) - 1 if self.solution_node else None,
            "solution_found": self.solution_node is not None
        }