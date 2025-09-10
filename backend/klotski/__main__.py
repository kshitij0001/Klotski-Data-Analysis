# backend/klotski/__main__.py

import sys
import os

# Add the project root directory to sys.path for absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


import argparse
from backend.klotski.utils import Block
from backend.klotski.solver import Solver

def main():
    parser = argparse.ArgumentParser(description="Klotski Puzzle Solver CLI")
    parser.add_argument('--winning_row', type=int, default=3, help='Winning block top-left row')
    parser.add_argument('--winning_col', type=int, default=1, help='Winning block top-left col')
    parser.add_argument('--preset', action='store_true', help='Use default puzzle preset')
    args = parser.parse_args()

    if args.preset:
        blocks = [
            Block(2,2,0,1),
            Block(2,1,2,0),
            Block(2,1,2,3),
            Block(1,1,0,0),
            Block(1,1,0,3),
            Block(1,1,1,0),
            Block(1,1,1,3),
            Block(1,1,2,1),
            Block(1,1,2,2),
            Block(1,1,3,1),
            Block(1,1,3,2),
            Block(1,1,4,0),
            Block(1,1,4,3),
        ]
    else:
        print("Please provide a puzzle or use --preset.")
        return

    solver = Solver(blocks, args.winning_row, args.winning_col)
    solver.solve()
    solution = solver.get_solution_boards()
    print(f"Solution found in {len(solution)-1} moves!" if solution else "No solution found.")

if __name__ == "__main__":
    main()