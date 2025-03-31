# main.py
import numpy as np
from ai_solver import solve_generated_level

def read_input(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    size = tuple(map(int, lines[0].strip().split()))
    start = tuple(map(int, lines[1].strip().split()))
    goal = tuple(map(int, lines[2].strip().split()))
    dungeon = [list(map(int, line.strip().split())) for line in lines[3:]]
    return np.array(dungeon), start, goal

def write_output(filename, path, outcome):
    with open(filename, 'w') as f:
        f.write("Path:\n")
        for step in path:
            f.write(f"{step}\n")
        f.write("\nOutcome:\n")
        f.write(outcome)

if __name__ == "__main__":
    dungeon, start, goal = read_input("input.txt")
    path, outcome = solve_generated_level(dungeon, start, goal)
    write_output("output.txt", path, outcome)
