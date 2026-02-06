# Wordle Solver

A collection of various Wordle solving algorithms implemented in C++ and Python, along with analysis tools for Wordle word usage patterns.

## Overview

This project contains multiple approaches to solving Wordle puzzles, ranging from simple greedy strategies to advanced dynamic programming techniques. It also includes tools for analyzing Wordle's historical answer patterns and word usage statistics.

## Solvers

### C++ Implementations

All C++ solvers are located in the `src/` directory:

- **`wordle_dp_solver.cpp`** - Dynamic programming solver that maximizes probability of success
- **`wordle_beam_solver.cpp`** - Beam search solver
- **`wordle_greedy_solver.cpp`** - Greedy algorithm solver
- **`wordle_monte_carlo_solver.cpp`** - Monte Carlo simulation-based solver
- **`wordle_random_play_solver.cpp`** - Random play solver (baseline)
- **`wordle_unused_first_solver.cpp`** - Solver that prioritizes unused words

### Python Implementations

Python scripts are located in the `scripts/` directory:

- **`wordle_dp_solver.py`** - Python implementation of the dynamic programming solver
- **`analyze_wordle_usage.py`** - Analyzes which words have been used as Wordle answers
- **`analyze_letter_independence.py`** - Analyzes letter independence patterns
- **`extract_wordle_data.py`** - Extracts Wordle data from HTML sources
- **`extract_wordle_words.py`** - Extracts word lists from Wordle data

## Data Files

The `data/` directory contains:
- `wordle_answers.csv` - Historical Wordle answers
- `wordle_words.txt` - Complete list of valid Wordle words
- `wordle_answer_list.html` - HTML source for answer list
- `historical_words.html` - HTML source for historical words

## Reports

The `reports/` directory contains analysis reports:
- `wordle_usage_report.txt` - Analysis of which words have been used as answers
- `letter_independence_report.txt` - Letter independence analysis

## Building

### C++ Solvers

The easiest way to build all solvers is using the provided Makefile:

```bash
make
```

This will compile all C++ solvers in the `src/` directory and place the executables in the root directory.

To compile a specific solver:

```bash
make wordle_dp_solver
```

To compile manually:

```bash
cd src
g++ -std=c++17 -O2 wordle_dp_solver.cpp -o ../wordle_dp_solver
```

To clean compiled binaries:

```bash
make clean
```

### Python Scripts

Python scripts require Python 3 and can be run directly:

```bash
cd scripts
python3 wordle_dp_solver.py
```

## Usage

### Running a Solver

Most solvers expect a word list file. By default, they look for `wordle_words.txt` in the current directory, but you can specify a different file:

```bash
./wordle_dp_solver --word-file data/wordle_words.txt
```

Some solvers accept additional parameters. Check the individual solver source code for details on expected input/output formats and command-line options.

Example with default word file:
```bash
./wordle_dp_solver
```

### Running Analysis Scripts

```bash
cd scripts
python3 analyze_wordle_usage.py
python3 analyze_letter_independence.py
```

Note: Analysis scripts may need to be run from the project root or may need paths adjusted to find data files.

## Project Structure

```
wordle_solver/
├── src/              # C++ solver implementations
├── scripts/          # Python scripts and solvers
├── data/             # Word lists and data files
├── reports/          # Analysis reports
└── README.md         # This file
```

## License

[Add your license here - MIT, GPL, etc.]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

