CXX = g++
CXXFLAGS = -std=c++17 -O2 -Wall
SRCDIR = src
BINDIR = .
DATADIR = data

# All C++ source files
SOURCES = $(wildcard $(SRCDIR)/*.cpp)
# Binary names (remove .cpp extension and src/ prefix)
BINARIES = $(patsubst $(SRCDIR)/%.cpp,$(BINDIR)/%,$(SOURCES))

.PHONY: all clean

all: $(BINARIES)

# Build rule for each solver
$(BINDIR)/%: $(SRCDIR)/%.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -f $(BINARIES)

# Individual solver targets
wordle_dp_solver: $(BINDIR)/wordle_dp_solver
wordle_beam_solver: $(BINDIR)/wordle_beam_solver
wordle_greedy_solver: $(BINDIR)/wordle_greedy_solver
wordle_monte_carlo_solver: $(BINDIR)/wordle_monte_carlo_solver
wordle_random_play_solver: $(BINDIR)/wordle_random_play_solver
wordle_unused_first_solver: $(BINDIR)/wordle_unused_first_solver

