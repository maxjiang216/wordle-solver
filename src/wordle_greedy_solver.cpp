#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <iomanip>
#include <cmath>
#include <chrono>

using namespace std;
using namespace std::chrono;

// Feedback pattern: 0=gray, 1=yellow, 2=green
typedef tuple<int, int, int, int, int> Feedback;

struct FeedbackHash {
    size_t operator()(const Feedback& f) const {
        return get<0>(f) + get<1>(f) * 3 + get<2>(f) * 9 + get<3>(f) * 27 + get<4>(f) * 81;
    }
};

// Compute Wordle feedback pattern
Feedback getFeedback(const string& guess, const string& answer) {
    int feedback[5] = {0, 0, 0, 0, 0};
    string answer_chars = answer;
    string guess_chars = guess;
    
    // First pass: mark greens
    for (int i = 0; i < 5; i++) {
        if (guess_chars[i] == answer_chars[i]) {
            feedback[i] = 2;
            answer_chars[i] = '\0';
            guess_chars[i] = '\0';
        }
    }
    
    // Second pass: mark yellows
    for (int i = 0; i < 5; i++) {
        if (guess_chars[i] != '\0') {
            size_t pos = answer_chars.find(guess_chars[i]);
            if (pos != string::npos) {
                feedback[i] = 1;
                answer_chars[pos] = '\0';
            }
        }
    }
    
    return make_tuple(feedback[0], feedback[1], feedback[2], feedback[3], feedback[4]);
}

class WordleGreedySolver {
private:
    vector<string> all_words;
    
    // Compute expected information gain (negative of expected remaining words)
    // Lower score = better (fewer expected remaining words)
    double computeScore(int guess_idx, const vector<int>& possible_word_indices) {
        const string& guess = all_words[guess_idx];
        
        // Group words by feedback pattern
        unordered_map<Feedback, int, FeedbackHash> feedback_counts;
        
        for (int answer_idx : possible_word_indices) {
            const string& answer = all_words[answer_idx];
            Feedback feedback = getFeedback(guess, answer);
            feedback_counts[feedback]++;
        }
        
        // Compute expected number of remaining words
        double expected_remaining = 0.0;
        int n_total = possible_word_indices.size();
        
        for (auto& [feedback, count] : feedback_counts) {
            // If this is a correct guess, we're done (0 remaining)
            bool is_correct = (get<0>(feedback) == 2 && get<1>(feedback) == 2 && 
                              get<2>(feedback) == 2 && get<3>(feedback) == 2 && 
                              get<4>(feedback) == 2);
            
            if (!is_correct) {
                double prob = (double)count / n_total;
                expected_remaining += prob * count;
            }
        }
        
        return expected_remaining;
    }
    
    // Find best guess using greedy information maximization
    int findBestGuess(const vector<int>& possible_word_indices, 
                     const vector<int>& candidate_guesses) {
        double best_score = 1e9;
        int best_guess_idx = -1;
        
        for (int guess_idx : candidate_guesses) {
            double score = computeScore(guess_idx, possible_word_indices);
            if (score < best_score) {
                best_score = score;
                best_guess_idx = guess_idx;
            }
        }
        
        return best_guess_idx;
    }
    
    // Simulate playing with greedy strategy and compute success rate
    double simulateStrategy(const vector<int>& possible_word_indices, int guesses_left, 
                           int depth = 0) {
        if (possible_word_indices.empty()) return 0.0;
        if (possible_word_indices.size() == 1) return guesses_left > 0 ? 1.0 : 0.0;
        if (guesses_left == 0) return 0.0;
        
        // Find best guess
        vector<int> candidate_guesses = possible_word_indices;
        if (possible_word_indices.size() > 100) {
            // For large sets, limit candidate guesses
            candidate_guesses.assign(possible_word_indices.begin(), 
                                    possible_word_indices.begin() + min(200, (int)possible_word_indices.size()));
        }
        
        int best_guess_idx = findBestGuess(possible_word_indices, candidate_guesses);
        if (best_guess_idx == -1) return 0.0;
        
        const string& guess = all_words[best_guess_idx];
        
        // Group words by feedback
        unordered_map<Feedback, vector<int>, FeedbackHash> feedback_groups;
        
        for (int answer_idx : possible_word_indices) {
            const string& answer = all_words[answer_idx];
            Feedback feedback = getFeedback(guess, answer);
            feedback_groups[feedback].push_back(answer_idx);
        }
        
        // Compute expected success probability
        double total_prob = 0.0;
        int n_total = possible_word_indices.size();
        
        for (auto& [feedback, word_indices] : feedback_groups) {
            bool is_correct = (get<0>(feedback) == 2 && get<1>(feedback) == 2 && 
                              get<2>(feedback) == 2 && get<3>(feedback) == 2 && 
                              get<4>(feedback) == 2);
            
            if (is_correct) {
                total_prob += (double)word_indices.size() / n_total;
            } else {
                double sub_prob = simulateStrategy(word_indices, guesses_left - 1, depth + 1);
                total_prob += ((double)word_indices.size() / n_total) * sub_prob;
            }
        }
        
        return total_prob;
    }
    
public:
    WordleGreedySolver(const vector<string>& words) : all_words(words) {}
    
    pair<double, string> solve(int max_guesses = 3) {
        vector<int> all_indices;
        for (int i = 0; i < (int)all_words.size(); i++) {
            all_indices.push_back(i);
        }
        
        // Find best first guess
        vector<int> candidate_guesses = all_indices;
        if (all_indices.size() > 200) {
            candidate_guesses.assign(all_indices.begin(), all_indices.begin() + min(500, (int)all_indices.size()));
        }
        
        int best_guess_idx = findBestGuess(all_indices, candidate_guesses);
        string best_guess = (best_guess_idx >= 0) ? all_words[best_guess_idx] : "";
        
        // Simulate to get actual success probability
        double prob = simulateStrategy(all_indices, max_guesses);
        
        return {prob, best_guess};
    }
    
    // Build strategy tree
    struct StrategyNode {
        string guess;
        double probability;
        int count;
        unordered_map<string, StrategyNode> children;
    };
    
    StrategyNode buildStrategyTree(const vector<int>& possible_word_indices, 
                                  int guesses_left, int max_depth, int depth = 0) {
        StrategyNode node;
        
        if (possible_word_indices.empty() || guesses_left == 0 || depth >= max_depth) {
            node.count = possible_word_indices.size();
            return node;
        }
        
        if (possible_word_indices.size() == 1) {
            node.guess = all_words[possible_word_indices[0]];
            node.count = 1;
            return node;
        }
        
        // Find best guess
        vector<int> candidate_guesses = possible_word_indices;
        if (possible_word_indices.size() > 100) {
            candidate_guesses.assign(possible_word_indices.begin(), 
                                    possible_word_indices.begin() + min(200, (int)possible_word_indices.size()));
        }
        
        int best_guess_idx = findBestGuess(possible_word_indices, candidate_guesses);
        if (best_guess_idx == -1) {
            node.count = possible_word_indices.size();
            return node;
        }
        
        node.guess = all_words[best_guess_idx];
        node.count = possible_word_indices.size();
        
        // Group by feedback
        unordered_map<Feedback, vector<int>, FeedbackHash> feedback_groups;
        const string& guess = node.guess;
        
        for (int answer_idx : possible_word_indices) {
            const string& answer = all_words[answer_idx];
            Feedback feedback = getFeedback(guess, answer);
            feedback_groups[feedback].push_back(answer_idx);
        }
        
        // Build children
        for (auto& [feedback, word_indices] : feedback_groups) {
            string feedback_str = "(" + to_string(get<0>(feedback)) + "," + 
                                to_string(get<1>(feedback)) + "," +
                                to_string(get<2>(feedback)) + "," +
                                to_string(get<3>(feedback)) + "," +
                                to_string(get<4>(feedback)) + ")";
            
            bool is_correct = (get<0>(feedback) == 2 && get<1>(feedback) == 2 && 
                              get<2>(feedback) == 2 && get<3>(feedback) == 2 && 
                              get<4>(feedback) == 2);
            
            if (is_correct) {
                StrategyNode win_node;
                win_node.guess = node.guess;
                win_node.count = word_indices.size();
                node.children[feedback_str + " (CORRECT)"] = win_node;
            } else {
                node.children[feedback_str] = buildStrategyTree(word_indices, guesses_left - 1, max_depth, depth + 1);
            }
        }
        
        return node;
    }
    
    void printTree(const StrategyNode& node, int indent = 0) {
        string prefix(indent * 2, ' ');
        
        if (!node.guess.empty()) {
            cout << prefix << "Guess: " << node.guess << endl;
            cout << prefix << "  Possible words: " << node.count << endl;
            
            if (!node.children.empty()) {
                int shown = 0;
                for (auto& [feedback, child] : node.children) {
                    if (shown++ >= 5) break; // Show top 5
                    cout << prefix << "  Feedback " << feedback << ":" << endl;
                    printTree(child, indent + 2);
                }
            }
        } else if (node.count > 0) {
            cout << prefix << "Remaining: " << node.count << " words" << endl;
        }
    }
};

int main(int argc, char* argv[]) {
    string word_file = "wordle_words.txt";
    bool show_tree = false;
    int max_depth = 2;
    
    // Parse arguments
    for (int i = 1; i < argc; i++) {
        if (string(argv[i]) == "--word-file" && i + 1 < argc) {
            word_file = argv[i + 1];
            i++;
        } else if (string(argv[i]) == "--show-tree") {
            show_tree = true;
        } else if (string(argv[i]) == "--max-depth" && i + 1 < argc) {
            max_depth = stoi(argv[i + 1]);
            i++;
        }
    }
    
    // Read word list
    ifstream file(word_file);
    if (!file.is_open()) {
        cerr << "Error: Could not open file " << word_file << endl;
        return 1;
    }
    
    vector<string> word_list;
    string word;
    while (file >> word) {
        if (word.length() == 5) {
            word_list.push_back(word);
        }
    }
    file.close();
    
    cout << "Loaded " << word_list.size() << " words" << endl;
    cout << "Using greedy information-maximizing strategy..." << endl;
    
    auto start = high_resolution_clock::now();
    
    WordleGreedySolver solver(word_list);
    auto [prob, best_guess] = solver.solve(3);
    
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);
    
    cout << "\nOptimal first guess: " << best_guess << endl;
    cout << fixed << setprecision(4);
    cout << "Success probability in 3 guesses: " << prob << " (" << (prob * 100) << "%)" << endl;
    cout << "Computation time: " << duration.count() << " ms" << endl;
    
    if (show_tree) {
        cout << "\nBuilding strategy tree..." << endl;
        vector<int> all_indices;
        for (int i = 0; i < (int)word_list.size(); i++) {
            all_indices.push_back(i);
        }
        auto tree = solver.buildStrategyTree(all_indices, 3, max_depth);
        cout << "\nStrategy tree:" << endl;
        solver.printTree(tree);
    }
    
    return 0;
}

