#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <iomanip>
#include <cmath>
#include <chrono>
#include <queue>

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

class WordleBeamSolver {
private:
    vector<string> all_words;
    int beam_width;
    
    // Compute entropy (information gain) for a guess
    // Higher entropy = better (more information gained)
    double computeEntropy(int guess_idx, const vector<int>& possible_word_indices) {
        const string& guess = all_words[guess_idx];
        
        // Group words by feedback pattern
        unordered_map<Feedback, int, FeedbackHash> feedback_counts;
        
        for (int answer_idx : possible_word_indices) {
            const string& answer = all_words[answer_idx];
            Feedback feedback = getFeedback(guess, answer);
            feedback_counts[feedback]++;
        }
        
        // Compute entropy (Shannon entropy)
        double entropy = 0.0;
        int n_total = possible_word_indices.size();
        
        for (auto& [feedback, count] : feedback_counts) {
            if (count > 0) {
                double prob = (double)count / n_total;
                entropy -= prob * log2(prob);
            }
        }
        
        return entropy;
    }
    
    // Get top K guesses by entropy
    vector<int> getTopKByEntropy(const vector<int>& possible_word_indices, 
                                 const vector<int>& candidate_guesses, 
                                 int k) {
        // Compute entropy for each candidate
        vector<pair<double, int>> entropy_scores;
        
        for (int guess_idx : candidate_guesses) {
            double entropy = computeEntropy(guess_idx, possible_word_indices);
            entropy_scores.push_back({entropy, guess_idx});
        }
        
        // Sort by entropy (descending)
        sort(entropy_scores.rbegin(), entropy_scores.rend());
        
        // Return top K
        vector<int> top_k;
        for (int i = 0; i < min(k, (int)entropy_scores.size()); i++) {
            top_k.push_back(entropy_scores[i].second);
        }
        
        return top_k;
    }
    
    // Recursively evaluate strategy with beam search
    pair<double, int> solveWithBeam(const vector<int>& possible_word_indices, 
                                    int guesses_left, 
                                    int depth = 0) {
        // Base cases
        if (possible_word_indices.empty()) {
            return {0.0, -1};
        }
        
        if (possible_word_indices.size() == 1) {
            return guesses_left > 0 ? make_pair(1.0, possible_word_indices[0]) : make_pair(0.0, -1);
        }
        
        if (guesses_left == 0) {
            return {0.0, -1};
        }
        
        // Determine candidate guesses
        vector<int> candidate_guesses = possible_word_indices;
        if (possible_word_indices.size() > 200) {
            // For large sets, limit initial candidates
            candidate_guesses.assign(possible_word_indices.begin(), 
                                    possible_word_indices.begin() + min(500, (int)possible_word_indices.size()));
        }
        
        // Get top beam_width guesses by entropy
        vector<int> top_guesses = getTopKByEntropy(possible_word_indices, candidate_guesses, beam_width);
        
        double best_prob = 0.0;
        int best_guess_idx = -1;
        
        // Evaluate each top guess with full lookahead
        for (int guess_idx : top_guesses) {
            double expected_prob = expectedSuccessProb(guess_idx, possible_word_indices, guesses_left, depth);
            
            if (expected_prob > best_prob) {
                best_prob = expected_prob;
                best_guess_idx = guess_idx;
                
                // Early termination
                if (best_prob >= 1.0 - 1e-9) {
                    break;
                }
            }
        }
        
        return {best_prob, best_guess_idx};
    }
    
    // Compute expected success probability with recursive evaluation
    double expectedSuccessProb(int guess_idx, const vector<int>& possible_word_indices, 
                              int guesses_left, int depth = 0) {
        const string& guess = all_words[guess_idx];
        
        // Group words by feedback pattern
        unordered_map<Feedback, vector<int>, FeedbackHash> feedback_groups;
        
        for (int answer_idx : possible_word_indices) {
            const string& answer = all_words[answer_idx];
            Feedback feedback = getFeedback(guess, answer);
            feedback_groups[feedback].push_back(answer_idx);
        }
        
        double total_prob = 0.0;
        int n_total = possible_word_indices.size();
        
        // Process each feedback group
        for (auto& [feedback, word_indices] : feedback_groups) {
            if (word_indices.empty()) continue;
            
            // Check if this is a correct guess (all greens)
            bool is_correct = (get<0>(feedback) == 2 && get<1>(feedback) == 2 && 
                              get<2>(feedback) == 2 && get<3>(feedback) == 2 && 
                              get<4>(feedback) == 2);
            
            if (is_correct) {
                total_prob += (double)word_indices.size() / n_total;
            } else {
                // Recursively solve with beam search
                auto [sub_prob, _] = solveWithBeam(word_indices, guesses_left - 1, depth + 1);
                total_prob += ((double)word_indices.size() / n_total) * sub_prob;
            }
        }
        
        return total_prob;
    }
    
public:
    WordleBeamSolver(const vector<string>& words, int beam_width = 10) 
        : all_words(words), beam_width(beam_width) {}
    
    pair<double, string> solve(int max_guesses = 3) {
        vector<int> all_indices;
        for (int i = 0; i < (int)all_words.size(); i++) {
            all_indices.push_back(i);
        }
        
        // Find best first guess using beam search
        vector<int> candidate_guesses = all_indices;
        if (all_indices.size() > 500) {
            candidate_guesses.assign(all_indices.begin(), all_indices.begin() + min(1000, (int)all_indices.size()));
        }
        
        vector<int> top_guesses = getTopKByEntropy(all_indices, candidate_guesses, beam_width);
        
        double best_prob = 0.0;
        int best_guess_idx = -1;
        
        cout << "Evaluating top " << top_guesses.size() << " guesses by entropy..." << endl;
        
        for (int i = 0; i < (int)top_guesses.size(); i++) {
            int guess_idx = top_guesses[i];
            cout << "  [" << (i+1) << "/" << top_guesses.size() << "] Evaluating: " 
                 << all_words[guess_idx] << " (entropy: " 
                 << fixed << setprecision(4) << computeEntropy(guess_idx, all_indices) << ")..." << flush;
            
            double prob = expectedSuccessProb(guess_idx, all_indices, max_guesses);
            
            cout << " prob: " << prob << endl;
            
            if (prob > best_prob) {
                best_prob = prob;
                best_guess_idx = guess_idx;
            }
        }
        
        string best_guess = (best_guess_idx >= 0) ? all_words[best_guess_idx] : "";
        return {best_prob, best_guess};
    }
    
    // Build strategy tree
    struct StrategyNode {
        string guess;
        double probability;
        double entropy;
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
        
        // Find best guess using beam search
        vector<int> candidate_guesses = possible_word_indices;
        if (possible_word_indices.size() > 200) {
            candidate_guesses.assign(possible_word_indices.begin(), 
                                    possible_word_indices.begin() + min(500, (int)possible_word_indices.size()));
        }
        
        vector<int> top_guesses = getTopKByEntropy(possible_word_indices, candidate_guesses, beam_width);
        
        double best_prob = 0.0;
        int best_guess_idx = -1;
        
        for (int guess_idx : top_guesses) {
            double prob = expectedSuccessProb(guess_idx, possible_word_indices, guesses_left, depth);
            if (prob > best_prob) {
                best_prob = prob;
                best_guess_idx = guess_idx;
            }
        }
        
        if (best_guess_idx == -1) {
            node.count = possible_word_indices.size();
            return node;
        }
        
        node.guess = all_words[best_guess_idx];
        node.count = possible_word_indices.size();
        node.probability = best_prob;
        node.entropy = computeEntropy(best_guess_idx, possible_word_indices);
        
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
            cout << prefix << "Guess: " << node.guess;
            if (node.probability > 0) {
                cout << " (prob: " << fixed << setprecision(4) << node.probability 
                     << ", entropy: " << setprecision(4) << node.entropy << ")";
            }
            cout << endl;
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
    int beam_width = 10;
    
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
        } else if (string(argv[i]) == "--beam-width" && i + 1 < argc) {
            beam_width = stoi(argv[i + 1]);
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
    cout << "Using beam search with width " << beam_width << " (entropy-based exploration)..." << endl;
    
    auto start = high_resolution_clock::now();
    
    WordleBeamSolver solver(word_list, beam_width);
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


