#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <iomanip>
#include <cstring>
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

class WordleSolver {
private:
    vector<string> all_words;
    unordered_map<string, int> word_to_idx;
    
    // Cache for DP: (word_set_hash, guesses_left) -> (prob, best_guess_idx)
    unordered_map<string, pair<double, int>> cache;
    
    // Cache for feedback computations
    unordered_map<string, Feedback> feedback_cache;
    
    int max_guesses_to_try;
    
    // Convert word set to hash string for caching
    string wordSetToHash(const vector<int>& word_indices) {
        string hash_str;
        for (int idx : word_indices) {
            hash_str += to_string(idx) + ",";
        }
        return hash_str;
    }
    
    // Get feedback with caching
    Feedback getFeedbackCached(const string& guess, const string& answer) {
        string key = guess + "|" + answer;
        auto it = feedback_cache.find(key);
        if (it != feedback_cache.end()) {
            return it->second;
        }
        Feedback f = getFeedback(guess, answer);
        feedback_cache[key] = f;
        return f;
    }
    
    // Recursive DP solver
    pair<double, int> solveRecursive(const vector<int>& possible_word_indices, int guesses_left) {
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
        
        // Check cache
        string cache_key = wordSetToHash(possible_word_indices) + "|" + to_string(guesses_left);
        auto cache_it = cache.find(cache_key);
        if (cache_it != cache.end()) {
            return cache_it->second;
        }
        
        double best_prob = 0.0;
        int best_guess_idx = -1;
        
        // Determine which guesses to try
        vector<int> guesses_to_try;
        if (max_guesses_to_try == -1) {
            // Try all words
            if (possible_word_indices.size() <= 50) {
                guesses_to_try = possible_word_indices;
                for (int i = 0; i < (int)all_words.size(); i++) {
                    if (find(possible_word_indices.begin(), possible_word_indices.end(), i) == possible_word_indices.end()) {
                        guesses_to_try.push_back(i);
                    }
                }
            } else if (possible_word_indices.size() <= 500) {
                guesses_to_try = possible_word_indices;
                for (int i = 0; i < min(500, (int)all_words.size()); i++) {
                    if (find(possible_word_indices.begin(), possible_word_indices.end(), i) == possible_word_indices.end()) {
                        guesses_to_try.push_back(i);
                    }
                }
            } else {
                guesses_to_try = possible_word_indices;
            }
        } else {
            // Limited search
            int limit = max_guesses_to_try;
            if (possible_word_indices.size() <= limit) {
                guesses_to_try = possible_word_indices;
            } else {
                guesses_to_try.assign(possible_word_indices.begin(), 
                                     possible_word_indices.begin() + limit);
                for (int i = 0; i < limit / 2 && i < (int)all_words.size(); i++) {
                    if (find(possible_word_indices.begin(), possible_word_indices.end(), i) == possible_word_indices.end()) {
                        guesses_to_try.push_back(i);
                    }
                }
            }
        }
        
        // Try each guess
        for (int guess_idx : guesses_to_try) {
            double expected_prob = expectedSuccessProb(guess_idx, possible_word_indices, guesses_left);
            
            if (expected_prob > best_prob) {
                best_prob = expected_prob;
                best_guess_idx = guess_idx;
                
                // Early termination
                if (best_prob >= 1.0 - 1e-9) {
                    break;
                }
            }
        }
        
        // Cache result
        cache[cache_key] = {best_prob, best_guess_idx};
        return {best_prob, best_guess_idx};
    }
    
    // Compute expected success probability for a guess
    double expectedSuccessProb(int guess_idx, const vector<int>& possible_word_indices, int guesses_left) {
        const string& guess = all_words[guess_idx];
        
        // Group words by feedback pattern
        unordered_map<Feedback, vector<int>, FeedbackHash> feedback_groups;
        
        for (int answer_idx : possible_word_indices) {
            const string& answer = all_words[answer_idx];
            Feedback feedback = getFeedbackCached(guess, answer);
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
                              get<4>(feedback) == 2) && 
                              find(word_indices.begin(), word_indices.end(), guess_idx) != word_indices.end();
            
            if (is_correct) {
                total_prob += (double)word_indices.size() / n_total;
            } else {
                // Recursively solve
                auto [sub_prob, _] = solveRecursive(word_indices, guesses_left - 1);
                total_prob += ((double)word_indices.size() / n_total) * sub_prob;
            }
        }
        
        return total_prob;
    }
    
public:
    WordleSolver(const vector<string>& words, int max_guesses = -1) 
        : all_words(words), max_guesses_to_try(max_guesses) {
        for (int i = 0; i < (int)words.size(); i++) {
            word_to_idx[words[i]] = i;
        }
    }
    
    pair<double, string> solve(int max_guesses = 3) {
        vector<int> all_indices;
        for (int i = 0; i < (int)all_words.size(); i++) {
            all_indices.push_back(i);
        }
        
        auto [prob, best_idx] = solveRecursive(all_indices, max_guesses);
        string best_guess = (best_idx >= 0) ? all_words[best_idx] : "";
        return {prob, best_guess};
    }
    
    size_t getCacheSize() const {
        return cache.size();
    }
};

int main(int argc, char* argv[]) {
    string word_file = "wordle_words.txt";
    int max_guesses_to_try = -1;
    
    // Parse arguments
    for (int i = 1; i < argc; i++) {
        if (string(argv[i]) == "--max-guesses-to-try" && i + 1 < argc) {
            max_guesses_to_try = stoi(argv[i + 1]);
            i++;
        } else if (string(argv[i]) == "--word-file" && i + 1 < argc) {
            word_file = argv[i + 1];
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
    if (max_guesses_to_try > 0) {
        cout << "Limiting to " << max_guesses_to_try << " guesses per step (approximate solution)" << endl;
    } else {
        cout << "Computing optimal strategy (this may take a while)..." << endl;
    }
    
    auto start = high_resolution_clock::now();
    
    WordleSolver solver(word_list, max_guesses_to_try);
    auto [prob, best_guess] = solver.solve(3);
    
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);
    
    cout << "\nOptimal first guess: " << best_guess << endl;
    cout << fixed << setprecision(4);
    cout << "Success probability in 3 guesses: " << prob << " (" << (prob * 100) << "%)" << endl;
    cout << "\nCache size: " << solver.getCacheSize() << " states" << endl;
    cout << "Computation time: " << duration.count() << " ms" << endl;
    
    return 0;
}

