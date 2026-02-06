#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <iomanip>
#include <random>
#include <chrono>
#include <cmath>

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

class WordleMonteCarloSolver {
private:
    vector<string> all_words;
    int max_guesses;
    int num_simulations;
    mt19937 rng;
    
    // Check if a word matches the feedback pattern
    bool wordMatchesFeedback(const string& word, const string& guess, const Feedback& feedback) {
        Feedback word_feedback = getFeedback(guess, word);
        return word_feedback == feedback;
    }
    
    // Simulate one game with random play
    bool simulateGame(int first_guess_idx, int answer_idx) {
        const string& first_guess = all_words[first_guess_idx];
        const string& answer = all_words[answer_idx];
        
        // First guess
        Feedback feedback = getFeedback(first_guess, answer);
        
        // Check if first guess is correct
        bool is_correct = (get<0>(feedback) == 2 && get<1>(feedback) == 2 && 
                          get<2>(feedback) == 2 && get<3>(feedback) == 2 && 
                          get<4>(feedback) == 2);
        if (is_correct) {
            return true;
        }
        
        // Find valid words (words matching the feedback)
        vector<int> valid_words;
        for (int i = 0; i < (int)all_words.size(); i++) {
            if (wordMatchesFeedback(all_words[i], first_guess, feedback)) {
                valid_words.push_back(i);
            }
        }
        
        // Random play for remaining guesses
        for (int guess_num = 2; guess_num <= max_guesses; guess_num++) {
            // Check if answer is still valid
            if (find(valid_words.begin(), valid_words.end(), answer_idx) == valid_words.end()) {
                return false;
            }
            
            // If only one word left, we win
            if (valid_words.size() == 1) {
                return true;
            }
            
            // Randomly pick a word from valid words
            uniform_int_distribution<int> dist(0, valid_words.size() - 1);
            int guess_idx = valid_words[dist(rng)];
            const string& guess = all_words[guess_idx];
            
            // Check if we guessed correctly
            if (guess_idx == answer_idx) {
                return true;
            }
            
            // Get feedback from this guess
            feedback = getFeedback(guess, answer);
            
            // Filter valid words to only those matching the new feedback
            vector<int> new_valid_words;
            for (int word_idx : valid_words) {
                if (wordMatchesFeedback(all_words[word_idx], guess, feedback)) {
                    new_valid_words.push_back(word_idx);
                }
            }
            valid_words = new_valid_words;
        }
        
        return false;
    }
    
    // Estimate success probability using Monte Carlo
    double estimateSuccessProb(int first_guess_idx) {
        int successes = 0;
        
        // Sample random answers
        uniform_int_distribution<int> answer_dist(0, all_words.size() - 1);
        
        for (int sim = 0; sim < num_simulations; sim++) {
            int answer_idx = answer_dist(rng);
            if (simulateGame(first_guess_idx, answer_idx)) {
                successes++;
            }
        }
        
        return (double)successes / num_simulations;
    }
    
public:
    WordleMonteCarloSolver(const vector<string>& words, int max_guesses = 3, 
                          int num_simulations = 10000, unsigned int seed = 42)
        : all_words(words), max_guesses(max_guesses), num_simulations(num_simulations), rng(seed) {}
    
    pair<double, string> solve(int top_k = 50) {
        // First, quickly estimate entropy for all words to find top candidates
        cout << "Computing entropy for all words to find top candidates..." << endl;
        
        vector<pair<double, int>> entropy_scores;
        
        for (int i = 0; i < (int)all_words.size(); i++) {
            // Quick entropy estimate: group by feedback patterns
            unordered_map<Feedback, int, FeedbackHash> feedback_counts;
            
            // Sample a subset for quick entropy estimate
            uniform_int_distribution<int> dist(0, all_words.size() - 1);
            for (int j = 0; j < min(500, (int)all_words.size()); j++) {
                int answer_idx = dist(rng);
                Feedback feedback = getFeedback(all_words[i], all_words[answer_idx]);
                feedback_counts[feedback]++;
            }
            
            // Compute entropy
            double entropy = 0.0;
            int n = min(500, (int)all_words.size());
            for (auto& [feedback, count] : feedback_counts) {
                if (count > 0) {
                    double prob = (double)count / n;
                    entropy -= prob * log2(prob);
                }
            }
            
            entropy_scores.push_back({entropy, i});
        }
        
        // Sort by entropy and take top K
        sort(entropy_scores.rbegin(), entropy_scores.rend());
        
        vector<int> top_candidates;
        for (int i = 0; i < min(top_k, (int)entropy_scores.size()); i++) {
            top_candidates.push_back(entropy_scores[i].second);
        }
        
        cout << "Evaluating top " << top_candidates.size() << " candidates with " 
             << num_simulations << " Monte Carlo simulations each..." << endl;
        
        double best_prob = 0.0;
        int best_guess_idx = -1;
        
        for (int i = 0; i < (int)top_candidates.size(); i++) {
            int guess_idx = top_candidates[i];
            cout << "  [" << (i+1) << "/" << top_candidates.size() << "] " 
                 << all_words[guess_idx] << "..." << flush;
            
            double prob = estimateSuccessProb(guess_idx);
            
            cout << " " << fixed << setprecision(4) << prob << " (" << (prob * 100) << "%)" << endl;
            
            if (prob > best_prob) {
                best_prob = prob;
                best_guess_idx = guess_idx;
            }
        }
        
        string best_guess = (best_guess_idx >= 0) ? all_words[best_guess_idx] : "";
        return {best_prob, best_guess};
    }
    
    // More detailed evaluation of a specific guess
    void evaluateGuess(const string& guess, int detailed_sims = 50000) {
        int guess_idx = -1;
        for (int i = 0; i < (int)all_words.size(); i++) {
            if (all_words[i] == guess) {
                guess_idx = i;
                break;
            }
        }
        
        if (guess_idx == -1) {
            cout << "Word not found: " << guess << endl;
            return;
        }
        
        cout << "\nEvaluating " << guess << " with " << detailed_sims << " simulations..." << endl;
        
        int successes = 0;
        uniform_int_distribution<int> answer_dist(0, all_words.size() - 1);
        
        auto start = high_resolution_clock::now();
        
        for (int sim = 0; sim < detailed_sims; sim++) {
            int answer_idx = answer_dist(rng);
            if (simulateGame(guess_idx, answer_idx)) {
                successes++;
            }
            
            if ((sim + 1) % 10000 == 0) {
                double current_prob = (double)successes / (sim + 1);
                cout << "  Progress: " << (sim + 1) << "/" << detailed_sims 
                     << " (current: " << fixed << setprecision(4) << current_prob << ")" << endl;
            }
        }
        
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(end - start);
        
        double prob = (double)successes / detailed_sims;
        cout << "\nSuccess probability: " << fixed << setprecision(4) << prob 
             << " (" << (prob * 100) << "%)" << endl;
        cout << "Computation time: " << duration.count() << " ms" << endl;
    }
};

int main(int argc, char* argv[]) {
    string word_file = "wordle_words.txt";
    string evaluate_word = "";
    int max_guesses = 3;
    int num_simulations = 10000;
    int top_k = 50;
    
    // Parse arguments
    for (int i = 1; i < argc; i++) {
        if (string(argv[i]) == "--word-file" && i + 1 < argc) {
            word_file = argv[i + 1];
            i++;
        } else if (string(argv[i]) == "--evaluate" && i + 1 < argc) {
            evaluate_word = argv[i + 1];
            i++;
        } else if (string(argv[i]) == "--max-guesses" && i + 1 < argc) {
            max_guesses = stoi(argv[i + 1]);
            i++;
        } else if (string(argv[i]) == "--simulations" && i + 1 < argc) {
            num_simulations = stoi(argv[i + 1]);
            i++;
        } else if (string(argv[i]) == "--top-k" && i + 1 < argc) {
            top_k = stoi(argv[i + 1]);
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
    cout << "Using Monte Carlo estimation with " << num_simulations << " simulations per guess" << endl;
    cout << "(Random play = randomly guess from words matching all clues)" << endl;
    
    auto start = high_resolution_clock::now();
    
    WordleMonteCarloSolver solver(word_list, max_guesses, num_simulations);
    
    if (!evaluate_word.empty()) {
        solver.evaluateGuess(evaluate_word, num_simulations * 5);
        return 0;
    }
    
    auto [prob, best_guess] = solver.solve(top_k);
    
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);
    
    cout << "\nOptimal first guess: " << best_guess << endl;
    cout << fixed << setprecision(4);
    cout << "Estimated success probability in " << max_guesses << " guesses (with random play): " 
         << prob << " (" << (prob * 100) << "%)" << endl;
    cout << "Computation time: " << duration.count() << " ms" << endl;
    
    return 0;
}

