#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <iomanip>
#include <cmath>
#include <chrono>
#include <random>

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

class WordleRandomPlaySolver {
private:
    vector<string> all_words;
    int max_guesses;
    
    // Memoization cache for simulateRandomPlay
    unordered_map<string, double> play_cache;
    
    // Check if a word matches the feedback pattern
    bool wordMatchesFeedback(const string& word, const string& guess, const Feedback& feedback) {
        Feedback word_feedback = getFeedback(guess, word);
        return word_feedback == feedback;
    }
    
    // Create cache key for valid_words + answer + guesses_left
    string createCacheKey(const vector<int>& valid_words, int answer_idx, int guesses_left) {
        string key = to_string(answer_idx) + "|" + to_string(guesses_left) + "|";
        for (int idx : valid_words) {
            key += to_string(idx) + ",";
        }
        return key;
    }
    
    // Count how many words match a given feedback pattern
    int countMatchingWords(const string& guess, const Feedback& feedback, 
                          const vector<int>& possible_word_indices) {
        int count = 0;
        for (int word_idx : possible_word_indices) {
            const string& word = all_words[word_idx];
            if (wordMatchesFeedback(word, guess, feedback)) {
                count++;
            }
        }
        return count;
    }
    
    // Compute success probability assuming random play after first guess
    // Returns: probability of success in max_guesses attempts
    double computeRandomPlaySuccess(int first_guess_idx, int max_guesses) {
        const string& first_guess = all_words[first_guess_idx];
        double total_success = 0.0;
        int n_answers = all_words.size();
        
        // For each possible answer
        for (int answer_idx = 0; answer_idx < n_answers; answer_idx++) {
            const string& answer = all_words[answer_idx];
            
            // Get feedback from first guess
            Feedback feedback = getFeedback(first_guess, answer);
            
            // If first guess is correct, we win immediately
            bool is_correct = (get<0>(feedback) == 2 && get<1>(feedback) == 2 && 
                              get<2>(feedback) == 2 && get<3>(feedback) == 2 && 
                              get<4>(feedback) == 2);
            
            if (is_correct) {
                total_success += 1.0;
                continue;
            }
            
            // Find all words that match this feedback pattern
            vector<int> matching_words;
            for (int i = 0; i < n_answers; i++) {
                const string& word = all_words[i];
                if (wordMatchesFeedback(word, first_guess, feedback)) {
                    matching_words.push_back(i);
                }
            }
            
            // If no words match, we lose
            if (matching_words.empty()) {
                continue;
            }
            
            // If answer is not in matching words, we lose
            if (find(matching_words.begin(), matching_words.end(), answer_idx) == matching_words.end()) {
                continue;
            }
            
            // Now simulate random play for remaining guesses
            double success_prob = simulateRandomPlay(matching_words, answer_idx, max_guesses - 1);
            total_success += success_prob;
        }
        
        return total_success / n_answers;
    }
    
    // Simulate random play: randomly guess from valid words
    double simulateRandomPlay(const vector<int>& valid_words, int answer_idx, int guesses_left) {
        if (guesses_left <= 0) {
            return 0.0;
        }
        
        // If only one word left, we win
        if (valid_words.size() == 1) {
            return 1.0;
        }
        
        // Check if answer is in valid words
        if (find(valid_words.begin(), valid_words.end(), answer_idx) == valid_words.end()) {
            return 0.0;
        }
        
        // Check cache
        string cache_key = createCacheKey(valid_words, answer_idx, guesses_left);
        auto it = play_cache.find(cache_key);
        if (it != play_cache.end()) {
            return it->second;
        }
        
        // Probability of guessing correctly = 1 / number of valid words
        double prob_correct = 1.0 / valid_words.size();
        
        // If we guess correctly, we win
        double total_success = prob_correct;
        
        // For each possible wrong guess, compute expected success
        // Group wrong guesses by the feedback pattern they produce
        unordered_map<Feedback, vector<int>, FeedbackHash> feedback_groups;
        
        for (int guess_idx : valid_words) {
            if (guess_idx == answer_idx) continue; // Already counted correct guess
            
            const string& guess = all_words[guess_idx];
            const string& answer = all_words[answer_idx];
            Feedback feedback = getFeedback(guess, answer);
            feedback_groups[feedback].push_back(guess_idx);
        }
        
        // For each feedback pattern from wrong guesses
        for (auto& [feedback, guess_indices] : feedback_groups) {
            // Find words that match this feedback (and are still valid)
            vector<int> new_valid_words;
            for (int word_idx : valid_words) {
                const string& word = all_words[word_idx];
                if (wordMatchesFeedback(word, all_words[guess_indices[0]], feedback)) {
                    new_valid_words.push_back(word_idx);
                }
            }
            
            // Continue with random play
            double sub_success = simulateRandomPlay(new_valid_words, answer_idx, guesses_left - 1);
            
            // Weight by probability of making any guess that produces this feedback
            double prob_this_feedback = (double)guess_indices.size() / valid_words.size();
            total_success += prob_this_feedback * sub_success;
        }
        
        // Cache result
        play_cache[cache_key] = total_success;
        return total_success;
    }
    
public:
    WordleRandomPlaySolver(const vector<string>& words, int max_guesses = 3) 
        : all_words(words), max_guesses(max_guesses) {}
    
    pair<double, string> solve() {
        double best_prob = 0.0;
        int best_guess_idx = -1;
        
        cout << "Evaluating all " << all_words.size() << " possible first guesses..." << endl;
        cout << "Assuming random play (random guess from valid words) after first guess" << endl;
        
        // Evaluate each possible first guess
        for (int i = 0; i < (int)all_words.size(); i++) {
            if ((i + 1) % 100 == 0) {
                cout << "  Progress: " << (i + 1) << "/" << all_words.size() 
                     << " (current best: " << (best_guess_idx >= 0 ? all_words[best_guess_idx] : "none")
                     << " with prob " << fixed << setprecision(4) << best_prob << ")" << endl;
            }
            
            double prob = computeRandomPlaySuccess(i, max_guesses);
            
            if (prob > best_prob) {
                best_prob = prob;
                best_guess_idx = i;
            }
        }
        
        string best_guess = (best_guess_idx >= 0) ? all_words[best_guess_idx] : "";
        return {best_prob, best_guess};
    }
    
    // Analyze a specific guess in detail
    void analyzeGuess(const string& guess) {
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
        
        cout << "\nAnalyzing guess: " << guess << endl;
        
        // Group answers by feedback
        unordered_map<Feedback, vector<int>, FeedbackHash> feedback_groups;
        
        for (int i = 0; i < (int)all_words.size(); i++) {
            const string& answer = all_words[i];
            Feedback feedback = getFeedback(guess, answer);
            feedback_groups[feedback].push_back(i);
        }
        
        cout << "\nFeedback distribution:" << endl;
        vector<pair<int, Feedback>> sorted_groups;
        for (auto& [feedback, words] : feedback_groups) {
            sorted_groups.push_back({words.size(), feedback});
        }
        sort(sorted_groups.rbegin(), sorted_groups.rend());
        
        double total_expected = 0.0;
        for (auto& [count, feedback] : sorted_groups) {
            bool is_correct = (get<0>(feedback) == 2 && get<1>(feedback) == 2 && 
                              get<2>(feedback) == 2 && get<3>(feedback) == 2 && 
                              get<4>(feedback) == 2);
            
            if (is_correct) {
                cout << "  All greens (CORRECT): " << count << " words" << endl;
                total_expected += (double)count / all_words.size();
            } else {
                // Compute expected success for this feedback group
                double group_success = 0.0;
                for (int answer_idx : feedback_groups[feedback]) {
                    // Find matching words
                    vector<int> matching;
                    for (int i = 0; i < (int)all_words.size(); i++) {
                        if (wordMatchesFeedback(all_words[i], guess, feedback)) {
                            matching.push_back(i);
                        }
                    }
                    
                    if (find(matching.begin(), matching.end(), answer_idx) != matching.end()) {
                        double success = simulateRandomPlay(matching, answer_idx, max_guesses - 1);
                        group_success += success;
                    }
                }
                group_success /= feedback_groups[feedback].size();
                
                cout << "  Feedback " << get<0>(feedback) << get<1>(feedback) << get<2>(feedback) 
                     << get<3>(feedback) << get<4>(feedback) << ": " << count << " words, "
                     << "expected success: " << fixed << setprecision(4) << group_success << endl;
                
                total_expected += ((double)count / all_words.size()) * group_success;
            }
        }
        
        cout << "\nOverall expected success probability: " << fixed << setprecision(4) 
             << total_expected << " (" << (total_expected * 100) << "%)" << endl;
    }
};

int main(int argc, char* argv[]) {
    string word_file = "wordle_words.txt";
    string analyze_word = "";
    int max_guesses = 3;
    
    // Parse arguments
    for (int i = 1; i < argc; i++) {
        if (string(argv[i]) == "--word-file" && i + 1 < argc) {
            word_file = argv[i + 1];
            i++;
        } else if (string(argv[i]) == "--analyze" && i + 1 < argc) {
            analyze_word = argv[i + 1];
            i++;
        } else if (string(argv[i]) == "--max-guesses" && i + 1 < argc) {
            max_guesses = stoi(argv[i + 1]);
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
    
    if (!analyze_word.empty()) {
        WordleRandomPlaySolver solver(word_list, max_guesses);
        solver.analyzeGuess(analyze_word);
        return 0;
    }
    
    cout << "Finding best first guess assuming random play after first guess..." << endl;
    cout << "(Random play = randomly guess from words matching all clues)" << endl;
    
    auto start = high_resolution_clock::now();
    
    WordleRandomPlaySolver solver(word_list, max_guesses);
    auto [prob, best_guess] = solver.solve();
    
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);
    
    cout << "\nOptimal first guess: " << best_guess << endl;
    cout << fixed << setprecision(4);
    cout << "Success probability in " << max_guesses << " guesses (with random play): " 
         << prob << " (" << (prob * 100) << "%)" << endl;
    cout << "Computation time: " << duration.count() << " ms" << endl;
    
    return 0;
}

