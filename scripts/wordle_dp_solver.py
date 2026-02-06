#!/usr/bin/env python3
"""
Dynamic Programming solver for Wordle to maximize probability of success in 3 guesses.
"""

from typing import List, Tuple, Dict, Set, FrozenSet
from collections import defaultdict
from functools import lru_cache
import sys


def get_feedback(guess: str, answer: str) -> Tuple[int, ...]:
    """
    Compute Wordle feedback pattern for a guess against an answer.
    Returns a tuple of 5 integers:
    2 = green (correct letter, correct position)
    1 = yellow (correct letter, wrong position)
    0 = gray (letter not in word)
    """
    feedback = [0] * 5
    answer_chars = list(answer)
    guess_chars = list(guess)
    
    # First pass: mark greens
    for i in range(5):
        if guess_chars[i] == answer_chars[i]:
            feedback[i] = 2
            answer_chars[i] = None  # Mark as used
            guess_chars[i] = None   # Mark as used
    
    # Second pass: mark yellows
    for i in range(5):
        if guess_chars[i] is not None:
            if guess_chars[i] in answer_chars:
                feedback[i] = 1
                # Remove one occurrence from answer_chars
                idx = answer_chars.index(guess_chars[i])
                answer_chars[idx] = None
    
    return tuple(feedback)


def filter_words(words: FrozenSet[str], guess: str, feedback: Tuple[int, ...]) -> FrozenSet[str]:
    """Filter words that are consistent with the given guess and feedback."""
    result = []
    for word in words:
        if get_feedback(guess, word) == feedback:
            result.append(word)
    return frozenset(result)


class WordleSolver:
    def __init__(self, word_list: List[str], max_guesses_to_try: int = None):
        self.all_words = [w.strip().lower() for w in word_list if w.strip()]
        self.word_set = frozenset(self.all_words)
        self.n_words = len(self.all_words)
        
        # Limit on number of guesses to try (None = try all)
        self.max_guesses_to_try = max_guesses_to_try
        
        # Memoization cache
        self.cache = {}
        
        # Cache for feedback computations (guess -> answer -> feedback)
        self.feedback_cache = {}
    
    def solve(self, max_guesses: int = 3) -> Tuple[float, str]:
        """
        Find the optimal strategy to maximize probability of success in max_guesses.
        Returns (success_probability, best_first_guess).
        """
        prob, guess = self._solve_recursive(self.word_set, max_guesses)
        return prob, guess
    
    def _solve_recursive(self, possible_words: FrozenSet[str], guesses_left: int) -> Tuple[float, str]:
        """
        Recursively solve for the best strategy.
        Returns (best_success_probability, best_guess).
        """
        # Base cases
        if len(possible_words) == 0:
            return 0.0, ""
        
        if len(possible_words) == 1:
            # If we have guesses left, we can guess the word
            return 1.0 if guesses_left > 0 else 0.0, next(iter(possible_words))
        
        if guesses_left == 0:
            return 0.0, ""
        
        # Check cache
        cache_key = (possible_words, guesses_left)
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        best_prob = 0.0
        best_guess = ""
        
        # Try each possible guess
        # Strategy: For optimal results, we should consider all words as potential guesses
        # But for efficiency with large word lists, we can limit the search
        
        if self.max_guesses_to_try is None:
            # Try all words for true optimality
            if len(possible_words) <= 50:
                guesses_to_try = list(possible_words) + [w for w in self.all_words if w not in possible_words]
            elif len(possible_words) <= 500:
                # For medium sets, try all possible words + sample from all words
                guesses_to_try = list(possible_words) + self.all_words[:500]
            else:
                # For very large sets, default to trying all possible words (hard mode)
                # This is still optimal if we're restricted to guessing valid words
                guesses_to_try = list(possible_words)
        else:
            # Limited search
            if len(possible_words) <= self.max_guesses_to_try:
                guesses_to_try = list(possible_words)
            else:
                guesses_to_try = list(possible_words)[:self.max_guesses_to_try] + self.all_words[:self.max_guesses_to_try//2]
        
        # Progress tracking for large computations
        if len(guesses_to_try) > 500:
            print(f"  Evaluating {len(guesses_to_try)} guesses for {len(possible_words)} possible words ({guesses_left} guesses left)...", end='\r', file=sys.stderr)
        
        for i, guess in enumerate(guesses_to_try):
            # Compute expected success probability for this guess
            expected_prob = self._expected_success_prob(guess, possible_words, guesses_left)
            
            if expected_prob > best_prob:
                best_prob = expected_prob
                best_guess = guess
                
                # Early termination: if we found a perfect solution, we're done
                if best_prob >= 1.0 - 1e-9:
                    break
        
        # Cache result
        self.cache[cache_key] = (best_prob, best_guess)
        return best_prob, best_guess
    
    def _expected_success_prob(self, guess: str, possible_words: FrozenSet[str], guesses_left: int) -> float:
        """
        Compute expected success probability for a given guess.
        Assumes uniform distribution over possible_words.
        """
        # Group possible answers by the feedback pattern they would produce
        feedback_groups = defaultdict(list)
        for answer in possible_words:
            # Use cached feedback if available
            cache_key = (guess, answer)
            if cache_key in self.feedback_cache:
                feedback = self.feedback_cache[cache_key]
            else:
                feedback = get_feedback(guess, answer)
                self.feedback_cache[cache_key] = feedback
            feedback_groups[feedback].append(answer)
        
        # Compute weighted average of success probabilities
        total_prob = 0.0
        n_total = len(possible_words)
        
        # For each feedback pattern, compute success probability
        for feedback, words in feedback_groups.items():
            if not words:
                continue
            
            remaining_words = frozenset(words)
            
            # If this feedback pattern means we guessed correctly (all greens)
            # and the guess is in the remaining words, we win immediately
            is_correct = all(f == 2 for f in feedback) and guess in remaining_words
            
            if is_correct:
                # We win for all words that produce this pattern (should be just the guess itself)
                total_prob += len(words) / n_total
            else:
                # Recursively solve for this feedback pattern
                sub_prob, _ = self._solve_recursive(remaining_words, guesses_left - 1)
                
                # Weight by probability of this feedback pattern
                weight = len(words) / n_total
                total_prob += weight * sub_prob
        
        return total_prob
    
    def get_strategy_tree(self, max_guesses: int = 3, max_depth: int = 3) -> Dict:
        """
        Build a strategy tree showing the optimal guesses at each step.
        """
        def build_tree(possible_words: FrozenSet[str], guesses_left: int, depth: int) -> Dict:
            if depth >= max_depth or guesses_left == 0 or len(possible_words) == 0:
                return {"words": list(possible_words), "count": len(possible_words)}
            
            if len(possible_words) == 1:
                return {"guess": next(iter(possible_words)), "words": list(possible_words), "count": 1}
            
            prob, guess = self._solve_recursive(possible_words, guesses_left)
            
            # Get feedback groups for this guess
            feedback_groups = defaultdict(list)
            for answer in possible_words:
                feedback = get_feedback(guess, answer)
                feedback_groups[feedback].append(answer)
            
            tree = {
                "guess": guess,
                "probability": prob,
                "count": len(possible_words),
                "children": {}
            }
            
            # Build subtrees for each feedback pattern (limit to top patterns for display)
            for feedback, words in list(feedback_groups.items())[:10]:  # Limit for readability
                remaining_words = frozenset(words)
                if guess in remaining_words:
                    tree["children"][str(feedback) + " (CORRECT)"] = {"guess": guess, "win": True}
                else:
                    tree["children"][str(feedback)] = build_tree(remaining_words, guesses_left - 1, depth + 1)
            
            return tree
        
        return build_tree(self.word_set, max_guesses, 0)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Find optimal Wordle strategy using DP')
    parser.add_argument('--max-guesses-to-try', type=int, default=None,
                        help='Limit number of guesses to try at each step (for faster computation)')
    parser.add_argument('--word-file', type=str, default='wordle_words.txt',
                        help='Path to word list file')
    args = parser.parse_args()
    
    # Read word list
    with open(args.word_file, 'r') as f:
        word_list = f.readlines()
    
    print(f"Loaded {len(word_list)} words")
    if args.max_guesses_to_try:
        print(f"Limiting to {args.max_guesses_to_try} guesses per step (approximate solution)")
    else:
        print("Computing optimal strategy (this may take a while)...")
    
    solver = WordleSolver(word_list, max_guesses_to_try=args.max_guesses_to_try)
    
    # Solve for 3 guesses
    prob, best_guess = solver.solve(max_guesses=3)
    
    print(f"\nOptimal first guess: {best_guess}")
    print(f"Success probability in 3 guesses: {prob:.4f} ({prob*100:.2f}%)")
    
    # Show some statistics
    print(f"\nCache size: {len(solver.cache)} states")
    
    # Build and display strategy tree (limited depth)
    print("\nBuilding strategy tree (first 2 levels)...")
    tree = solver.get_strategy_tree(max_guesses=3, max_depth=2)
    
    def print_tree(node, indent=0):
        prefix = "  " * indent
        if "win" in node:
            print(f"{prefix}✓ WIN! (guessed correctly)")
        elif "guess" in node:
            print(f"{prefix}Guess: {node['guess']}")
            if "probability" in node:
                print(f"{prefix}  Success prob: {node['probability']:.4f}")
            print(f"{prefix}  Possible words: {node['count']}")
            if "children" in node:
                for feedback, child in list(node['children'].items())[:5]:  # Show top 5
                    print(f"{prefix}  Feedback {feedback}:")
                    print_tree(child, indent + 2)
        elif "words" in node:
            print(f"{prefix}Remaining: {node['count']} words")
            if node['count'] <= 5:
                print(f"{prefix}  {', '.join(node['words'])}")
    
    print("\nStrategy tree:")
    print_tree(tree)


if __name__ == "__main__":
    main()

