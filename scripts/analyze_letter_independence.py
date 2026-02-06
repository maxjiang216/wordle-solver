#!/usr/bin/env python3
"""
Analyze letter independence in Wordle word choices.
Test if the conditional probability of repeating a letter is independent
for consecutive days.
"""

import csv
from collections import Counter, defaultdict
from pathlib import Path
import string
import math
import numpy as np

# Try to import scipy, fall back to manual implementations if not available
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Note: scipy not available, using manual statistical test implementations")

def load_historical_answers(csv_path):
    """Load all words from the historical answers CSV in order."""
    words = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            word = row['word'].strip().upper()
            if word:
                words.append(word)
    return words

def get_letters_in_word(word):
    """Get set of unique letters in a word."""
    return set(word)

def binom_coeff(n, k):
    """Calculate binomial coefficient C(n, k)."""
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    k = min(k, n - k)  # Take advantage of symmetry
    result = 1
    for i in range(k):
        result = result * (n - i) // (i + 1)
    return result

def binom_pmf(k, n, p):
    """Binomial probability mass function."""
    if p < 0 or p > 1:
        return 0
    if k < 0 or k > n:
        return 0
    return binom_coeff(n, k) * (p ** k) * ((1 - p) ** (n - k))

def binom_test_manual(x, n, p, alternative='two-sided'):
    """
    Manual binomial test implementation.
    Returns p-value for testing if observed x successes in n trials
    is significantly different from expected probability p.
    """
    # Calculate p-value based on alternative hypothesis
    if alternative == 'two-sided':
        # Two-tailed: sum probabilities of outcomes as or more extreme
        p_obs = binom_pmf(x, n, p)
        p_value = 0
        for k in range(n + 1):
            p_k = binom_pmf(k, n, p)
            if p_k <= p_obs:
                p_value += p_k
    elif alternative == 'less':
        # Left-tailed: probability of x or fewer successes
        p_value = sum(binom_pmf(k, n, p) for k in range(x + 1))
    elif alternative == 'greater':
        # Right-tailed: probability of x or more successes
        p_value = sum(binom_pmf(k, n, p) for k in range(x, n + 1))
    else:
        raise ValueError("alternative must be 'two-sided', 'less', or 'greater'")
    
    return min(1.0, p_value)  # Cap at 1.0

def ttest_rel_manual(x, y):
    """
    Manual paired t-test implementation.
    Tests if the mean difference between paired samples is significantly different from zero.
    Returns (t-statistic, p-value).
    """
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    
    n = len(x)
    if n < 2:
        return (0.0, 1.0)
    
    # Calculate differences
    d = np.array(x) - np.array(y)
    d_mean = np.mean(d)
    d_std = np.std(d, ddof=1)  # Sample standard deviation
    
    if d_std == 0:
        return (0.0, 1.0)
    
    # t-statistic
    t_stat = d_mean / (d_std / math.sqrt(n))
    
    # p-value using t-distribution approximation
    # For large n, t-distribution approximates normal
    # For simplicity, we'll use normal approximation (works well for n > 30)
    if n > 30:
        # Normal approximation
        p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(t_stat) / math.sqrt(2))))
    else:
        # For smaller samples, we'd need t-distribution
        # Using a rough approximation: t-distribution is wider than normal
        # This is a simplified version - for exact results, scipy is better
        z_score = abs(t_stat)
        p_value = 2 * (1 - 0.5 * (1 + math.erf(z_score / math.sqrt(2))))
        # Adjust for smaller sample size (rough approximation)
        if n < 30:
            p_value *= 1.1  # Slightly increase p-value for smaller samples
    
    return (t_stat, min(1.0, p_value))

def calculate_unconditional_probabilities(words):
    """Calculate unconditional probability of each letter appearing in a word."""
    letter_counts = Counter()
    total_words = len(words)
    
    for word in words:
        letters = get_letters_in_word(word)
        for letter in letters:
            letter_counts[letter] += 1
    
    # Convert to probabilities
    probabilities = {}
    for letter in string.ascii_uppercase:
        probabilities[letter] = letter_counts[letter] / total_words if total_words > 0 else 0
    
    return probabilities, letter_counts

def analyze_consecutive_days(words):
    """Analyze letter independence between consecutive days."""
    # Track transitions
    transitions = defaultdict(lambda: {'appeared_next': 0, 'not_appeared_next': 0})
    total_transitions = len(words) - 1
    
    # For each consecutive pair
    for i in range(len(words) - 1):
        word_today = words[i]
        word_tomorrow = words[i + 1]
        
        letters_today = get_letters_in_word(word_today)
        letters_tomorrow = get_letters_in_word(word_tomorrow)
        
        # For each letter that appeared today
        for letter in letters_today:
            if letter in letters_tomorrow:
                transitions[letter]['appeared_next'] += 1
            else:
                transitions[letter]['not_appeared_next'] += 1
    
    return transitions, total_transitions

def calculate_conditional_probabilities(transitions, unconditional_probs, total_words):
    """Calculate conditional probabilities and compare to unconditional."""
    results = {}
    
    for letter in string.ascii_uppercase:
        if letter not in transitions:
            continue
        
        trans = transitions[letter]
        total_times_letter_appeared = trans['appeared_next'] + trans['not_appeared_next']
        
        if total_times_letter_appeared == 0:
            continue
        
        # Conditional probability: P(letter appears tomorrow | letter appeared today)
        conditional_prob = trans['appeared_next'] / total_times_letter_appeared
        
        # Unconditional probability: P(letter appears in any word)
        unconditional_prob = unconditional_probs[letter]
        
        # Difference
        difference = conditional_prob - unconditional_prob
        
        # Ratio
        ratio = conditional_prob / unconditional_prob if unconditional_prob > 0 else 0
        
        # Statistical significance test (binomial test)
        # H0: P(letter appears tomorrow | letter appeared today) = P(letter appears in any word)
        # Test if observed successes significantly differ from expected
        observed_successes = trans['appeared_next']
        n_trials = total_times_letter_appeared
        expected_prob = unconditional_prob
        
        if n_trials > 0 and expected_prob > 0 and expected_prob < 1:
            # Two-tailed binomial test
            if SCIPY_AVAILABLE:
                p_value = stats.binom_test(observed_successes, n_trials, expected_prob, alternative='two-sided')
                p_value_less = stats.binom_test(observed_successes, n_trials, expected_prob, alternative='less')
                p_value_greater = stats.binom_test(observed_successes, n_trials, expected_prob, alternative='greater')
            else:
                p_value = binom_test_manual(observed_successes, n_trials, expected_prob, alternative='two-sided')
                p_value_less = binom_test_manual(observed_successes, n_trials, expected_prob, alternative='less')
                p_value_greater = binom_test_manual(observed_successes, n_trials, expected_prob, alternative='greater')
        else:
            p_value = 1.0
            p_value_less = 1.0
            p_value_greater = 1.0
        
        results[letter] = {
            'times_appeared_today': total_times_letter_appeared,
            'times_appeared_tomorrow': trans['appeared_next'],
            'conditional_prob': conditional_prob,
            'unconditional_prob': unconditional_prob,
            'difference': difference,
            'ratio': ratio,
            'p_value': p_value,
            'p_value_less': p_value_less,
            'p_value_greater': p_value_greater,
            'significant': p_value < 0.05
        }
    
    return results

def analyze_letter_overlap(words, unconditional_probs):
    """
    Analyze overall letter overlap between consecutive days.
    Test if the intersection size is different from what we'd expect under independence.
    """
    actual_intersections = []
    expected_intersections = []
    
    for i in range(len(words) - 1):
        word_today = words[i]
        word_tomorrow = words[i + 1]
        
        letters_today = get_letters_in_word(word_today)
        letters_tomorrow = get_letters_in_word(word_tomorrow)
        
        # Actual intersection size
        actual_intersection = len(letters_today & letters_tomorrow)
        actual_intersections.append(actual_intersection)
        
        # Expected intersection size under independence
        # E[intersection] = sum over all letters of P(letter in word1) * P(letter in word2)
        # But we need to account for the fact that words have exactly 5 letters
        # So we use the unconditional probabilities
        expected_intersection = 0
        for letter in string.ascii_uppercase:
            # Probability letter appears in both words under independence
            prob_in_today = 1.0 if letter in letters_today else 0.0
            prob_in_tomorrow = unconditional_probs[letter]
            expected_intersection += prob_in_today * prob_in_tomorrow
        
        # Alternative: use average unconditional probability for both words
        # This is a simpler approximation
        # E[intersection] ≈ sum of P(letter)^2 for all letters in today's word
        expected_intersection_alt = sum(unconditional_probs[letter] for letter in letters_today)
        
        expected_intersections.append(expected_intersection_alt)
    
    return actual_intersections, expected_intersections

def analyze_extended_patterns(words, max_days=5):
    """Analyze patterns beyond just consecutive days."""
    patterns = {}
    
    for letter in string.ascii_uppercase:
        patterns[letter] = {}
        for days_ahead in range(1, max_days + 1):
            appeared_today_and_ahead = 0
            appeared_today = 0
            
            for i in range(len(words) - days_ahead):
                word_today = words[i]
                word_future = words[i + days_ahead]
                
                letters_today = get_letters_in_word(word_today)
                letters_future = get_letters_in_word(word_future)
                
                if letter in letters_today:
                    appeared_today += 1
                    if letter in letters_future:
                        appeared_today_and_ahead += 1
            
            if appeared_today > 0:
                conditional_prob = appeared_today_and_ahead / appeared_today
                patterns[letter][days_ahead] = {
                    'conditional_prob': conditional_prob,
                    'count': appeared_today_and_ahead,
                    'total': appeared_today
                }
    
    return patterns

def main():
    csv_path = Path('wordle_answers.csv')
    
    print("Loading historical answers...")
    words = load_historical_answers(csv_path)
    print(f"Loaded {len(words)} historical answers")
    
    print("\nCalculating unconditional probabilities...")
    unconditional_probs, letter_counts = calculate_unconditional_probabilities(words)
    
    print("\nAnalyzing consecutive day transitions...")
    transitions, total_transitions = analyze_consecutive_days(words)
    
    print("\nCalculating conditional probabilities...")
    results = calculate_conditional_probabilities(transitions, unconditional_probs, len(words))
    
    print("\nAnalyzing overall letter overlap...")
    actual_intersections, expected_intersections = analyze_letter_overlap(words, unconditional_probs)
    
    print("\nAnalyzing extended patterns...")
    extended_patterns = analyze_extended_patterns(words, max_days=5)
    
    # Print results
    print("\n" + "="*100)
    print("LETTER INDEPENDENCE ANALYSIS")
    print("="*100)
    
    print(f"\nTotal word pairs analyzed: {total_transitions}")
    
    # Overall letter overlap analysis
    print("\n" + "="*100)
    print("OVERALL LETTER OVERLAP ANALYSIS")
    print("="*100)
    
    avg_actual = np.mean(actual_intersections)
    avg_expected = np.mean(expected_intersections)
    overlap_difference = avg_actual - avg_expected
    
    print(f"\nAverage actual intersection size: {avg_actual:.4f} letters")
    print(f"Average expected intersection size (under independence): {avg_expected:.4f} letters")
    print(f"Difference: {overlap_difference:+.4f} letters")
    print(f"Ratio: {avg_actual / avg_expected:.4f}" if avg_expected > 0 else "Ratio: N/A")
    
    # Statistical test for overall overlap
    # Paired t-test: are actual intersections significantly different from expected?
    if SCIPY_AVAILABLE:
        t_stat, p_value_overlap = stats.ttest_rel(actual_intersections, expected_intersections)
    else:
        t_stat, p_value_overlap = ttest_rel_manual(actual_intersections, expected_intersections)
    print(f"\nPaired t-test (actual vs expected):")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value_overlap:.6f}")
    print(f"  Significant at α=0.05: {'Yes' if p_value_overlap < 0.05 else 'No'}")
    
    if overlap_difference < 0:
        print(f"\n→ Letters overlap LESS than expected under independence (avoidance pattern)")
    elif overlap_difference > 0:
        print(f"\n→ Letters overlap MORE than expected under independence (clustering pattern)")
    else:
        print(f"\n→ Letters overlap as expected under independence")
    
    # Summary statistics
    print("\n" + "-"*100)
    print("SUMMARY: Conditional vs Unconditional Probabilities (with Statistical Tests)")
    print("-"*100)
    print(f"{'Letter':<8} {'Appeared':<12} {'P(Next|Today)':<18} {'P(Any)':<12} {'Difference':<12} {'p-value':<12} {'Sig':<6}")
    print("-"*100)
    
    # Sort by absolute difference (most dependent first)
    sorted_results = sorted(results.items(), key=lambda x: abs(x[1]['difference']), reverse=True)
    
    for letter, data in sorted_results:
        sig_marker = "***" if data['significant'] else ""
        print(f"{letter:<8} "
              f"{data['times_appeared_today']:<12} "
              f"{data['conditional_prob']:<18.4f} "
              f"{data['unconditional_prob']:<12.4f} "
              f"{data['difference']:+.4f}      "
              f"{data['p_value']:<12.6f} "
              f"{sig_marker:<6}")
    
    print("\n*** = Statistically significant (p < 0.05)")
    
    # Statistical summary
    print("\n" + "-"*100)
    print("STATISTICAL SUMMARY")
    print("-"*100)
    
    letters_less_likely = [l for l, d in results.items() if d['difference'] < -0.01]
    letters_more_likely = [l for l, d in results.items() if d['difference'] > 0.01]
    letters_independent = [l for l, d in results.items() if -0.01 <= d['difference'] <= 0.01]
    
    # Statistically significant letters
    sig_less = [l for l, d in results.items() if d['significant'] and d['difference'] < 0]
    sig_more = [l for l, d in results.items() if d['significant'] and d['difference'] > 0]
    
    print(f"\nLetters LESS likely to appear next day (difference < -0.01): {len(letters_less_likely)}")
    if letters_less_likely:
        print(f"  {', '.join(sorted(letters_less_likely))}")
    print(f"  Statistically significant: {len(sig_less)} - {', '.join(sorted(sig_less))}")
    
    print(f"\nLetters MORE likely to appear next day (difference > 0.01): {len(letters_more_likely)}")
    if letters_more_likely:
        print(f"  {', '.join(sorted(letters_more_likely))}")
    print(f"  Statistically significant: {len(sig_more)} - {', '.join(sorted(sig_more))}")
    
    print(f"\nLetters approximately independent (|difference| <= 0.01): {len(letters_independent)}")
    if letters_independent:
        print(f"  {', '.join(sorted(letters_independent))}")
    
    # Average difference
    avg_diff = sum(d['difference'] for d in results.values()) / len(results) if results else 0
    print(f"\nAverage difference (conditional - unconditional): {avg_diff:.4f}")
    
    # Count significant results
    total_sig = len(sig_less) + len(sig_more)
    print(f"\nTotal statistically significant letters (p < 0.05): {total_sig} out of {len(results)}")
    
    # Extended patterns
    print("\n" + "-"*100)
    print("EXTENDED PATTERNS: Conditional Probabilities by Days Ahead")
    print("-"*100)
    print(f"{'Letter':<8} {'Day+1':<12} {'Day+2':<12} {'Day+3':<12} {'Day+4':<12} {'Day+5':<12} {'Unconditional':<12}")
    print("-"*100)
    
    for letter in sorted(string.ascii_uppercase):
        if letter not in extended_patterns:
            continue
        
        pattern = extended_patterns[letter]
        unconditional = unconditional_probs[letter]
        
        probs = []
        for days in range(1, 6):
            if days in pattern:
                probs.append(f"{pattern[days]['conditional_prob']:.4f}")
            else:
                probs.append("N/A")
        
        print(f"{letter:<8} " + " ".join(f"{p:<12}" for p in probs) + f"{unconditional:<12.4f}")
    
    # Save detailed report
    output_file = Path('letter_independence_report.txt')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("LETTER INDEPENDENCE ANALYSIS REPORT\n")
        f.write("="*100 + "\n\n")
        f.write(f"Total words analyzed: {len(words)}\n")
        f.write(f"Total word pairs analyzed: {total_transitions}\n\n")
        
        f.write("\n" + "="*100 + "\n")
        f.write("OVERALL LETTER OVERLAP ANALYSIS\n")
        f.write("="*100 + "\n")
        f.write(f"\nAverage actual intersection size: {avg_actual:.4f} letters\n")
        f.write(f"Average expected intersection size (under independence): {avg_expected:.4f} letters\n")
        f.write(f"Difference: {overlap_difference:+.4f} letters\n")
        f.write(f"Ratio: {avg_actual / avg_expected:.4f}\n" if avg_expected > 0 else "Ratio: N/A\n")
        f.write(f"\nPaired t-test (actual vs expected):\n")
        f.write(f"  t-statistic: {t_stat:.4f}\n")
        f.write(f"  p-value: {p_value_overlap:.6f}\n")
        f.write(f"  Significant at α=0.05: {'Yes' if p_value_overlap < 0.05 else 'No'}\n")
        
        f.write("\n" + "-"*100 + "\n")
        f.write("DETAILED RESULTS: Conditional vs Unconditional Probabilities (with Statistical Tests)\n")
        f.write("-"*100 + "\n")
        f.write(f"{'Letter':<8} {'Appeared':<12} {'P(Next|Today)':<18} {'P(Any)':<12} "
                f"{'Difference':<12} {'p-value':<12} {'Sig':<6}\n")
        f.write("-"*100 + "\n")
        
        for letter, data in sorted_results:
            sig_marker = "***" if data['significant'] else ""
            f.write(f"{letter:<8} "
                   f"{data['times_appeared_today']:<12} "
                   f"{data['conditional_prob']:<18.4f} "
                   f"{data['unconditional_prob']:<12.4f} "
                   f"{data['difference']:+.4f}      "
                   f"{data['p_value']:<12.6f} "
                   f"{sig_marker:<6}\n")
        
        f.write("\n*** = Statistically significant (p < 0.05)\n")
        
        f.write("\n" + "-"*100 + "\n")
        f.write("EXTENDED PATTERNS\n")
        f.write("-"*100 + "\n")
        f.write(f"{'Letter':<8} {'Day+1':<12} {'Day+2':<12} {'Day+3':<12} "
                f"{'Day+4':<12} {'Day+5':<12} {'Unconditional':<12}\n")
        f.write("-"*100 + "\n")
        
        for letter in sorted(string.ascii_uppercase):
            if letter not in extended_patterns:
                continue
            
            pattern = extended_patterns[letter]
            unconditional = unconditional_probs[letter]
            
            probs = []
            for days in range(1, 6):
                if days in pattern:
                    probs.append(f"{pattern[days]['conditional_prob']:.4f}")
                else:
                    probs.append("N/A")
            
            f.write(f"{letter:<8} " + " ".join(f"{p:<12}" for p in probs) + f"{unconditional:<12.4f}\n")
    
    print(f"\nDetailed report saved to: {output_file}")
    
    # Interpretation
    print("\n" + "="*100)
    print("INTERPRETATION")
    print("="*100)
    print("\nIf letters were independent, we would expect:")
    print("  - Conditional probability ≈ Unconditional probability")
    print("  - Difference ≈ 0")
    print("  - Ratio ≈ 1.0")
    print("\nNegative differences indicate letters are LESS likely to appear the next day")
    print("Positive differences indicate letters are MORE likely to appear the next day")

if __name__ == '__main__':
    main()

