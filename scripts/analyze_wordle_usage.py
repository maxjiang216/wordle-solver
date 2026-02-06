#!/usr/bin/env python3
"""
Analyze Wordle word usage:
- Find unused words (in wordle_words.txt but not in wordle_answers.csv)
- Categorize words by how many times they've been used historically
"""

import csv
from collections import Counter, defaultdict
from pathlib import Path

def load_historical_answers(csv_path):
    """Load all words from the historical answers CSV."""
    words = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            word = row['word'].strip().upper()
            if word:
                words.append(word)
    return words

def load_possible_words(txt_path):
    """Load all possible words from the text file."""
    words = set()
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            word = line.strip().upper()
            if word:
                words.add(word)
    return words

def categorize_by_usage(word_counts):
    """Categorize words by their usage count."""
    categories = defaultdict(list)
    for word, count in word_counts.items():
        categories[count].append(word)
    
    # Sort words within each category alphabetically
    for count in categories:
        categories[count].sort()
    
    return categories

def main():
    # File paths
    csv_path = Path('wordle_answers.csv')
    txt_path = Path('wordle_words.txt')
    
    print("Loading historical answers...")
    historical_words = load_historical_answers(csv_path)
    print(f"Loaded {len(historical_words)} historical answers")
    
    print("\nLoading possible words...")
    possible_words = load_possible_words(txt_path)
    print(f"Loaded {len(possible_words)} possible words")
    
    # Count word usage
    print("\nAnalyzing word usage...")
    word_counts = Counter(historical_words)
    historical_word_set = set(historical_words)
    
    # Find unused words
    unused_words = possible_words - historical_word_set
    unused_words = sorted(list(unused_words))
    
    # Categorize by usage
    categories = categorize_by_usage(word_counts)
    
    # Print results
    print("\n" + "="*80)
    print("WORDLE WORD USAGE ANALYSIS")
    print("="*80)
    
    print(f"\nTotal historical answers: {len(historical_words)}")
    print(f"Unique words used: {len(historical_word_set)}")
    print(f"Total possible words: {len(possible_words)}")
    print(f"Unused words: {len(unused_words)}")
    
    # Show unused words
    print("\n" + "-"*80)
    print("UNUSED WORDS (never used as answers):")
    print("-"*80)
    if unused_words:
        print(f"\nTotal: {len(unused_words)} words")
        # Print in columns for better readability
        cols = 8
        for i in range(0, len(unused_words), cols):
            row = unused_words[i:i+cols]
            print("  " + "  ".join(f"{word:8s}" for word in row))
    else:
        print("  None - all possible words have been used!")
    
    # Show usage categories
    print("\n" + "-"*80)
    print("WORD USAGE CATEGORIES (by number of times used):")
    print("-"*80)
    
    # Sort categories by count (descending)
    sorted_counts = sorted(categories.keys(), reverse=True)
    
    for count in sorted_counts:
        words = categories[count]
        print(f"\nUsed {count} time{'s' if count > 1 else ''}: {len(words)} word{'s' if len(words) > 1 else ''}")
        if len(words) <= 20:
            # Show all words if 20 or fewer
            print("  " + ", ".join(words))
        else:
            # Show first 10 and last 10 if more than 20
            print("  " + ", ".join(words[:10]) + " ... " + ", ".join(words[-10:]))
    
    # Summary statistics
    print("\n" + "-"*80)
    print("SUMMARY STATISTICS:")
    print("-"*80)
    print(f"Words used once: {len(categories.get(1, []))}")
    print(f"Words used twice: {len(categories.get(2, []))}")
    print(f"Words used 3+ times: {sum(len(categories[c]) for c in sorted_counts if c >= 3)}")
    print(f"Most used word(s): {', '.join(categories[max(sorted_counts)])} ({max(sorted_counts)} times)")
    
    # Save detailed report to file
    output_file = Path('wordle_usage_report.txt')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("WORDLE WORD USAGE ANALYSIS REPORT\n")
        f.write("="*80 + "\n\n")
        f.write(f"Total historical answers: {len(historical_words)}\n")
        f.write(f"Unique words used: {len(historical_word_set)}\n")
        f.write(f"Total possible words: {len(possible_words)}\n")
        f.write(f"Unused words: {len(unused_words)}\n\n")
        
        f.write("\n" + "-"*80 + "\n")
        f.write("UNUSED WORDS (never used as answers):\n")
        f.write("-"*80 + "\n")
        if unused_words:
            for word in unused_words:
                f.write(f"{word}\n")
        else:
            f.write("None - all possible words have been used!\n")
        
        f.write("\n" + "-"*80 + "\n")
        f.write("WORD USAGE CATEGORIES:\n")
        f.write("-"*80 + "\n")
        for count in sorted_counts:
            words = categories[count]
            f.write(f"\nUsed {count} time{'s' if count > 1 else ''}: {len(words)} word{'s' if len(words) > 1 else ''}\n")
            for word in words:
                f.write(f"  {word}\n")
    
    print(f"\nDetailed report saved to: {output_file}")

if __name__ == '__main__':
    main()


