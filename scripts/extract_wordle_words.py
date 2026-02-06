#!/usr/bin/env python3
"""
Extract Wordle words from wordle_answer_list.html
"""

import re
from html.parser import HTMLParser


class WordleWordExtractor(HTMLParser):
    """HTML parser to extract words from <li class="invert light"> tags."""
    
    def __init__(self):
        super().__init__()
        self.words = []
        self.in_li_invert_light = False
        self.in_a_tag = False
        self.current_word = None
    
    def handle_starttag(self, tag, attrs):
        if tag == 'li':
            # Check if this li has class="invert light"
            attrs_dict = dict(attrs)
            classes = attrs_dict.get('class', [])
            if isinstance(classes, str):
                classes = classes.split()
            if 'invert' in classes and 'light' in classes:
                self.in_li_invert_light = True
        elif tag == 'a' and self.in_li_invert_light:
            self.in_a_tag = True
    
    def handle_endtag(self, tag):
        if tag == 'li':
            if self.in_li_invert_light and self.current_word:
                word = self.current_word.strip().lower()
                # Only add 5-letter words
                if len(word) == 5 and word.isalpha():
                    self.words.append(word)
                self.current_word = None
            self.in_li_invert_light = False
            self.in_a_tag = False
        elif tag == 'a':
            self.in_a_tag = False
    
    def handle_data(self, data):
        if self.in_a_tag and self.in_li_invert_light:
            if self.current_word is None:
                self.current_word = data
            else:
                self.current_word += data


def extract_wordle_words(html_file):
    """
    Extract all Wordle words from the HTML file.
    
    Args:
        html_file: Path to the HTML file
        
    Returns:
        List of word strings
    """
    with open(html_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    parser = WordleWordExtractor()
    parser.feed(content)
    
    return parser.words


def main():
    html_file = 'wordle_answer_list.html'
    
    print(f"Extracting words from {html_file}...")
    words = extract_wordle_words(html_file)
    
    print(f"Found {len(words)} words")
    
    # Save to a text file (one word per line)
    output_file = 'wordle_words.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        for word in words:
            f.write(word + '\n')
    
    print(f"Words saved to {output_file}")
    
    # Also print first 20 words as a sample
    print("\nFirst 20 words:")
    for word in words[:20]:
        print(f"  {word}")
    
    if len(words) > 20:
        print(f"\n... and {len(words) - 20} more words")
    
    return words


if __name__ == '__main__':
    main()

