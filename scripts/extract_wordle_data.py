#!/usr/bin/env python3
"""
Extract Wordle historical answers from HTML and create a clean CSV file.
"""

import re
import csv
from datetime import datetime
from collections import OrderedDict

def parse_date(date_str, year):
    """Parse date string like 'Feb. 02' or 'Jun. 19' into a datetime object."""
    month_map = {
        'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
        'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
    }
    
    # Remove periods and split
    parts = date_str.replace('.', '').strip().split()
    if len(parts) != 2:
        return None
    
    month_name, day = parts
    month = month_map.get(month_name, None)
    if month is None:
        return None
    
    try:
        day_num = int(day)
        return datetime(year, month, day_num)
    except ValueError:
        return None

def extract_wordle_data(html_file):
    """Extract Wordle answers with dates from HTML file."""
    print(f"Reading HTML file...")
    with open(html_file, 'r', encoding='utf-8') as f:
        html = f.read()
    print(f"HTML file read, size: {len(html):,} characters")
    
    # Create a mapping of month sections
    month_map = {
        'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
        'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
    }
    
    # Find all month/year headers and their positions
    print("Finding month/year section boundaries...")
    month_year_pattern = r'All\s+([A-Z][a-z]+)\s+(\d{4})\s+Wordle\s+Answers'
    section_boundaries = []
    for match in re.finditer(month_year_pattern, html):
        month_name = match.group(1)
        year = int(match.group(2))
        section_boundaries.append((match.start(), year, month_map.get(month_name)))
    
    print(f"Found {len(section_boundaries)} month sections")
    
    # Now find all word entries
    print("Finding word entries...")
    # Pattern: date, number, word in sequence within a table row
    # The structure is: Date\n</td> <td>...Number\n</td> <td>...Word\n</td>
    # Match text before closing tags
    pattern = r'([A-Z][a-z]+\.?\s+\d{1,2})\s*</td>.*?(\d+)\s*</td>.*?([A-Z]{5})\s*</td>'
    
    all_matches = []
    print("  Searching with pattern (this may take a moment)...")
    match_count = 0
    for match in re.finditer(pattern, html, re.DOTALL):
        date_str, puzzle_num, word = match.groups()
        pos = match.start()
        match_count += 1
        
        # Find which section this belongs to
        year = None
        for section_pos, section_year, section_month in section_boundaries:
            if pos >= section_pos:
                year = section_year
            else:
                break
        
        if year:
            all_matches.append((pos, date_str.strip(), puzzle_num.strip(), word.strip(), year))
        
        if match_count % 100 == 0:
            print(f"    Processed {match_count} matches, {len(all_matches)} valid entries so far...")
    
    print(f"  Total matches found: {match_count}, valid entries: {len(all_matches)}")
    
    print(f"Found {len(all_matches)} potential entries")
    
    # Process matches
    print("Processing and validating entries...")
    all_entries = []
    for pos, date_str, puzzle_num, word, year in all_matches:
        parsed_date = parse_date(date_str, year)
        if parsed_date:
            all_entries.append({
                'date': parsed_date,
                'puzzle_number': int(puzzle_num),
                'word': word
            })
    
    print(f"Valid entries: {len(all_entries)}")
    
    # Sort by date
    print("Sorting entries by date...")
    all_entries.sort(key=lambda x: x['date'])
    
    return all_entries

def write_csv(data, output_file):
    """Write Wordle data to CSV file."""
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['date', 'puzzle_number', 'word'])
        writer.writeheader()
        writer.writerows(data)

def main():
    html_file = 'historical_words.html'
    output_file = 'wordle_answers.csv'
    
    print(f"Extracting Wordle data from {html_file}...")
    data = extract_wordle_data(html_file)
    
    print(f"Found {len(data)} Wordle answers")
    
    if data:
        # Convert dates to strings for CSV
        csv_data = []
        for entry in data:
            csv_data.append({
                'date': entry['date'].strftime('%Y-%m-%d'),
                'puzzle_number': entry['puzzle_number'],
                'word': entry['word']
            })
        
        write_csv(csv_data, output_file)
        print(f"Data written to {output_file}")
        
        # Print first few entries
        print("\nFirst 10 entries:")
        for entry in csv_data[:10]:
            print(f"  {entry['date']} - Puzzle #{entry['puzzle_number']}: {entry['word']}")
        
        print(f"\nLast 10 entries:")
        for entry in csv_data[-10:]:
            print(f"  {entry['date']} - Puzzle #{entry['puzzle_number']}: {entry['word']}")
    else:
        print("No data extracted. Check the HTML structure.")

if __name__ == '__main__':
    main()

