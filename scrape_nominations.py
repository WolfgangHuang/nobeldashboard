import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from typing import Dict, List, Optional
import logging
from tqdm import tqdm

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NobelNominationScraper:
    """Scraper for Nobel Prize nomination archive."""
    
    def __init__(self, base_url: str = "https://www.nobelprize.org/nomination/archive/show.php"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def fetch_nomination(self, nomination_id: int) -> Optional[BeautifulSoup]:
        """
        Fetch a single nomination page.
        
        Args:
            nomination_id: The nomination ID to fetch
            
        Returns:
            BeautifulSoup object or None if request fails
        """
        url = f"{self.base_url}?id={nomination_id}"
        
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            # Check if page exists (might return 200 but show "not found")
            if "not found" in response.text.lower() or len(response.text) < 100:
                return None
                
            return BeautifulSoup(response.content, 'html.parser')
            
        except requests.RequestException as e:
            logger.warning(f"Failed to fetch ID {nomination_id}: {e}")
            return None
    
    def parse_nomination(self, soup: BeautifulSoup, nomination_id: int) -> Optional[Dict]:
        """
        Parse nomination data from BeautifulSoup object with strict schema.
        
        Args:
            soup: BeautifulSoup object of the page
            nomination_id: The nomination ID
            
        Returns:
            Dictionary with nomination data or None if parsing fails
        """
        try:
            import re
            from collections import OrderedDict
            
            # Initialize data with strict schema - ORDERED!
            data = OrderedDict()
            
            # 1. NOMINATION fields first
            data['nomination_id'] = nomination_id
            data['nomination_title'] = ''
            data['nomination_year'] = ''
            data['nomination_number'] = ''
            data['nomination_num_nominees'] = 0
            data['nomination_num_nominators'] = 0
            data['nomination_category_from_title'] = ''
            data['nomination_motivation'] = ''
            data['nomination_comments'] = ''
            
            # Define allowed fields for nominees and nominators
            ALLOWED_NOMINEE_FIELDS = {
                'Name': 'name',
                'Gender': 'gender',
                'Year, Birth': 'year_birth',
                'Year, Death': 'year_death',
                'Profession': 'profession',
                'University': 'university',
                'City': 'city',
                'Country': 'country'
            }
            
            ALLOWED_NOMINATOR_FIELDS = {
                'Name': 'name',
                'Gender': 'gender',
                'Year, Birth': 'year_birth',
                'Year, Death': 'year_death',
                'Profession': 'profession',
                'University': 'university',
                'City': 'city',
                'Country': 'country',
                'Comments': 'comments'
            }
            
            # Storage for nominees and nominators (will be added in correct order later)
            nominees_data = {}
            nominators_data = {}
            
            # Storage for other information that doesn't fit schema
            other_info = []
            
            # Extract title line (contains category and sometimes year)
            title = soup.find('h1')
            if title:
                title_text = title.get_text(strip=True)
                data['nomination_title'] = title_text
                
                # Try to extract category from title
                categories = ['Physics', 'Chemistry', 'Physiology or Medicine', 'Literature', 'Peace']
                for cat in categories:
                    if cat in title_text:
                        data['nomination_category_from_title'] = cat
                        break
            
            # Find the main table with nomination data
            table = soup.find('table')
            if not table:
                return None
            
            rows = table.find_all('tr')
            
            # Extract title from first row if not found in h1
            if not data['nomination_title']:
                if len(rows) > 0:
                    first_row_cells = rows[0].find_all('td')
                    if first_row_cells:
                        title_text = first_row_cells[0].get_text(strip=True)
                        data['nomination_title'] = title_text
                        
                        # Try to extract category from title
                        categories = ['Physics', 'Chemistry', 'Physiology or Medicine', 'Literature', 'Peace']
                        for cat in categories:
                            if cat in title_text:
                                data['nomination_category_from_title'] = cat
                                break
            
            nominee_count = 0
            nominator_count = 0
            
            i = 0
            while i < len(rows):
                row = rows[i]
                cells = row.find_all('td')
                
                if len(cells) == 0:
                    i += 1
                    continue
                
                first_cell_text = cells[0].get_text(strip=True)
                
                # Skip the title row (Row 0)
                if 'Nomination for Nobel Prize' in first_cell_text:
                    i += 1
                    continue
                
                # NOMINATION INFO
                if first_cell_text == 'Year:' and len(cells) >= 2:
                    data['nomination_year'] = cells[1].get_text(strip=True)
                    
                elif first_cell_text == 'Number:' and len(cells) >= 2:
                    data['nomination_number'] = cells[1].get_text(strip=True)
                    
                elif first_cell_text == 'Motivation:' and len(cells) >= 2:
                    data['nomination_motivation'] = cells[1].get_text(strip=True)
                    
                elif first_cell_text == 'Comments:' and len(cells) >= 2:
                    # Only if this is a top-level comment (not inside nominator section)
                    if len(cells) == 2:
                        data['nomination_comments'] = cells[1].get_text(strip=True)
                
                # NOMINEE SECTION
                elif first_cell_text.startswith('Nominee'):
                    nominee_count += 1
                    prefix = f'nominee_{nominee_count}'
                    
                    # Initialize nominee fields
                    nominee_dict = OrderedDict()
                    nominee_dict[f'{prefix}_name'] = ''
                    nominee_dict[f'{prefix}_id'] = ''
                    nominee_dict[f'{prefix}_gender'] = ''
                    nominee_dict[f'{prefix}_year_birth'] = ''
                    nominee_dict[f'{prefix}_year_death'] = ''
                    nominee_dict[f'{prefix}_profession'] = ''
                    nominee_dict[f'{prefix}_university'] = ''
                    nominee_dict[f'{prefix}_city'] = ''
                    nominee_dict[f'{prefix}_country'] = ''
                    nominee_dict[f'{prefix}_awarded_prizes'] = ''
                    
                    # List to collect awarded prizes for this nominee
                    awarded_prizes = set()
                    
                    # Nominee data usually follows in subsequent rows
                    i += 1
                    while i < len(rows):
                        row = rows[i]
                        cells = row.find_all('td')
                        
                        if len(cells) == 0:
                            i += 1
                            continue
                        
                        # Handle single-cell rows (might be awards or empty rows)
                        if len(cells) == 1:
                            cell_text = cells[0].get_text(strip=True)
                            
                            # Check for awarded prizes
                            if 'Awarded the' in cell_text and 'Nobel Prize' in cell_text:
                                match = re.search(r'Nobel Prize in ([A-Za-z\s]+)\s+(\d{4})', cell_text)
                                if match:
                                    category = match.group(1).strip()
                                    year = match.group(2)
                                    awarded_prizes.add(f"{category} {year}")
                            
                            i += 1
                            
                            # Check if next row starts a new section
                            if i < len(rows):
                                next_row = rows[i]
                                next_cells = next_row.find_all('td')
                                if next_cells and (next_cells[0].get_text(strip=True).startswith('Nominee') or 
                                                next_cells[0].get_text(strip=True) == 'Nominator:'):
                                    i -= 1
                                    break
                            continue
                        
                        label = cells[0].get_text(strip=True).rstrip(':')
                        value = cells[1].get_text(strip=True)
                        
                        # Clean up double spaces in value
                        value = ' '.join(value.split())
                        
                        # Check if we've moved to a new section
                        if label.startswith('Nominee') or label == 'Nominator':
                            i -= 1
                            break
                        
                        # Only extract allowed fields
                        if label in ALLOWED_NOMINEE_FIELDS:
                            field_name = ALLOWED_NOMINEE_FIELDS[label]
                            nominee_dict[f'{prefix}_{field_name}'] = value
                            
                            # Extract ID from link if this is the Name field
                            if label == 'Name':
                                link = cells[1].find('a')
                                if link and link.get('href'):
                                    id_match = re.search(r'id=(\d+)', link.get('href'))
                                    if id_match:
                                        nominee_dict[f'{prefix}_id'] = id_match.group(1)
                        else:
                            # Collect unrecognized fields
                            if label and value and label not in ['', ' ']:
                                other_info.append(f"nominee_{nominee_count}_{label}: {value}")
                        
                        i += 1
                    
                    # Add collected prizes for this nominee
                    if awarded_prizes:
                        nominee_dict[f'{prefix}_awarded_prizes'] = ', '.join(sorted(awarded_prizes))
                    
                    # Store nominee data
                    nominees_data[nominee_count] = nominee_dict
                    
                    continue
                
                # NOMINATOR SECTION
                # NOMINATOR SECTION
                elif first_cell_text == 'Nominator:':
                    nominator_count += 1
                    prefix = f'nominator_{nominator_count}'
                    
                    # Initialize nominator fields
                    nominator_dict = OrderedDict()
                    nominator_dict[f'{prefix}_name'] = ''
                    nominator_dict[f'{prefix}_id'] = ''
                    nominator_dict[f'{prefix}_gender'] = ''
                    nominator_dict[f'{prefix}_year_birth'] = ''
                    nominator_dict[f'{prefix}_year_death'] = ''
                    nominator_dict[f'{prefix}_profession'] = ''
                    nominator_dict[f'{prefix}_university'] = ''
                    nominator_dict[f'{prefix}_city'] = ''
                    nominator_dict[f'{prefix}_country'] = ''
                    nominator_dict[f'{prefix}_comments'] = ''
                    nominator_dict[f'{prefix}_awarded_prizes'] = ''
                    
                    # List to collect awarded prizes for this nominator
                    awarded_prizes_nominator = set()
                    
                    # Special case: sometimes all nominator data is in one row with many cells
                    if len(cells) > 2:
                        # Parse inline nominator data
                        j = 1
                        while j < len(cells):
                            cell_text = cells[j].get_text(strip=True)
                            
                            # Check if this cell contains awarded prize info
                            if 'Awarded the' in cell_text and 'Nobel Prize' in cell_text:
                                match = re.search(r'Nobel Prize in ([A-Za-z\s]+)\s+(\d{4})', cell_text)
                                if match:
                                    category = match.group(1).strip()
                                    year = match.group(2)
                                    awarded_prizes_nominator.add(f"{category} {year}")
                                j += 1
                                continue
                            
                            label = cell_text.rstrip(':')
                            if j + 1 < len(cells):
                                value = cells[j + 1].get_text(strip=True)
                                
                                # Clean up double spaces in value
                                value = ' '.join(value.split())
                                
                                if label in ALLOWED_NOMINATOR_FIELDS:
                                    field_name = ALLOWED_NOMINATOR_FIELDS[label]
                                    nominator_dict[f'{prefix}_{field_name}'] = value
                                    
                                    # Extract ID from link if this is the Name field
                                    if label == 'Name':
                                        link = cells[j + 1].find('a')
                                        if link and link.get('href'):
                                            id_match = re.search(r'id=(\d+)', link.get('href'))
                                            if id_match:
                                                nominator_dict[f'{prefix}_id'] = id_match.group(1)
                                else:
                                    # Collect unrecognized fields
                                    if label and value:
                                        other_info.append(f"nominator_{nominator_count}_{label}: {value}")
                                
                                j += 2
                            else:
                                j += 1
                        
                        # After inline parsing, continue to check subsequent rows for more data and awarded prizes
                        i += 1
                        while i < len(rows):
                            row = rows[i]
                            cells = row.find_all('td')
                            
                            if len(cells) == 0:
                                i += 1
                                continue
                            
                            # Handle single-cell rows for awarded prizes
                            if len(cells) == 1:
                                cell_text = cells[0].get_text(strip=True)
                                
                                # Check for awarded prizes
                                if 'Awarded the' in cell_text and 'Nobel Prize' in cell_text:
                                    match = re.search(r'Nobel Prize in ([A-Za-z\s]+)\s+(\d{4})', cell_text)
                                    if match:
                                        category = match.group(1).strip()
                                        year = match.group(2)
                                        awarded_prizes_nominator.add(f"{category} {year}")
                                
                                i += 1
                                
                                # Check if next row starts a new section
                                if i < len(rows):
                                    next_row = rows[i]
                                    next_cells = next_row.find_all('td')
                                    if next_cells and (next_cells[0].get_text(strip=True).startswith('Nominee') or 
                                                    next_cells[0].get_text(strip=True) == 'Nominator:'):
                                        i -= 1
                                        break
                                continue
                            
                            # Handle multi-cell rows (Name:, Gender:, etc.) - these are duplicates after inline
                            if len(cells) >= 2:
                                label = cells[0].get_text(strip=True).rstrip(':')
                                
                                # Check if we've moved to a new section
                                if label.startswith('Nominee') or label == 'Nominator':
                                    i -= 1
                                    break
                                
                                # Skip these rows as we already parsed them inline
                                i += 1
                                continue
                            
                            i += 1
                        
                        # Add collected prizes for this nominator
                        if awarded_prizes_nominator:
                            nominator_dict[f'{prefix}_awarded_prizes'] = ', '.join(awarded_prizes_nominator)
                        
                    else:
                        # Nominator data follows in subsequent rows
                        i += 1
                        while i < len(rows):
                            row = rows[i]
                            cells = row.find_all('td')
                            
                            if len(cells) == 0:
                                i += 1
                                continue
                            
                            # Handle single-cell rows for awarded prizes
                            if len(cells) == 1:
                                cell_text = cells[0].get_text(strip=True)
                                
                                # Check for awarded prizes
                                if 'Awarded the' in cell_text and 'Nobel Prize' in cell_text:
                                    match = re.search(r'Nobel Prize in ([A-Za-z\s]+)\s+(\d{4})', cell_text)
                                    if match:
                                        category = match.group(1).strip()
                                        year = match.group(2)
                                        awarded_prizes_nominator.add(f"{category} {year}")
                                
                                i += 1
                                
                                # Check if next row starts a new section
                                if i < len(rows):
                                    next_row = rows[i]
                                    next_cells = next_row.find_all('td')
                                    if next_cells and (next_cells[0].get_text(strip=True).startswith('Nominee') or 
                                                    next_cells[0].get_text(strip=True) == 'Nominator:'):
                                        i -= 1
                                        break
                                continue
                            
                            if len(cells) < 2:
                                i += 1
                                if i < len(rows):
                                    next_row = rows[i]
                                    next_cells = next_row.find_all('td')
                                    if next_cells and (next_cells[0].get_text(strip=True).startswith('Nominee') or 
                                                    next_cells[0].get_text(strip=True) == 'Nominator:'):
                                        i -= 1
                                        break
                                continue
                            
                            label = cells[0].get_text(strip=True).rstrip(':')
                            value = cells[1].get_text(strip=True)
                            
                            # Clean up double spaces in value
                            value = ' '.join(value.split())
                            
                            # Check if we've moved to a new section
                            if label.startswith('Nominee') or label == 'Nominator':
                                i -= 1
                                break
                            
                            if label in ALLOWED_NOMINATOR_FIELDS:
                                field_name = ALLOWED_NOMINATOR_FIELDS[label]
                                nominator_dict[f'{prefix}_{field_name}'] = value
                                
                                # Extract ID from link if this is the Name field
                                if label == 'Name':
                                    link = cells[1].find('a')
                                    if link and link.get('href'):
                                        id_match = re.search(r'id=(\d+)', link.get('href'))
                                        if id_match:
                                            nominator_dict[f'{prefix}_id'] = id_match.group(1)
                            else:
                                # Collect unrecognized fields
                                if label and value and label not in ['', ' ']:
                                    other_info.append(f"nominator_{nominator_count}_{label}: {value}")
                            
                            i += 1
                        
                        # Add collected prizes for this nominator
                        if awarded_prizes_nominator:
                            nominator_dict[f'{prefix}_awarded_prizes'] = ', '.join(sorted(awarded_prizes_nominator))                        
                        
                        continue
                    
                    # Store nominator data
                    nominators_data[nominator_count] = nominator_dict
                
                i += 1
            
            # Update counts
            data['nomination_num_nominees'] = nominee_count
            data['nomination_num_nominators'] = nominator_count
            
            # 2. Add NOMINATOR fields (all nominators)
            for nom_num in sorted(nominators_data.keys()):
                data.update(nominators_data[nom_num])
            
            # 3. Add NOMINEE fields (all nominees)
            for nominee_num in sorted(nominees_data.keys()):
                data.update(nominees_data[nominee_num])
            
            # 4. Add other information at the end
            if other_info:
                data['other_information'] = ' | '.join(other_info)
            else:
                data['other_information'] = ''
            
            return data if len(data) > 3 else None
            
        except Exception as e:
            logger.error(f"Failed to parse nomination {nomination_id}: {e}")
            import traceback
            traceback.print_exc()
            return None
            


    def debug_single_page(self, nomination_id: int):
        """
        Debug function to see what's being extracted from a single page.
        
        Args:
            nomination_id: The nomination ID to debug
        """
        soup = self.fetch_nomination(nomination_id)
        if not soup:
            print(f"Failed to fetch nomination {nomination_id}")
            return
        
        print(f"\n{'='*80}")
        print(f"DEBUG: Nomination ID {nomination_id}")
        print(f"{'='*80}\n")
        
        # Print raw HTML structure
        table = soup.find('table')
        if table:
            rows = table.find_all('tr')
            print(f"Found {len(rows)} rows in table\n")
            
            for i, row in enumerate(rows):
                cells = row.find_all('td')
                if cells:
                    print(f"Row {i}: ", end="")
                    for j, cell in enumerate(cells):
                        print(f"[{j}]: {cell.get_text(strip=True)[:50]}", end=" | ")
                    print()
        
        # Print parsed data
        print(f"\n{'-'*80}")
        print("PARSED DATA:")
        print(f"{'-'*80}\n")
        
        parsed = self.parse_nomination(soup, nomination_id)
        if parsed:
            import json
            print(json.dumps(parsed, indent=2, ensure_ascii=False))
        else:
            print("Failed to parse any data")
        
        print(f"\n{'='*80}\n")
        
    def scrape_range(
        self, 
        start_id: int = 0, 
        end_id: int = 23566,
        delay: float = 0.5,
        save_interval: int = 100
    ) -> pd.DataFrame:
        """
        Scrape a range of nomination IDs.
        
        Args:
            start_id: Starting nomination ID
            end_id: Ending nomination ID
            delay: Delay between requests in seconds
            save_interval: Save progress every N nominations
            
        Returns:
            DataFrame with all scraped nominations
        """
        nominations = []
        
        logger.info(f"Starting scrape from ID {start_id} to {end_id}")
        
        for nomination_id in tqdm(range(start_id, end_id + 1), desc="Scraping nominations"):
            soup = self.fetch_nomination(nomination_id)
            
            if soup:
                parsed_data = self.parse_nomination(soup, nomination_id)
                if parsed_data:
                    nominations.append(parsed_data)
            
            # Polite delay
            time.sleep(delay)
            
            # Save progress periodically
            if len(nominations) > 0 and len(nominations) % save_interval == 0:
                self._save_progress(nominations, f"nominations_progress_{len(nominations)}.csv")
        
        logger.info(f"Scraping complete. Found {len(nominations)} valid nominations.")
        
        return pd.DataFrame(nominations)
    
    def _save_progress(self, nominations: List[Dict], filename: str):
        """Save progress to CSV file."""
        try:
            df = pd.DataFrame(nominations)
            df.to_csv(filename, index=False, encoding='utf-8', sep=';')
            logger.info(f"Progress saved to {filename}")
        except Exception as e:
            logger.error(f"Failed to save progress: {e}")


def main():
    """Main execution function."""
    
    # ============================================================
    # EINSTELLUNGEN HIER ÄNDERN
    # ============================================================
    TEST_START = 1
    TEST_END = 500
    DELAY = 0.05
    RUN_FULL_SCRAPE = True
    SAVE_INTERVAL = 5000
    OUTPUT_PREFIX = "nominations_full"
    
    # DEBUG MODE - teste einzelne Seite
    DEBUG_MODE = False          # Auf True für Debugging
    DEBUG_ID = 19562            # Die ID zum Debuggen
    # ============================================================
    
    scraper = NobelNominationScraper()
    
    # Debug single page
    if DEBUG_MODE:
        print("Running in DEBUG mode...")
        scraper.debug_single_page(DEBUG_ID)
        
        # Ask if user wants to continue with scraping
        cont = input("\nContinue with scraping? (yes/no): ")
        if cont.lower() != 'yes':
            return

    if RUN_FULL_SCRAPE:
        logger.info("Running FULL scrape (0-23566)...")
        df = scraper.scrape_range(
            start_id=0, 
            end_id=23566, 
            delay=DELAY,
            save_interval=SAVE_INTERVAL
        )
        output_csv = f"{OUTPUT_PREFIX}_full.csv"
        output_excel = f"{OUTPUT_PREFIX}_full.xlsx"
    else:
        logger.info(f"Running TEST scrape ({TEST_START}-{TEST_END})...")
        df = scraper.scrape_range(
            start_id=TEST_START, 
            end_id=TEST_END, 
            delay=DELAY,
            save_interval=SAVE_INTERVAL
        )
        output_csv = f"{OUTPUT_PREFIX}_test.csv"
        output_excel = f"{OUTPUT_PREFIX}_test.xlsx"
    
    if not df.empty:
        print("\n" + "="*60)
        print("SCRAPING RESULTS")
        print("="*60)
        print(f"Total nominations found: {len(df)}")
        print(f"\nColumns: {len(df.columns)}")
        print(df.columns.tolist())
        print(f"\nFirst few rows:")
        print(df.head())
        
        # Check for multiple nominees
        if 'num_nominees' in df.columns:
            max_nominees = df['num_nominees'].max()
            print(f"\nMax nominees in single nomination: {max_nominees}")
            print(f"Nominations with multiple nominees: {(df['num_nominees'] > 1).sum()}")
        
        # Save results
        df.to_csv(output_csv, index=False, encoding='utf-8', sep=';')
        df.to_excel(output_excel, index=False)
        
        print(f"\n✓ Data saved to:")
        print(f"  - {output_csv}")
        print(f"  - {output_excel}")
        print("="*60)
    else:
        logger.error("No data scraped. Please check the URL and page structure.")


if __name__ == "__main__":
    main()