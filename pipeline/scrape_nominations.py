"""
Scraper for the Nobel Prize nomination archive (nobelprize.org), run by hand.

Nominations become public after 50 years, so each year a new batch appears with
IDs above the existing ones. The default mode only fetches those and merges them
with data/raw/nominations_full.csv.

Usage (from the project root):
    python -m pipeline.scrape_nominations                # new IDs only, merged
    python -m pipeline.scrape_nominations --full         # everything from ID 0
    python -m pipeline.scrape_nominations --range 1 500  # test run
    python -m pipeline.scrape_nominations --debug 19562  # show parsing of one page

Results go to data/derived/. After checking them, copy nominations_full.csv to
data/raw/ (what the app reads) and commit it.
"""

import argparse
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
import pandas as pd
import time
from typing import Dict, List, Optional
import logging
from tqdm import tqdm

from nbldash import paths

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

CATEGORIES = ['Physics', 'Chemistry', 'Physiology or Medicine', 'Literature', 'Peace', 'Economic Sciences']

# IDs are not contiguous (a gap of ~300 empty IDs precedes the 1975 batch), so the
# end of the archive is assumed only after this many empty IDs in a row.
STOP_AFTER_EMPTY = 1000


class NobelNominationScraper:
    """Scraper for Nobel Prize nomination archive."""
    
    def __init__(self, base_url: str = "https://www.nobelprize.org/nomination/archive/show.php"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'nbldata.org nomination-archive scraper (+https://nbldata.org)'
        })
        # Transient errors are retried with backoff instead of looking like an empty page.
        retry = Retry(total=5, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504],
                      respect_retry_after_header=True)
        self.session.mount("https://", HTTPAdapter(max_retries=retry))
    
    def fetch_nomination(self, nomination_id: int) -> Optional[BeautifulSoup]:
        """
        Fetch a single nomination page.
        
        Args:
            nomination_id: The nomination ID to fetch
            
        Returns:
            BeautifulSoup object, or None if no nomination has this ID (the
            archive answers 200 with an empty table then).

        Raises:
            requests.RequestException if the page could not be fetched even after
            retries -- the caller records the ID instead of treating it as empty.
        """
        url = f"{self.base_url}?id={nomination_id}"
        response = self.session.get(url, timeout=20)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        if self._title_row_text(soup) is None:
            return None
        return soup

    @staticmethod
    def _title_row_text(soup: BeautifulSoup) -> Optional[str]:
        """Text of the table's "Nomination for ..." row, or None on an empty page."""
        table = soup.find('table')
        if not table:
            return None
        first_row = table.find('tr')
        text = first_row.get_text(strip=True) if first_row else ''
        return text if text.startswith('Nomination for') else None
    
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
            
            # Find the main table with nomination data
            table = soup.find('table')
            if not table:
                return None
            
            rows = table.find_all('tr')
            
            # Title from the table's first row ("Nomination for Nobel Prize in ...").
            # Not from <h1>: the site's first h1 is now just "Nobel Prize", which
            # would leave every nomination without a category.
            title_text = self._title_row_text(soup) or ''
            data['nomination_title'] = title_text
            for cat in CATEGORIES:
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
                            nominator_dict[f'{prefix}_awarded_prizes'] = ', '.join(sorted(awarded_prizes_nominator))
                        
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

                        # Store it here: the `continue` skips the store below.
                        nominators_data[nominator_count] = nominator_dict
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
        end_id: Optional[int] = None,
        delay: float = 0.5,
        save_interval: int = 1000
    ) -> pd.DataFrame:
        """
        Scrape a range of nomination IDs.

        Args:
            start_id: Starting nomination ID
            end_id: Last ID to try; None = continue until STOP_AFTER_EMPTY empty IDs in a row
            delay: Delay between requests in seconds
            save_interval: Save progress every N new nominations

        Returns:
            DataFrame with all scraped nominations
        """
        nominations = []
        failed_ids = []
        empty_in_a_row = 0
        last_saved = 0

        logger.info(f"Starting scrape from ID {start_id} to {end_id if end_id is not None else 'the end'}")

        nomination_id = start_id
        with tqdm(desc="Scraping nominations") as progress:
            while end_id is None or nomination_id <= end_id:
                try:
                    soup = self.fetch_nomination(nomination_id)
                except requests.RequestException as e:
                    logger.warning(f"Failed to fetch ID {nomination_id}: {e}")
                    failed_ids.append(nomination_id)
                    soup = None

                if soup:
                    empty_in_a_row = 0
                    parsed_data = self.parse_nomination(soup, nomination_id)
                    if parsed_data:
                        nominations.append(parsed_data)
                else:
                    empty_in_a_row += 1
                    if end_id is None and empty_in_a_row >= STOP_AFTER_EMPTY:
                        logger.info(f"{STOP_AFTER_EMPTY} empty IDs in a row - stopping at ID {nomination_id}")
                        break

                progress.update(1)
                progress.set_postfix(found=len(nominations), failed=len(failed_ids))

                # Polite delay
                time.sleep(delay)

                # Save progress periodically
                if len(nominations) - last_saved >= save_interval:
                    self._save_progress(nominations, "nominations_progress.csv")
                    last_saved = len(nominations)

                nomination_id += 1

        logger.info(f"Scraping complete. Found {len(nominations)} valid nominations.")
        if failed_ids:
            logger.error(f"{len(failed_ids)} IDs could not be fetched and are missing: {failed_ids[:50]}")

        return pd.DataFrame(nominations)

    def _save_progress(self, nominations: List[Dict], filename: str):
        """Save progress to CSV file."""
        try:
            df = pd.DataFrame(nominations)
            paths.DERIVED_DIR.mkdir(parents=True, exist_ok=True)
            df.to_csv(paths.derived(filename), index=False, encoding='utf-8', sep=';')
            logger.info(f"Progress saved to {filename}")
        except Exception as e:
            logger.error(f"Failed to save progress: {e}")


def merge_with_existing(df_new: pd.DataFrame) -> pd.DataFrame:
    """
    Append newly scraped nominations to data/raw/nominations_full.csv.

    The existing column order is kept; columns only the new rows have (e.g. a
    nomination with more nominees than ever before) are appended at the end.
    """
    df_old = pd.read_csv(paths.raw("nominations_full.csv"), sep=';', dtype=str, keep_default_na=False)
    df_new = df_new.astype(str).replace({'nan': ''})
    df_new = df_new[~df_new['nomination_id'].isin(df_old['nomination_id'])]
    columns = list(df_old.columns) + [c for c in df_new.columns if c not in df_old.columns]
    merged = pd.concat([df_old, df_new], ignore_index=True).reindex(columns=columns).fillna('')
    return merged


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--full", action="store_true", help="scrape every ID from 0")
    mode.add_argument("--range", nargs=2, type=int, metavar=("START", "END"), help="test run over an ID range")
    mode.add_argument("--debug", type=int, metavar="ID", help="print raw rows and parsed data of one page")
    parser.add_argument("--delay", type=float, default=0.5, help="seconds between requests (default 0.5)")
    args = parser.parse_args()

    scraper = NobelNominationScraper()

    if args.debug is not None:
        scraper.debug_single_page(args.debug)
        return

    if args.full:
        logger.info("Running FULL scrape from ID 0...")
        df = scraper.scrape_range(start_id=0, delay=args.delay)
        output_name = "nominations_full"
    elif args.range:
        start, end = args.range
        logger.info(f"Running TEST scrape ({start}-{end})...")
        df = scraper.scrape_range(start_id=start, end_id=end, delay=args.delay)
        output_name = "nominations_test"
    else:
        existing = pd.read_csv(paths.raw("nominations_full.csv"), sep=';', usecols=['nomination_id'])
        start = int(existing['nomination_id'].max()) + 1
        logger.info(f"Running INCREMENTAL scrape from ID {start} (existing data ends at {start - 1})...")
        df_new = scraper.scrape_range(start_id=start, delay=args.delay)
        if df_new.empty:
            logger.info("No new nominations.")
            return
        paths.DERIVED_DIR.mkdir(parents=True, exist_ok=True)
        df_new.to_csv(paths.derived("nominations_new.csv"), index=False, encoding='utf-8', sep=';')
        df = merge_with_existing(df_new)
        output_name = "nominations_full"

    if df.empty:
        logger.error("No data scraped. Please check the URL and page structure.")
        return

    print("\n" + "="*60)
    print("SCRAPING RESULTS")
    print("="*60)
    print(f"Total nominations: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    if 'nomination_year' in df.columns:
        years = pd.to_numeric(df['nomination_year'], errors='coerce')
        print(f"Years: {int(years.min())}-{int(years.max())}")
    if 'nomination_num_nominees' in df.columns:
        num = pd.to_numeric(df['nomination_num_nominees'], errors='coerce')
        print(f"Max nominees in single nomination: {int(num.max())}")
        print(f"Nominations with multiple nominees: {int((num > 1).sum())}")

    # Save results to data/derived/; after checking them, copy the CSV to
    # data/raw/nominations_full.csv (what the app reads) and commit it.
    paths.DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    output_csv = paths.derived(f"{output_name}.csv")
    # CRLF like the existing file (first written on Windows), so that after an
    # incremental run the git diff shows only the appended rows.
    df.to_csv(output_csv, index=False, encoding='utf-8', sep=';', lineterminator='\r\n')
    print(f"\nData saved to {output_csv}")
    print("="*60)


if __name__ == "__main__":
    main()
