from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import os

import argparse
import time
import re
import csv
import os
from datetime import datetime
import pytz
import sys
import pandas as pd

import logging
from pymongo import MongoClient
from dotenv import load_dotenv

import subprocess
import gridfs


class FileManager:
    def __init__(self, db):
        """Initialize FileManager with MongoDB database connection"""
        self.db = db
        self.fs = gridfs.GridFS(db)

    def store_file_gridfs(self, filepath, tags=None, description=None):
        """Store a file in GridFS with metadata"""
        try:
            with open(filepath, 'rb') as f:
                file_data = f.read()
            
            # Create metadata document
            metadata = {
                "filename": os.path.basename(filepath),
                "upload_date": datetime.now(pytz.UTC),
                "tags": tags or [],
                "description": description
            }
            
            # Store file in GridFS
            file_id = self.fs.put(
                file_data,
                filename=metadata["filename"],
                metadata=metadata
            )
            
            # Store metadata in separate collection
            metadata["file_id"] = file_id
            self.db.file_metadata.insert_one(metadata)
            
            return file_id
            
        except Exception as e:
            raise Exception(f"Error storing file in GridFS: {e}")

    def find_files(self, query):
        """Find files based on metadata query"""
        try:
            return self.db.file_metadata.find(query)
        except Exception as e:
            raise Exception(f"Error finding files: {e}")

    def read_text_file(self, file_id):
        """Read text file content from GridFS"""
        try:
            grid_out = self.fs.get(file_id)
            return grid_out.read().decode('utf-8')
        except Exception as e:
            raise Exception(f"Error reading file from GridFS: {e}")
        
########################################################################


class BookmakerScraper:
    def __init__(self, base_dir='bookmaker'):
        self.base_dir = base_dir
        self.setup_logging()
        self.urls = {
            'nba': "https://www.oddsportal.com/basketball/usa/nba/",
            'nfl': "https://www.oddsportal.com/american-football/usa/nfl/",
            'cfb': "https://www.oddsportal.com/american-football/usa/ncaa/",
            'ncaa': "https://www.oddsportal.com/american-football/usa/ncaa/"
        }

        db_user = os.environ.get('MONGODB_User')
        # Get password from environment or .env file
        db_password = os.environ.get('MONGODB_PW')
        host_loc = os.environ.get('MONGODB_IP')
        if not db_password:
            self.logger.error("MongoDB password not found in environment variables")
            raise ValueError("MongoDB password not found")
        
        # MongoDB Atlas setup
        mongodb_uri = f"mongodb://{db_user}:{db_password}@{host_loc}/"
        db_name = 'Bookmaker_db'
        
        try:
            self.mongo_client = MongoClient(mongodb_uri, 
                                          serverSelectionTimeoutMS=5000,
                                          directConnection=True)
            
            self.db = self.mongo_client[db_name]
            self.mongo_client.server_info()
            # Initialize FileManager
            self.file_manager = FileManager(self.db)
            self.logger.info("Successfully connected to MongoDB Atlas")
        except Exception as e:
            self.logger.error(f"Failed to connect to MongoDB Atlas: {e}")
            raise

###########################################################################

    def write_csv_to_mongodb(self, csv_file, sport_name, current_time):
        """Bulk write CSV data to MongoDB"""
        try:
            # Read CSV file
            df = pd.read_csv(csv_file)
            
            # Convert dates to datetime objects
            df['Match Date'] = pd.to_datetime(df['Match Date'], format='%d %b %Y')
            
            # Create documents for bulk insert
            documents = []
            for _, row in df.iterrows():
                document = {
                    'sport_name': sport_name,
                    'capture_time': current_time,
                    'match_date': row['Match Date'],
                    'team1': row['Team 1'],
                    'team2': row['Team 2'],
                    'odds1': float(row['Odds 1']),
                    'odds2': float(row['Odds 2']),
                    'game_tag': row['game tag']
                }
                documents.append(document)
            
            # Create collection if it doesn't exist
            collection_name = f'{sport_name}_bookmaker_odds'
            if collection_name not in self.db.list_collection_names():
                self.db.create_collection(collection_name)
            
            # Create indexes
            self.db[collection_name].create_index([
                ('sport_name', 1),
                ('capture_time', 1),
                ('game_tag', 1)
            ])
            
            # Bulk write to MongoDB
            if documents:
                result = self.db[collection_name].insert_many(documents)
                self.logger.info(f"Successfully bulk wrote {len(result.inserted_ids)} documents to MongoDB")
                return result.inserted_ids
            
            return []
            
        except Exception as e:
            self.logger.error(f"Error bulk writing CSV data to MongoDB: {e}")
            raise
    
    ###########################################################################

    def setup_logging(self):
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
        log_file = os.path.join(self.base_dir, 'scraper.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_file)
            ]
        )
        self.logger = logging.getLogger(__name__)

    ###########################################################################

    def get_chrome_options(self):
        """Return configured Chrome options"""
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        options.binary_location = os.environ.get("CHROME_BIN", "/usr/bin/chromium")
        options.add_argument('--window-size=1920,1080')
        options.add_argument('--disable-extensions')
        options.add_argument('--disable-dev-tools')
        options.add_argument('--ignore-certificate-errors')
        options.add_argument('--allow-running-insecure-content')
        
        prefs = {
            'profile.default_content_settings': {
                'images': 2,
                'plugins': 2,
                'popups': 2,
                'geolocation': 2,
                'notifications': 2
            },
            'profile.managed_default_content_settings': {
                'javascript': 1
            }
        }
        options.add_experimental_option('prefs', prefs)
        return options
        
    def create_directories(self,sport_tag):
        """Create necessary directories for data storage"""
        root_dir = self.base_dir
        Dir = os.path.join(root_dir, f'{sport_tag}')
        os.makedirs(Dir, exist_ok=True)
        return Dir

    def get_file_paths(self, sport_str):
        """Generate file paths for data storage"""
        
        Dir = self.create_directories(sport_str)
        filename = "BM"
        
        raw_txt = os.path.join(Dir, f'raw_{filename}.txt')
        cleaned_txt = os.path.join(Dir, f'cleaned_{filename}.txt')
        csv_file = os.path.join(Dir, f'{filename}.csv')
        
        return raw_txt, cleaned_txt, csv_file

###########################################################################

    @staticmethod
    def check_browser_availability():
        """Check if Chrome browser is available and can properly parse web structure"""
        print('Checking browser availability')
        driver = None
        try:
            scraper = BookmakerScraper()
            options = scraper.get_chrome_options()
            service = Service(executable_path=os.environ.get("CHROMEDRIVER_PATH", "/usr/bin/chromedriver"))
            
            print("Creating Chrome driver...")
            driver = webdriver.Chrome(service=service, options=options)
            
            print("Setting page load timeout...")
            driver.set_page_load_timeout(30)
            
            print("Attempting to access Apple.com...")
            driver.get('https://www.apple.com')
            
            # List of elements to check
            structure_checks = {
                'body': (By.TAG_NAME, "body"),
                'head': (By.TAG_NAME, "head"),
                'title': (By.TAG_NAME, "title"),
                'navigation': (By.TAG_NAME, "nav"),
                'links': (By.TAG_NAME, "a"),
                'main content': (By.TAG_NAME, "main"),
                'headers': (By.CSS_SELECTOR, "h1, h2, h3"),
                'paragraphs': (By.TAG_NAME, "p")
            }
            
            print("\nChecking web structure elements:")
            wait = WebDriverWait(driver, 10)
            
            for element_name, (by, selector) in structure_checks.items():
                try:
                    elements = wait.until(EC.presence_of_all_elements_located((by, selector)))
                    count = len(elements)
                    print(f"✓ Found {count} {element_name} element{'s' if count != 1 else ''}")
                except Exception as e:
                    print(f"✗ Failed to find {element_name} elements: {str(e)}")
                    return False
            
            # Check if we can interact with elements
            print("\nTesting element interaction capabilities:")
            try:
                # Test text extraction
                title = driver.title
                print(f"✓ Successfully extracted page title: {title}")
                
                # Test attribute extraction
                links = driver.find_elements(By.TAG_NAME, "a")
                if links:
                    href = links[0].get_attribute('href')
                    print(f"✓ Successfully extracted link attribute")
                
                # Test CSS property extraction
                body = driver.find_element(By.TAG_NAME, "body")
                style = body.value_of_css_property('display')
                print(f"✓ Successfully extracted CSS property")
                
                # Test JavaScript execution
                viewport_height = driver.execute_script("return window.innerHeight;")
                print(f"✓ Successfully executed JavaScript")
                
            except Exception as e:
                print(f"✗ Failed element interaction test: {str(e)}")
                return False
            
            # Check page response
            print("\nChecking page response:")
            try:
                # Get page source
                page_source = driver.page_source
                if len(page_source) > 0:
                    print(f"✓ Successfully retrieved page source ({len(page_source)} characters)")
                
                # Check HTTP status using JavaScript
                status = driver.execute_script(
                    "return window.performance.getEntries()[0].responseStatus"
                )
                if status:
                    print(f"✓ Page response status available")
                
            except Exception as e:
                print(f"✗ Failed page response check: {str(e)}")
                return False
            
            print("\n✓ All browser functionality checks passed successfully")
            return True
            
        except Exception as e:
            print(f"\n✗ Chrome browser not available or connection failed: {str(e)}")
            print("Error details:")
            import traceback
            traceback.print_exc()
            return False
            
        finally:
            if driver:
                try:
                    driver.quit()
                    print("\nBrowser closed successfully")
                except Exception as e:
                    print(f"Error closing browser: {str(e)}")

###########################################################################

    @staticmethod
    def clean_text(text):
        """Clean text from non-text characters and extra whitespace"""
        cleaned = re.sub(r'[^\w\s\-–.]', '', text)
        cleaned = re.sub(r'\s+', ' ', cleaned)
        return cleaned.strip()

    def extract_games(self, driver):
        """Extract games data from the webpage"""
        games_data = []
        rows = driver.find_elements(By.XPATH, "//div[contains(@class, 'row')]")
        
        for row in rows:
            text = row.text.strip()
            if not text:
                continue
                
            Search_text = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 
                          "Today", "Tomorrow"]
                          
            if any(date_indicator in text for date_indicator in Search_text):
                current_date = text
                games_data.append(current_date)
                continue
                
            if text:
                cleaned_text = self.clean_text(text)
                if cleaned_text:
                    games_data.append(cleaned_text)
        
        return games_data

    def scrape_games(self, url, filename, max_retries=3):
        """Scrape games data from the specified URL with retry mechanism"""
        last_exception = None
        
        for attempt in range(max_retries):
            options = self.get_chrome_options()
            service = Service(executable_path=os.environ.get("CHROMEDRIVER_PATH", "/usr/bin/chromedriver"))
            driver = None
            
            try:
                driver = webdriver.Chrome(service=service, options=options)
                driver.set_page_load_timeout(300)  # 5 minutes timeout
                
                self.logger.info(f'Attempt {attempt + 1}: Accessing {url}')
                driver.get(url)
                
                # Wait for initial load with explicit wait
                wait = WebDriverWait(driver, 60)
                wait.until(EC.presence_of_element_located((By.XPATH, "//div[contains(@class, 'row')]")))
                
                # Dynamic content loading with progress check
                total_wait = 0
                rows_count = 0
                while total_wait < 120:  # Maximum 2 minutes wait
                    time.sleep(10)
                    total_wait += 10
                    current_rows = len(driver.find_elements(By.XPATH, "//div[contains(@class, 'row')]"))
                    
                    if current_rows > rows_count:
                        rows_count = current_rows
                    elif current_rows > 5:  # We have enough rows and they've stopped increasing
                        break
                
                games_data = self.extract_games(driver)
                if games_data:
                    self.save_games_to_file(games_data, filename)
                    return True
                    
            except Exception as e:
                last_exception = e
                self.logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                time.sleep(30 * (attempt + 1))  # Exponential backoff
                
            finally:
                if driver:
                    try:
                        driver.quit()
                    except:
                        pass
        
        if last_exception:
            self.logger.error(f"All attempts failed. Last error: {str(last_exception)}")
            raise last_exception

    @staticmethod
    def save_games_to_file(games_data, filename):
        """Save games data to a file"""
        with open(filename, 'w', encoding='utf-8') as f:
            for line in games_data:
                f.write(line + '\n')
        print(f"Data saved.")

    def clean_file(self, input_file, output_file):
        """Clean the raw data file"""
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        def is_date(text):
            if not text:
                return False
            parts = text.split()
            if len(parts) < 2:
                return False
            base_text = text.split('-')[0].strip()
            base_parts = base_text.split()
            return (any(month in base_text for month in months) and 
                    base_parts[0].isdigit() and 
                    1 <= int(base_parts[0]) <= 31)

        content = re.sub(r'^.*?(?=(?:\d{2} [A-Za-z]{3}|Today|Tomorrow))', '', content, flags=re.DOTALL)

        pattern = (
            r'('
            r'(?:\d{2} [A-Za-z]{3}(?:\s+\d{4})?[^\n]*|Today[^\n]*|Tomorrow[^\n]*)'
            r')'
            r'\s*\n1\n2\n'
            r'(?:B\'s\n)?'
            r'((?:.*?\n)*?)'
            r'(?=(?:\d{2} [A-Za-z]{3}|Today|Tomorrow)|$)'
        )
        
        cleaned_content = []
        matches = re.finditer(pattern, content, re.MULTILINE | re.DOTALL)
        
        for match in matches:
            date_section = match.group(1)
            games_section = match.group(2)
            
            if '-' in date_section:
                date_section = date_section.split('-')[0].strip()
            
            cleaned_content.append(date_section)
            cleaned_content.append("1")
            cleaned_content.append("2")
            
            game_lines = [line for line in games_section.split('\n') if line.strip()]
            
            current_game = []
            skip_current = False
            
            for line in game_lines:
                if (re.match(r'\d{2} [A-Za-z]{3}(?:\s+\d{4})?|Today|Tomorrow', line) 
                    and len(current_game) == 0):
                    continue
                    
                current_game.append(line)
                
                if len(current_game) == 6:
                    if is_date(current_game[4]) or is_date(current_game[5]):
                        current_game = current_game[:4]
                    cleaned_content.extend(current_game)
                    current_game = []
                    skip_current = False
            
            if current_game and not skip_current:
                cleaned_content.extend(current_game)

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(line for line in cleaned_content if line.strip()))

    def process_txt_to_csv(self, input_file, output_file):
        """Convert cleaned text file to CSV format"""
        games = []
        match_date = ""
        current_year = datetime.now().year
        
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        def fix_date(date_str):
            if not date_str:
                return date_str
        
            
            if 'Today' in date_str:
                date_part = date_str.split('Today, ')[1]
                return f"{date_part} {current_year}"
            elif 'Tomorrow' in date_str:
                date_part = date_str.split('Tomorrow, ')[1]
                return f"{date_part} {current_year}"
            return date_str

        def is_team_name(value):
            if not value:
                return False
            return not any(char.isdigit() for char in str(value))
        
        def is_valid_odd(odd):
            if not odd:  # Handle empty strings
                return False
            try:
                float(odd)  # Try to convert to float
                return True
            except ValueError:
                return False
            
        def create_game_tag(team1, team2, date):
            # Create the game tag by combining teams with 'vs' and the date
            team1 = team1.split()[-1]
            team2 = team2.split()[-1]
                # Convert date from "05 Jan 2025" to "20250105"
            try:
                date_obj = datetime.strptime(date, '%d %b %Y')
                formatted_date = date_obj.strftime('%Y%m%d')
            except ValueError:
                formatted_date = date  # Keep original if conversion fails
            return f"{team1}vs{team2}{formatted_date}"
        
        with open(input_file, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                
                if not line:
                    i += 1
                    continue
                
                if any(month in line for month in months) and any(c.isdigit() for c in line):
                    if ('NBA' not in line or 'NFL' not in line or 'NCAA' not in line) and 'Leagues' not in line:
                        match_date = fix_date(line)
                    i += 1
                    continue
                    
                if line in ['1', '2']:
                    i += 1
                    continue
                
                if ':' in line:
                    if i + 5 >= len(lines):
                        i += 1
                        continue
                        
                    team1 = lines[i + 1].strip()
                    score_or_dash = lines[i + 2].strip()
                    team2 = lines[i + 3].strip()
                    
                    if score_or_dash.isdigit():
                        i += 6
                        continue
                    
                    odds1 = lines[i + 4].strip()
                    odds2 = lines[i + 5].strip()
                    
                    if any(month in odds2 for month in months) and any(c.isdigit() for c in odds2):
                        match_date = fix_date(odds2)
                        i += 1
                        continue
                    
                    odds1 = '' if odds1 == '-' else odds1
                    odds2 = '' if odds2 == '-' else odds2
                    
                    if (is_team_name(team1) and is_team_name(team2) and 
                        team2 != "B's" and 
                        not any(month in team2 for month in months) and
                        is_valid_odd(odds1) and is_valid_odd(odds2)):

                        game_tag = create_game_tag(team1, team2, match_date)
                        games.append([
                            match_date,
                            team1,
                            team2,
                            odds1,
                            odds2,
                            game_tag
                        ])
                    
                    i += 6
                else:
                    i += 1

        # Write to CSV files
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Match Date', 'Team 1', 'Team 2', 'Odds 1', 'Odds 2', 'game tag'])
            writer.writerows(games)

###########################################################################

    def run_scraper(self, sport):
        """Main function to run the scraper"""
        current_time = datetime.now(pytz.UTC)
        
        url = self.urls.get(sport.lower())
        if not url:
            self.logger.error(f"No URL found for sport: {sport}")
            return

        raw_txt, cleaned_txt, csv_file = self.get_file_paths(sport)

        try:
            self.scrape_games(url, raw_txt)
            self.clean_file(raw_txt, cleaned_txt)
            self.process_txt_to_csv(cleaned_txt, csv_file)

            inserted_ids = self.write_csv_to_mongodb(csv_file, sport, current_time)

            self.logger.info(f"Successfully processed and stored data for {sport}")
            return inserted_ids

        except Exception as e:
            self.logger.error(f"Error processing sport {sport}: {e}")
            raise

###########################################################################

    def read_data(self, sport_name, start_date=None, end_date=None, full_output=False):
        
        if start_date is None:
            start_date = '2024-01-01'

        if end_date is None:
            end_date = datetime.now(pytz.UTC)
        
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date).tz_localize(pytz.UTC)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date).tz_localize(pytz.UTC)

        try:
            query = {
                "sport_name": sport_name,
                "capture_time": {
                    "$gte": start_date,
                    "$lte": end_date
                }
            }
            
            # If not full_output, only query for CSV files
            if not full_output:
                query["file_type"] = "csv"
            
            files = self.file_manager.find_files(query)
            
            if full_output:
                result = {
                    "raw_txt": [],
                    "cleaned_txt": [],
                    "csv": []
                }
            else:
                result = []  # For CSV-only output
            
            for file_info in files:
                content = self.file_manager.read_text_file(file_info['_id'])
                file_data = {
                    "content": content,
                    "capture_time": file_info['capture_time'],
                    "metadata": file_info
                }
                
                if full_output:
                    result[file_info['file_type']].append(file_data)
                else:
                    result.append(file_data)
            
            return result

        except Exception as e:
            self.logger.error(f"Error retrieving files: {e}")
            raise
        

def main(looping=False):
    parser = argparse.ArgumentParser(description='Sports odds scraper from OddsPortal')
    parser.add_argument('-M', '--mode', required=True, choices=['collect', 'read'],
                       help='Operation mode: collect or read data')
    parser.add_argument('-s', '--sport', 
                       type=str,
                       nargs='+',
                       default=['nba', 'nfl', 'cfb'],
                       help='Sport(s) to scrape (e.g., nba nfl cfb). Default: nba nfl cfb')
    parser.add_argument('--start_date', default=None,
                       help='Start date for reading data (YYYY-MM-DD)')
    parser.add_argument('--end_date', default=None, help='End date for reading data (YYYY-MM-DD)')
    
    args = parser.parse_args()
    sports = [sport.lower() for sport in args.sport]
    
    scraper = BookmakerScraper()
    
    if args.mode == 'collect':
        print('Data collecting mode')
        if not scraper.check_browser_availability():
            print('Please install chrome in your system.')
            sys.exit(1)
        
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"\nStarting scraping cycle at {current_time}")
            
        for sport in sports:
            print(f"\nScraping {sport.upper()}...")
            try:
                scraper.run_scraper(sport)
            except ValueError as e:
                print(f"Error with {sport}: {e}")
                continue
            except Exception as e:
                print(f"Unexpected error with {sport}: {e}")
                continue
            print('Exiting scraper')
                
    elif args.mode == 'read':
        print('Data reading mode')
        if not args.end_date:
            print("End date is required for read mode")
            return
            
        for sport in sports:
            print(f"\nReading {sport.upper()} data...")
            try:
                game_dfs = scraper.read_data(
                    sport=sport,
                    start_date=args.start_date,
                    end_date=args.end_date
                )
                
                if not game_dfs:
                    print(f"No data found for {sport} with the given criteria")
                else:
                    print(f"\nFound {len(game_dfs)} different games for {sport}")
                    for idx, df in enumerate(game_dfs, 1):
                        print(f"\nGame {idx}: {df['game tag'].iloc[0]}")
                        print(f"Number of records: {len(df)}")
                        print(df)
                        
            except ValueError as e:
                print(f"Error with {sport}: {e}")
                continue
            except Exception as e:
                print(f"Error processing {sport}: {str(e)}")
                continue


if __name__ == "__main__":
    main()
