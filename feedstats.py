import argparse
import dns.resolver
import logging
import multiprocessing
import re
import sys
import xml.etree.ElementTree as ET
from datetime import timedelta, datetime
from urllib.parse import unquote, urljoin, urlparse, urlunparse

import pytz
import requests
from bs4 import BeautifulSoup
from dateutil.parser import parse
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import urllib3


# Setting up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')

# Disable SSL verification warnings
#urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


FEED_SUFFIXES = [
    "feed/", "feed.xml", "index.xml", "default", "rss.xml", "atom.xml", "rss/", "feed",
    "rss", "all.atom.xml", "feed.rss", "?format=rss", "feed.atom", "?feed=rss2",
    "blog?format=rss", "?type=rss", "feed/?type=rss", "blog-feed.xml", "index.rss",
    "atom", "atom/", "index.atom", "posts.atom", "blog.atom", "all.xml", "blog.xml",
    "blog.rss", "all.rss.xml", "posts.xml", "posts.rss", "articles.xml", "rss.php",
    "articles.atom", "blog/?feed=rss2", "blog", "all.rss", "wordpress/?feed=rss2",
    "feed.php", "all"
]

TZINFOS = {
    "UT": pytz.UTC,                             # Coordinated Universal Time
    "EEST": pytz.timezone("Europe/Athens"),     # Eastern European Summer Time
    "EET": pytz.timezone("Europe/Athens"),      # Eastern European Time
    "CDT": pytz.timezone("US/Central"),         # Central Daylight Time
    "CST": pytz.timezone("US/Central"),         # Central Standard Time
    "EDT": pytz.timezone("US/Eastern"),         # Eastern Daylight Time
    "EST": pytz.timezone("US/Eastern"),         # Eastern Standard Time
    "PDT": pytz.timezone("US/Pacific"),         # Pacific Daylight Time
    "PST": pytz.timezone("US/Pacific"),         # Pacific Standard Time
    "MDT": pytz.timezone("US/Mountain"),        # Mountain Daylight Time
    "MST": pytz.timezone("US/Mountain"),        # Mountain Standard Time
    "AEST": pytz.timezone("Australia/Sydney"),  # Australian Eastern Standard Time
    "AEDT": pytz.timezone("Australia/Sydney")   # Australian Eastern Daylight Time
}

USER_AGENT = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36'
TIMEOUT = 30
BASE_URL = 'http://sanitizer.blogtrottr.com/sanitize?url='
HEADERS = {'User-Agent': USER_AGENT}
MAX_PROCESSES = 300  # Set a maximum number of processes
from requests.packages.urllib3.util.retry import Retry

RETRY_STRATEGY = Retry(
    # Total number of retries
    total=3,
    
    # Status codes to trigger a retry
    status_forcelist=[408, 413, 423, 429, 440, 449, 500, 502, 503, 504],
    
    # Methods to be retried
    allowed_methods=["HEAD", "GET", "OPTIONS"],
    
    # Exponential backoff (wait time between retries doubles)
    backoff_factor=1,  # You can adjust this, e.g., 0.5 for shorter waits, 2 for longer waits.
    
    # Respect Retry-After header (use header's value for delay if available)
    respect_retry_after_header=True
)


# EXCEPTIONS
class HTTPError(Exception):
    """Exception raised for errors in the HTTP response."""
    pass

class XMLParseError(Exception):
    """Exception raised for errors in the XML parsing."""
    pass

class DateExtractionError(Exception):
    """Exception raised for errors during date extraction."""
    pass

class ProcessUrlError(Exception):
    """Exception raised for errors while processing URLs."""
    pass


# UTILITY FUNCTIONS
def get_http_session():
    """Return a requests Session with a retry strategy."""
    session = requests.Session()
    session.mount("http://", HTTPAdapter(max_retries=RETRY_STRATEGY))
    session.mount("https://", HTTPAdapter(max_retries=RETRY_STRATEGY))
    return session

def sanitize_url(url):
    """Sanitize the provided URL if it starts with the BASE_URL."""
    if url.startswith(BASE_URL):
        return unquote(url.replace(BASE_URL, ''))
    return url



def discover_feed(base_url, session, switch_to_http=False, force_https=False):
    logging.info(f"Starting feed discovery for base URL: {base_url}")

    for suffix in FEED_SUFFIXES:
        try_url = urljoin(base_url, suffix)
        
        logging.info(f"Trying feed URL: {try_url}")
        
        response, error = get_http_response(try_url, session, switch_to_http, force_https)

        if response and response.content:
            temp_soup = BeautifulSoup(response.content, determine_parser(response.content.decode('utf-8', 'ignore')))
            if temp_soup.find(['rss', 'feed', 'rdf']):
                logging.info(f"Valid feed discovered at {try_url}")
                return temp_soup, try_url, ""
            else:
                logging.warning(f"No valid feed structure discovered at {try_url}")

        else:
            logging.error(f"Failed to fetch content from {try_url}. Error: {error}")

    error_message = f"No Atom or RSS feed found at URL (auto-detection attempted!) {base_url}"
    logging.error(error_message)
    return None, "", error_message

def fetch_feed_url(soup, url, session, original_url, switch_to_http=False, force_https=False, auto_discover_feed=True, follow_feed_redirects=False):

    def fetch_content(url):
        return get_http_response(url, session, switch_to_http, force_https)

    feed_url_element = None

    if soup is None and not auto_discover_feed:
        error_message = f"No Atom or RSS feed found at URL {url} (original url: {original_url}) and auto-discovery is disabled"
        return None, url, error_message
    if soup:
        feed_url_element = soup.find('link', type='application/atom+xml') or soup.find('link', type='application/rss+xml')

    # If there's no feed_url_element found and auto_discover_feed is False, then return with an error
    if not feed_url_element and not auto_discover_feed:
        error_message = f"No direct or indirect Atom, RSS, or RDF feed found (URL: {url})"
        return None, url, error_message

    if not feed_url_element and auto_discover_feed:
        soup, url, error_message = discover_feed(original_url, session)
        if soup is None:
            error_message = f"No Atom or RSS feed found at URL {url} (original url: {original_url}). Additionally, auto-discovery failed with error: {error_message}"
            return None, url, error_message

    if feed_url_element:
        url = urljoin(url, feed_url_element.get('href'))

        if follow_feed_redirects:
            feed_response, error = fetch_content(url)
            if feed_response:
                url = handle_redirection(feed_response, url)
            elif error:
                logging.warning(f"Error following feed URL redirect (url: {url}): {error}")

    response, error = fetch_content(url)

    try:
        soup = BeautifulSoup(response.content, determine_parser(response.content.decode('utf-8', 'ignore')))
        if not soup:
            error_message = f"Failed to parse content into soup for URL {url}."
            logging.error(error_message)
            return None, url, error_message

        # Check for valid feed elements
        if not soup.find(['rss', 'feed', 'rdf']):
            error_message = f"{response.status_code} Not a valid Atom, RSS, or RDF feed XML: {response.url}"
            logging.error(error_message)
            
            # If auto_discover_feed is enabled, try searching with different suffixes
            if auto_discover_feed:
                logging.info(f"Attempting to auto-discover feed for URL {url} as the provided feed URL was not valid.")
                soup, url, error_message = discover_feed(original_url, session)
                if soup is None:
                    error_message = f"No Atom or RSS feed found at URL {url} after attempting auto-discovery. Original error: {error_message}"
                    logging.error(error_message)
                    return None, url, error_message
            else:
                return None, url, error_message

        return soup, url, ""

    except ValueError as e:  # More specific exception, you can add more as needed
        error_message = f"Error parsing XML from URL {url}: {e}"
        logging.error(error_message)
        return None, url, error_message

def filter_dates(dates, url):
    if len(dates) < 2:
        return dates

    # Calculate intervals between dates in days
    intervals = [(dates[i] - dates[i - 1]).days for i in range(1, len(dates))]

    # Compute the first and third quartiles of the intervals
    q1 = sorted(intervals)[int(len(intervals) / 4)]
    q3 = sorted(intervals)[int(3 * len(intervals) / 4)]
    iqr = q3 - q1

    # Determine the multiplier to mimic a 15-year threshold
    # Approximating that multiplier * IQR ~ 15 years (in days)
    TARGET_THRESHOLD_IN_DAYS = 15 * 365.25
    multiplier = TARGET_THRESHOLD_IN_DAYS / iqr

    # Compute the bounds for outliers
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr

    # Filter the dates
    filtered_dates = [dates[0]]  # Always keep the first date

    for i in range(1, len(dates)):
        # If the current interval is not an outlier, keep the date
        if lower_bound <= intervals[i - 1] <= upper_bound:
            filtered_dates.append(dates[i])
        else:
            logging.info(f"Discarding date {dates[i]} from URL: {url} due to being an outlier in the distribution.")

    return filtered_dates

def get_sorted_dates_from_soup(soup, url, heuristic_date_parsing, filter_dates_enabled):
    """Extract and return sorted dates from soup."""
    dates, num_posts = extract_dates_and_count_posts(soup, url, heuristic_date_parsing)
    dates.sort()
    if filter_dates_enabled:
        filtered_dates = filter_dates(dates, url)
    else:
        filtered_dates = dates

    return filtered_dates, num_posts

def custom_date_parser(date_str, heuristic_date_parsing, url):

    # Heuristic Parsing Logic
    if heuristic_date_parsing:
        # Handle day anomalies
        day_replacements = {
            "Tues": "Tue",
            "Thurs": "Thu",
            "Thur": "Thu",
            "Fru": "Fri"
        }

        for k, v in day_replacements.items():
            if k in date_str:
                date_str = date_str.replace(k, v)
                logging.info(f"Detected day anomaly, replaced '{k}' with '{v}'. New date string: {date_str}")

        # Handle dates with comma-separated timezone offset
        if ", -" in date_str or ", +" in date_str:
            date_str = date_str.replace(", -", "-").replace(", +", "+")

        # Handle invalid month
        if '-81-' in date_str:
            logging.info(f"Encountered invalid month in date: {date_str}. Skipping.")
            return None

        # Handle hour 24
        if " 24:" in date_str:
            date_str = date_str.replace(" 24:", " 00:")

            # Try parsing with fuzzy=True after adjusting hour 24
            try:
                parsed_date = parse(date_str, tzinfos=TZINFOS, fuzzy=True)
                return parsed_date + timedelta(days=1)  # Add a day
            except Exception as e:
                logging.warning(f"Failed to parse date after hour 24 transformation: {date_str}. Error: {e}")
                return None

    # General Parsing Logic
    try:
        return parse(date_str, tzinfos=TZINFOS)
    except:
        logging.error(f"Invalid date encountered for URL: {url}. Date string: {date_str}")
        return None

def extract_dates_and_count_posts(soup, url, heuristic_date_parsing):
    """Extract dates from the soup object and count the number of posts."""

    dates = []
    posts = []

    # RSS Feed Handling
    if soup.find('rss'):
        posts = soup.find_all('item')
        for post in posts:
            # Use a lambda function for case-insensitive search for pubDate element
            pub_date_element = post.find(lambda tag: tag.name and tag.name.lower() == 'pubdate')
            if pub_date_element and pub_date_element.text:
                try:
                    parsed_date = custom_date_parser(pub_date_element.text, heuristic_date_parsing, url)
                    if parsed_date:
                        dates.append(parsed_date)
                    else:
                        logging.error(f"Unable to parse RSS post date for URL {url}: {pub_date_element.text}")

                except ValueError as e:
                    logging.error(f"Invalid date in RSS post for URL {url}: {pub_date_element.text} - Error: {e}")

    # Atom Handling
    elif soup.find('feed'):
        posts = soup.find_all(['entry', 'item'])
        atom_dates = []
        for post in posts:
            date_text = post.find('published').text if post.find('published') else None
            if date_text:
                try:
                    parsed_date = custom_date_parser(date_text, heuristic_date_parsing, url)
                    if parsed_date:
                        atom_dates.append(parsed_date)
                    else:
                        logging.error(f"Unable to parse Atom post date for URL {url}: {date_text}")
                except ValueError as e:
                    logging.error(f"Invalid date in Atom post for URL {url}: {date_text} - Error: {e}")
        if atom_dates:
            dates.extend(atom_dates)
        else:
            updated_dates = [parse(post.updated.text, tzinfos=TZINFOS) for post in posts if post.updated and post.updated.text and not any([post.find('pubDate'), post.find('published')])]
            if updated_dates:
                dates.extend(updated_dates)

    # RDF Handling
    elif soup.find('rdf:RDF', namespaces={'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#'}):
        posts = soup.find_all(['item', 'rdf:Description'], namespaces={'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#'})

        for post in posts:
            date_element = post.find('dc:date', namespaces={'dc': 'http://purl.org/dc/elements/1.1/'})
            date_text = date_element.text if date_element else None
            if date_text:
                try:
                    parsed_date = custom_date_parser(date_text, heuristic_date_parsing, url)
                    if parsed_date:
                        dates.append(parsed_date)
                    else:
                        logging.error(f"Unable to parse RDF post date for URL {url}: {date_text}")

                except ValueError as e:
                    logging.error(f"Invalid date in RDF post for URL {url}: {date_text} - Error: {e}")

    dates = [date.replace(tzinfo=pytz.UTC) if date.tzinfo is None else date for date in dates]

    return dates, len(posts)

def combine_status_and_error(status_code_str, error_message):
    """Combine the status code and error message, if both are available."""
    if status_code_str and error_message:
        return f"{status_code_str} - {error_message}"
    return status_code_str or error_message

def determine_parser(content):
    """Determine whether the content should be parsed as XML or HTML."""
    content_lower = content.lower()

    # Check for XML declaration
    is_xml = '<?xml' in content_lower

    # Check for common HTML indicators
    is_html = '<!doctype html>' in content_lower or '<html' in content_lower

    # Determine parser type
    if is_xml and not is_html:
        return 'xml'
    elif not is_xml and is_html:
        return 'html.parser'
    elif is_xml and is_html:
        return 'xml'
    else:
        return 'html.parser'

def format_output(avg_posts_per_day_str, avg_posts_per_week_str, original_url, newest_post, oldest_post, num_posts_str, status_or_error, url, blogging_platform, bid_url, handle_blogtrottr):
    base_output = f'{avg_posts_per_day_str}\t{avg_posts_per_week_str}\t{original_url}\t{newest_post}\t{oldest_post}\t{num_posts_str}\t{status_or_error}\t{url}\t{blogging_platform}'
    if handle_blogtrottr:
        base_output += f'\t{bid_url}'
    return base_output

def detect_blogging_platform(soup):
    """Detect the blogging platform from the soup."""
    generator_tag = soup.find('generator')
    
    if not generator_tag:
        return ""  # return empty string if platform can't be determined
    
    tag_text = generator_tag.text.strip().replace('\n', ' ')  # Replacing newlines with spaces

    # List of patterns for known platforms and their corresponding names
    known_platform_patterns = [
        (re.compile(r'substack', re.I), "Substack"),
        (re.compile(r'Medium', re.I),   "Medium"),
        #(re.compile(r'wordpress\.org', re.I), "WordPress")
        # Add more platforms as needed in the format
        # (compiled_regex, "ReturnText"),
    ]
    
    for pattern, platform_name in known_platform_patterns:
        if pattern.search(tag_text):
            return platform_name

    # If text doesn't match any known pattern
    # Return the text if it exists, else return the entire tag
    #return tag_text if tag_text else str(generator_tag).replace('\n', ' ')
    return str(generator_tag).replace('\n', ' ')

def extract_feed_data(soup, url, heuristic_date_parsing, filter_dates_enabled):
    dates, num_posts_str = get_sorted_dates_from_soup(soup, url, heuristic_date_parsing, filter_dates_enabled)
    oldest_post = dates[0].astimezone(pytz.UTC).isoformat(timespec='seconds') + 'Z' if dates else ''
    newest_post = dates[-1].astimezone(pytz.UTC).isoformat(timespec='seconds') + 'Z' if dates else ''

    date_diff = dates[-1] - dates[0] if dates else timedelta(days=0)
    avg_posts_per_day_str, avg_posts_per_week_str = "", ""
    try:
        if date_diff.days == 0:
            avg_posts_per_day = len(dates)
            avg_posts_per_week = len(dates) * 7
        else:
            avg_posts_per_day = len(dates) / date_diff.days
            avg_posts_per_week = avg_posts_per_day * 7
        avg_posts_per_day_str = f'{avg_posts_per_day:.8f}'
        avg_posts_per_week_str = f'{avg_posts_per_week:.8f}'
    except ZeroDivisionError as e:
        logging.error(f"Error calculating averages: {str(e)}")
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")

    return avg_posts_per_day_str, avg_posts_per_week_str, newest_post, oldest_post, num_posts_str

def handle_discovery(url, session, switch_to_http=False, force_https=False, auto_discover_feed=False):
    if not auto_discover_feed:
        return None, url, "Auto-discovery is disabled."

    soup, url, error_message_from_feed = discover_feed(url, session, switch_to_http, force_https)
    if soup is None:
        return None, url, error_message_from_feed

    return soup, url, ""

def extract_errors(text):
    """Extract specific error messages following the 'Caused by' pattern."""
    # Regular expression pattern to capture errors after "Caused by"
    pattern = r"Caused by ([^\()]+)"
    matches = re.findall(pattern, text)
    if matches:
        return matches[0]
    else:
        return text


def get_http_response(url, session, switch_to_http=False, force_https=False):
    """Fetch the HTTP response using requests with a retry strategy."""
    
    parsed_url = urlparse(url)
    original_scheme = parsed_url.scheme

    # Function to try fetching content and returning the response and error
    def try_fetch(u):
        try:
            r = session.get(u, timeout=TIMEOUT, headers=HEADERS, verify=True)
            r.raise_for_status()
            return r, ""
        except requests.RequestException as e:
            if e.response and e.response.status_code:
                # Extract only the primary error message for clarity
                error_msg = f"{e.response.status_code} {e.response.reason}"
                return None, error_msg
            else:
                # If there's no status code, extract detailed error messages
                detailed_errors = extract_errors(str(e)) + ": " + str(e)
                return None, detailed_errors

    # Main fetching
    response, error = try_fetch(url)

    # Check if we should handle alternate schemes due to error
    if error:
        alt_url = None

        if force_https and original_scheme == 'http':
            alt_url = urlunparse(parsed_url._replace(scheme='https'))
        
        elif switch_to_http and original_scheme == 'https':
            alt_url = urlunparse(parsed_url._replace(scheme='http'))

        if alt_url:
            logging.warning(f"Encountered error with {original_scheme.upper()}. Trying alternate URL: {alt_url}")
            alt_response, alt_error = try_fetch(alt_url)

            if alt_response:
                logging.info(f"Successful request with alternate URL. Original URL: {url}. Accessed URL: {alt_url}")
                return alt_response, ""
            else:
                logging.error(f"Failed request with alternate URL: {alt_url}. Error: {alt_error}")
                return None, alt_error

    return response, error

def handle_redirection(response, url):
    if response.url != url:
        logging.info(f"URL Redirected: {url} -> {response.url}")
        return response.url
    return url


def fetch_content(url, session, switch_to_http=False, force_https=False):
    # Fetch content from the URL and handle redirections.
    response, error = get_http_response(url, session, switch_to_http, force_https)
    
    if response:
        url = handle_redirection(response, url)
        return response, f"{response.status_code}", url

    else:
        # If there's no response, return the error as the status_or_error
        return None, error, url

import logging

def process_url(url, heuristic_date_parsing, handle_blogtrottr, bid=None, filter_dates_enabled=False, log_external=False, 
                switch_to_http=False, force_https=False, auto_discover_feed=False, follow_feed_redirects=False):

    original_url = url
    if handle_blogtrottr:
        url = sanitize_url(url)

    bid_url = f"https://blogtrottr.com/subscription/{bid}" if bid else ""
    session = get_http_session()
    response, error_or_status, url = fetch_content(url, session, switch_to_http, force_https)

    # If there's an error in the HTTP response, log and return
    if not response or not response.ok:
        error_msg = f"HTTP error encountered at {url}. Error: {response.status_code if response else error_or_status}"
        logging.error(error_msg)
        return format_output("", "", original_url, "", "", "", error_or_status, url, "", bid_url, handle_blogtrottr)

    # Parse the fetched content
    soup = BeautifulSoup(response.content, determine_parser(response.content.decode('utf-8', 'ignore')))

    # Check for valid feed tags and attempt discovery if none are found
    if not soup.find(['rss', 'feed', 'rdf']):
        logging.info(f"Initial attempt to find feed at {url} failed. Trying further discovery.")
        soup, url, error_message_from_fetch = fetch_feed_url(soup, url, session, original_url, switch_to_http, force_https, auto_discover_feed, follow_feed_redirects)
        
        if not soup:
            error_msg = f"Failed to discover feed at {url}. Error: {error_message_from_fetch}"
            logging.error(error_msg)
            return format_output("", "", original_url, "", "", "", error_message_from_fetch, url, "", bid_url, handle_blogtrottr)

    # Log if the feed URL has changed during processing
    if url != original_url:
        logging.info(f"Feed URL updated: {original_url} -> {url}")

    # Process and return the parsed feed data
    blogging_platform = detect_blogging_platform(soup)
    avg_posts_per_day_str, avg_posts_per_week_str, newest_post, oldest_post, num_posts_str = extract_feed_data(soup, url, heuristic_date_parsing, filter_dates_enabled)
    
    return format_output(avg_posts_per_day_str, avg_posts_per_week_str, original_url, newest_post, oldest_post, num_posts_str, str(response.status_code), url, blogging_platform, bid_url, handle_blogtrottr)

def extract_urls_from_outline(outline_element, handle_blogtrottr):
    urls_and_bids = []
    
    xml_url = outline_element.get('xmlUrl')
    bid = outline_element.get('{http://blogtrottr.com/ns/opml/1.0}id') if handle_blogtrottr else None

    if xml_url:
        urls_and_bids.append((xml_url, bid))
    
    # Check for nested outlines
    for child_outline in outline_element.findall('outline'):
        urls_and_bids.extend(extract_urls_from_outline(child_outline))
    
    return urls_and_bids

def is_valid_url(url):
    parsed = urlparse(url)
    return parsed.scheme in ('http', 'https')

def parse_command_line_arguments():
    parser = argparse.ArgumentParser(description='Process URLs and extract feed data.', allow_abbrev=False)
    parser.add_argument('-n', '--num_processes', type=int, default=multiprocessing.cpu_count(), help='Number of processes to use. Default is the number of CPUs available.')
    parser.add_argument('--filter-dates', action='store_true', default=False, help='Enable date filtering to remove outliers. Default is disabled.')
    parser.add_argument('--heuristic-date-parsing', action='store_true', default=False, help='If set, try hard to parse dates that are otherwise invalid.')
    parser.add_argument('--blogtrottr', action='store_true', default=False, help='Enable handling for Blogtrottr specifics. Default is disabled.')
    parser.add_argument('--log-external', action='store_true', default=False, help='Enable logging from external libraries. Default is disabled.')
    parser.add_argument('--switch-to-http', action='store_true', default=False, help='Switch to HTTP on error with HTTPS (and revert on error). Default is disabled.')
    parser.add_argument('--force-https', action='store_true', default=False, help='Unconditionally switch to HTTPS (but revert to HTTP on error). Default is disabled.')
    parser.add_argument('--auto-discover-feed', action='store_true', default=False, help='Automatically attempt to discover feeds by appending common feed paths. Default is disabled.')
    parser.add_argument('--follow-feed-redirects', action='store_true', default=False, help='Follow redirects for detected feed URLs')
    parser.add_argument('--no-header', action='store_true', default=False, help='Suppress the header in the TSV output. Default is to show the header.')

    args = parser.parse_args()

    # Add additional checks for valid arguments if needed. 
    # For example:
    if args.num_processes <= 0:
        print("Error: Number of processes must be positive.")
        sys.exit(1)

    # Add other checks as necessary

    return args

def print_results(results):
    for result in results:
        if result:
            print(result)

def generate_header(blogtrottr=False):
    base_header = ['Average posts per day', 'Average posts per week', 'Feed URL', 
                   'Newest post date', 'Oldest post date', 'Number of posts', 
                   'HTTP response code', 'Processed URL', 'Blogging Platform']
    if blogtrottr:
        base_header.append('Blogtrottr URL')
    return '\t'.join(base_header)

def execute_pooling(num_processes, urls_and_bids, args):
    pool = multiprocessing.Pool(processes=num_processes)
    results = pool.starmap(process_url, [
        (url, args.heuristic_date_parsing, args.blogtrottr, bid, args.filter_dates, 
         args.log_external, args.switch_to_http, args.force_https, 
         args.auto_discover_feed, args.follow_feed_redirects) 
        for url, bid in urls_and_bids
    ])
    pool.close()
    pool.join()
    return results

def configure_multiprocessing(num_processes, max_allowed, data_length):
    final_processes = min(num_processes, max_allowed, data_length)
    if final_processes < num_processes:
        logging.info(f"Using {final_processes} processes instead of the requested {num_processes} due to the number of input lines.")
    return final_processes

def parse_input_data():
    try:
        input_data = sys.stdin.read()
        root = ET.fromstring(input_data)
        if root.tag.lower() == 'opml':
            main_outline = root.find('./body/outline')
            if main_outline is not None:
                return extract_urls_from_outline(main_outline, args.blogtrottr)
        return []  # Not OPML? Return an empty list and let the ParseError handle it.
    except ET.ParseError:
        # Parsing as plain URLs
        return [(line.strip(), None) for line in input_data.splitlines() if is_valid_url(line.strip())]
    except EOFError:
        logging.error("Encountered EOFError when reading input data.")
        return []

def main():
    args = parse_command_line_arguments()

    urls_and_bids = parse_input_data()
    num_processes = configure_multiprocessing(args.num_processes, MAX_PROCESSES, len(urls_and_bids))
    results = execute_pooling(num_processes, urls_and_bids, args)

    if not args.no_header:
        print(generate_header(args.blogtrottr))
    print_results(results)

if __name__ == '__main__':
    main()
