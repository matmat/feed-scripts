import argparse
import logging
import multiprocessing
import re
import sys
import xml.etree.ElementTree as ET
from datetime import timedelta, datetime
import argparse
import logging
import multiprocessing
import re
import sys
import xml.etree.ElementTree as ET
from datetime import timedelta, datetime
import argparse
import logging
import multiprocessing
import re
import sys
import xml.etree.ElementTree as ET
from datetime import timedelta, datetime
from urllib.parse import unquote, urljoin, urlparse, urlunparse, parse_qs
from dateutil.tz import tzoffset

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
    "PDT": tzoffset("PDT", -7 * 3600),          # Pacific Daylight Time,  7 hours in seconds 
    "PST": tzoffset("PST", -8 * 3600),          # Pacific Standard Time, 8 hours in seconds
    "MDT": pytz.timezone("US/Mountain"),        # Mountain Daylight Time
    "MST": pytz.timezone("US/Mountain"),        # Mountain Standard Time
    "AEST": pytz.timezone("Australia/Sydney"),  # Australian Eastern Standard Time
    "AEDT": pytz.timezone("Australia/Sydney")   # Australian Eastern Daylight Time
}

USER_AGENT = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36'
#USER_AGENT = 'Googlebot'
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

def is_retryable_exception(exception):
    """Check if the exception is retryable."""
    retryable_exceptions = (
        urllib3.exceptions.ProtocolError      # For lower-level protocol errors (from urllib3)
    )
    return isinstance(exception, retryable_exceptions)


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
    """Extract the actual URL from the provided blogtrottr sanitize URL."""
    parsed_url = urlparse(url)
    
    # Check if the domain is "sanitizer.blogtrottr.com", regardless of the scheme
    if parsed_url.netloc == "sanitizer.blogtrottr.com":
        query_params = parse_qs(parsed_url.query)
        if 'url' in query_params:
            return unquote(query_params['url'][0])
    return url

def discover_feed(base_url, session, switch_to_http=False, force_https=False):
    logging.info(f"Starting feed discovery for base URL: {base_url}")

    for suffix in FEED_SUFFIXES:
        try_url = urljoin(base_url, suffix)
        
        logging.info(f"Trying feed URL: {try_url}")
        
        response, error = get_http_response(try_url, session, switch_to_http, force_https)

        if response and response.content:
            temp_soup = BeautifulSoup(response.content, determine_parser(response.content.decode('utf-8', 'ignore')))
            if temp_soup.find(lambda tag: tag.name and tag.name.lower() in ['rss', 'feed', 'rdf']):
                logging.info(f"Valid feed discovered at {try_url}")
                return temp_soup, try_url, ""
            else:
                logging.warning(f"No valid feed structure discovered at {try_url}")

        else:
            logging.error(f"Failed to fetch content (discover_feed()) from {try_url}. Error: {error}")

    error_message = f"No Atom or RSS feed found at URL (auto-detection attempted!) {base_url}"
    logging.error(error_message)
    return None, try_url, error_message

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

    response, error = fetch_content(url)
    
    if not response:
        logging.error(f"Failed to fetch content (fetch_feed_url()) for URL {url}. Error: {error}")
        return None, url, error

    try:
        soup = BeautifulSoup(response.content, determine_parser(response.content.decode('utf-8', 'ignore')))
        if not soup:
            error_message = f"Failed to parse content into soup for URL {url}."
            logging.error(error_message)
            return None, url, error_message

        # Check for valid feed elements
        if not soup.find(lambda tag: tag.name and tag.name.lower() in ['rss', 'feed', 'rdf']):
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

def remove_leading_symbol(date_str, symbol, url):
    if date_str.startswith(symbol):
        date_str = date_str[1:].strip()
        logging.info(f"Detected and removed '{symbol}' symbol for {url} Date string changed: \"{date_str}\"")
    return date_str

def handle_day_anomalies(date_str, url):
    cur_date_str = date_str
    day_replacements = {
        "Monday":       "Mon",
        "Tuesday":      "Tue",
        "Tues":         "Tue",
        "Wednesday":    "Wed",
        "Thursday":     "Thu",
        "Thurs":        "Thu",
        "Thur":         "Thu",
        "Friday":       "Fri",
        "Fru":          "Fri",
    }
    for k, v in day_replacements.items():
        if k in date_str:
            date_str = date_str.replace(k, v)
            logging.info(f"Detected day anomaly for {url} Date string changed: \"{cur_date_str}\" -> \"{date_str}\"")
            return date_str
    return date_str

def handle_timezone_anomaly(date_str, url):
    cur_date_str = date_str
    timezone_pattern = re.compile(r' \(Pacific (Daylight|Standard) Time\)$')
    if timezone_pattern.search(date_str):
        date_str = timezone_pattern.sub('', date_str)
        logging.info(f"Detected and removed invalid tz information for {url} Date string changed: \"{cur_date_str}\" -> \"{date_str}\"")
    return date_str

def handle_comma_separated_tz(date_str, url):
    cur_date_str = date_str
    if ", -" in date_str or ", +" in date_str:
        date_str = date_str.replace(", -", "-").replace(", +", "+")
        logging.info(f"Detected date with comma-separated tz offset for {url} Date string changed: \"{cur_date_str}\" -> \"{date_str}\"")
    return date_str

def handle_large_hours(date_str, url):
    cur_date_str = date_str
    hour_pattern = re.compile(r'(\d{1,2}T|\s)([0-9]{2,3}):([0-9]{2})')  # Adjusted pattern to capture 2 or 3 digits for hours
    
    match = hour_pattern.search(date_str)
    if match:
        hour_val = int(match.group(2))
        if hour_val >= 24:  # Adjusting for hours 24 or greater
            adjusted_hour = hour_val % 24
            days_to_add = hour_val // 24  # Number of whole days in the hour value

            date_str = hour_pattern.sub(f"{match.group(1)}{adjusted_hour:02}:{match.group(3)}", date_str)

            if days_to_add > 0:
                try:
                    parsed_date = parse(date_str, tzinfos=TZINFOS)
                    parsed_date += timedelta(days=days_to_add)  # Add the days
                    date_str = parsed_date.isoformat()  # Convert datetime to ISO 8601 format string
                    logging.info(f"Detected hour anomaly for {url}. Date string changed (added {days_to_add} day(s)): \"{cur_date_str}\" -> \"{date_str}\"")
                except Exception as e:
                    logging.warning(f"Failed to parse date after large hour adjustment for {url}: \"{date_str}\". Error: {e}")
                    return cur_date_str  # Return the original string if parsing fails

    return date_str

def handle_comma_after_month(date_str, url):
    cur_date_str = date_str
    # Regular expression pattern to match dates in the format "Thu, May 28, 2015, 15:24 - 0700"
    pattern = re.compile(r'(\w+, \w+ \d+), (\d+), (\d+:\d+ [+-]\d+)')
    if pattern.search(date_str):
        date_str = pattern.sub(r'\1 \2 \3', date_str)
        logging.info(f"Detected and adjusted date format with comma after month for {url} Date string changed: \"{cur_date_str}\" -> \"{date_str}\"")
    return date_str

def handle_space_in_tz(date_str, url):
    cur_date_str = date_str
    # Regular expression pattern to match dates in the format "Thu, May 28, 2015, 15:24 - 0700"
    pattern = re.compile(r'(\d+:\d+) ([+-]) (\d+)')
    if pattern.search(date_str):
        date_str = pattern.sub(r'\1\2\3', date_str)
        logging.info(f"Detected and adjusted date format with space in timezone offset for {url} Date string changed: \"{cur_date_str}\" -> \"{date_str}\"")
    return date_str

def custom_date_parser(date_str, heuristic_date_parsing, url):
    # Heuristic Parsing Logic
    if heuristic_date_parsing:
        date_str = remove_leading_symbol(date_str, '>', url)
        date_str = handle_day_anomalies(date_str, url)
        date_str = handle_comma_after_month(date_str, url)
        date_str = handle_space_in_tz(date_str, url)  # Add this line
        date_str = handle_timezone_anomaly(date_str, url)
        date_str = handle_comma_separated_tz(date_str, url)
        date_str = handle_large_hours(date_str, url)


        # If invalid month is detected
        if '-81-' in date_str:
            logging.info(f"Encountered invalid month in date for {url}, skipping")
            return None

    # General Parsing Logic
    try:
        returned_date = parse(date_str, tzinfos=TZINFOS)
        return returned_date
    except Exception as e:
        logging.error(f"Invalid date encountered for URL: {url}. Date string: \"{date_str}\". Error: {e}")
        return None

def valid_name(name):
    if not name:
        return False
    if isinstance(name, str):
        return 'item' in name.lower()
    return False

def find_date_tag(element, *tag_names):
    """Helper function to find date tags ignoring namespaces."""
    for tag_name in tag_names:
        date_element = element.find(lambda tag: tag.name and tag.name.split(":")[-1].lower() == tag_name.lower())
        if date_element and date_element.text:
            return date_element
    return None

def extract_dates_and_count_posts(soup, url, heuristic_date_parsing):
    """Extract dates from the soup object and count the number of posts."""
    dates = []

    def should_fallback_to_published(dates):
        if len(dates) <= 1:
            return False
        return max(dates) - min(dates) <= timedelta(minutes=5)  # Replace with your CONFIGURABLE_INTERVAL

    # RSS Feed Handling
    if soup.find('rss'):
        posts = soup.find_all('item')
        logging.info(f"Found {len(posts)} RSS items for URL {url}")

        # Extract dates from the preferred tag 'updated'
        #preferred_dates = [find_date_tag(post, 'updated').text for post in posts if find_date_tag(post, 'updated')]
        preferred_dates = []
        for post in posts:
            date_tag = find_date_tag(post, 'updated')
            if date_tag:
                try:
                    parsed_date = custom_date_parser(date_tag.text, heuristic_date_parsing, url)
                    preferred_dates.append(parsed_date)
                except Exception as e:
                    logging.error(f"Failed to parse date '{date_tag.text}' from URL {url}. Error: {e}")

        # Fallback logic
        if len(preferred_dates) == len(posts) and should_fallback_to_published(preferred_dates):
            logging.info(f"RSS <updated> dates are within the CONFIGURABLE_INTERVAL for URL {url}. Falling back to using <pubdate>/<date> timestamps.")
            for post in posts:
                date_element = find_date_tag(post, 'pubdate', 'date')
                if date_element and date_element.text:
                    try:
                        parsed_date = custom_date_parser(date_element.text, heuristic_date_parsing, url)
                        if parsed_date:
                            dates.append(parsed_date)
                        else:
                            logging.error(f"Unable to parse RSS {date_element.name} date for URL {url}: {date_element.text}")
                    except ValueError as e:
                        logging.error(f"Invalid date in RSS {date_element.name} for URL {url}: {date_element.text} - Error: {e}")
        else:
            for post in posts:
                date_element = find_date_tag(post, 'updated', 'pubdate', 'date')
                if date_element and date_element.text:
                    try:
                        parsed_date = custom_date_parser(date_element.text, heuristic_date_parsing, url)
                        if parsed_date:
                            dates.append(parsed_date)
                        else:
                            logging.error(f"Unable to parse RSS {date_element.name} date for URL {url}: {date_element.text}")
                    except ValueError as e:
                        logging.error(f"Invalid date in RSS {date_element.name} for URL {url}: {date_element.text} - Error: {e}")

        #If no dates found for individual posts, look for channel-wide date
        #if len(dates) == 0 or len(dates) != len(posts):
        if len(dates) == 0:
            channel = soup.find('channel')
            date_tags_preference_order = ['updated', 'lastBuildDate', 'pubdate', 'date']
            for tag_name in date_tags_preference_order:
                date_element = find_date_tag(channel, tag_name)
                if date_element:
                    try:
                        parsed_date = custom_date_parser(date_element.text, heuristic_date_parsing, url)
                        if parsed_date:
                            dates.extend([parsed_date] * len(posts))
                            logging.info(f"Using top-level RSS {tag_name} date ({parsed_date}) for all items in URL {url}")
                            break  # If a valid date is found, break out of the loop
                        else:
                            logging.error(f"Unable to parse RSS channel {tag_name} date for URL {url}: {date_element.text}")
                    except ValueError as e:
                        logging.error(f"Invalid date in RSS channel {tag_name} for URL {url}: {date_element.text} - Error: {e}")

    # Atom Feed Handling
    elif soup.find('feed'):
        posts = soup.find_all(['entry', 'item'])
        logging.info(f"Found {len(posts)} Atom items for URL {url}")

        # Extract dates from the preferred tags 'updated' and 'modified'
        #preferred_dates = [find_date_tag(post, 'updated', 'modified').text for post in posts if find_date_tag(post, 'updated', 'modified')]

        preferred_dates = []
        for post in posts:
            date_tag = find_date_tag(post, 'updated', 'modified')
            if date_tag:
                try:
                    parsed_date = custom_date_parser(date_tag.text, heuristic_date_parsing, url)
                    preferred_dates.append(parsed_date)
                except Exception as e:
                    logging.error(f"Failed to parse date '{date_tag.text}' from URL {url}. Error: {e}")

        #print("preferred_dates: ", preferred_dates)

        # Fallback logic
        if len(preferred_dates) == len(posts) and should_fallback_to_published(preferred_dates):
            logging.info(f"Atom <updated>/<modified> dates are within the CONFIGURABLE_INTERVAL for URL {url}. Falling back to using <published>/<created> timestamps.")
            for post in posts:
                date_element = find_date_tag(post, 'published', 'created')
                if date_element and date_element.text:
                    try:
                        parsed_date = custom_date_parser(date_element.text, heuristic_date_parsing, url)
                        if parsed_date:
                            dates.append(parsed_date)
                        else:
                            logging.error(f"Unable to parse Atom {date_element.name} date for URL {url}: {date_element.text}")
                    except ValueError as e:
                        logging.error(f"Invalid date in Atom {date_element.name} for URL {url}: {date_element.text} - Error: {e}")
        else:
            for post in posts:
                date_element = find_date_tag(post, 'updated', 'modified', 'published', 'created')
                if date_element and date_element.text:
                    try:
                        parsed_date = custom_date_parser(date_element.text, heuristic_date_parsing, url)
                        if parsed_date:
                            dates.append(parsed_date)
                        else:
                            logging.error(f"Unable to parse Atom {date_element.name} date for URL {url}: {date_element.text}")
                    except ValueError as e:
                        logging.error(f"Invalid date in Atom {date_element.name} for URL {url}: {date_element.text} - Error: {e}")

        # If there are no posts or if all posts are missing dates
        if len(posts) == 0 or all(date is None for date in preferred_dates):
            feed_date_element = find_date_tag(soup.find('feed'), 'updated')
            if feed_date_element:
                try:
                    parsed_date = custom_date_parser(feed_date_element.text, heuristic_date_parsing, url)
                    if parsed_date:
                        dates.extend([parsed_date] * len(posts))  # Apply the same date to all posts
                        logging.info(f"Using feed-level Atom <updated> date ({parsed_date}) for all items in URL {url}")
                    else:
                        logging.error(f"Unable to parse Atom feed <updated> date for URL {url}: {feed_date_element.text}")
                except ValueError as e:
                    logging.error(f"Invalid date in Atom feed <updated> for URL {url}: {feed_date_element.text} - Error: {e}")

    # RDF Feed Handling (Unchanged)
    elif soup.find(lambda tag: tag.name and tag.name.lower() in ['rdf', 'rdf:rdf']):
        posts = soup.find_all('item')
        logging.info(f"Found {len(posts)} RDF items for URL {url}")
        for post in posts:
            date_element = post.find('dc:date')
            if date_element:
                date_text = date_element.text
                try:
                    parsed_date = custom_date_parser(date_text, heuristic_date_parsing, url)
                    if parsed_date:
                        dates.append(parsed_date)
                    else:
                        logging.error(f"Unable to parse RDF post date for URL {url}: {date_text}")
                except ValueError as e:
                    logging.error(f"Invalid date in RDF post for URL {url}: {date_text} - Error: {e}")

        logging.info(f"Found {len(dates)} dates in RDF items for URL {url}")

    # Ensure all dates have a timezone
    dates = [date.replace(tzinfo=pytz.UTC) if date.tzinfo is None else date for date in dates]

    if len(dates) != len(posts):
        logging.warning(f"Number of dates ({len(dates)}) doesn't match the number of posts ({len(posts)}) for URL {url}")

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
    
    # Check for RDF tag, which indicates it might be an XML-based RSS feed
    is_rdf = '<rdf:rdf' in content_lower

    # Determine parser type
    if (is_xml or is_rdf) and not is_html:
        return 'xml'
    elif not is_xml and is_html:
        return 'html.parser'
    elif (is_xml or is_rdf) and is_html:
        return 'xml'
    else:
        return 'html.parser'


def format_output(avg_posts_per_day_str, avg_posts_per_week_str, original_url, newest_post, oldest_post, num_posts_str, status_or_error, url, blogging_platform, bid_url, handle_blogtrottr):
    base_output = f'{avg_posts_per_day_str}\t{avg_posts_per_week_str}\t{original_url}\t{newest_post}\t{oldest_post}\t{num_posts_str}\t{status_or_error}\t{url}\t{blogging_platform}'
    if handle_blogtrottr:
        base_output += f'\t{bid_url}'
    return base_output

import re

def detect_blogging_platform(soup):
    """Detect the blogging platform from the soup."""
    generator_tag = soup.find('generator')
    if not generator_tag:
        return ""  # return empty string if platform can't be determined

    tag_text = generator_tag.text.strip().replace('\n', ' ')  # Replacing newlines with spaces

    known_platform_patterns = [
        (re.compile(r'substack', re.I), "Substack"),
        (re.compile(r'Medium', re.I), "Medium"),
        #(re.compile(r'wordpress\.org', re.I), "WordPress"),
        # Add more platforms as needed in the format (compiled_regex, "ReturnText"),
    ]

    for pattern, platform_name in known_platform_patterns:
        if pattern.search(tag_text):
            return platform_name

    if generator_tag.attrs:
        return str(generator_tag).replace('\n', ' ')  # return the entire tag if it contains attributes

    return tag_text

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

            # Check and log redirection after a successful request
            handle_redirection(r, u)
            
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

    # Log the original error that causes trying alternate schemes
    if error:
        logging.error(f"Failed request with original URL: {url}. Error: {error}")
    
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

def process_url(url, heuristic_date_parsing, handle_blogtrottr, bid=None, filter_dates_enabled=False, log_external=False, 
                switch_to_http=False, force_https=False, auto_discover_feed=False, follow_feed_redirects=False):

    original_url = url
    if handle_blogtrottr:
        url = sanitize_url(url)

    bid_url = f"https://blogtrottr.com/subscription/{bid}" if bid else ""
    session = get_http_session()
    
    response, error_or_status, url = fetch_content(url, session, switch_to_http, force_https)

    # If there's an error in the HTTP response and auto_discovery is not enabled, log and return
    if not response or not response.ok:
        error_msg = f"HTTP error encountered at {url}. Error: {response.status_code if response else error_or_status}"
        logging.error(error_msg)
        
        if not auto_discover_feed:
            return format_output("", "", original_url, "", "", "", error_or_status, url, "", bid_url, handle_blogtrottr)
        
        # If auto_discovery is enabled, attempt discovery despite the HTTP error
        logging.info("Trying auto-discovery despite HTTP error due to auto-discover-feed option being set.")
        soup, url, error_message_from_fetch = fetch_feed_url(None, url, session, original_url, switch_to_http, force_https, auto_discover_feed, follow_feed_redirects)
        
        if not soup:
            error_msg = f"Failed to discover feed for {original_url} after HTTP error."
            logging.error(error_msg)
            return format_output("", "", original_url, "", "", "", error_message_from_fetch, url, "", bid_url, handle_blogtrottr)
    else:
        # Parse the fetched content
        soup = BeautifulSoup(response.content, determine_parser(response.content.decode('utf-8', 'ignore')))

        # Check for valid feed tags and attempt discovery if none are found
        if not soup.find(lambda tag: tag.name and tag.name.lower() in ['rss', 'feed', 'rdf']):
        #if not find_date_tag(soup, 'rss', 'feed', 'rdf'):

            logging.info(f"Initial attempt to find feed at {url} failed. Trying further discovery.")
            soup, url, error_message_from_fetch = fetch_feed_url(soup, url, session, original_url, switch_to_http, force_https, auto_discover_feed, follow_feed_redirects)
            
            if not soup:
                error_msg = f"Failed to discover feed for {original_url}"
                logging.error(error_msg)
                return format_output("", "", original_url, "", "", "", error_message_from_fetch, url, "", bid_url, handle_blogtrottr)

    # Log if the feed URL has changed during processing
    if url != original_url:
        logging.info(f"Feed URL updated: {original_url} -> {url}")

    # Process and return the parsed feed data
    blogging_platform = detect_blogging_platform(soup)
    avg_posts_per_day_str, avg_posts_per_week_str, newest_post, oldest_post, num_posts_str = extract_feed_data(soup, url, heuristic_date_parsing, filter_dates_enabled)
    
    return format_output(avg_posts_per_day_str, avg_posts_per_week_str, original_url, newest_post, oldest_post, num_posts_str, str(response.status_code if response else error_or_status), url, blogging_platform, bid_url, handle_blogtrottr)

def extract_urls_from_outline(outline_element, handle_blogtrottr):
    urls_and_bids = []
    
    xml_url = outline_element.get('xmlUrl')
    bid = outline_element.get('{http://blogtrottr.com/ns/opml/1.0}id') if handle_blogtrottr else None

    if xml_url:
        urls_and_bids.append((xml_url, bid))
    
    # Check for nested outlines
    for child_outline in outline_element.findall('outline'):
        urls_and_bids.extend(extract_urls_from_outline(child_outline, handle_blogtrottr))
    
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
    # Begin newly added lines
    if data_length == 0:
        logging.error("No valid URLs found in the input data.")
        sys.exit(1)
    # End newly added lines

    final_processes = min(num_processes, max_allowed, data_length)
    if final_processes < num_processes:
        logging.info(f"Using {final_processes} processes instead of the requested {num_processes} due to the number of input lines.")
    return final_processes

def parse_input_data(args):
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

    urls_and_bids = parse_input_data(args)
    num_processes = configure_multiprocessing(args.num_processes, MAX_PROCESSES, len(urls_and_bids))
    results = execute_pooling(num_processes, urls_and_bids, args)

    if not args.no_header:
        print(generate_header(args.blogtrottr))
    print_results(results)

if __name__ == '__main__':
    main()
