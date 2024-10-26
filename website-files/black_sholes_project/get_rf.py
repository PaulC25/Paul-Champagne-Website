import requests
from bs4 import BeautifulSoup


# URL of the page to scrape
URL = 'https://www.cnbc.com/quotes/US3M'

def extract_treasury_rate(url):
    '''
    Parameters: url (string) - URL of the webpage to scrape
    Returns: (float) - The 3-month Treasury Bill rate as a decimal
    Does: Scrapes the specified URL for the 3-month Treasury Bill rate and returns it as a decimal.
          It uses BeautifulSoup and regular expressions to extract the rate from the HTML content.
    '''
    
    response = requests.get(url)
    response.raise_for_status()

    # Parse the HTML content
    soup = BeautifulSoup(response.text, 'html.parser')

    yield_element = soup.find('span', class_='QuoteStrip-lastPrice')

    if yield_element:
        yield_text = yield_element.get_text().strip('%')  # Remove the '%' symbol
        return round(float(yield_text) / 100 ,6) # Convert to decimal
    else:
        return 'Yield not found'


