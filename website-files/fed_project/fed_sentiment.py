import os
import PyPDF2
from textblob import TextBlob
import pandas as pd
import matplotlib.pyplot as plt

SPEECH_DIR = 'fed_speeches'
INDEX_DIR = 'index_data'


def read_pdf(pdf):
    '''
    Parameters: pdf (string) - path to the PDF file
    Returns: text (string) - the text extracted from the PDF
    Does: Opens the PDF file and extracts the text from it.
    '''
    text = ""
    
    # Open the PDF file
    with open(pdf, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        num_pages = len(pdf_reader.pages)
        
        for page_number in range(num_pages):
            page = pdf_reader.pages[page_number]
            text += page.extract_text()
    
    return text

def get_filenames(dirname):
    ''' 
    Parameters: dirname (string) - path to the directory
    Returns: filelist (list) - list of filenames in the directory
    Does: Gets the filenames in the directory.
    '''
    
    # Get the filenames in the directory
    filelist = []
    files = os.listdir(dirname)
    for file in files:
        path = dirname + "/" + file
        if not os.path.isdir(path) and not file.startswith("."):
            filelist.append(path)
    return filelist

def get_date_fromfile(file):
    '''
    Parameters: file (string) - path to the file
    Returns: date (int) - date of the file
    Does: Gets the date of the file.
    '''
    
    # Get the date of the file
    date = ''.join(filter(str.isdigit, file.split('/')[-1]))
    return int(date)


def text_from_pdf(pdf):
    '''
    Parameters: pdf (string) - path to the PDF file
    Returns: text (string) - the text extracted from the PDF
    Does: Opens the PDF file and extracts the text from it.
    '''
    text = ""
    
    # Open the PDF file
    with open(pdf, "rb") as file:
       pdf_reader = PyPDF2.PdfReader(file)
       num_pages = len(pdf_reader.pages)
       
       # Extract the text from each page
       for page_number in range(num_pages):
           page = pdf_reader.pages[page_number]
           text += page.extract_text()
   
    return text

def get_sentiment_score(text):
    '''
    Parameters: text (string) - the text to analyze
    Returns: polarity (float) - the sentiment polarity of the text
    Does: Analyzes the sentiment of the text and returns the polarity score.
    '''
   
    # Analyze the sentiment of the text
    blob = TextBlob(text)
    return blob.sentiment.polarity

def sentiment_scores_pdfs(pdf_file_list):
    '''
    Parameters: pdf_file_list (list) - list of paths to PDF files
    Returns: sentiment_scores (list) - list of sentiment scores for each PDF file
    Does: Gets the sentiment scores for each PDF file.
    '''
    sentiment_scores = []
    
    # Get the sentiment scores for each PDF file
    for file in pdf_file_list:
        file_text = text_from_pdf(file)
        sentiment_score = get_sentiment_score(file_text)
        sentiment_scores.append(sentiment_score)
    
    return sentiment_scores

def normalize(lst):
    '''
    Parameters: lst (list) - list of numbers
    Returns: normalized_lst (list) - list of normalized numbers
    Does: Normalizes the numbers in the list.
    '''
    
    # Normalize the numbers in the list
    xmin = min(lst)
    xmax = max(lst)
    normalized_lst = []
    
    # Normalize the numbers in the list
    for i in lst:
        normalized_lst.append((i-xmin) / (xmax-xmin))
        
    return normalized_lst

def store_dates(files):
    '''
    Parameters: files (list) - list of paths to files
    Returns: dates (list) - list of dates of the files
    Does: Gets the dates of the files.
    '''
    dates = []
    
    # Get the dates of the files
    for file in files:
        date = get_date_fromfile(file)
        dates.append(date)
    
    return dates


def extract_prices(directory):
    '''
    Parameters: directory (string) - path to the directory
    Returns: price_df (DataFrame) - DataFrame of prices
    Does: Extracts the prices from the CSV files in the directory and returns a DataFrame.
    '''
    price_data = {}
    
    # Extract the prices from the CSV files in the directory
    for file in os.listdir(directory):
        if file.endswith(".csv"):
        
            filepath = os.path.join(directory, file)
            
            filename = file[:-4]
            
            index_df = pd.read_csv(filepath, encoding = 'utf-8')
            
            index_df = index_df.dropna(subset='Adj Close')
            
            price_data[filename] = index_df['Adj Close']
            
           
    # Add the dates to the DataFrame        
    price_data["Date"] = index_df['Date']
        
    price_df = pd.DataFrame(price_data)

    # Convert the dates to integers
    price_df['Date'] = pd.to_datetime(price_df['Date'])
    price_df['Date'] = price_df['Date'].dt.strftime('%Y%m%d')
    price_df['Date'] = price_df['Date'].astype(int)
    
    return price_df

def df_roc(df, columns = ['QQQ', 'DIA', 'SPY', 'VIX']):
    '''
    Parameters: df (DataFrame) - DataFrame of prices
    columns (list) - list of column names
    Returns: df (DataFrame) - DataFrame of prices with rate of change columns'''
    
    # Add the rate of change columns
    for col in columns:
        df[f'{col} Rate of Change'] = df[col].pct_change().shift(-1) * 100
    
    return df

def plot_scatter(x, y, title, x_label, y_label):
    '''
    Parameters: x (list) - list of x values
    y (list) - list of y values
    title (string) - title of the plot'''

    # Plot the scatter plot
    plt.scatter(x, y)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()

def main():
    files = get_filenames(SPEECH_DIR)
    ordered_files = sorted(files, key=get_date_fromfile)
    dates = store_dates(ordered_files)
    sentiment_scores = sentiment_scores_pdfs(ordered_files)
    normalized_sentscores = normalize(sentiment_scores)
    df = extract_prices(INDEX_DIR)
    filtered_df = df[df['Date'].isin(dates)]
    roc_df = df_roc(filtered_df)
    roc_df['Polarity Score'] = normalized_sentscores
    corr_df = roc_df.iloc[:-1]
    
    # Calculate the correlation between the sentiment scores and the rate of change
    corr_spy = corr_df['Polarity Score'].corr(corr_df['SPY Rate of Change'])
    corr_dow = corr_df['Polarity Score'].corr(corr_df['DIA Rate of Change'])
    corr_vix = corr_df['Polarity Score'].corr(corr_df['VIX Rate of Change'])
    corr_qqq = corr_df['Polarity Score'].corr(corr_df['QQQ Rate of Change'])
    
    # Print the correlation values
    print(f"Correlation with S&P500: {corr_spy}\n")
    print(f"Correlation with DOW: {corr_dow}\n")
    print(f"Correlation with NASDAQ: {corr_qqq}\n")
    print(f"Correlation with VIX: {corr_vix}\n")
    
    # Plot the scatter plots
    plot = plot_scatter
    plot(corr_df['Polarity Score'], corr_df['SPY Rate of Change'], 'Fed Speech Sentiment Score vs. S&P500 Rate of Change', 'Normalized Fed Speech Sentiment Scores', '% Change in Index')
    plot(corr_df['Polarity Score'], corr_df['DIA Rate of Change'], 'Fed Speech Sentiment Score vs. DOW Rate of Change', 'Normalized Fed Speech Sentiment Scores', '% Change in Index')
    plot(corr_df['Polarity Score'], corr_df['QQQ Rate of Change'], 'Fed Speech Sentiment Score vs. NASDAQ Rate of Change', 'Normalized Fed Speech Sentiment Scores', '% Change in Index')
    plot(corr_df['Polarity Score'], corr_df['VIX Rate of Change'], 'Fed Speech Sentiment Score vs. VIX Rate of Change', 'Normalized Fed Speech Sentiment Scores', '% Change in Index')

if __name__ == '__main__':
    main()

