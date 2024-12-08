import requests
import re
import pandas as pd
from time import sleep
from bs4 import BeautifulSoup
from random import randrange
from numpy import nan
from tqdm import tqdm

# Section of code is to get the titles of the table
def parse_titles(titles:BeautifulSoup):
    """
        Searches for all tags labeled 'th' in a BeautifulSoup object, recursively. Returns the title text of each th tag in a list.
        
        Args:
            titles (BeautifulSoup). As long as the 'th' tags desired are isolated in the BS object (and no other unintended titles), this code should work as intended.
        Returns:
            titles (list). List of title strings.
    """
    t = list()
    for title in titles.find_all('th'):
        text = title.text
        if text in ['\xa0', "#"]: continue # We don't need the checkbox or result index
        t.append(text)
    titles = t
    return titles
def parse_table_entry_data(row:BeautifulSoup):
    """
        Turns a table row and turns it into a list containing each data entry.

        Args: 
            row (BeautifulSoup) that needs to be turned into a list.
        Returns: 
            row_entries (list)
    
    """
    entries = row.find_all('td', recursive=False)
    e = list()
    for entry in entries:
        e.append(entry.text)
    return e


def fix_construction_entry(entry:str):
    key_titles = [
        'Worker Height Above Ground/Floor:',
        'Cause:',
        'Fatality Cause:'
    ]
    for title in key_titles:
        entry = entry.replace(title, f'\n{title}', 1)
    return entry

def drop_columns(df, columns_to_drop):
    for col in columns_to_drop:
        try:
            df = df.drop(col, axis=1)
        except KeyError:
            continue
    return df
    
def parse_time_of_incident(abstract:str):
    """
        Searches for an incident time given the abstract text. Returns pd.NA if a time was not found.
        Args:
            abstract (string)
        Returns:
            time_of_incident (string)
    """
    if type(abstract) == type(pd.NA): return pd.NA

    time_of_incident = re.search(r"(([0-1][0-9])|([1-9]))\s*:\s*[0-9]{2}\s*([AaPp]\.?[Mm]\.?)?", abstract) # Searches for time format hh:mm p.m. in all variations
    if time_of_incident is None:
        time_of_incident = re.search(r"(([0-1][0-9])|([1-9]))\s*([AaPp]\.?[Mm]\.?)", abstract) # Searches for time format h p.m in all vairations
        if time_of_incident is None:    return pd.NA

    time_of_incident = time_of_incident.group(0)
    time_of_incident = re.sub(r'[Aa]\.?', 'a.', time_of_incident)
    time_of_incident = re.sub(r'[Pp]\.?', 'p.', time_of_incident)
    time_of_incident = re.sub(r'[Mm]\.?', 'm.', time_of_incident)
    time_of_incident = re.sub(r'\s*:\s*', ':', time_of_incident)

    if not any(x in time_of_incident for x in ['a.m.', 'p.m.']):    
        time_of_incident = time_of_incident.strip() + ' a.m.' # Adding default case
    if ":" not in time_of_incident:
        time_of_incident = re.sub(r'([0-1]?[0-9])(\s*)([ap].m.)', r'\1:00 \3', time_of_incident, flags=re.IGNORECASE)
    return time_of_incident


def get_accident_search_results(webpage_content:BeautifulSoup) -> pd.DataFrame:
    """
        The function searches for the accident results on the single page and transforms it into a Pandas Dataframe.

        Args:
            webpage_content (BeautifulSoup)
        Returns:
            accident_table (Pandas DataFrame)
    """
    # Results table has a blank id label. Might change if they update their website
    results_table = webpage_content.find(name='table', attrs={'aria-label' : ''})
    titles = results_table.thead.tr # Gets the table head tag, which holds the table row that has the header informatino we're looking for
    table_body = results_table.find_all(name='tr', recursive=False)

    results = list()
    for result in table_body:
        results.append(parse_table_entry_data(result)[2:]) # the [2:] is to skip the checkbox and result index
    titles = parse_titles(titles)
    return pd.DataFrame(data=results, columns=titles)


def get_employee_details_results(webpage_content:BeautifulSoup) -> pd.DataFrame:
    employee_details = webpage_content.find('table', attrs={'name':'accidentEmployeeDetails'})
    if employee_details == None:    return None
    employee_details = employee_details.tbody
    table_body = employee_details.find_all('tr', recursive=False)
    titles = parse_titles(table_body.pop(0))

    results = list()
    for result in table_body:
        entry = parse_table_entry_data(result)
        entry[-1] = fix_construction_entry(entry[-1]) # The tags on the last column mess up the way the data is processed, wanted to fix it with newlines
        results.append(entry)

    return pd.DataFrame(data=results, columns=titles)

def get_accident_details_results(webpage_content:BeautifulSoup) -> pd.DataFrame:
    accident_details = webpage_content.find('table', attrs={'name' : 'accidentDetails'})
    if accident_details == None:    return None
    accident_details = accident_details.tbody
    titles = parse_titles(accident_details)
    table_body = accident_details.find_all('tr')[1:] # In this section, index 0 is the title column

    results = list()
    for result in table_body:
        entry = parse_table_entry_data(result)
        results.append(entry)
    return pd.DataFrame(data=results, columns=titles)

def get_abstract_and_keywords_results(webpage_content:BeautifulSoup) -> pd.DataFrame:
    abstract = webpage_content.find('p')
    if abstract != None and 'Abstract' in abstract.text:
        abstract = abstract.text.replace('Abstract: \n', '')
    else:   
        abstract = pd.NA

    time_of_incident = parse_time_of_incident(abstract=abstract)

    keywords = webpage_content.find('div')
    if keywords != None and 'Keywords' in keywords.text:
        keywords = keywords.text.replace('Keywords: \n', '')
    else:   
        keywords = pd.NA
    return pd.DataFrame([[abstract, keywords, time_of_incident]], columns=['Abstract', 'Keywords', 'Time of Incident'])

def get_site_address_results(webpage_content:BeautifulSoup) -> pd.DataFrame:
    site_locations = webpage_content.find_all('p')
    site_location = None
    for site in site_locations:
        if "Site Address: \n" in site.text:
            site_location = site
            break
    if site_location == None:   return None

    site_location.strong.decompose()
    site_location = site_location.get_text(separator='\n').strip()
    site_location = site_location.lstrip(':').strip()
    site_location = site_location.replace('\t', '')
    bad_string = lambda mystr : mystr.isspace() or not mystr 
    site_location = '\n'.join([x.strip() for x in site_location.split('\n') if not bad_string(x)])

    zip_code = site_location[-5:]
    if not zip_code.isnumeric():
        raise RuntimeError(f"Could not properly find zip code. Instead got the string: {zip_code}")

    return pd.DataFrame([[site_location, zip_code]], columns=['Site Address', 'Zip Code'])

def get_accident_summary_results(webpage_content:BeautifulSoup) -> pd.DataFrame:
    accident_overview = webpage_content.find('table', attrs={'name': 'accidentOverview'})
    if accident_overview == None:
        return None
    accident_overview = accident_overview.tbody
    table_body = accident_overview.find_all('tr')
    try:
        titles = parse_titles(table_body[0])
        inspection_entry = table_body[1] # In this section, index 0 is the title column
        results = [parse_table_entry_data(inspection_entry)]
        return pd.DataFrame(data=results, columns=titles)
    except KeyError:
        return None



######################################################### MAIN LINES OF CODE ##################################################################
# NOTE: at the end of the url link, notice the additional &p_show= parameter. That is so every result is on one long webpage.
def main():
    PARAMETERS = {
        'naics' : 2373,
        'endmonth' : 1,
        'endday' : 1,
        'endyear' : 1984,
        'startmonth' : 12,
        'startday' : 31,
        'startyear' : 2025
    }

    NHEADERS = {
        'User-Agent' : 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:130.0) Gecko/20100101 Firefox/130.0'
    }

    print("Performing The Search...")
    url = f"https://www.osha.gov/ords/imis/AccidentSearch.search?p_logger=1&acc_description=&acc_Abstract=&acc_keyword=&sic=&naics={PARAMETERS['naics']}&Office=All&officetype=All&endmonth={PARAMETERS['endmonth']}&endday={PARAMETERS['endday']}&endyear={PARAMETERS['endyear']}&startmonth={PARAMETERS['startmonth']}&startday={PARAMETERS['startday']}&startyear={PARAMETERS['startyear']}&InspNr=&p_show=300000"
    # Make sure when doing your webpage request that you have the most current User Agent, or there's a possibility for your request being blocked 

    accidentSearch_content = requests.get(url=url, headers=NHEADERS)
    accidentSearch_content.raise_for_status() # If the webpage wasn't properly pulled, an error will occur
    accidentSearch_content = BeautifulSoup(accidentSearch_content.content, 'html.parser')
    accident_table = get_accident_search_results(webpage_content=accidentSearch_content)


    print("Scraping Data for Each Accident. This May Take A While...")
    final_employee_table = pd.DataFrame()
    for i in tqdm(range(accident_table.shape[0])):
        summary_id = accident_table['Summary Nr'][i]
        url = f"https://www.osha.gov/ords/imis/accidentsearch.accident_detail?id={summary_id}"
        accidentDetail_content = requests.get(url=url, headers=NHEADERS)
        accidentDetail_content.raise_for_status() # If the webpage wasn't properly pulled, an error will occur
        accidentDetail_content = BeautifulSoup(accidentDetail_content.content, 'html.parser')
        accidentDetail_content = accidentDetail_content.find('div', {'class' : 'table-responsive'})
        
        employee_table = get_employee_details_results(webpage_content=accidentDetail_content)
        if (employee_table) is not None:
            
            employee_table['Summary Nr'] = pd.Series(summary_id, index=employee_table.index, dtype='category')

            employee_table
            columns_to_drop = ["Inspection Nr"]
            employee_table = drop_columns(employee_table, columns_to_drop)

        final_employee_table = pd.concat([final_employee_table, employee_table], axis=0)


        accident_details_table = get_accident_details_results(webpage_content=accidentDetail_content)
        if (accident_details_table) is not None:
            columns_to_drop = ["Fatality"]
            accident_details_table = drop_columns(accident_details_table, columns_to_drop)

        abstract_keywords_table = get_abstract_and_keywords_results(webpage_content=accidentDetail_content)
        
        accident_overview_table = get_accident_summary_results(webpage_content=accidentDetail_content)
        if (accident_overview_table) is not None:
            columns_to_drop = ['SIC', 'NAICS', 'Date Opened']
            accident_overview_table = drop_columns(accident_overview_table, columns_to_drop)

        inspection_number = accident_overview_table['Inspection Nr'][0]
        url = f'https://www.osha.gov/ords/imis/establishment.inspection_detail?id={inspection_number}'

        inspectionDetail_content = requests.get(url=url, headers=NHEADERS)
        inspectionDetail_content.raise_for_status() # If the webpage wasn't properly pulled, an error will occur
        inspectionDetail_content = BeautifulSoup(inspectionDetail_content.content, 'html.parser')

        site_address_table = get_site_address_results(webpage_content=inspectionDetail_content)

        tables = [accident_overview_table, abstract_keywords_table, accident_details_table, site_address_table]
        final_table = pd.DataFrame()
        for table in tables:
            if (table) is not None: final_table = pd.concat([final_table, table], axis=1)
        final_table.index = [i]

        accident_table = pd.concat([accident_table, final_table], axis=1)
        sleep(randrange(0, 20) / 100)
        accident_table = accident_table.loc[:, ~accident_table.columns.duplicated()].combine_first(final_table)


    secondary_columns = accident_table.columns
    primary_columns = ['Summary Nr', 'Event Date', 'Zip Code', 'Time of Incident', 'Event Description', 'Abstract',  'Keywords', 'Site Address']
    secondary_columns = [x for x in secondary_columns if x not in primary_columns]
    columns = primary_columns + secondary_columns

    accident_table = accident_table[columns]
    final_employee_table = final_employee_table.set_index('Summary Nr')
    print(accident_table)
    print(final_employee_table)    

    accident_table = accident_table.replace([r'\xa0', nan, r'^\s*$', 'Â', ' '], pd.NA, regex=True)
    final_employee_table = final_employee_table.replace([r'\xa0', nan, r'^\s*$', 'Â', ' '], pd.NA, regex=True)
    accident_table.to_csv("accident_data.csv", index=False, encoding='utf-8')
    final_employee_table.to_csv("employee_data.csv", encoding='utf-8')


if __name__ == "__main__":
    main()