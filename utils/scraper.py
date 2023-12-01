from bs4 import BeautifulSoup
import requests
import time
import pandas as pd
import re
import json
import warnings

# Function to get all the vocabulary of a page
def vocabList(link, url, end, list_):
    retries = 3  # Number of retries if a connection error occurs
    while retries > 0:
        try:
            response = requests.get(link, stream=True,verify=False)
            # Check if the request was successful
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")
                voc = soup.find_all("div", {"class": "SearchContainer"})

                if len(voc) > 0:
                    end += 1
                    index = 0
                    for i in range(0, len(voc)):
                        link = voc[i].a["href"]
                        vocabPage(url + link, list_, index)
                return list_, end
            else:
                print(f"Failed to fetch data from {link}. Status code: {response.status_code}")
                retries -= 1
        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")
            retries -= 1
            time.sleep(5)  # Wait for a few seconds before retrying

    return list_, end

# Get all the info from the vocabulary page
def vocabPage(link, list_, index):
    # Pause the code for half a sec
    time.sleep(0.500)
    # Connect to the URL
    response = requests.get(link, stream=True, verify=False)
    # Parse HTML and save to BeautifulSoup object
    soup = BeautifulSoup(response.text, "html.parser")
     # Suppress only the specific warning
    warnings.filterwarnings("ignore", message="Unverified HTTPS request")
    # Restore warning settings afterward
    warnings.resetwarnings()
    
    # Get the title and prefix from the vocabulary page
    title = soup.find("h1")
    prefix = title.span.extract().text.strip()
    title = title.text.strip()
    prefix = prefix.replace("(", "").replace(")", "")
    # Get the URI and Languages from the vocabulary page
    uri = "URI"
    languages = " "
    for child in soup("tbody")[0].find_all("tr"):
        # Get the URI
        if child.td.text.strip() == "URI":
            uri = child.find_all("td")[1].text.strip() 
        # Get the Languages
        if child.td.text.strip() == "Language":
            language = child.find_all("td")[1]
            # Append the Languages with a space as separator
            for childL in language.find_all("a"):
                nameL = childL.find("div", {"class": "agentThumbPrefUri"}).text.strip()
                languages += nameL + " "

    # Get the latest versions and save only the last version's information
    script = soup("script", {"src": None})[3].text.strip()
    versions = re.compile("{\"events\":(.|\n|\r)*?}]}").search(script)
    if versions is not None:
        versions = json.loads(versions.group(0))["events"]
        # Find the last version
        last_version = None
        for version in versions[::-1]:  # Start from the end to get the last version
            if "link" in version.keys():
                last_version = version
                break

        # If the last version is found, create a dictionary and add it to the list
        if last_version is not None:
            versionName = last_version.get("title", f"{prefix}_LastVersion")
            versionName = versionName.replace(" ", "-").replace("\\", "").replace("/", "").replace(":", "").replace("*", "").replace("?", "").replace("\"", "").replace("<", "").replace(">", "").replace("|", "")
            versionDate = last_version.get("start", "")
            last_version_link = last_version.get("link", "")

            versionD = {
                "prefix": prefix,
                "URI": uri,
                "Title": title,
                "Languages": languages,
                "VersionName": versionName,
                "VersionDate": versionDate,
                "Link": last_version_link,
                "Folder": "LOV_Full"
            }
            # Add the last version to the list
            list_.append(versionD)
            # Update the index for the next element of the list
            index += 1

    return list_, index

def scraper():
    # Create the DataFrame to save the LOVs' vocabs' information
    df = pd.DataFrame(columns=["prefix", "URI", "Title", "Languages", "VersionName", "VersionDate", "Link", "Folder"])

    # Set the URL you want to webscrape from
    url = "https://lov.linkeddata.es"
    # Set the starting and ending page to scrape, that updates dynamically
    page = 1
    end = 56
    
    # Scrape every page from the vocabs tab of LOV
    while page < end:
        # Get the #page with the vocabs list
        link = url + "/dataset/lov/vocabs?&page=" + str(page)
        # Examine the list of vocabs
        list_, end = vocabList(link, url, end, list())
        # Add the list of that page to the DataFrame, if there are vocabularies in that page
        if len(list_) > 0:
            temp_df = pd.DataFrame(list_, columns=["prefix", "URI", "Title", "Languages", "VersionName", "VersionDate", "Link", "Folder"])
            df = pd.concat([df, temp_df], ignore_index=True)
        else:
            # No more vocabs found, break the loop
            break
        # Iterate to the next page
        page += 1

    # Save DataFrame to CSV and Excel files
    csv_file = "files/LOV_vocabs.csv"
    excel_file = "files/LOV_vocabs.xlsx"
    df.to_csv(csv_file, index=False)  # Save as CSV without index
    df.to_excel(excel_file, index=False)  # Save as Excel without index

    print(f"Data saved to {csv_file} and {excel_file}")
    # Return the DataFrame for RapidMiner visualization
    return df
