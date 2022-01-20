import time
from bs4 import BeautifulSoup
from selenium import webdriver
import sys
import os
import numpy as np
import pandas as pd

def snapdeal_scraper(link):
    alltitles = []
    allreviews = []
    allstars = []
    allvotes = []
    path = 'chromedriver'
    driver = webdriver.Chrome(os.path.join(os.getcwd(), 'chromedriver'))
    #If necessary, define the chrome path explicitly
    for page_num in range(1,2):

        # Navigating to Review Page

        driver.get(link.format(page_num))

        time.sleep(1)
        soup = BeautifulSoup(driver.page_source, 'lxml')

        # Iterating through Reviews

        for items in soup.findAll("div", class_= 'commentlist first jsUserAction'):

            # Accesing Various Attributes of the Reviews

            title = items.select_one(".head").text
            review = items.select_one("p").text
            stars = len(items.find_all("i", class_="active"))
            votes = items.select_one(".hf-review").text[0]

            # Storing the Reviews
            
            alltitles.append(title)
            allreviews.append(review)
            allstars.append(stars)
            allvotes.append(votes)

     # Creating main DataFrame

    mainreviews = pd.DataFrame([])
    mainreviews['Title'] = np.array(alltitles)
    mainreviews['comment'] = np.array(allreviews)
    mainreviews['Stars'] = np.array(allstars)
    mainreviews['votes'] = np.array(allvotes)
    print("Number of Reviews Extracted from Snapdeal are:", mainreviews.shape[0])

    # Saving to csv file

    driver.quit()
    mainreviews.to_csv('snapdealreviews.csv')
    return mainreviews
# snapdeal_scraper(link1)
