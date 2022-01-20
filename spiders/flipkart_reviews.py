import time
from bs4 import BeautifulSoup
from selenium import webdriver
import sys
import os
import numpy as np
import pandas as pd

def flipkart_scraper(link):
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
        [item.click() for item in driver.find_elements_by_class_name("_1EPkIx")]
        time.sleep(1)
        soup = BeautifulSoup(driver.page_source, 'lxml')

        # Iterating through Reviews

        for items in soup.findAll("div", class_= '_1PBCrt'):
            title = items.select_one("p._2xg6Ul").text
            review = ' '.join(items.select_one(".qwjRop").text.split())
            stars = items.select_one("div", class_="hGSR34 E_uFuv").text[0]
            votes = items.select_one("._2ZibVB").text

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
    print("Number of Reviews Extracted from Flipkart are:", mainreviews.shape[0])

    # Saving the reviews to .csv File

    mainreviews.to_csv('flipkartreviews.csv')
    return mainreviews
    driver.quit()
