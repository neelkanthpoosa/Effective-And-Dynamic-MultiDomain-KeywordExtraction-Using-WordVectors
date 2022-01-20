import time
from bs4 import BeautifulSoup
from selenium import webdriver
import sys
import os
import numpy as np
import pandas as pd
import time
from difflib import SequenceMatcher
from spiders.flipkart_reviews import flipkart_scraper
from spiders.snapdeal_reviews import snapdeal_scraper
from spiders.newamazon_reviews import amazon_scraper

# Function for Multi Domain Scraping

def getreviews(link):
    driver = webdriver.Chrome(os.path.join(os.getcwd(), 'spiders/chromedriver'))

    # Opening the product link
    driver.get(link)
    soup = BeautifulSoup(driver.page_source, 'lxml')
    time.sleep(5)

    # Finding the product title

    titles = soup.findAll('h1',class_='_9E25nV')
    for tit in titles:
        flipkartitle = tit.text
    snaplist = flipkartitle.split()
    snapsearch = ""
    amazonsearch = ""

    # Finding the Model Number

    stats = soup.findAll("tr", class_= '_3_6Uyw row')
    for x in stats:
        if(x.text.startswith("Model Number")):
            model = x.text[12:]
            print("Model is:",model)

    # Building Flipkart Review Page link

    flipreviewlink = driver.find_elements_by_css_selector('.col._39LH-M a')[-1].get_attribute('href') + "&page={}"

    # Running Flipkart Scraper

    flipkartreviews = flipkart_scraper(flipreviewlink)

    # Bulding Snapdeal and Amazon search Query

    for g in snaplist[0:3]:
        amazonsearch = amazonsearch + "+" + g
    for h in snaplist:
        snapsearch = snapsearch + "+" + h


    driver.quit()
    modellist = model.split()
    search = ""
    for i in modellist:
        search = search + "+" + i
    search = search[1:]
    driver = webdriver.Chrome(os.path.join(os.getcwd(), 'spiders/chromedriver'))
    amazon = 'https://www.amazon.in/s?k={}&ref=nb_sb_noss_2'
    snapdeal = 'https://www.snapdeal.com/search?keyword={}&santizedKeyword=G50-80&catId=0&categoryId=0&suggested=false&vertical=p&noOfResults=20&searchState=&clickSrc=go_header&lastKeyword=&prodCatId=&changeBackToAll=false&foundInAll=false&categoryIdSearched=&cityPageUrl=&categoryUrl=&url=&utmContent=&dealDetail=&sort=rlvncy'

    # Searching using Search Query on Amazon

    driver.get(amazon.format(amazonsearch[1:]))
    soup = BeautifulSoup(driver.page_source, 'lxml')

    # Browsing through Products on Amazon

    products = driver.find_elements_by_css_selector('.a-link-normal.a-text-normal')
    for w in products:
        productlink = w.get_attribute('href')

        # Opening Product page on Amazon
        driver.get(productlink)
        time.sleep(10)
        break

    # Finding Review Page link for Amazon

    reviewlink = driver.find_element_by_css_selector('a.a-link-emphasis')
    amazonreviewlink = reviewlink.get_attribute("href")
    amazonreviewlink = amazonreviewlink.replace('_dp','_arp').replace('show_all_btm','paging_btm_next_{}')

    # Running Amazon Scraper

    amazon_scraper(amazonreviewlink + "&pageNumber={}")

    # Running Search Query on Snapdeal

    driver = webdriver.Chrome(os.path.join(os.getcwd(), 'spiders/chromedriver'))
    driver.get(snapdeal.format(snapsearch[1:].replace('/','%2F').replace(',','%2C')))

    # Finding Review Link on snapdeal

    products = driver.find_elements_by_css_selector('.product-tuple-image  a')
    for w in products:
        snaplink = w.get_attribute('href')
        # driver.get(snaplink + "/reviews?page=1&sortBy=HELPFUL")
        break

    # Running Snapdeal Scraper

    try:
        snapdeal_scraper(snaplink + "/reviews?page={}&sortBy=HELPFUL")
    except:
        pass
    driver.quit()
