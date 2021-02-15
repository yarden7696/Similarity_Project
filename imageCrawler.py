import logging
import pandas as pd
import os.path as osp
from argparse import ArgumentParser

from icrawler.builtin import (BaiduImageCrawler, BingImageCrawler,
                              FlickrImageCrawler, GoogleImageCrawler,
                              GreedyImageCrawler, UrlListCrawler)


def main(city):
    bing_crawler = BingImageCrawler(
        downloader_threads=2,
        #put the name of directory as city
        storage={'root_dir': 'images/' + city},
        log_level=logging.INFO)
    search_filters = dict(
        type='photo',
        license='commercial',
        layout='wide',
        size='large',
        date='pastmonth')
    bing_crawler.crawl(city, max_num=20, filters=None)


if __name__ == '__main__':
    df = pd.read_excel('first_300_des.xlsx', engine='openpyxl')
    cities = df['city'].tolist()
    for i in range(len(cities)):
        main(cities[i])