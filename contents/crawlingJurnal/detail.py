import scrapy
import pandas as pd

class Spider(scrapy.Spider):
    name = 'detail'
    data_csv = pd.read_json('jurnal.json').values
    start_urls = [ link[0] for link in data_csv ]

    def parse(self, response):
        yield {
            'Judul': response.css('#content_journal > ul > li > div:nth-child(2) > a::text').extract(),
            'Abstraksi': response.css('#content_journal > ul > li > div:nth-child(4) > div:nth-child(2) > p::text').extract(),
        }