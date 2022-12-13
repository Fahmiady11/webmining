from tracemalloc import start
import scrapy

class LinkSpider(scrapy.Spider):
    name='link'
    start_urls=[]
    for i in range(1, 15+1):
        start_urls.append(f'https://www.tripadvisor.co.id/Attraction_Review-g3384116-d3619178-Reviews-Wisata_Bahari_Lamongan-Lamongan_East_Java_Java.html')
        

    def parse(self, response):
        count=0
        link=[]
        for jurnal in response.css('#tab-data-qa-reviews-0 > div > div.LbPSX > div:nth-child(11) > div:nth-child(2) > div > div.tgunb.j > div'):
            count+=1
            for j in range(1,6):
                yield {
                    'link': response.css(f'div:nth-child({j})').get(),
                }