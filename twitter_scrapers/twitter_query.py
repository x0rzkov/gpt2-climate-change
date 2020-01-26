# pulled from Teams/hashtags files
terms = [
    'climate change',
    'global warming',
    '#climatechange',
    '#climatestrike',
    '#climate',
    '#greennewdeal',
    '#globalwarming',
    '#climatebrawl',
    '#climatecrisis',
    '#climatehoax',
    '#climateemergency',
    '#climatescam',
    '#gretathunberg', #maybe not this one
    '#climatecult',
    '#greenhypocrisy',
    '#climateaction',
    '#climatetownhall',
    '#climatechangeisreal',
    '#actonclimate',
    '#climatechangethefacts',
    '#peoplesclimate',
    '#climatehustle',
    '#parisagreement',
    '#climatemarch' # basically the max limit of terms we can include..
]

# remove duplicate values
query = ' OR '.join(list(set(terms)))[:500]
query += ' -filter:retweets'
