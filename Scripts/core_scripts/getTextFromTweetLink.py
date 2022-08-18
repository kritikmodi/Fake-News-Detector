CONSUMER_KEY = "eJL1xOgPnXVx0DzCr5pGa8lNv"
CONSUMER_SECRET = "iDBuPdCEXZQDzsRqvNtkVcIhcdvlT8x8aW74VTm1EqXcIaPmrZ"
OAUTH_TOKEN = "1420293020080082948-Qo8PBaf5oXA1xrPryabo3C3g09xdBf"
OAUTH_TOKEN_SECRET = "HzW0KllX3NRls7pOKA0cPIbNEyAFWON9wgVODcRwlrVBi"
BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAABKRewEAAAAAem8kRGNx2U5tGTmD%2BtikqmENETI%3DCqzEWLXGEmM8WKNXRRnW0Tke4QlWw2sihgwtjpVYwnAR0QD6bo"
import tweepy
import requests
import re
api = tweepy.Client(bearer_token = BEARER_TOKEN)

def get_tweet(link):
    id = int(link.split('/')[-1])
    tweetContent = api.get_tweet(id)
    tc = str(tweetContent)
    start = tc.index("'")
    end = tc.index("'",start+1)
    substring = tc[start+1:end]
    print(substring)
    
link = "https://twitter.com/kunalstwt/status/1557403946842558464"
get_tweet(link)
