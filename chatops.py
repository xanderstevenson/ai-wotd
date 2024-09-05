#!/Users/alexstev/Documents/CiscoDevNet/code/ai-wod/venv/bin/python3
import json
import requests
import re
import time
import subprocess
from requests_oauthlib import OAuth1Session
import os
from terms import return_word
from datetime import datetime
from passwords import (
    profile_id,
    li_access_token,
    TEAMS_ACCESS_TOKEN,
    x_api_key,
    x_api_secret_key,
)

# Path to save tokens
token_file = "tokens.json"


# Simple Bot Function for passing messages to a room
def send_it(token, room_id, message):

    header = {"Authorization": "Bearer %s" % token, "Content-Type": "application/json"}

    data = {"roomId": room_id, "text": message, "attachments": card}

    return requests.post(
        "https://api.ciscospark.com/v1/messages/",
        headers=header,
        data=json.dumps(data),
        verify=True,
    )


def save_tokens(token_data):
    with open(token_file, "w") as f:
        json.dump(token_data, f)


def load_tokens():
    if os.path.exists(token_file):
        with open(token_file, "r") as f:
            return json.load(f)
    return None


def get_oauth_session():
    # Check if tokens exist
    tokens = load_tokens()
    if tokens:
        # If tokens exist, create a session with them
        oauth = OAuth1Session(
            x_api_key,
            client_secret=x_api_secret_key,
            resource_owner_key=tokens["oauth_token"],
            resource_owner_secret=tokens["oauth_token_secret"],
        )
    else:
        # Run the authorization process
        request_token_url = "https://api.twitter.com/oauth/request_token?oauth_callback=oob&x_auth_access_type=write"
        oauth = OAuth1Session(x_api_key, client_secret=x_api_secret_key)
        fetch_response = oauth.fetch_request_token(request_token_url)

        resource_owner_key = fetch_response.get("oauth_token")
        resource_owner_secret = fetch_response.get("oauth_token_secret")
        print("Got OAuth token: %s" % resource_owner_key)

        base_authorization_url = "https://api.twitter.com/oauth/authorize"
        authorization_url = oauth.authorization_url(base_authorization_url)
        print("Please go here and authorize: %s" % authorization_url)
        verifier = input("Paste the PIN here: ")

        access_token_url = "https://api.twitter.com/oauth/access_token"
        oauth = OAuth1Session(
            x_api_key,
            client_secret=x_api_secret_key,
            resource_owner_key=resource_owner_key,
            resource_owner_secret=resource_owner_secret,
            verifier=verifier,
        )
        oauth_tokens = oauth.fetch_access_token(access_token_url)

        # Save tokens for future use
        save_tokens(oauth_tokens)

        oauth = OAuth1Session(
            x_api_key,
            client_secret=x_api_secret_key,
            resource_owner_key=oauth_tokens["oauth_token"],
            resource_owner_secret=oauth_tokens["oauth_token_secret"],
        )

    return oauth


def post_tweet(text):
    oauth = get_oauth_session()

    # Set the tweet payload
    payload = {"text": text}

    # Post the tweet
    response = oauth.post(
        "https://api.twitter.com/2/tweets",
        json=payload,
    )

    if response.status_code != 201:
        raise Exception(
            "Request returned an error: {} {}".format(
                response.status_code, response.text
            )
        )

    print("Response code: {}".format(response.status_code))
    json_response = response.json()
    print(json.dumps(json_response, indent=4, sort_keys=True))


if __name__ == "__main__":

    # Command line arguments parsing
    from argparse import ArgumentParser

    # Bot Testing room
    teams_room = (
        "Y2lzY29zcGFyazovL3VzL1JPT00vNjRiNTY1NDAtNjU4NS0xMWVmLTk3ZDMtODFhYjdmM2ZkMGIz"
    )
    the_message = ""
    # fetch random dictionary containing word as key and definition as value
    random_word = return_word()
    random_word_name = random_word["name"]
    word = "\n" + random_word["name"] + "\n"
    word_url = random_word["url"]
    definition = random_word["definition"]
    wiki_link_text = f"Click to Learn about '{random_word_name}'"

    card = [
        {
            "contentType": "application/vnd.microsoft.card.adaptive",
            "content": {
                "type": "AdaptiveCard",
                "version": "1.2",
                "body": [
                    # {
                    #     "type": "Image",
                    #     "url": "https://raw.githubusercontent.com/xanderstevenson/word-of-the-day/main/media/AI-WOTD.png",  # Replace with your image URL
                    #     "horizontalAlignment": "center",
                    #     "width": "100px",
                    #     "height": "auto",
                    # },
                    {
                        "type": "TextBlock",
                        "text": "AI Daily Dose",
                        "size": "ExtraLarge",
                        "horizontalAlignment": "center",
                        "fontType": "Default",
                        "color": "Warning",
                        "weight": "Bold",
                        "wrap": True,
                        "style": "Emphasis",
                    },
                    {
                        "type": "TextBlock",
                        "text": word,
                        "size": "ExtraLarge",
                        "separator": True,
                        "horizontalAlignment": "center",
                        "fontType": "Default",
                        "isSubtle": True,
                        "color": "Good",
                        "weight": "Bolder",
                        "wrap": True,
                    },
                    {
                        "type": "TextBlock",
                        "text": definition,
                        "size": "Medium",
                        "horizontalAlignment": "center",
                        "fontType": "Default",
                        "isSubtle": True,
                        "wrap": True,
                    },
                    {
                        "type": "ActionSet",
                        "horizontalAlignment": "center",
                        "actions": [
                            {
                                "type": "Action.OpenUrl",
                                "url": word_url,
                                "title": wiki_link_text,
                                "style": "positive",
                                "horizontalAlignment": "center",
                            }
                        ],
                    },
                    {
                        "type": "TextBlock",
                        "text": "Reply below and let's discuss this AI concept!",
                        "size": "Small",
                        "horizontalAlignment": "center",
                        "fontType": "Default",
                        "isSubtle": True,
                        "wrap": True,
                        "color": "Warning",
                    },
                ],
            },
        }
    ]

    # Now let's post our message to Webex Teams
    res = send_it(TEAMS_ACCESS_TOKEN, teams_room, the_message)
    if res.status_code == 200:
        print(f"{word} was successfully posted to Webex Teams on {datetime.now()}")

    else:
        print("Failed with status code: %d" % res.status_code)
        if res.status_code == 404:
            print(
                "Please check that the bot is in the room you're attempting to post to..."
            )
        elif res.status_code == 400:
            print(
                "Please check the identifier of the room you're attempting to post to..."
            )
        elif res.status_code == 401:
            print("Please check if the access token is correct...")

    # Sleep for 2 seconds
    time.sleep(3)

    # Format the tweet text to match the Webex message

    random_word_x = random_word_name.lower().replace(" ", "")
    # remove abbreviations for linkedin hashtag
    # remove everything between ()
    random_word_x = re.sub("\(.*?\)", "()", random_word_x)
    # remove (), -.  and /
    random_word_lx = random_word_x.replace("(", "").replace(")", "")
    random_word_x = random_word_x.replace("-", "").replace("/", "")
    random_word_x = random_word_x.replace(".", "")

    tweet_text = (
        "--------------------\n"
        f"AI Word of the Day\n"
        "--------------------\n\n"
        f"{random_word_name}\n\n"
        f"{definition}\n\n"
        f"Learn more: {word_url}\n\n"
        f"#AI #WordOfTheDay #Cisco #DevNet #{random_word_x}"
    )
    # Path to the JSON file
    export_file = "exported_tweet.json"

    # Dictionary to store the tweet_text
    export_data = {
        "tweet_text": tweet_text,
    }

    # Save the tweet_text to the JSON file
    with open(export_file, "w") as f:
        json.dump(export_data, f)

    # Run the second script
    # subprocess.run(
    #     [
    #         "/Users/alexstev/Documents/CiscoDevNet/code/ai-wotd/venv/bin/python3",
    #         "post_to_twitter.py",
    #     ]
    # )
