import json
from requests_oauthlib import OAuth1Session
import os
from passwords import x_api_key, x_api_secret_key


# Path to the JSON file
import_file = "exported_tweet.json"

# Load the tweet_text
with open(import_file, "r") as f:
    data = json.load(f)

tweet_text = data["tweet_text"]

# Replace with your API keys
consumer_key = x_api_key
consumer_secret = x_api_secret_key

# Path to save tokens
token_file = "tokens.json"


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
            consumer_key,
            client_secret=consumer_secret,
            resource_owner_key=tokens["oauth_token"],
            resource_owner_secret=tokens["oauth_token_secret"],
        )
    else:
        # Run the authorization process
        request_token_url = "https://api.twitter.com/oauth/request_token?oauth_callback=oob&x_auth_access_type=write"
        oauth = OAuth1Session(consumer_key, client_secret=consumer_secret)
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
            consumer_key,
            client_secret=consumer_secret,
            resource_owner_key=resource_owner_key,
            resource_owner_secret=resource_owner_secret,
            verifier=verifier,
        )
        oauth_tokens = oauth.fetch_access_token(access_token_url)

        # Save tokens for future use
        save_tokens(oauth_tokens)

        oauth = OAuth1Session(
            consumer_key,
            client_secret=consumer_secret,
            resource_owner_key=oauth_tokens["oauth_token"],
            resource_owner_secret=oauth_tokens["oauth_token_secret"],
        )

    return oauth


def post_tweet():
    oauth = get_oauth_session()

    # Set the tweet payload
    payload = {"text": tweet_text}

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
    post_tweet()
