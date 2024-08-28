import json
import requests
from passwords import organization_id, li_access_token, profile_id


def post_to_linkedin(author_id, li_access_token, message):
    url = "https://api.linkedin.com/v2/ugcPosts"

    headers = {
        "Content-Type": "application/json",
        "X-Restli-Protocol-Version": "2.0.0",
        "Authorization": f"Bearer {li_access_token}",
    }

    post_data = {
        "author": "urn:li:organization:104917889",
        "lifecycleState": "PUBLISHED",
        "specificContent": {
            "com.linkedin.ugc.ShareContent": {
                "shareCommentary": {"text": message},
                "shareMediaCategory": "NONE",
            }
        },
        "visibility": {"com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"},
    }

    response = requests.post(url, headers=headers, json=post_data)
    return response


if __name__ == "__main__":
    # Message to post
    message = "Hello, LinkedIn! This is a test post from my Python script."

    # Set to True for organization, False for profile
    use_organization = False

    if use_organization:
        author_id = f"organization:{organization_id}"
    else:
        author_id = f"person:{profile_id}"

    # Post to LinkedIn
    response = post_to_linkedin(author_id, li_access_token, message)

    # Print the response
    if response.status_code == 201:
        print(f"Post successful! Status code: {response.status_code}")
    else:
        print(f"Post failed. Status code: {response.status_code}")
        print("Response details:", response.json())
