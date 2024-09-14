#!/Users/alexstev/Documents/CiscoDevNet/code/wod/venv/bin/python3
import random
from tinydb import TinyDB, Query

# Database connection
db = TinyDB("db-test.json")
User = Query()


def return_word():
    word_list = [
        {
            "name": "Supervised Learning",
            "definition": "A type of machine learning where a model is trained on labeled data, meaning that each training example is paired with an output label.",
            "url": "https://en.wikipedia.org/wiki/Supervised_learning",
            "id": 0,
        },
        {
            "name": "Unsupervised Learning",
            "definition": "A type of machine learning where a model is trained on unlabeled data and must find patterns and relationships within the data on its own.",
            "url": "https://en.wikipedia.org/wiki/Unsupervised_learning",
            "id": 1,
        },
        {
            "name": "Reinforcement Learning",
            "definition": "A type of machine learning where an agent learns to make decisions by performing actions and receiving rewards or penalties based on those actions.",
            "url": "https://en.wikipedia.org/wiki/Reinforcement_learning",
            "id": 2,
        },
        {
            "name": "Indirect normativity",
            "definition": "Indirect normativity is an approach to the AI alignment problem that attempts to specify AI values indirectly, such as by reference to what a rational agent would value under idealized conditions, rather than via direct specification. It is a method of control by which the motivation of the superintelligence is shaped not directly, but indirectly. The approach specifies a process for the superintelligence to determine beneficial goals rather than specifying them directly.",
            "url": "https://ordinaryideas.wordpress.com/2012/04/21/indirect-normativity-write-up/",
            "id": 60,
        },
    ]

    # Select a candidate word from word list
    candidate_word = random.choice(word_list)

    # Query db for current used word list
    used_list_search = db.search(User.used.exists())
    if used_list_search:
        used_list = used_list_search[0]["used"]
    else:
        used_list = []

    # If length of word list and length of used list is equal, erase used list and start over
    if len(used_list) == len(word_list):
        used_list = []

    # If random choice is already in used list, choose again
    while candidate_word["id"] in used_list:
        candidate_word = random.choice(word_list)

    # Solidify choice
    word = candidate_word

    # Add new choice to used list and to db
    used_list.append(word["id"])
    db.upsert({"used": used_list}, User.used.exists())

    return word
