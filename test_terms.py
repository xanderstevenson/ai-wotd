#!/Users/alexstev/Documents/CiscoDevNet/code/wod/venv/bin/python3
import random

# database connection.
from tinydb import TinyDB, Query

db = TinyDB("db.json")
User = Query()

##                        ATTENTION DEVELOPERS!!!

##     Hi, please add new definitions to the end of the list.
##     A Wikipedia entry is the default url unless it is lacking, absent or clearly outdone.

##     Use SINGLE QUOTES inside the "definition" if quotes are needed. Double quotes nested inside double quotes will throw an error when run.


def return_word():

    word_list = [
        # {
        #     "name": "",
        #     "definition": "",
        #     "url": "",
        #     "id": 1
        # },
        # {
        #     "name": "",
        #     "definition": "",
        #     "url": "",
        #     "id": 2
        # },
        # {
        #     "name": "",
        #     "definition": "",
        #     "url": "",
        #     "id": 3
        # },
        # {
        #     "name": "",
        #     "definition": "",
        #     "url": "",
        #     "id": 4
        # },
        # {
        #     "name": "",
        #     "definition": "",
        #     "url": "",
        #     "id": 5
        # },
        # {
        #     "name": "",
        #     "definition": "",
        #     "url": "",
        #     "id": 6
        # },
        # {
        #     "name": "",
        #     "definition": "",
        #     "url": "",
        #     "id": 7
        # },
        # {
        #     "name": "",
        #     "definition": "",
        #     "url": "",
        #     "id": 8
        # },
        # {
        #     "name": "",
        #     "definition": "",
        #     "url": "",
        #     "id": 9
        # },
        # {
        #     "name": "",
        #     "definition": "",
        #     "url": "",
        #     "id": 0
        # },
        #     "name": "",
        #     "definition": "",
        #     "url": "",
        #     "id": 1
        # },
        # {
        #     "name": "",
        #     "definition": "",
        #     "url": "",
        #     "id": 2
        # },
        # {
        #     "name": "",
        #     "definition": "",
        #     "url": "",
        #     "id": 3
        # },
        # {
        #     "name": "",
        #     "definition": "",
        #     "url": "",
        #     "id": 4
        # },
        # {
        #     "name": "",
        #     "definition": "",
        #     "url": "",
        #     "id": 5
        # },
        # {
        #     "name": "",
        #     "definition": "",
        #     "url": "",
        #     "id": 6
        # },
        # {
        #     "name": "",
        #     "definition": "",
        #     "url": "",
        #     "id": 7
        # },
        # {
        #     "name": "",
        #     "definition": "",
        #     "url": "",
        #     "id": 8
        # },
        # {
        #     "name": "",
        #     "definition": "",
        #     "url": "",
        #     "id": 9
        # },
        # {
        #     "name": "",
        #     "definition": "",
        #     "url": "",
        #     "id": 0
        # },
        # {
        #     "name": "",
        #     "definition": "",
        #     "url": "",
        #     "id": 1
        # },
        # {
        #     "name": "",
        #     "definition": "",
        #     "url": "",
        #     "id": 2
        # },
        # {
        #     "name": "",
        #     "definition": "",
        #     "url": "",
        #     "id": 3
        # },
        # {
        #     "name": "",
        #     "definition": "",
        #     "url": "",
        #     "id": 4
        # },
        # {
        #     "name": "",
        #     "definition": "",
        #     "url": "",
        #     "id": 5
        # },
        # {
        #     "name": "",
        #     "definition": "",
        #     "url": "",
        #     "id": 6
        # },
        # {
        #     "name": "",
        #     "definition": "",
        #     "url": "",
        #     "id": 7
        # },
        # {
        #     "name": "",
        #     "definition": "",
        #     "url": "",
        #     "id": 8
        # },
        # {
        #     "name": "",
        #     "definition": "",
        #     "url": "",
        #     "id": 9
        # },
        # {
        #     "name": "",
        #     "definition": "",
        #     "url": "",
        #     "id": 0
        # },
        #     "name": "",
        #     "definition": "",
        #     "url": "",
        #     "id": 1
        # },
        # {
        #     "name": "",
        #     "definition": "",
        #     "url": "",
        #     "id": 2
        # },
        # {
        #     "name": "",
        #     "definition": "",
        #     "url": "",
        #     "id": 3
        # },
        # {
        #     "name": "",
        #     "definition": "",
        #     "url": "",
        #     "id": 4
        # },
        # {
        #     "name": "",
        #     "definition": "",
        #     "url": "",
        #     "id": 5
        # },
        # {
        #     "name": "",
        #     "definition": "",
        #     "url": "",
        #     "id": 6
        # },
        # {
        #     "name": "",
        #     "definition": "",
        #     "url": "",
        #     "id": 7
        # },
        # {
        #     "name": "",
        #     "definition": "",
        #     "url": "",
        #     "id": 8
        # },
        # {
        #     "name": "",
        #     "definition": "",
        #     "url": "",
        #     "id": 9
        # },
        # {
        #     "name": "",
        #     "definition": "",
        #     "url": "",
        #     "id": 0
        # },
        # {
        #     "name": "",
        #     "definition": "",
        #     "url": "",
        #     "id": 1
        # },
        # {
        #     "name": "",
        #     "definition": "",
        #     "url": "",
        #     "id": 2
        # },
        # {
        #     "name": "",
        #     "definition": "",
        #     "url": "",
        #     "id": 3
        # },
        # {
        #     "name": "",
        #     "definition": "",
        #     "url": "",
        #     "id": 4
        # },
        # {
        #     "name": "",
        #     "definition": "",
        #     "url": "",
        #     "id": 5
        # },
        # {
        #     "name": "",
        #     "definition": "",
        #     "url": "",
        #     "id": 6
        # },
        # {
        #     "name": "",
        #     "definition": "",
        #     "url": "",
        #     "id": 7
        # },
        # {
        #     "name": "",
        #     "definition": "",
        #     "url": "",
        #     "id": 8
        # },
        # {
        #     "name": "",
        #     "definition": "",
        #     "url": "",
        #     "id": 9
        # },
        # {
        #     "name": "",
        #     "definition": "",
        #     "url": "",
        #     "id": 0
        # },
        #     "name": "",
        #     "definition": "",
        #     "url": "",
        #     "id": 1
        # },
        # {
        #     "name": "",
        #     "definition": "",
        #     "url": "",
        #     "id": 2
        # },
        # {
        #     "name": "",
        #     "definition": "",
        #     "url": "",
        #     "id": 3
        # },
        # {
        #     "name": "",
        #     "definition": "",
        #     "url": "",
        #     "id": 4
        # },
        # {
        #     "name": "",
        #     "definition": "",
        #     "url": "",
        #     "id": 5
        # },
        # {
        #     "name": "",
        #     "definition": "",
        #     "url": "",
        #     "id": 6
        # },
        # {
        #     "name": "",
        #     "definition": "",
        #     "url": "",
        #     "id": 7
        # },
        # {
        #     "name": "",
        #     "definition": "",
        #     "url": "",
        #     "id": 8
        # },
        # {
        #     "name": "",
        #     "definition": "",
        #     "url": "",
        #     "id": 9
        # },
        # {
        #     "name": "",
        #     "definition": "",
        #     "url": "",
        #     "id": 0
        # },
        # {
        #     "name": "",
        #     "definition": "",
        #     "url": "",
        #     "id": 1
        # },
        # {
        #     "name": "",
        #     "definition": "",
        #     "url": "",
        #     "id": 2
        # },
        # {
        #     "name": "",
        #     "definition": "",
        #     "url": "",
        #     "id": 3
        # },
        # {
        #     "name": "",
        #     "definition": "",
        #     "url": "",
        #     "id": 4
        # },
        # {
        #     "name": "",
        #     "definition": "",
        #     "url": "",
        #     "id": 5
        # },
        # {
        #     "name": "",
        #     "definition": "",
        #     "url": "",
        #     "id": 6
        # },
        # {
        #     "name": "",
        #     "definition": "",
        #     "url": "",
        #     "id": 7
        # },
        # {
        #     "name": "",
        #     "definition": "",
        #     "url": "",
        #     "id": 8
        # },
        # {
        #     "name": "",
        #     "definition": "",
        #     "url": "",
        #     "id": 9
        # },
        # {
        #     "name": "",
        #     "definition": "",
        #     "url": "",
        #     "id": 0
        # },
        #     "name": "",
        #     "definition": "",
        #     "url": "",
        #     "id": 1
        # },
        # {
        #     "name": "",
        #     "definition": "",
        #     "url": "",
        #     "id": 2
        # },
        # {
        #     "name": "",
        #     "definition": "",
        #     "url": "",
        #     "id": 3
        # },
        # {
        #     "name": "",
        #     "definition": "",
        #     "url": "",
        #     "id": 4
        # },
        # {
        #     "name": "",
        #     "definition": "",
        #     "url": "",
        #     "id": 5
        # },
        # {
        #     "name": "",
        #     "definition": "",
        #     "url": "",
        #     "id": 6
        # },
        # {
        #     "name": "",
        #     "definition": "",
        #     "url": "",
        #     "id": 7
        # },
        # {
        #     "name": "",
        #     "definition": "",
        #     "url": "",
        #     "id": 8
        # },
        # {
        #     "name": "",
        #     "definition": "",
        #     "url": "",
        #     "id": 9
        # },
        # {
        #     "name": "",
        #     "definition": "",
        #     "url": "",
        #     "id": 0
        # },
        # {
        #     "name": "",
        #     "definition": "",
        #     "url": "",
        #     "id": 1
        # },
        # {
        #     "name": "",
        #     "definition": "",
        #     "url": "",
        #     "id": 2
        # },
        # {
        #     "name": "",
        #     "definition": "",
        #     "url": "",
        #     "id": 3
        # },
        # {
        #     "name": "",
        #     "definition": "",
        #     "url": "",
        #     "id": 4
        # },
        # {
        #     "name": "",
        #     "definition": "",
        #     "url": "",
        #     "id": 5
        # },
        # {
        #     "name": "",
        #     "definition": "",
        #     "url": "",
        #     "id": 6
        # },
        # {
        #     "name": "",
        #     "definition": "",
        #     "url": "",
        #     "id": 7
        # },
        # {
        #     "name": "",
        #     "definition": "",
        #     "url": "",
        #     "id": 8
        # },
        # {
        #     "name": "",
        #     "definition": "",
        #     "url": "",
        #     "id": 9
        # },
        # {
        #     "name": "",
        #     "definition": "",
        #     "url": "",
        #     "id": 0
        # },
        #     "name": "",
        #     "definition": "",
        #     "url": "",
        #     "id": 1
        # },
        # {
        #     "name": "",
        #     "definition": "",
        #     "url": "",
        #     "id": 2
        # },
        # {
        #     "name": "",
        #     "definition": "",
        #     "url": "",
        #     "id": 3
        # },
        # {
        #     "name": "",
        #     "definition": "",
        #     "url": "",
        #     "id": 4
        # },
        # {
        #     "name": "",
        #     "definition": "",
        #     "url": "",
        #     "id": 5
        # },
        # {
        #     "name": "",
        #     "definition": "",
        #     "url": "",
        #     "id": 6
        # },
        # {
        #     "name": "",
        #     "definition": "",
        #     "url": "",
        #     "id": 7
        # },
        # {
        #     "name": "",
        #     "definition": "",
        #     "url": "",
        #     "id": 8
        # },
        # {
        #     "name": "",
        #     "definition": "",
        #     "url": "",
        #     "id": 9
        # },
        # {
        #     "name": "",
        #     "definition": "",
        #     "url": "",
        #     "id": 0
        # },
    ]

    # Note -- if you want to run this app in GitHub only, remove all lines regarding variables 'used_list' and 'items'

    # select a candidate word from word list
    candidate_word = random.choice(word_list)
    # query db for current used word list
    # used_list_search = db.search(User.used.exists())
    # used_list = used_list_search[0]["used"]
    # # if length of word list and length of used list is equal, erase used list and start over
    # if len(used_list) == len(word_list):
    #     used_list = []
    # # if random choice is already in used list, choose again
    # while candidate_word["id"] in used_list:
    #     candidate_word = random.choice(word_list)
    # solidify choice
    word = candidate_word
    # add new choice to used list and to db
    # remove these three lines to test without writing to db
    # used_list.append(word["id"])
    # items = {'used':used_list}
    # db.update(items)

    return word
