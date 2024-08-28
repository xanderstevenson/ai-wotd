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
            "name": "Neural Network",
            "definition": "A series of algorithms that attempt to recognize underlying relationships in a set of data through a process that mimics the way the human brain operates.",
            "url": "https://en.wikipedia.org/wiki/Artificial_neural_network",
            "id": 3,
        },
        {
            "name": "Deep Learning",
            "definition": "A subset of machine learning that uses algorithms based on neural networks with many layers (deep networks) to model complex patterns in data.",
            "url": "https://en.wikipedia.org/wiki/Deep_learning",
            "id": 4,
        },
        {
            "name": "Natural Language Processing",
            "definition": "A field of artificial intelligence that focuses on the interaction between computers and humans through natural language.",
            "url": "https://en.wikipedia.org/wiki/Natural_language_processing",
            "id": 5,
        },
        {
            "name": "Feature Engineering",
            "definition": "The process of using domain knowledge to extract features (attributes, variables) from raw data to improve the performance of machine learning models.",
            "url": "https://en.wikipedia.org/wiki/Feature_engineering",
            "id": 6,
        },
        {
            "name": "Overfitting",
            "definition": "A modeling error that occurs when a machine learning model learns the training data too well, capturing noise and details that do not generalize to new data.",
            "url": "https://en.wikipedia.org/wiki/Overfitting",
            "id": 7,
        },
        {
            "name": "Underfitting",
            "definition": "A modeling error that occurs when a machine learning model is too simple to capture the underlying patterns in the data.",
            "url": "https://en.wikipedia.org/wiki/Underfitting",
            "id": 8,
        },
        {
            "name": "Cross-Validation",
            "definition": "A technique for assessing how the results of a statistical analysis will generalize to an independent data set, used to prevent overfitting.",
            "url": "https://en.wikipedia.org/wiki/Cross-validation_(statistics)",
            "id": 9,
        },
        {
            "name": "Gradient Descent",
            "definition": "An optimization algorithm used to minimize the loss function in machine learning models by iteratively moving towards the steepest descent of the loss function.",
            "url": "https://en.wikipedia.org/wiki/Gradient_descent",
            "id": 10,
        },
        {
            "name": "Support Vector Machine",
            "definition": "A supervised learning model that finds the hyperplane that best separates different classes in a dataset.",
            "url": "https://en.wikipedia.org/wiki/Support_vector_machine",
            "id": 11,
        },
        {
            "name": "Decision Tree",
            "definition": "A supervised learning model that makes decisions based on a series of binary choices, often visualized as a tree-like diagram.",
            "url": "https://en.wikipedia.org/wiki/Decision_tree_learning",
            "id": 12,
        },
        {
            "name": "Random Forest",
            "definition": "An ensemble learning method that combines multiple decision trees to improve classification or regression accuracy.",
            "url": "https://en.wikipedia.org/wiki/Random_forest",
            "id": 13,
        },
        {
            "name": "K-Nearest Neighbors",
            "definition": "A simple algorithm that classifies a data point based on how its neighbors are classified, typically used for classification and regression.",
            "url": "https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm",
            "id": 14,
        },
        {
            "name": "Dimensionality Reduction",
            "definition": "The process of reducing the number of features (dimensions) in a dataset while preserving its important characteristics.",
            "url": "https://en.wikipedia.org/wiki/Dimensionality_reduction",
            "id": 15,
        },
        {
            "name": "Clustering",
            "definition": "A type of unsupervised learning that groups similar data points into clusters based on their features.",
            "url": "https://en.wikipedia.org/wiki/Cluster_analysis",
            "id": 16,
        },
        {
            "name": "Principal Component Analysis",
            "definition": "A dimensionality reduction technique that transforms data into a set of orthogonal (uncorrelated) components, capturing the most variance in the data.",
            "url": "https://en.wikipedia.org/wiki/Principal_component_analysis",
            "id": 17,
        },
        {
            "name": "Hyperparameter Tuning",
            "definition": "The process of finding the best set of hyperparameters for a machine learning model to optimize its performance.",
            "url": "https://en.wikipedia.org/wiki/Hyperparameter_optimization",
            "id": 18,
        },
        {
            "name": "Ensemble Learning",
            "definition": "A technique that combines predictions from multiple models to improve accuracy and robustness.",
            "url": "https://en.wikipedia.org/wiki/Ensemble_learning",
            "id": 19,
        },
        {
            "name": "Bagging",
            "definition": "A machine learning ensemble method that improves the stability and accuracy of algorithms by training multiple models on different subsets of the data and combining their predictions.",
            "url": "https://en.wikipedia.org/wiki/Bootstrap_aggregating",
            "id": 20,
        },
        {
            "name": "Boosting",
            "definition": "An ensemble learning technique that combines multiple weak learners to create a strong learner by iteratively correcting the errors of the weak models.",
            "url": "https://en.wikipedia.org/wiki/Boosting_(machine_learning)",
            "id": 21,
        },
        {
            "name": "AdaBoost",
            "definition": "A specific type of boosting algorithm that adjusts the weights of incorrectly classified instances to improve the performance of the model.",
            "url": "https://en.wikipedia.org/wiki/AdaBoost",
            "id": 22,
        },
        {
            "name": "Gradient Boosting",
            "definition": "An ensemble technique that builds models sequentially, each new model trying to correct errors made by the previous ones, using gradient descent to minimize errors.",
            "url": "https://en.wikipedia.org/wiki/Gradient_boosting",
            "id": 23,
        },
        {
            "name": "Convolutional Neural Network",
            "definition": "A deep learning algorithm specifically designed for processing structured grid data like images, using convolutional layers to automatically and adaptively learn spatial hierarchies of features.",
            "url": "https://en.wikipedia.org/wiki/Convolutional_neural_network",
            "id": 24,
        },
        {
            "name": "Recurrent Neural Network",
            "definition": "A type of neural network where connections between nodes form directed cycles, allowing the network to maintain a memory of previous inputs, useful for sequence prediction tasks.",
            "url": "https://en.wikipedia.org/wiki/Recurrent_neural_network",
            "id": 25,
        },
        {
            "name": "Long Short-Term Memory",
            "definition": "A special kind of recurrent neural network architecture designed to better retain long-term dependencies by using gates to control the flow of information.",
            "url": "https://en.wikipedia.org/wiki/Long_short-term_memory",
            "id": 26,
        },
        {
            "name": "Autoencoder",
            "definition": "A type of neural network used to learn efficient codings of input data, typically for dimensionality reduction or feature learning.",
            "url": "https://en.wikipedia.org/wiki/Autoencoder",
            "id": 27,
        },
        {
            "name": "Transfer Learning",
            "definition": "A technique in machine learning where a model developed for a particular task is reused as the starting point for a model on a different but related task.",
            "url": "https://en.wikipedia.org/wiki/Transfer_learning",
            "id": 28,
        },
        {
            "name": "Bayesian Network",
            "definition": "A graphical model that represents the probabilistic relationships among a set of variables using directed acyclic graphs.",
            "url": "https://en.wikipedia.org/wiki/Bayesian_network",
            "id": 29,
        },
        {
            "name": "Markov Chain",
            "definition": "A stochastic model that describes a sequence of possible events where the probability of each event depends only on the state attained in the previous event.",
            "url": "https://en.wikipedia.org/wiki/Markov_chain",
            "id": 30,
        },
        {
            "name": "Hidden Markov Model",
            "definition": "A statistical model where the system being modeled is assumed to be a Markov process with unobserved (hidden) states, commonly used in temporal pattern recognition.",
            "url": "https://en.wikipedia.org/wiki/Hidden_Markov_model",
            "id": 31,
        },
        {
            "name": "Q-Learning",
            "definition": "A model-free reinforcement learning algorithm that seeks to learn the value of an action in a particular state in order to maximize the total reward.",
            "url": "https://en.wikipedia.org/wiki/Q-learning",
            "id": 32,
        },
        {
            "name": "Generative Adversarial Network",
            "definition": "A class of machine learning frameworks where two neural networks, a generator and a discriminator, compete against each other to generate realistic data.",
            "url": "https://en.wikipedia.org/wiki/Generative_adversarial_network",
            "id": 33,
        },
        {
            "name": "Hyperparameter",
            "definition": "A parameter whose value is set before the learning process begins and governs the training process, such as learning rate or number of epochs.",
            "url": "https://en.wikipedia.org/wiki/Hyperparameter_optimization",
            "id": 34,
        },
        {
            "name": "Epoch",
            "definition": "One complete pass through the entire training dataset during the training process of a machine learning model.",
            "url": "https://en.wikipedia.org/wiki/Epoch_(machine_learning)",
            "id": 35,
        },
        {
            "name": "Activation Function",
            "definition": "A mathematical function applied to a neural network's node to introduce non-linearity, helping the network learn complex patterns.",
            "url": "https://en.wikipedia.org/wiki/Activation_function",
            "id": 36,
        },
        {
            "name": "Softmax",
            "definition": "An activation function often used in the output layer of a neural network to produce a probability distribution over multiple classes.",
            "url": "https://en.wikipedia.org/wiki/Softmax_function",
            "id": 37,
        },
        {
            "name": "Loss Function",
            "definition": "A function that measures the difference between the predicted output and the actual output during training, guiding the optimization process.",
            "url": "https://en.wikipedia.org/wiki/Loss_function",
            "id": 38,
        },
        {
            "name": "Stochastic Gradient Descent",
            "definition": "An optimization method that reduces the loss by updating the model's parameters using only a small, random subset of the training data.",
            "url": "https://en.wikipedia.org/wiki/Stochastic_gradient_descent",
            "id": 39,
        },
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
