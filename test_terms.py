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
            "name": "Natural Language Processing (NLP)",
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
            "name": "Principal Component Analysis (PCA)",
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
            "name": "Convolutional Neural Network (CNN)",
            "definition": "A deep learning algorithm specifically designed for processing structured grid data like images, using convolutional layers to automatically and adaptively learn spatial hierarchies of features.",
            "url": "https://en.wikipedia.org/wiki/Convolutional_neural_network",
            "id": 24,
        },
        {
            "name": "Recurrent Neural Network (RNN)",
            "definition": "A type of neural network where connections between nodes form directed cycles, allowing the network to maintain a memory of previous inputs, useful for sequence prediction tasks.",
            "url": "https://en.wikipedia.org/wiki/Recurrent_neural_network",
            "id": 25,
        },
        {
            "name": "Long Short-Term Memory (LSTM)",
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
            "name": "Generative Adversarial Network (GAN)",
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
        {
            "name": "Feature engineering",
            "definition": "Feature engineering is a preprocessing step in supervised machine learning and statistical modeling which transforms raw data into a more effective set of inputs. Each input comprises several attributes, known as features. By providing models with relevant information, feature engineering significantly enhances their predictive accuracy and decision-making capability.",
            "url": "https://en.wikipedia.org/wiki/Feature_engineering",
            "id": 40,
        },
        {
            "name": "Hyperparameter optimization",
            "definition": "In machine learning, hyperparameter optimization or tuning is the problem of choosing a set of optimal hyperparameters for a learning algorithm. A hyperparameter is a parameter whose value is used to control the learning process.\n\nHyperparameter optimization determines the set of hyperparameters that yields an optimal model which minimizes a predefined loss function on a given data set. The objective function takes a set of hyperparameters and returns the associated loss. Cross-validation is often used to estimate this generalization performance, and therefore choose the set of values for hyperparameters that maximize it.",
            "url": "https://en.wikipedia.org/wiki/Hyperparameter_optimization",
            "id": 41,
        },
        {
            "name": "Feature extraction vs. Feature selection",
            "definition": "Feature extraction transforms the original features in a dataset into a new set of features, often reducing dimensionality and capturing the most important information. This process can involve techniques like Principal Component Analysis (PCA), which combines features into fewer, more informative ones.\n\nFeature selection involves choosing a subset of the most relevant features from the original dataset without altering them. The goal is to reduce the feature space by eliminating irrelevant or redundant features, which can enhance model performance and reduce overfitting.\n\nIn essence, feature extraction creates new features by transforming data, while feature selection narrows down the features to the most important ones, both aiming to improve the efficiency and accuracy of machine learning models.",
            "url": "https://www.geeksforgeeks.org/difference-between-feature-selection-and-feature-extraction/",
            "id": 42,
        },
        {
            "name": "Transformer",
            "definition": "A transformer is a deep learning architecture developed by researchers at Google and based on the multi-head attention mechanism, proposed in a 2017 paper ‘Attention Is All You Need’. Text is converted to numerical representations called tokens, and each token is converted into a vector via looking up from a word embedding table. At each layer, each token is then contextualized within the scope of the context window with other (unmasked) tokens via a parallel multi-head attention mechanism allowing the signal for key tokens to be amplified and less important tokens to be diminished.\n\nTransformers have the advantage of having no recurrent units, and therefore require less training time than earlier recurrent neural architectures (RNNs) such as long short-term memory (LSTM). Later variations have been widely adopted for training large language models (LLM) on large (language) datasets, such as the Wikipedia corpus and Common Crawl.\n\nTransformers were first developed as an improvement over previous architectures for machine translation, but have found many applications since then. They are used in large-scale natural language processing, computer vision (vision transformers), reinforcement learning, audio, multi-modal processing, robotics, and even playing chess. It has also led to the development of pre-trained systems, such as generative pre-trained transformers (GPTs) and BERT (Bidirectional Encoder Representations from Transformers).",
            "url": "https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)",
            "id": 43,
        },
        {
            "name": "Feedforward neural network (FNN)",
            "definition": "A feedforward neural network (FNN) is one of the two broad types of artificial neural network, characterized by direction of the flow of information between its layers. Its flow is uni-directional, meaning that the information in the model flows in only one direction—forward—from the input nodes, through the hidden nodes (if any) and to the output nodes, without any cycles or loops, in contrast to recurrent neural networks, which have a bi-directional flow. Modern feedforward networks are trained using the backpropagation method and are colloquially referred to as the ‘vanilla’ neural networks.",
            "url": "https://en.wikipedia.org/wiki/Feedforward_neural_network",
            "id": 44,
        },
        {
            "name": "Named-entity recognition (NER)",
            "definition": "Named-entity recognition (NER) (also known as (named) entity identification, entity chunking, and entity extraction) is a subtask of information extraction that seeks to locate and classify named entities mentioned in unstructured text into pre-defined categories such as person names, organizations, locations, medical codes, time expressions, quantities, monetary values, percentages, etc.",
            "url": "https://en.wikipedia.org/wiki/Named-entity_recognition",
            "id": 45,
        },
        {
            "name": "Machine learning (ML)",
            "definition": "Machine learning (ML) is the study of computer algorithms that can improve automatically through experience and by the use of data. It is seen as a part of artificial intelligence. Machine learning algorithms build a model based on sample data, known as training data, in order to make predictions or decisions without being explicitly programmed to do so. Machine learning algorithms are used in a wide variety of applications, such as in medicine, email filtering, speech recognition, and computer vision, where it is difficult or unfeasible to develop conventional algorithms to perform the needed tasks.\n\nA subset of machine learning is closely related to computational statistics, which focuses on making predictions using computers; but not all machine learning is statistical learning. The study of mathematical optimization delivers methods, theory and application domains to the field of machine learning. \n\nData mining is a related field of study, focusing on exploratory data analysis through unsupervised learning. Some implementations of machine learning use data and neural networks in a way that mimics the working of a biological brain. In its application across business problems, machine learning is also referred to as predictive analytics.",
            "url": "https://en.wikipedia.org/wiki/Machine_learning",
            "id": 46,
        },
        {
            "name": "Artificial empathy (AE)",
            "definition": "Artificial empathy (AE) or computational empathy is the development of AI systems—such as companion robots or virtual agents—that can detect emotions and respond to them in an empathic way.\n\nAlthough such technology can be perceived as scary or threatening, it could also have a significant advantage over humans for roles in which emotional expression can be important, such as in the health care sector. For example, care-givers who perform emotional labor above and beyond the requirements of paid labor can experience chronic stress or burnout, and can become desensitized to patients.\n\nA broader definition of artificial empathy is ‘the ability of nonhuman models to predict a person's internal state (e.g., cognitive, affective, physical) given the signals (s)he emits (e.g., facial expression, voice, gesture) or to predict a person's reaction (including, but not limited to internal states) when he or she is exposed to a given set of stimuli (e.g., facial expression, voice, gesture, graphics, music, etc.)’.",
            "url": "https://en.wikipedia.org/wiki/Artificial_empathy",
            "id": 47,
        },
        {
            "name": "Onboard intelligence",
            "definition": "The capability of a device to solve a problem itself without passing the request to another computer in the network. Onboard intelligence also refers to artificial intelligence that's built into a device design, rather than outsourced to remote technology.",
            "url": "https://www.techopedia.com/definition/33391/onboard-intelligence",
            "id": 48,
        },
        {
            "name": "AI winter",
            "definition": "In the history of artificial intelligence, an AI winter is a period of reduced funding and interest in artificial intelligence research. The term was coined by analogy to the idea of a nuclear winter. The field has experienced several hype cycles, followed by disappointment and criticism, followed by funding cuts, followed by renewed interest years or decades later.\n\nEnthusiasm and optimism about AI has generally increased since its low point in the early 1990s. Beginning about 2012, interest in artificial intelligence (and especially the sub-field of machine learning) from the research and corporate communities led to a dramatic increase in funding and investment.",
            "url": "https://en.wikipedia.org/wiki/AI_winter",
            "id": 49,
        },
        {
            "name": "Expert system",
            "definition": "In artificial intelligence, an expert system is a computer system emulating the decision-making ability of a human expert. Expert systems are designed to solve complex problems by reasoning through bodies of knowledge, represented mainly as if–then rules rather than through conventional procedural code. The first expert systems were created in the 1970s and then proliferated in the 1980s. Expert systems were among the first truly successful forms of artificial intelligence (AI) software. \n\nAn expert system is divided into two subsystems: the inference engine and the knowledge base. The knowledge base represents facts and rules. The inference engine applies the rules to the known facts to deduce new facts. Inference engines can also include explanation and debugging abilities.",
            "url": "https://en.wikipedia.org/wiki/Expert_system",
            "id": 50,
        },
        {
            "name": "John McCarthy",
            "definition": "John McCarthy (September 4, 1927 – October 24, 2011) was an American computer scientist and cognitive scientist. He was one of the founders of the discipline of artificial intelligence. He co-authored the document that coined the term ‘artificial intelligence’ (AI), developed the programming language family Lisp, significantly influenced the design of the language ALGOL, popularized time-sharing, and invented garbage collection.\n\nSeveral Key Theme’s in McCarthy’s Work\n\n- The idea that the world can be represented symbolically for a machine and reasoned about by that machine.\n- The division of the AI problem into two parts: epistemological (representation) and heuristic (search).\n- The importance of formal (mathematical) representation, particularly the use of logic for representation.\n- The use of reification—the conceptual turning of abstract structures (such as situations) into concrete instances that can be reasoned about. The situation calculus, where the state of the universe is compressed into a single constant, is an example of such reification.\n- The development of abstract and general mechanisms (proof checkers, logic, circumscription, common sense) in contrast with domain-specific tools and languages.\n- The relatively complete analysis of artificial domains (programming languages, puzzles, chess) instead of a (necessarily) incomplete analysis of natural situations.\n- The importance of environments for interactive exploration. This is illustrated not so much by an explicit campaign position of McCarthy's as by the systems whose creation he led, such as the AI lab, Lisp, and timesharing.\n\nAwards and honors\n\n- Turing Award from the Association for Computing Machinery (1971)\n- Kyoto Prize (1988)\n- National Medal of Science (USA) in Mathematical, Statistical, and Computational Sciences (1990)\n- Inducted as a Fellow of the Computer History Museum ‘for his co-founding of the fields of Artificial Intelligence (AI) and timesharing systems, and for major contributions to mathematics and computer science’. (1999)\n- Benjamin Franklin Medal in Computer and Cognitive Science from the Franklin Institute (2003)\n- Inducted into IEEE Intelligent Systems' AI's Hall of Fame (2011), for the ‘significant contributions to the field of AI and intelligent systems’.",
            "url": "https://en.wikipedia.org/wiki/John_McCarthy_(computer_scientist)",
            "id": 51,
        },
        {
            "name": "Artificial general intelligence (AGI)",
            "definition": "Artificial general intelligence (AGI) is the ability of an intelligent agent to understand or learn any intellectual task that a human being can. It is a primary goal of some artificial intelligence research and a common topic in science fiction and futures studies. AGI can also be referred to as strong AI, full AI, or general intelligent action, although some academic sources reserve the term ‘strong AI’ for computer programs that experience sentience or consciousness.\n\nIn contrast to strong AI, weak AI or ‘narrow AI’ is not intended to have general cognitive abilities; rather, weak AI is any program that is designed to solve exactly one problem. (Academic sources reserve ‘weak AI’ for programs that do not experience consciousness or do not have a mind in the same sense people do.) A 2020 survey identified 72 active AGI R&D projects spread across 37 countries.",
            "url": "https://en.wikipedia.org/wiki/Artificial_general_intelligence",
            "id": 52,
        },
        {
            "name": "Superintelligence",
            "definition": "A superintelligence is a hypothetical agent that possesses intelligence far surpassing that of the brightest and most gifted human minds. ‘Superintelligence’ may also refer to a property of problem-solving systems (e.g., superintelligent language translators or engineering assistants) whether or not these high-level intellectual competencies are embodied in agents that act in the world. A superintelligence may or may not be created by an intelligence explosion and associated with a technological singularity. University of Oxford philosopher Nick Bostrom defines superintelligence as ‘any intellect that greatly exceeds the cognitive performance of humans in virtually all domains of interest’. \n\nTechnological researchers disagree about how likely present-day human intelligence is to be surpassed. Some argue that advances in artificial intelligence (AI) will probably result in general reasoning systems that lack human cognitive limitations. Others believe that humans will evolve or directly modify their biology so as to achieve radically greater intelligence. A number of futures studies scenarios combine elements from both of these possibilities, suggesting that humans are likely to interface with computers, or upload their minds to computers, in a way that enables substantial intelligence amplification.\n\nSome researchers believe that superintelligence will likely follow shortly after the development of artificial general intelligence. The first generally intelligent machines are likely to immediately hold an enormous advantage in at least some forms of mental capability, including the capacity of perfect recall, a vastly superior knowledge base, and the ability to multitask in ways not possible to biological entities. This may give them the opportunity to—either as a single being or as a new species—become much more powerful than humans, and to displace them.\n\nA number of scientists and forecasters argue for prioritizing early research into the possible benefits and risks of human and machine cognitive enhancement, because of the potential social impact of such technologies in both raising the level of collective human intelligence and in intelligently ushering the age of super intelligence as a whole.",
            "url": "https://en.wikipedia.org/wiki/Superintelligence",
            "id": 53,
        },
        {
            "name": "Human Level Machine Intelligence (HLMI)",
            "definition": "Human-Level Machine Intelligence (HLMI), also known as Human-Level Artificial intelligence, refers to computer systems that can operate with the intelligence of an average human being. These programs can complete tasks or make decisions as successfully as the average human can.",
            "url": "https://arxiv.org/abs/2108.03793",
            "id": 54,
        },
        {
            "name": "Seed AI",
            "definition": "Seed AI is a hypothesized type of strong artificial intelligence capable of recursive self-improvement. Having improved itself it would become better at improving itself, potentially leading to an exponential increase in intelligence. No such AI currently exists, but researchers are working to make seed AI a reality.\n\nModern compilers optimize for efficiency and raw speed, but this is not sufficient for the sort of open-ended recursive self-enhancement needed to create superintelligence, as a true seed AI would need to be able to do. Existing optimizers can transform code into a functionally equivalent, more efficient form, but cannot identify the intent of an algorithm and rewrite it for more effective results. The optimized version of a given compiler may compile faster, but it cannot compile better. That is, an optimized version of a compiler will never spot new optimization tricks that earlier versions failed to see or innovate new ways of improving its own program. Seed AI must be able to understand the purpose behind the various elements of its design, and design entirely new modules that will make it genuinely more intelligent and more effective in fulfilling its purpose.\n\nCreating seed AI is the goal of several organizations. The Singularity Institute for Artificial Intelligence is the most prominent of those explicitly working to create seed AI. Others include the Artificial General Intelligence Research Institute, creator of the Novamente AI engine, and Adaptive Artificial Intelligence Incorporated.",
            "url": "https://towardsdatascience.com/seed-ai-history-philosophy-and-state-of-the-art-22b5294b21b5",
            "id": 55,
        },
        {
            "name": "Orthogonality thesis",
            "definition": "The Orthogonality Thesis states that an agent can have any combination of intelligence level and final goal, that is, its Utility Functions and General Intelligence can vary independently of each other. This is in contrast to the belief that, because of their intelligence, AIs will all converge to a common goal.",
            "url": "https://en.wikipedia.org/wiki/Existential_risk_from_artificial_general_intelligence#Orthogonality_thesis",
            "id": 56,
        },
        {
            "name": "Liquid Neural Network",
            "definition": "Liquid neural networks are a class of artificial intelligence systems that learn on the job, even after their training. In other words, they utilize ‘liquid’ algorithms that continuously adapt to new information, such as a new environment, just like the brains of living organisms.\n\nRecent research has demonstrated the efficiency of a new kind of very small—20,000 parameter—machine-learning system called a liquid neural network. They showed that drones equipped with these excelled in navigating complex, new environments with precision, even edging out state-of-the art systems. The systems were able to make decisions that led them to a target in previously unexplored forests and city spaces, and they could do it in the presence of added noise and other difficulties.",
            "url": "https://spectrum.ieee.org/liquid-neural-networks",
            "id": 57,
        },
        {
            "name": "AI alignment",
            "definition": "In the field of artificial intelligence (AI), AI alignment research aims to steer AI systems towards humans' intended goals, preferences, or ethical principles. An AI system is considered aligned if it advances the intended objectives. A misaligned AI system pursues some objectives, but not the intended ones.",
            "url": "https://en.wikipedia.org/wiki/AI_alignment",
            "id": 58,
        },
        {
            "name": "AIOps",
            "definition": "AIOps is artificial intelligence for IT operations. It refers to the strategic use of AI, machine learning (ML), and machine reasoning (MR) technologies throughout IT operations to simplify and streamline processes and optimize the use of IT resources.",
            "url": "https://www.cisco.com/c/en/us/solutions/artificial-intelligence/what-is-aiops.html",
            "id": 59,
        },
        {
            "name": "Indirect normativity",
            "definition": "Indirect normativity is an approach to the AI alignment problem that attempts to specify AI values indirectly, such as by reference to what a rational agent would value under idealized conditions, rather than via direct specification. It is a method of control by which the motivation of the superintelligence is shaped not directly, but indirectly. The approach specifies a process for the superintelligence to determine beneficial goals rather than specifying them directly.",
            "url": "https://ordinaryideas.wordpress.com/2012/04/21/indirect-normativity-write-up/",
            "id": 60,
        },
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

    # Testing without DB
    candidate_word = random.choice(word_list)
    word = candidate_word

    return word
