import pandas as pd

with open('/Users/mine/Dropbox/Thesis/Recommender-BayesianDeep/Bayesian Deep/RecommenderSystems/citeulike-a/citations.dat','r') as f:
    next(f) # skip first row
    citations = pd.DataFrame(l.rstrip().split() for l in f)

with open('/Users/mine/Dropbox/Thesis/Recommender-BayesianDeep/Bayesian Deep/RecommenderSystems/citeulike-a/item-tag.dat','r') as f:
    next(f) # skip first row
    itemtag = pd.DataFrame(l.rstrip().split() for l in f)


with open('/Users/mine/Dropbox/Thesis/Recommender-BayesianDeep/Bayesian Deep/RecommenderSystems/citeulike-a/mult.dat','r') as f:
    next(f) # skip first row
    mult = pd.DataFrame(l.rstrip().split() for l in f)


with open('/Users/mine/Dropbox/Thesis/Recommender-BayesianDeep/Bayesian Deep/RecommenderSystems/citeulike-a/tags.dat','r') as f:
    next(f) # skip first row
    tags = pd.DataFrame(l.rstrip().split() for l in f)


with open('/Users/mine/Dropbox/Thesis/Recommender-BayesianDeep/Bayesian Deep/RecommenderSystems/citeulike-a/users.dat','r') as f:
    next(f) # skip first row
    users = pd.DataFrame(l.rstrip().split() for l in f)



with open('/Users/mine/Dropbox/Thesis/Recommender-BayesianDeep/Bayesian Deep/RecommenderSystems/citeulike-a/vocabulary.dat','r') as f:
    next(f) # skip first row
    vocabulary = pd.DataFrame(l.rstrip().split() for l in f)


#print(citations)