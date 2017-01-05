import pandas as pd

with open('./citeulike-a/citations.dat','r') as f:
    next(f) # skip first row
    citations = pd.DataFrame(l.rstrip().split() for l in f)

with open('./citeulike-a/item-tag.dat','r') as f:
    next(f) # skip first row
    itemtag = pd.DataFrame(l.rstrip().split() for l in f)


with open('./citeulike-a/mult.dat','r') as f:
    next(f) # skip first row
    mult = pd.DataFrame(l.rstrip().split() for l in f)


with open('./citeulike-a/tags.dat','r') as f:
    next(f) # skip first row
    tags = pd.DataFrame(l.rstrip().split() for l in f)


with open('./citeulike-a/users.dat','r') as f:
    next(f) # skip first row
    users = pd.DataFrame(l.rstrip().split() for l in f)



with open('./citeulike-a/vocabulary.dat','r') as f:
    next(f) # skip first row
    vocabulary = pd.DataFrame(l.rstrip().split() for l in f)


#print(citations)