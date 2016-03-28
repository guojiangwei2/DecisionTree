import math


# find most common value for an attribute
# here, data is a list, each item is a list represent an observation,
# attributes is the column name, target is the target column name to calculate
def majority(attributes, data, target):
    idx = attributes.index(target)
    lst = [item[idx] for item in data]
    sorted_lst = sorted([[lst.count(k), k] for k in set(lst)], reverse=True)
    return sorted_lst[0][1]


# Calculates the entropy of the given data set for the target attr
def entropy(attributes, data, targetAttr):
    # find index of the target attribute
    idx = attributes.index(targetAttr)
    # Calculate the frequency of each of the values in the target attr
    valLst = [item[idx] for item in data]
    valFreq = dict([[k, valLst.count(k)] for k in set(valLst)])
    # Calculate the entropy of the data for the target attr
    dataEntropy = 0.0
    for freq in valFreq.values():
        dataEntropy += (-float(freq)/len(data)) * math.log(float(freq)/len(data), 2)
    return dataEntropy


def gain(attributes, data, attr, targetAttr):
    """
    Calculates the information gain (reduction in entropy) that would
    result by splitting the data on the chosen attribute (attr).
    """
    # find index of the attribute
    idx = attributes.index(attr)
    # Calculate the frequency of each of the values in the target attribute
    valLst = [item[idx] for item in data]
    valFreq = dict([[k, valLst.count(k)] for k in set(valLst)])
    # Calculate the sum of the entropy for each subset of records weighted
    # by their probability of occuring in the training set.
    subsetEntropy = 0.0
    for val in valFreq.keys():
        valProb = round(valFreq[val]) / sum(valFreq.values())
        dataSubset = [entry for entry in data if entry[i] == val]
        subsetEntropy += valProb * entropy(attributes, dataSubset, targetAttr)
    # Subtract the entropy of the chosen attribute from the entropy of the
    # whole data set with respect to the target attribute (and return it)
    return (entropy(attributes, data, targetAttr) - subsetEntropy)


# choose best attibute
def chooseAttr(data, attributes, target):
    res = sorted([[gain(attributes, data, attr, target), attr]
                  for attr in attributes], reverse=True)
    return res[0][1]


# get values in the column of the given attribute
def getValues(data, attributes, attr):
    idx = attributes.index(attr)
    values = list(set([item[idx] for item in data]))
    return values


# get the subset of the target node
def getExamples(data, attributes, best, val):
    """
    find the subset that the best attribute equals to val,
    subset is a list of item that exclude the split value
    """
    idx = attributes.index(best)
    # rm_val_by_idx: remove the value by index and return the remaing value
    # lst = [1,2,3,4]; idx = -1; rm_val_by_idx(lst, idx) # return [1, 2, 3]
    rm_val_by_idx = lambda val, idx: val.pop(idx) and val
    examples = [rm_val_by_idx(item, i) for item in data if item[i] == val]
    return examples


def makeTree(data, attributes, target, recursion):
    recursion += 1
    # Returns a new decision tree based on the examples given.
    # slice operator completely copy shallow list structures
    data = data[:]
    idx = attributes.index(target)
    vals = [record[idx] for record in data]
    default = majority(attributes, data, target)
    # If the dataset is empty or the attributes list is empty, return the
    # default value. When checking the attributes list for emptiness, we
    # need to subtract 1 to account for the target attribute.
    if not data or (len(attributes) - 1) <= 0:
        return default
    # If all the records in the dataset have the same classification,
    # return that classification.
    elif len(set(vals)) == 1:
        return vals[0]
    else:
        # Choose the next best attribute to best classify our data
        best = chooseAttr(data, attributes, target)
        # Create a new decision tree/node with the best attribute and an empty
        # dictionary object--we'll fill that up next.
        tree = {best:{}}
        # Create a new decision tree/sub-node for each of the values in the
        # best attribute field
        for val in getValues(data, attributes, best):
            # Create a subtree for the current value under the "best" field
            examples = getExamples(data, attributes, best, val)
            newAttr = attributes[:]
            newAttr.remove(best)
            subtree = makeTree(examples, newAttr, target, recursion)
            # Add the new subtree to the empty dictionary object in our new
            # tree/node we just created.
            tree[best][val] = subtree
    return tree
