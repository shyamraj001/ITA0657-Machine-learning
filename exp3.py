import math
import pandas as pd
data = pd.DataFrame([
    ['Sunny','Hot','High','Weak','No'],
    ['Sunny','Hot','High','Strong','No'],
    ['Overcast','Hot','High','Weak','Yes'],
    ['Rain','Mild','High','Weak','Yes'],
    ['Rain','Cool','Normal','Weak','Yes'],
    ['Rain','Cool','Normal','Strong','No'],
    ['Overcast','Cool','Normal','Strong','Yes'],
    ['Sunny','Mild','High','Weak','No'],
    ['Sunny','Cool','Normal','Weak','Yes'],
    ['Rain','Mild','Normal','Weak','Yes'],
    ['Sunny','Mild','Normal','Strong','Yes'],
    ['Overcast','Mild','High','Strong','Yes'],
    ['Overcast','Hot','Normal','Weak','Yes'],
    ['Rain','Mild','High','Strong','No']
], columns=['Outlook','Temperature','Humidity','Wind','PlayTennis'])
def entropy(target_col):
    elements, counts = zip(*target_col.value_counts().items())
    entropy = sum([(-counts[i]/sum(counts)) * math.log2(counts[i]/sum(counts)) for i in range(len(elements))])
    return entropy
def info_gain(data, split_attribute, target_name="PlayTennis"):
    total_entropy = entropy(data[target_name])
    vals, counts = zip(*data[split_attribute].value_counts().items())
    weighted_entropy = sum([(counts[i]/sum(counts)) * entropy(data.where(data[split_attribute]==vals[i]).dropna()[target_name]) for i in range(len(vals))])
    return total_entropy - weighted_entropy
def id3(data, originaldata, features, target_attribute_name="PlayTennis", parent_node_class=None):
    # If all target values are the same, return that value
    if len(data[target_attribute_name].unique()) <= 1:
        return data[target_attribute_name].iloc[0]
    elif len(data)==0:
        return originaldata[target_attribute_name].mode()[0]
    elif len(features) ==0:
        return parent_node_class
    else:
        parent_node_class = data[target_attribute_name].mode()[0]
        item_values = [info_gain(data, feature, target_attribute_name) for feature in features]
        best_feature_index = item_values.index(max(item_values))
        best_feature = features[best_feature_index]
        tree = {best_feature:{}}
        features = [i for i in features if i != best_feature] 
        for value in data[best_feature].unique():
            sub_data = data.where(data[best_feature]==value).dropna()
            subtree = id3(sub_data, data, features, target_attribute_name, parent_node_class)
            tree[best_feature][value] = subtree
        return tree
features = data.columns[:-1].tolist()
tree = id3(data, data, features)
print("Decision Tree:", tree)
def classify(sample, tree):
    for attribute, branches in tree.items():
        value = sample[attribute]
        if value in branches:
            result = branches[value]
            if isinstance(result, dict):
                return classify(sample, result)
            else:
                return result
    return None
sample = {'Outlook':'Sunny','Temperature':'Cool','Humidity':'High','Wind':'Strong'}
print("New Sample Classification:", classify(sample, tree))