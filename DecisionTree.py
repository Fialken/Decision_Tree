import math
import random
from copy import deepcopy


def best_attribute_split(df,dy):
    best = df.columns[0]
    size = dy.size
    entropy_class = entropy(dy)
    val_gain = float('-inf')

    for attri in df.columns:
        tmp_val = 0

        for val in df[attri].unique():
            subset_dy = dy.loc[df[df[attri] == val].index]
            tmp_val += ((len(subset_dy)/size) * entropy(subset_dy))

        tmp_val = entropy_class - tmp_val
        if val_gain < tmp_val:
            val_gain = tmp_val
            best = attri

    return best

def entropy(attribute):
    size = attribute.size
    ent_val = 0
    for val in attribute.unique():
        count = 0
        for i in attribute.index:
            if attribute[i] == val:
                count += 1
        if(count == 0): # only true when val == nan
            count = attribute.isna().sum()
        p = count/size
        ent_val += (p * math.log2(p))
    return -ent_val

def all_classification_equal(setX, setY):
    #True if all classification are equal, False otherwise
    classif = setY[setX.index[0]]
    for i in setX.index: # get the index of line i presente in sub set (can be 1,3,..,13; not necessary linear 1..n)
        if classif != setY[i]:
            return False
    return True

def most_common_output(indices,setY):
    outputs = {}
    for i in indices:
        if setY[i] not in outputs.keys():
            outputs[setY[i]] = 1
        else: outputs[setY[i]] += 1
   
    max_value = max(outputs.values())
    keys_with_max_value = [key for key, value in outputs.items() if value == max_value]

    return random.choice(keys_with_max_value)

def new_case(curr_node, best_node, max_counter):
    '''
    this funcion is called when a new case is seen at DT.predict()
    in that case we will choose the classification that occurs the most in subtree from the current node
    '''
    if type(curr_node) == LeafNode:  # If the node is a leaf
        return curr_node
    
    for node in curr_node.splits.values():
        leaf_node = new_case(node, best_node, max_counter)
        if leaf_node.counter > max_counter:
            best_node = leaf_node
            max_counter = leaf_node.counter
            
    return best_node


class Evaluation:
    def accuracy(df_pred, df_class):
        assert(len(df_pred) == len(df_class))
        well_classified = 0
        i = 0
        for classif in df_class:
            if df_pred[i] == classif: 
                well_classified += 1
            i += 1
        print(well_classified/len(df_class))


    def confusion_matrix(df_pred, df_class):
        assert(len(df_pred) == len(df_class))
        well_classified = {}
        wrong_classified = {}
        #create dict for TP | FP
        for val in list(set(df_pred)):
            assert(val in list(set(df_pred)))
            well_classified[val] = 0
            wrong_classified[val] = 0

        i = 0
        for classif in df_class:
            if df_pred[i] == classif: 
                well_classified[classif] += 1
            else:
                wrong_classified[classif] += 1
            i += 1

        Evaluation.print_matrix(well_classified, wrong_classified)
    
    def print_matrix(well_classified:dict, wrong_classified:dict):
        spaces = []

        res = "Real classification was correctly predicted or wrongly\n"
        #header
        res += f"class->"
        for key in well_classified.keys():
            res += f" | {key}"
            spaces.append(len(key))
        
        #correctly classified
        res += " |\ncorrect"
        i = 0
        for val in well_classified.values():
            space = " "*int((spaces[i]-len(str(val)))/2)
            res += f" | {space}{val}{space}" if ((spaces[i]-len(str(val))) % 2 == 0) else f" | {space}{val}{space} "
            i += 1
        
        #wrongly classified
        res += " |\nwrong  "
        i = 0
        for val in wrong_classified.values():
            space = " "*int((spaces[i]-len(str(val)))/2)
            res += f" | {space}{val}{space}" if ((spaces[i]-len(str(val))) % 2 == 0) else f" | {space}{val}{space} "
            i += 1
        res += " |\n"
        print(res)



class LeafNode:

    def __init__(self, classif, size, val_ori):
        self.classif = classif
        self.counter = size
        self.origin_value = val_ori


class Node:
    
    def __init__(self, attrib, val_ori=None):
        self.attribute = attrib
        self.origin_value = val_ori
        self.splits = {}


class DTree:

    def __init__(self):
        self.root = None
        self.num_nodes = 0


    def create_DTree(self, dx_train, dy_train, curr_node=None):

        # calculate best attribute to split based in information gain and create Nodes
        best_attrib = best_attribute_split(dx_train,dy_train)
        #create a Node for the split
        if curr_node==None: 
            node = Node(best_attrib)
            self.root = node
        else: 
            curr_node.attribute = best_attrib
            node = curr_node

        dfa = deepcopy(dx_train)
        dfa.drop(columns=best_attrib,inplace=True)

        for val in dx_train[best_attrib].unique(): #analise if when the best attribute as the value = val it classifies every case with the same type
            sub_set = dfa[dx_train[best_attrib] == val]


            if len(sub_set) == 0:
                #create a leaf node with the most common output
                node.splits[val] = LeafNode(most_common_output(dx_train.index,dy_train),len(dx_train.index),val) #add to 'split' dictionary a leafnode for val
                self.num_nodes += 1

            elif all_classification_equal(sub_set,dy_train): 
                #create a leaf node
                node.splits[val] = LeafNode(dy_train[sub_set.index[0]],len(sub_set.index),val) #add to split dictionary a leafnode for val
                self.num_nodes += 1

            elif len(sub_set.columns) == 0:
                #create a leaf node with the most common output
                node.splits[val] = LeafNode(most_common_output(sub_set.index,dy_train),len(sub_set.index),val) #add to 'split' dictionary a leafnode for val
                self.num_nodes += 1
            
            else:
                #need to split again
                node.splits[val] = Node(None,val_ori=val)
                self.create_DTree(sub_set, dy_train, curr_node=node.splits[val])
                self.num_nodes += 1


    def predict(self,df):
        pred = []
        for i in df.index: #each row in df to classify
            curr_node = self.root
            while type(curr_node) != LeafNode:
                try:
                    curr_node = curr_node.splits[df[curr_node.attribute][i]]
                except:
                    curr_node = new_case(curr_node,None, float("-inf"))
            pred.append(curr_node.classif)
            
        return pred


    def __str__(self):
        return print_tree(self.root,0)



def print_tree(node,depth):
    if type(node) == LeafNode:
        space = "    "*(depth-1)
        return f"{space}{node.origin_value}: {node.classif} ({node.counter})\n"
    
    else:
        space = "    "*depth
        res = f"{space}<{node.attribute}>\n"

        for key in node.splits.keys():

            if type(node.splits[key]) == Node:
                res += f"{space}    {key}:\n"

            res += print_tree(node.splits[key],depth+2)

    return res