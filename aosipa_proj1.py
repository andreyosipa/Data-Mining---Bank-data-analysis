import pandas as pd
import numpy as np
import time
import copy
from sets import Set

class Tree(object):
	#Tree object to save fp-growth tree.
    def __init__(self,itemset):
        self.child = []
        #In terms of fp-growth data is parameter that contains 
        #attribute name. Count is same as count in definition of 
        #fp-growth.
        self.data = itemset
        self.data_count = 1
 
    def createChild(self,child_itemset):
        self.child.append(Tree(child_itemset))
 
    def inc_count(self):
        self.data_count = self.data_count + 1

def join_lists(l1, l2):
	#Function to join two lists together if they have len(l1)-1 common
	#elements. Used to generate k-candidate of two k-1 frequent
	#itemsets. List are supposed to have same length.
	#Input: List l1
	#		List l2
	#Output: None or List
    l = []
    f = 0
    for idx in range(len(l1)):
        if l1[idx] != l2[idx]:
            f = f + 1
        if l1[idx] != None and l2[idx] != None and l1[idx] != l2[idx]:
            return None
        else:
            if l1[idx] == None:
                l = l + [l2[idx]]
            else:
                l = l + [l1[idx]]
    if f==2:
        return l
    else:
        return None

def eq_pattern(pat, value):
	#Function to find if two patterns are equal. Patterns are lists 
	#with elements that correspond to ssame attributes at same 
	#positions. It is needed if we have same attribute values for
	#different attributes. Lists in the input are supposed to have
	#same length.
	#Input: List pat
	#		List value
	#Output: 0 or 1.
    for idx in range(len(pat)):
        if pat[idx] != None:
            if pat[idx] != value[idx]:
                return 0
    return 1

def freq_patterns_apriori(data, min_sup, pat_len):
	#Function to find frequent patterns with apriori algorithm.
	#Input: Pandas DataFrame data
	#		integer min_sup
	#		integer pat_len
	#Output: List of frequent patterns of length pat_len.
    columns = data.keys()
    #mining 1-itemsets:
    one_itemsets = []
    features = len(data.columns.values)
    itemset = []
    for i in range(features):
        itemset = itemset + [None]
    for col_idx in range(features):
        itemsets_tmp = {}
        for index in range(len(data)):
            itemset[col_idx] = data.get_value(index, data.columns.values[col_idx])
            if itemset[col_idx] in itemsets_tmp.keys():
                itemsets_tmp[itemset[col_idx]] = itemsets_tmp[itemset[col_idx]] + 1
            else:
                itemsets_tmp[itemset[col_idx]] = 1
        for key in itemsets_tmp.keys():
            if itemsets_tmp[key] < min_sup:
                del itemsets_tmp[key]
            else:
                itemset[col_idx] = key
                one_itemsets = one_itemsets + [copy.copy(itemset)]
        itemset[col_idx] = None
    data_indexes = range(len(data))
    #Creating k-itemsets from k-1 itemsets.
    for k in range(2, pat_len + 1):
    	#Variable to include rows that contain t least on frequent 
    	#k-itemset.
        rows_to_include = Set([])
        kitemsets = []
        for index1 in range(len(one_itemsets)):
            for index2 in range(index1 + 1,len(one_itemsets)):
            	#Generating itemset.
                itemset = join_lists(one_itemsets[index1], one_itemsets[index2])
                if itemset != None:
                    #Prune step. Check all its k-1 itemsets if they 
                    #are frequent.
                    prune = False
                    for attribute in range(len(itemset)):
                        if itemset[attribute] != None:
                            p_itemset = copy.copy(itemset)
                            p_itemset[attribute] = None
                            prune = not p_itemset in one_itemsets
                            if prune:
                                break
                    if (not itemset in kitemsets) and (not prune):
                        #Check support of new itemset.
                        count = 0
               			#Variable to store rows` indexes that contain
               			#this frequent itemset.
                        rows_to_include_itemset = Set([])
                        for index in data_indexes:
                            row = []
                            for idx in range(features):
                                row = row + [data.get_value(index, data.columns.values[idx])]
                            if eq_pattern(itemset, row):
                                count = count + 1
                                rows_to_include_itemset.add(index)
                        if count >= min_sup:
                            kitemsets = kitemsets + [copy.copy(itemset)]
                            rows_to_include = rows_to_include | rows_to_include_itemset
        one_itemsets = copy.copy(kitemsets)
        data_indexes = list(rows_to_include)
    return kitemsets

def freq_patterns_fpgrowth(data, min_sup, pat_len):
	#Function to find frequent patterns with apriori algorithm.
	#Input: Pandas DataFrame data
	#		integer min_sup
	#		integer pat_len
	#Output: Tree.
    
	#Generating one-itemsets.
    one_itemsets = []
    for col in data.keys():
        itemsets_tmp = {}
        for index in range(len(data)):
            tmp = data.get_value(index, col)
            if tmp in itemsets_tmp.keys():
                itemsets_tmp[tmp] = itemsets_tmp[tmp] + 1
            else:
                itemsets_tmp[tmp] = 1
        for key in itemsets_tmp.keys():
            if itemsets_tmp[key] < min_sup:
                del itemsets_tmp[key]
            else:
                one_itemsets = one_itemsets + [[col, key, itemsets_tmp[key]]]
    #Sorting one-itemsets in order of their support.
    for idx1 in range(len(one_itemsets)):
        for idx2 in range(len(one_itemsets)):
            if one_itemsets[idx1][2] > one_itemsets[idx2][2]:
                one_itemsets[idx1] = one_itemsets[idx1] + one_itemsets[idx2]
                one_itemsets[idx2] = one_itemsets[idx1][:3]
                one_itemsets[idx1] = one_itemsets[idx1][3:]
    #Removing support variable from one-itemsets list.
    one_itemsets = [x[:-1] for x in one_itemsets]
    columns = []
    #As one-itemsets may correspond not to every attribute in data
    #we find attributes we need to look at.
    for idx in range(len(one_itemsets)):
        columns = columns + [one_itemsets[idx][0]]
    root = Tree('null')
    for idx in range(len(data)):
        cur_root = root
        row = []
        #Getting needed information from data row.
        for col in columns:
            row = row + [[col, data.get_value(idx, col)]]
        row = [x for x in row if x in one_itemsets]
        #Sort the row in correspondence with 1-itemset frequency order
        for idx1 in range(len(row)):
            for idx2 in range(len(row)):
                if one_itemsets.index(row[idx1]) < one_itemsets.index(row[idx2]):
                    t = copy.copy(row[idx2])
                    row[idx2] = row[idx1]
                    row[idx1] = t
        #Adding new transaction to the tree.
        cur_root = root
        while len(row)>0:
            create = True
            for order in range(len(row)):
                if create:
                    for child in cur_root.child:
                        if child.data == row[order]:
                            child.inc_count()
                            del row[order]
                            cur_root = child
                            create = False
                            break
            if create:
                cur_root.createChild(row[0])
                del row[0]
                cur_root = cur_root.child[-1]
    return root

def test(data_name='adult.data',separator=','):
	data = pd.read_csv(data_name,sep=separator)
	start = time.time()
	t = freq_patterns_apriori(data, 20000, 3)
	end = time.time()
	print end - start

	start = time.time()
	tree = freq_patterns_fpgrowth(data, 20000, 3)
	end = time.time()
	print end - start

	print "Number of 3-itemsets: ", len(t)