import re
import pandas as pd
import numpy as np
import networkx as nx 
import matplotlib.pyplot as plt

alphabet = [chr(i) for i in range(65,91)]
alphabet.extend([chr(i) for i in range(97,123)])
alphabet.extend([chr(i) for i in range(48,58)])
epsilon = '#'
precedence = {'*':3, '|':2, '.':1, '(':1}

# Shunting-yard Algorithm
def toPostfix(infix):
    stack = []
    postfix = []
    for token in infix:
        if token in alphabet: postfix.append(token)
        elif token == '(': stack.append(token)
        elif token == ')':
            top = stack.pop()
            while top != '(':
                postfix.append(top)
                top = stack.pop()
        else:
            while len(stack) != 0 and (precedence[stack[-1:][0]] >= precedence[token]):
                postfix.append(stack.pop())
            
            stack.append(token)

    while len(stack) != 0:
        postfix.append(stack.pop())

    return postfix

# To update indices(column and index) of the dataframe
def updateDataFrame(df, by=1):
    df.columns = np.array(df.columns)+by
    df.index = np.array(df.index)+by
    return df

# To obtain the least and highest node (i.e., digit) in indices of the dataframe
def rangeDataFrame(df):
    n = set(list(df.columns)+list(df.index))
    return min(n), max(n)

# Thompson's Algorithm
def toNFA(regex):
    keys=list(set(re.sub('[^A-Za-z0-9]+', '', regex)+epsilon))
    stack = []
    node = -1
    for i in regex:
        if i in keys:
            # node 1 to node 2 for i
            nfa = pd.DataFrame(columns=[node+1], index=[node+2], data=i)
            # {i: {node+1:node+2}}
            stack.append(nfa)
            node += 2

        elif i == '|':
            nfa1 = stack.pop()
            nfa2 = stack.pop()

            start = nfa2.columns.min()

            nfa2 = updateDataFrame(nfa2, 1)
            nfa1 = updateDataFrame(nfa1, 1)
            n2start, n2end = rangeDataFrame(nfa2)
            n1start, n1end = rangeDataFrame(nfa1)
            nfa = nfa2.append(nfa1)

            nfa[start], nfa[n1end], nfa[n2end] = np.nan, np.nan, np.nan
            nfa.reindex(set(nfa.index.to_list()+[n2start, n1start, n1end+1]))

            nfa.loc[n2start, start] = epsilon
            nfa.loc[n1start, start] = epsilon
            nfa.loc[n1end+1, n1end] = epsilon
            nfa.loc[n1end+1, n2end] = epsilon

            stack.append(nfa)
            node = n1end+1

        elif i == '.':
            nfa1 = stack.pop()
            nfa2 = stack.pop()

            nfa1 = updateDataFrame(nfa1, -1)
            nfa = nfa2.append(nfa1)

            stack.append(nfa)
            node -= 1

        elif i == '*':
            nfa1 = stack.pop()
            start, end = rangeDataFrame(nfa1)
            nfa1 = updateDataFrame(nfa1, 1)
            n1start, n1end = rangeDataFrame(nfa1)

            nfa[start], nfa[n1end] = np.nan, np.nan
            nfa.reindex(set(nfa.index.to_list()+[n1start, n1end+1]))
            nfa.loc[n1start, start] = epsilon
            nfa.loc[n1end+1, n1end] = epsilon
            nfa.loc[n1start, n1end] = epsilon
            nfa.loc[n1end+1, start] = epsilon

            stack.append(nfa)
            node += 2

    adj = stack[0]
    nodes = set(adj.index.to_list() + adj.columns.to_list())
    adj.reindex(nodes)
    for col in list(nodes): 
        if col not in adj.columns.to_list(): nfa[col] = np.nan
    return adj, max(list(nodes))

def eClosure(nfa):
    states = {}
    for state in nfa.columns.to_list():
        states[state] = [state] + nfa[nfa[state]==epsilon].index.to_list()


    for state in nfa.columns.to_list():
        successors = states[state].copy()
        while len(successors):
            s = successors.pop()
            successors = list(set(successors + [i for i in states[s] if i not in states[state]]))
            states[state] = list(set(states[state] + states[s]))

    return states

# e-NFA to DFA
def toDFA(nfa, accept, keys):
    e = eClosure(nfa)
    # Rename each new (combined) state formed for DFA to incrementing whole number
    alias = {0: tuple(e[0])}
    dfa = pd.DataFrame(index=keys, columns=[0,])
    node = [set(e[0]),]

    # For each new state generated for DFA, starting with initial state e-closure of e-NFA
    for n in node:
        
        # For each symbol in alphabet (here, alphabet: non-operation symbols used in regex among [A-Za-z0-9])
        for key in keys:
            
            transitions = []
            # For each NFA state combined as single DFA state (n)
            for i in n:
                # For given symbol (key), get all states reached from current state
                transitions = list(set(list(transitions) + nfa[nfa[i]==key].index.to_list()))
            
            successor = set()
            # For each transition, get all states possible to reach with epsilon
            for t in transitions:
                if t in e.keys(): successor = set(list(successor) + e[t])

            # dfa.loc[key, [tuple(n),]] = successor
            if len(successor):
                if successor not in node:
                    node += [successor,]
                    alias[max(alias.keys()) + 1] = tuple(successor)

                for k, v in alias.items():
                    if v == tuple(n):
                        pre = k
                    if v == tuple(successor):
                        post = k
                dfa.loc[key, pre] = post
    
    final = []
    for k, v in alias.items():
        if accept in v: final.append(k)
    # print(alias)
    return dfa, final

def display(dfa, accept):
    dg = nx.DiGraph()

    edge_labels = {}
    for col in dfa.columns:
        for i, v in dfa[dfa[col].notnull()][col].items():
            dg.add_edge(col, int(v))
            edge_labels[tuple([col, int(v)])] = i

    color_map = []
    for node in dg:
        if node in accept:
            color_map.append('#c8f902')
        else:
            color_map.append('#c4daef')

    pos = nx.spring_layout(dg)
    nx.draw(dg, pos, node_color=color_map, with_labels=True, node_size=500)
    nx.draw_networkx_edge_labels(dg, pos, edge_labels=edge_labels)
    plt.show(block=False)

def recognize(dfa, accept, word, keys):
    # if re.match(regex, word): print('Word Recognized')
    current = 0
    for c in word:
        if c not in keys or np.isnan(dfa.loc[c, current]):
            print('Word Not Recognized')
            return
        current = int(dfa.loc[c, current])
    if current in accept: print('Word Recognized')
    else: print('Word Not Recognized')
    

regex = input('Input: ').replace(" ", "")
postfix = toPostfix(regex)
regex=''.join(postfix)
keys = list(set(re.sub('[^A-Za-z0-9]+', '', regex)))
# print(regex)
adj, accept = toNFA(regex)
# print(adj)
dfa, accept = toDFA(adj, accept, keys)
# print(dfa)
display(dfa, accept)

while True:
    word = input('\nEnter the word: ').replace(" ", "")
    recognize(dfa, accept, word, keys)

    if(input('Do you want to continue (y/n): ').lower() == 'n'): break
