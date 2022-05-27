#!/usr/bin/python -tt
# -*- coding: utf-8 -*-
######################################
#   Author: Nguyen Duc Duy - UNITN
#	This script load a comment file and create a graph for each comment. Then build up composition graph.
######################################

import codecs
from textblob import TextBlob
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from itertools import combinations
import re
import matplotlib.pyplot as plt


import networkx as nx

stopwords = set(stopwords.words('english')) # Add more stopword to ignore her

# Build a graph from a text
# Param: text: the text content
# Return: None if the graph is empty/unable to build | Otherwise, the built graph.
def build_graph(text):
    sentences = []
    ln = text
    # Extract noun phrase
    if ln:
        for sen in sent_tokenize(ln):
            # print sen;
            for np in TextBlob(sen).noun_phrases:
                ln = ln.replace(np, np.replace(' ', '_'))
                ln = ln.replace('<blockquote>', '')
                ln = ln.replace('</blockquote>', '; ')
                ln = re.sub('<a.*>.*?</a>', '', ln)
    sentences.extend(sent_tokenize(ln))

    G = nx.Graph(); # Initialize the graph

    # Get nodes and edges from sentence
    for sen in sentences:
        text = TextBlob(sen);  # Convert to textblob
        #words = text.words;
        preferred_words = [w.lemmatize().lower() for w, t in text.tags if t in ['NN', 'NNS', 'JJ']]
        filtered_words = [w for w in preferred_words if w not in stopwords];
        G.add_nodes_from(filtered_words) # Add nodes from filtered words
        # Update nodes's weight
        for node in filtered_words:
            try:
                G.node[node]['weight'] += 1
            except KeyError:
                G.node[node]['weight'] = 1
        print 'Nodes ', G.nodes();

        edges = combinations(filtered_words, 2);
        filtered_edges = list(edges);
        G.add_edges_from(filtered_edges) # Add edges from the combination of words co-occurred in the same sentence
        # Update edges's weight
        for u,v in filtered_edges:
            try:
                G.edge[u][v]['weight'] += 1
            except KeyError:
                G.edge[u][v]['weight'] = 1
        print 'Edges ', G.edges(), '\n'

    if len(G.nodes()) == 0:
        return None
    return G

if __name__=="__main__":

    key_graphs = [];
    # Read from the file
    with codecs.open("comments.txt", "rb", "utf8") as comment_file:
        for comment in comment_file:
            g = build_graph(comment.strip())
            if g!=None:
                key_graphs.append(g);

    print "Number of extracted graph: ", len(key_graphs);

    """
        for g in key_graphs:
        pos = nx.spring_layout(g)
        nx.draw_networkx_nodes(g, pos,
                              node_color='r',
                              node_size=500,
                              alpha=0.8)

        # edges
        nx.draw_networkx_edges(g, pos,
                               width=1.0,
                               alpha=0.5)
        nx.draw_networkx_labels(g, pos)

    plt.axis('off')
    plt.show();
    """


