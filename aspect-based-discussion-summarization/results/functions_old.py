#!/usr/bin/python -tt
# -*- coding: utf-8 -*-
######################################
#  Author: Nguyen Duc Duy - UNITN
#  Created on Mon Jun 19 10:51:42 2017
#   FUNCTIONS FOR TEXT SUMMARIZATION
######################################
from Graph import Graph;
from Edge import Edge;
from Node import Node;
from sets import Set;
import json;
import nltk;
from datetime import datetime;
from datetime import timedelta;
from nltk.corpus import brown;
from nltk.tag import UnigramTagger;
import string;
import re;
from collections import Counter;
import time;
import nltk.chunk.named_entity;
import random;

######################################
# CONST DECLARATION
NUM_MIN_NODES = 20	# Define minimum number of node. If total number of nodes is  < this number, pruning will not be performed
NUM_MAX_NODES = 200; 	# Define maximum number of nodes to keep
NUM_MIN_EDGES = 30;		# Define minimum number of edge. The value could not more than (NUM_MIN_NODES)*(NUM_MIN_NODES-1). If total number of edge is  < this number, pruning will not be performed
NUM_MAX_EDGES = 300; 	# Define maximum number of edge to keep. The value could not more than (NUM_MAX_NODES)*(NUM_MAX_NODES-1)
EDGE_FREQ_MIN = 3;	# Minimum frequency that an edge is required to be. Being smaller, it will be eliminated.
NODE_FREQ_MIN = 3;	# Minimum frequency that a node is required to be. Being smaller, it will be eliminated.
MIN_WORD_LENGTH = 4;	# Minimum nunber of character of a word, accepted to enter the graph

preferedTags = Set(['NN','NNP','NE']);

exceptionsWords = ['a','the','aboard','about','above','across','after','against','along','amid','among','anti','around','as','at','before','behind','below','beneath','beside','besides','between','beyond','but','by','concerning','considering','despite','down','during','except','excepting','excluding','following','for','from','in','inside','into','like','minus','near','of','off','on','onto','opposite','outside','over','past','per','plus','regarding','round','save','since','than','through','to','toward','towards','under','underneath','unlike','until','up','upon','versus','via','with','within','without'];
regex = re.compile('[%s]' % re.escape(string.punctuation)) #see documentation here: http://docs.python.org/2/library/string.html
######################################
#### From a corpus of Json, it make a set of Node and a list of Edge
# Input: a LIST of Json
# Output: a TUPLE contain 2 set:
#        - [0] is set of Nodes, under the form NODE 
#        - [1] is set of Edges, under the form EDGE
def TextMasher(inpJSls,tagger,lemmatizer):
	print "Started Text Mashing...";
	start_time = time.time();
	#print "[--!--] Enter time point 0 - Begin Mashing:" + str(time.time() - start_time);
	docls = []; #List of all decoded JSON. Type: <type 'dict'>. Access by: ellement['key']
	nodes = Set([]);
	edges = Set([]);
	node_counter = Counter();
	edge_counter = Counter();
	count = 0;
	#print "[--!--] Enter time point 1 - start processing corpus:" + str(time.time() - start_time);
	for js in inpJSls:
		#print "[--!--] Enter time point 1.1 - Start preprocessing" + str(time.time() - start_time);
		count += 1;
		#jl = json.loads(js.replace('\n',''));
		#docls.append(jl);
		docls.append(js);
		# Tokenize
		#text = nltk.word_tokenize(jl['content']);
		ctn = js['content'].encode('ascii', 'ignore');
		repls = ('\n' , ''), ('            ' , '');
		text = reduce(lambda a, kv: a.replace(*kv), repls, ctn);
		#print text;
		raw_tokens = nltk.word_tokenize(text);
		clean_tokens = [];
		# Removing punctuation
		for token in raw_tokens: 
			new_token = regex.sub(u'', token)
			if not new_token == u'':
				lemma = lemmatizer.lemmatize(new_token.lower());
				lemmaUp = lemma.title();
				if (new_token.istitle()):
					clean_tokens.append(lemmaUp); # associate with lemmatize
				else:
					clean_tokens.append(lemma);
		# POS tag
		#print "[--!--] Enter time point 1.2 - Ended tokenize, begin POS tag:" + str(time.time() - start_time);
		pos = tagger.tag(clean_tokens);
		#print "[--!--] Enter time point 1.3 - Ended pos tag:" + str(time.time() - start_time);
		#Named entities detect and merge known tags
		chunks = nltk.ne_chunk(pos);
		# Merge Named entities and put them a tag as NE
		nes = []; # The list of NEs and its tag. From sentence: 'Donald Trump killed John Lenon in White House'
					#	Output look like: [[('Donald', 'NNP')], [('Trump', 'NNP')], [('John', 'NNP'), ('Lenon', 'NNP')], [('White', 'NNP'), ('House', 'NNP')]]
		if isinstance(chunks, nltk.tree.Tree):               
			traverse(chunks,nes);
		nes = [item for sublist in nes for item in sublist]; # Flattenning it
		pos_size =len(pos); 
		toEliItems = set([]); # Items to delete from POS
		toAddItems = []; 	# INDEX of Items to add to POS
		discard_flag = False; # to discard a element if it is arealy in previous pair. E.g: Ben Jim House Beem ->'Ben Jim', 'House Beem', Instead of 'Ben Jim', 'Jim House', 'House Beem'
		for i in xrange(0,len(nes)-2):
			if (nes[i][1] != nes[i+1][1] or discard_flag): continue;
			if (discard_flag): 
				discard_flag = False;
				continue;
			po_index = pos.index(nes[i]);
			if (po_index >= 0 and pos[po_index+1] == nes[i+1]): # the next one is also NE
				toAddItems.append((nes[i][0]+'_'+ nes[i+1][0],'NE'));
				toEliItems.update([po_index,po_index+1]);
				discard_flag = True;
		count_1 =0;
		for i in toEliItems:		#delete items
			pos.pop(i-count_1);
			count_1 = count_1 +1;
		pos.extend(toAddItems);
		# Remove stopwords and filter
		#words = [po[0] for po in pos if ((po[1] in preferedTags) and (len(po[0])>MIN_WORD_LENGTH) and (po[0].lower() not in nltk.corpus.stopwords.words('english')))];
		words=Set([]);
		for po in pos:
			if ((po[1] in preferedTags) and (len(po[0])>MIN_WORD_LENGTH) and (po[0].lower() not in nltk.corpus.stopwords.words('english'))):
				node_counter[po[0]] +=1; 			# Update the counter as new word come
				words.add(po[0]);
		# COnstruct bigram and transform to output format
		#bigrms = nltk.bigrams(words);
		#bims = nltk.bigrams(node_counter.elements());
		#print "[--!--] Enter time point 1.4 - Begin building tuples:" + str(time.time() - start_time);
		bims = [(element,subelement) for element in words for subelement in words if not(element is subelement)];
		#edges = [];
		for grm in bims:
			#edges.append(grm[0]+ ' ' +grm[1]); # a space
			edge_counter[grm[0]+ ' ' +grm[1]] +=1; 	# Update the counter as new co-occorence come
			
		#print node_counter;
		#print edge_counter;
		#print "[--!--] Enter time point 1.5 - start comverting node to set:" + str(time.time() - start_time);
		nodes = nodeCounterToSet(node_counter);
		edges = edgeCounterToSet(edge_counter);
		#print "[--!--] Enter time point 1.6 - End comverting node to set:" + str(time.time() - start_time);
		if (count % 100 == 0):
			print " - " + str(count) + " records processed!"
	print "Text Mashing Finished!";	
	print " - Returned " + str(len(nodes)) + " nodes and " + str(len(edges)) + " edges";
	#print "[--!--] Enter time point 2 - Ready to return:" + str(time.time() - start_time);
	# Merge nodes accoring to modal
	return (nodes,edges);

	
######################################
#### The function will remove nodes and edges whose saftisfy some conditions. The priority of removal is:
# 0. If number of nodes < NUM_MIN_NODES the SKIP the pruning!
# 1. Edges whose timeAdded < EDGE_TIME_LIMIT (compare to corrent time)
# 2. Nodes whose timeAdded < NODE_TIME_LIMIT (compare to corrent time)
# 3. Edges whose efreq < EDGE_FREQ_MIN
# 4. Nodes whose nfreq < NODE_FREQ_MIN
# After every above steps, If the total number of node still > NUM_MAX_NODES, the program will continue to 4, then if it still not saftisfied, it loop again until number of node not exceed NUM_MAX_NODES
#    Please note that all realted edges will also be eliminated, during step of GraphRefine. See. Graph.py->GraphRefine()
# Input: TheGraph
# Output: Null
def GraphPrunning(graph,logCollection):
	print 'Started Graph Pruning...';
	count = 0;
	countDeletedNodes=0;
	countDeletedEdges=0;
	currentTime = datetime.now();
	current_EDGE_FREQ_MIN = EDGE_FREQ_MIN/2; # To start from EDGE_FREQ_MIN not EDGE_FREQ_MIN+0.1
	current_NODE_FREQ_MIN = NODE_FREQ_MIN/2; # To start from NODE_FREQ_MIN not NODE_FREQ_MIN+0.1
	current_EDGE_TIME_LIMIT = EDGE_TIME_LIMIT+(EDGE_TIME_LIMIT*10/100); # similar above
	current_NODE_TIME_LIMIT = NODE_TIME_LIMIT+(NODE_TIME_LIMIT*10/100); # similar above
	Flag_kill_by_frequency=True;
	# First kill node and edge acoording to its age, after 5 iteration, Flag_kill_by_frequency will turn true, and pruning by Frequency begin
	eledges =[];
	elnodes =[];
	while (True):	
		# EDGE_FREQ_MIN & NODE_FREQ_MIN dynamically add 1 after a loop
		# NODE_TIME_LIMIT & NODE_TIME_LIMIT are narrowed down by delta being devide by COUNT after every loop
		count +=1;
		
		if (count>5): Flag_kill_by_frequency=True;
		
		print ' Entered pruning iteration ' + str(count);
		print ' - Current graph status:' + graph.toString();
		print ' - current_EDGE_FREQ_MIN = ' + str(current_EDGE_FREQ_MIN) + '. current_NODE_FREQ_MIN= ' + str(current_NODE_FREQ_MIN);
		# Check the number of node is < than NUM_MIN_NODES
		if (len(graph.nodes) < NUM_MIN_NODES or len(graph.edges) < NUM_MIN_EDGES):
			break; # Skip the prunning
		if (current_EDGE_TIME_LIMIT > EDGE_TIME_LIMIT*30/100): current_EDGE_TIME_LIMIT-=(EDGE_TIME_LIMIT*10/100);
		if (current_NODE_TIME_LIMIT > NODE_TIME_LIMIT*30/100): current_NODE_TIME_LIMIT-=(NODE_TIME_LIMIT*10/100);
		current_node_time_checkpoint = currentTime - timedelta(minutes=current_NODE_TIME_LIMIT);
		current_edge_time_checkpoint = currentTime - timedelta(minutes=current_EDGE_TIME_LIMIT);
		print ' - Now: ' + str(datetime.now());
		print ' - Node time checkpoint: ' +  str(current_node_time_checkpoint);
		print ' - Edge time checkpoint: ' +  str(current_edge_time_checkpoint);

		#Kill old Edges
		for edge in graph.edges:
			if (edge.gettimeAdded() < current_edge_time_checkpoint):
				eledges.append(edge);
		graph.RemoveEdges(eledges,logCollection);
		print '   + ' + str(len(eledges)) + ' edge(s) removed in Kill old Edges.';
		countDeletedEdges += len(eledges);
		eledges =[]; # release memory
		if (len(graph.nodes) < NUM_MAX_NODES or len(graph.edges) < NUM_MAX_EDGES): # check graph size again
			graph.updateNodeEdgeList();
			break; # Skip the prunning
		
		#Kill old Nodes
		for node in graph.nodes:
			if (node.gettimeAdded() < current_node_time_checkpoint):
				elnodes.append(node);
		graph.RemoveNodes(elnodes,logCollection);
		print '   + ' + str(len(elnodes)) + ' node(s) removed in Kill old Nodes.';
		countDeletedNodes += len(elnodes);
		elnodes =[]; # release memory
		if (len(graph.nodes) < NUM_MAX_NODES or len(graph.edges) < NUM_MAX_EDGES): # check graph size again
			graph.updateNodeEdgeList();
			break; # Skip the prunning
		
		
		if (Flag_kill_by_frequency== True):
			current_EDGE_FREQ_MIN *=2;
			current_NODE_FREQ_MIN *=2;
			#Kill low frequenced Edges
			for edge in graph.edges:
				if (edge.getfreq() < current_EDGE_FREQ_MIN and edge.gettimeAdded() > current_edge_time_checkpoint):
					eledges.append(edge);
			graph.RemoveEdges(eledges,logCollection);
			print '   + ' + str(len(eledges)) + ' edge(s) removed in Kill low frequenced Edges.';
			countDeletedEdges += len(eledges);
			eledges =[]; # release memory
			if (len(graph.nodes) < NUM_MAX_NODES or len(graph.edges) < NUM_MAX_EDGES): # check graph size again
				graph.updateNodeEdgeList();
				break; # Skip the 
			
			
			#Kill low frequenced Nodes
			for node in graph.nodes:
				if (node.getfreq() < current_NODE_FREQ_MIN and node.gettimeAdded() > current_node_time_checkpoint):
					elnodes.append(node);
			graph.RemoveNodes(elnodes,logCollection);
			print '   + ' + str(len(elnodes)) + ' node(s) removed in Kill low frequenced Nodes.';
			countDeletedNodes += len(elnodes);
			elnodes =[]; # release memory
			if (len(graph.nodes) < NUM_MAX_NODES or len(graph.edges) < NUM_MAX_EDGES): # check graph size again
				graph.updateNodeEdgeList();
				break; # Skip the prunning
				
		# Worst case, obsolute limit for the prunning
		if (count > MAXIMUM_PRINNING_ITERATIONS):
			print "Some stupid things happend in the loop of GraphPrunning. Count >20!"
			graph.updateNodeEdgeList();
			break;
		print ' - Current graph status:' + graph.toString();
		graph.printNodes();
		graph.printEdges();
	# end while	
	
	print "Graph Pruning Finished!";
	print ' - Current graph status:' + graph.toString();
	print " - Total " + str(countDeletedNodes) + " nodes and " + str(countDeletedEdges) + " edges were eliminated.";

	
######################################
#### The function will build a UNIGRAM tagger from Brown Corpus, that specified to deal with incorrect capitalize
# Input: Nothing
# Output: The tagger
def BuildTagger():
	sentences = brown.tagged_sents();
	train_set = [];
	for sentence in sentences:
		ls=[];
		for w,tag in sentence:
			if (not(w in exceptionsWords)):
				ls.append((w,tag));
			else:
				ls.append((w.capitalize(),tag));
		train_set.append(ls);

	default_tagger = nltk.DefaultTagger('NN');
	bigram_tagger = nltk.BigramTagger(train_set,backoff=default_tagger);
	trigram_tagger = nltk.TrigramTagger(train_set,backoff=bigram_tagger);
	
	return nltk.UnigramTagger(train_set,backoff=trigram_tagger);

######################################
### Convert a NODE counter (strings) into SET OF NODES
# Input: The counter
# Output: A set of Nodes
def nodeCounterToSet(nCounter):
	res = Set([]);
	for key in nCounter:
		newNode = Node(key);
		newNode.setwfreq(nCounter[key]);
		if not(newNode.isDump()):
			res.add(newNode);
	
	return res;

######################################
### Convert a EDGE counter (strings) into SET OF EDGES
# Input: The counter
# Output: A set of Edges
def edgeCounterToSet(nCounter):
	res = Set([]);
	for key in nCounter:
		newEdge = Edge(key);
		newEdge.setefreq(nCounter[key]);
		if not(newEdge.isDump()):
			res.add(newEdge);
	
	return res;
	
######################################
### NLTK tree traverse and return list of named entities
# Input: A node (type NLTK Tree) and output list
# Output: Nothing. Data come to outls as a list.
def traverse(t,outls):
	try: 
		t.label()
	except AttributeError: 
		return
	else: 
		#print  t.label();
		if t.label() in set(['LOCATION', 'ORGANIZATION', 'PERSON', 'DURATION', 'DATE', 'CARDINAL', 'PERCENT', 'MONEY', 'MEASURE', 'FACILITY', 'GPE']) : 
			outls.append(t.leaves());
			#print t.leaves();
		else: 
			for child in t: 
				traverse(child,outls)
		
######################################
### MERGE NODES acoordin to the model contructed from bigram of the entire corpus of data
# Input: Set of nodes (with full information), Set of edges (with full information), a classifier that input 2 words as a tuple, and output prediction. True, then these words sould be merged.
# Output: Set of merge_nodes (with modified information), Set of merge_edges (with modified information)
# How it work?
# 1. Find the mergeable pair of node, by putting each pair of node1,node2 to the classifier. Node that the graph was designed to be directed graph, therefore if two nodes (A,B) are merge then (B,A) need to be merged too.
# The program will disregard node have '_' since it was merged beforehand. This step will return a list of tuple(n1,n2) is nodes to merge
# 2. Dealing with overlap. There will be a case happend to node A: (A,1); (A,n); (1,A); (n,A) where n indicate more than 1 nodes are determined to merge with A
# Assumption 1: Node A can not merge with itself
# Assumption 2: Contradiction: (A,B) and (B,A) can not exist at the same time. Since A --> B is an orded relation, we can not merge them if they have the realtion to each other in both way
# Altgoritm:	Given: The classifier say: TRUE to input (A,B) means there is an directed relation, A-->B. To select best A->x to merge:
#				i. Find a list of all node whose appeared in left hand side of the relation is minimum (chose value = 1), called A_1eft_list
#				ii. Find a list of all node whose appeared in right hand side of the relation is minimum (chose value = 1), called A_right_list
#				iii. Map each element i in A_1eft_list to each element j in A_right_list
#					IF relation i-->i exist THEN continue (assumption 1)
#					IF relation i-->j exist THEN
#						IF j-->i exist THEN contradiction, go to next element
#						ELSE IF there are more than 1 element left THEN randomly choose 1 of them
#							  ELSE: Keep the only one
#					
# 3. Merge Nodes: the idea is the new node contais combination of information belong to its ancestor, as well as all connecton freom-to these two nodes will be directed to new node
#			- new node freq = sum; 
#			- date= max of two date, 
#			- all connection from and to these 2 nodes direct to new node
#			- delete all these edges
#			- IF there are edge between these 2 nodes THEN delete it
#			- Delete two nodes
# 4. Run Graph.GraphRefine() to delete all trash
def BigramNodeMerge(nodes,edges,classifier):


	# Find mergable nodes pair
	mergeable_list =[];
	co_left = Counter();	# (A,x) Counter to compute number of its second element, from 1 to n. Key is A, value id 1..n
	co_right = Counter();	# (x,A) Counter to compute number of its first element, from 1 to n. Key is A, value id 1..n
	for node1 in nodes:
		word1 = node1.getWord();
		for node2 in nodes:
			word2 = node2.getWord();
			if (word1==word2 or word1.find('_')>=0 or word2.find('_')>=0): continue; 
			if (classifier.classify({'word1':word1, 'word2':word2})): # Merge able
				mergeable_list.append((node1,node2));
				co_left[node1] +=1;
				co_right[node2] +=1;
	print 'mergeable_list: ' +str(mergeable_list);
	# Deal with overlap - CURRENTLY NOT APPLICABLE
	A_1eft_list = [];
	A_right_list = [];
	# get elements that count ==1
	for e in co_left.elements():
		if (co_left[e]==1): A_1eft_list.append(e); 
	
	for e in co_right.elements():
		if (co_right[e]==1): A_right_list.append(e); 
	print 'co_left: ' +str(co_left);
	print 'co_right: ' +str(co_right);
	print 'A_1eft_list: ' +str(A_1eft_list);
	print 'A_right_list: ' +str(A_right_list);
	final_mergeable_set = [];
	for node_left in A_1eft_list:
		local_mergeable_list = [];
		print '     In merge node: ' + node_left.toString();
		for node_right in A_right_list:
			if (node_left.getword() == node_right.getword()): 
				print '     mergeable escape: duplicate label';
				continue;
			if ((node_left,node_right) in mergeable_list):
				local_mergeable_list.append((node_left,node_right));
#				if ((node_right,node_left) in mergeable_list): 
#					print '     mergeable escape: contradiction';
#					continue;
#				else: 
#					local_mergeable_list.append((node_left,node_right));
		print '     local_mergeable_list: ' + str(local_mergeable_list);
		if (local_mergeable_list == []): continue;
		else: final_mergeable_set.add(random.choice(local_mergeable_list));
	#Merging items in set
	print 'final_mergeable_set: ' + str(final_mergeable_set);
	for (node1,node2) in final_mergeable_set:
		# disregard if any node node in node list
		if ((node1 not in nodes) or (node2 not in nodes)): continue;
		node1_in_edges = []; # list of edges that point TO node1
		node1_out_edges= []; # list of edges that point FROM node1
		node2_in_edges = []; # list of edges that point TO node2
		node2_out_edges= []; # list of edges that point FROM node2
		for ed in edges: 
			# delete connection betweent these node, if exist
			if (ed.getWords() == (node1.getWord(),node2.getWord())): 
				edges.remove(ed);
			# identify incoming nodes to node1
			if (ed.getWord0() == node1.getWord()): node1_out_edges.append(ed);
			if (ed.getWord1() == node1.getWord()): node1_in_edges.append(ed);
			if (ed.getWord0() == node2.getWord()): node2_in_edges.append(ed);
			if (ed.getWord1() == node2.getWord()): node2_out_edges.append(ed);
		# Start merging
		#	initialize new node
		new_node = Node(node1.getWord() + '_' + node2.getWord());
		new_node.setwfreq(node1.getfreq() + node2.getfreq());
		if (node1.gettimeAdded() < node2.gettimeAdded()):
			new_node.setTime(node2.gettimeAdded());
		else:
			new_node.setTime(node1.gettimeAdded());
		if (new_node.isDump()): continue;
		#	redirect edges:
		new_edges = [];
		# 	Incomming nodes
		for ed in (node1_in_edges+node2_in_edges):
			newEdge = Edge(ed.getWord0()+ " " + new_node.getWord());
			newEdge.setefreq(ed.getfreq());
			newEdge.setTime(ed.gettimeAdded());
			if not(newEdge.isDump()):
				res.add(newEdge);
			new_edges.append(newEdge);
		# 	Outgoing nodes
		for ed in (node1_out_edges+node2_out_edges):
			newEdge = Edge(new_node.getWord()+ " " + ed.getWord1());
			newEdge.setefreq(ed.getfreq());
			newEdge.setTime(ed.gettimeAdded());
			if not(newEdge.isDump()):
				new_edges.append(newEdge);
		# Now interfere the data !!!!!!!!!!!!!!!!
		edges = edges.difference(set(node1_in_edges + node1_out_edges + node2_in_edges + node2_out_edges));
		nodes = nodes.difference(set([node1,node2]));
		edges.update(new_edges);
		nodes.add(new_node);
		print '@@@' + new_node.toString();
		print '* Merged ' + node1.getWord() + ' and ' + node2.getWord();
	return (nodes,edges);

######################################
### MERGE NODES THAT IS THE CAPITALIZED VERSION OF OTHER. For example: Debate and debate
# Input: Set of nodes (with full information), Set of edges (with full information)
# Output: Set of merge_nodes (with modified information), Set of merge_edges (with modified information)
def MergeDuplicate(nodes,edges,lemmatizer):
	new_nodes = [];
	eli_nodes = [];
	new_edges = [];
	eli_edges = [];
	for node1 in nodes:
		word1 = node1.getWord();
		node1_in_edges = []; # list of edges that point TO node1
		node1_out_edges= []; # list of edges that point FROM node1
		node2_in_edges = []; # list of edges that point TO node2
		node2_out_edges= []; # list of edges that point FROM node2
		for node2 in nodes:
			word2 = node2.getWord();
			if (word1 != word2 and (node1 not in eli_nodes) and (node2 not in eli_nodes) and (word1.lower() == word2.lower() or lemmatizer.lemmatize(word1.lower()) == lemmatizer.lemmatize(word2.lower()))):
				# Start merging
				#	initialize new node
				new_node = Node(word1.title());
				new_node.setwfreq(node1.getfreq() + node2.getfreq());
				if (node1.gettimeAdded() < node2.gettimeAdded()):
					new_node.setTime(node2.gettimeAdded());
				else:
					new_node.setTime(node1.gettimeAdded());
				if (new_node.isDump()): continue;
				new_nodes.append(new_node);
				#	redirect edges:
				
				# 	Incomming nodes
				for ed in (node1_in_edges+node2_in_edges):
					newEdge = Edge(ed.getWord0()+ " " + new_node.getWord());
					newEdge.setefreq(ed.getfreq());
					newEdge.setTime(ed.gettimeAdded());
					if not(newEdge.isDump()):
						new_edges.append(newEdge);
				# 	Outgoing nodes
				for ed in (node1_out_edges+node2_out_edges):
					newEdge = Edge(new_node.getWord()+ " " + ed.getWord1());
					newEdge.setefreq(ed.getfreq());
					newEdge.setTime(ed.gettimeAdded());
					if not(newEdge.isDump()):
						new_edges.append(newEdge);
				eli_nodes = eli_nodes +[node1,node2];
				eli_edges = eli_edges + node1_in_edges + node1_out_edges + node2_in_edges + node2_out_edges;
	# Now interfere the data !!!!!!!!!!!!!!!!
	for ed in eli_nodes:
		edges.discard(ed);
	for nd in eli_nodes:
		nodes.discard(nd);

	edges.update(new_edges);
	nodes.update(new_nodes);
	return (nodes,edges);		