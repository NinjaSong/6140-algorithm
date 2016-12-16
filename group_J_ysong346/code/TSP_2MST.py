import numpy as np
import csv
import networkx as nx
import pandas as pd
import time
import sys
import random
import os


#----------------------------------------------------
# set up global dict first
Parent = dict()
Rank = dict()

def MakeSet(node):
    # Parent is the parent record for all nodes
    # Rank is the rank
    Parent[node] = node
    Rank[node] = 0
    
def Find(node):
    
    if Parent[node] != node:
        Parent[node] = Find(Parent[node])
    return Parent[node]
    
    '''
    if X[node] == node:
        return node
    else:
        return Find(X[node])
    '''
def Union(node1,node2):
    root1 = Find(node1)
    root2 = Find(node2)
    if root1 != root2:
        if Rank[root1] == Rank[root2]:
            Parent[root2] = root1
        else:
            Parent[root1] = root2
            if Rank[root1] == Rank[root2]:
                Rank[root2] = Rank[root2] + 1

def parseEdges_new(data_file):
    map_data0 = []
    with open(data_file, 'r') as data:
        for line in data:
            ##print type(data),'begin'
            line = line.strip('\n')
            ##line = data.readline()
            ###print line
            map_data0.append(line)
        ###print 'DATA_init=',map_data0
    dms = len(map_data0)-6
    ###print 'Dimension = ', dms 
    map_data0 = map_data0[5:-1]
    #print map_data0
    map_data_num = np.zeros([dms,3])
    for i in range(len(map_data0)):
        #print map_data0[i]
        line_map = str(map_data0[i]).split(' ')
        map_data_num[i][0] = int(line_map[0])
        map_data_num[i][1] = float(line_map[1])
        map_data_num[i][2] = float(line_map[2])
        
    ###print map_data_num,'haha'
    map_data = np.zeros([dms,dms])
    #print type(map_data)
    graph_produced = []
    graph_produced_double = []
    for i in range(dms):
        for j in range(dms):
            map_data[i][j] = np.sqrt((map_data_num[i][1]-map_data_num[j][1])**2+(map_data_num[i][2]-map_data_num[j][2])**2)
            if i !=j:
                graph_produced_double.append((i,j,map_data[i][j]))
            if j > i:
                graph_produced.append((i,j,map_data[i][j]))
            #Edge_len = 
    ##print map_data[0],'002'
    ###print len(graph_produced),graph_produced,'gp'
    ###print len(graph_produced_double),graph_produced_double,'gp_double'
#------------
    G=nx.Graph()
    Gd=nx.Graph()
    ##data0 = pd.read_csv(graph_file, header=None)
    ##data = data0.values[1:]
    ##print type(data)
    #total_V = data0.values[0][0].split(' ')[0]
    #total_E = data0.values[0][0].split(' ')[1]
    Edge_list = set()
    Edge_list_double = set()
    #print len(data)
    ##graph = np.zeros([len(data),3],dtype=int)
    ###for i in range(len(data)):
    for i in range(len(graph_produced)):
        ###data_temp = data[i][0].split(' ')
        ##graph[i,0] = int(data_temp[0])
        ###u = int(data_temp[0])
        u = int(graph_produced[i][0])
        ##graph[i,1] = int(data_temp[1])
        ###v = int(data_temp[1])
        v = int(graph_produced[i][1])
        ##graph[i,2] = int(data_temp[2])
        weight = float(graph_produced[i][2])
        if G.has_edge(u, v):
            G[u][v]['weight'] = min(G[u][v]['weight'],weight)
        else:
            G.add_edge(u, v, weight = weight)
    ###print 'Graph of library is:', G.number_of_edges(),'G is',G

    ###for i in range(len(data)):
    for i in range(len(graph_produced)):
        #data_temp = data[i][0].split(' ')
        #u = int(data_temp[0])
        #v = int(data_temp[1])
        u = int(graph_produced[i][0])
        v = int(graph_produced[i][1])
        Edge_list.add((G[u][v]['weight'],u,v))
    ##print Edge_list
    #Edge_list = sorted(Edge_list)
    #--------------------------------
    '''
    for i in range(len(graph_produced_double)):
        ud = int(graph_produced_double[i][0])
        vd = int(graph_produced_double[i][1])
        weightd = float(graph_produced_double[i][2])
        if Gd.has_edge(ud, vd):
            Gd[ud][vd]['weight'] = min(G[ud][vd]['weight'],weightd)
        else:
            Gd.add_edge(ud, vd, weight = weightd)
    for i in range(len(graph_produced_double)):
        ud = int(graph_produced_double[i][0])
        vd = int(graph_produced_double[i][1])
        Edge_list_double.add((Gd[ud][vd]['weight'],ud,vd))
    #--------------------------------
    '''
    return Edge_list, dms, graph_produced_double#, Edge_list_double and dimensions of graph: the number of nodes of the graph.

def computeMST(Edge_list):
    Edge_list = sorted(Edge_list)
    nodes_pool = set()
    for i in range(len(Edge_list)):
        nodes_pool.add(Edge_list[i][1])
        nodes_pool.add(Edge_list[i][2])
    nodes_num = len(nodes_pool)
    #print nodes_num,'haha'
    for j in range(nodes_num):
        MakeSet(j)
    MST=set()
    total_weight = 0
    for edge in Edge_list:
        # Edge_list has beed sorted by weight already!
        weight = edge[0]
        node1 = edge[1]
        node2 = edge[2]
        if Find(node1) != Find(node2):
            Union(node1,node2)
            MST.add(edge)
            total_weight = total_weight + weight
    return total_weight,MST # MST is a set storing set((weight,u,v))

#def find_cycle(u_new,v_new,MSTmatrix,):
def find_cycle(G,s,S=None):
    # DFS algo
    if S is None:
        S = set()
    S.add(s) # Record the node we have visited before.
    for u in G[s]:
        if u in S:
            continue
        find_cycle(G, u, S)
    return S


def MST2_alg(data_file):
    ###G, Gd = parseEdges_new(data_file)
    G, dms, FullGraph_Double = parseEdges_new(data_file)
    ###print G,'dms:',dms
    ###print Gd
    MSTweight,MST = computeMST(G)
    ###MSTweightd,MSTd = computeMST(Gd)
    ###print MSTweight
    ###print MSTweightd
    ###print MST # MST is the edge_weights and their corresponding vertices!
    ###print len(MST)
    ###print MSTd
    ###print len(MSTd)
    ###print len(MST)
    TSP_result = []
    return MST, dms, FullGraph_Double
    '''
    map_data = []
    with open(data_file, 'rb') as data:
        for line in data:
            line = line.strip('\n')
            ##line = data.readline()
            #print line
            map_data.append(line)
        print 'DATA_init=',map_data
    dms = len(map_data)-6
    print 'Dimension = ', dms
    '''
    
def dfs_it(graph_dfs, root):
    vtd = []
    stack = [root, ]
    while stack:
        node = stack.pop()
        if node not in vtd:
            vtd.append(node)
            stack.extend([x for x in graph_dfs[node] if x not in vtd])
    return vtd






def main_work(data_file,ipf,cutoff_time,randSeed):
    MST, dms, FullGraph_Double = MST2_alg(data_file)
    a, b = data_file.split(".")
    city_name = a.split("/")
    city = city_name[-1]
    #print TSP_result, dms, 'cp22'
    # graph_dfs must be sorted by ascending order for each node's connection.

    graph_dfs_n = {}
    for i in range(dms):
        graph_dfs_n[i] = []
    #print list(MST)[0],list(MST)[1]
    MST_list = list(MST)
    for i in range(dms):
        for MST_edge in MST_list:
            if i == MST_edge[1]:        
                graph_dfs_n[i].append(MST_edge[2])
            if i == MST_edge[2]:
                graph_dfs_n[i].append(MST_edge[1])
    for i in range(dms):
        graph_dfs_n[i].sort()
                
    ###print graph_dfs_n
    start_time = time.time()
    TOTAL_PATH_WEIGHT = 1000000000000000
    random.seed(randSeed)
    
    trace_line = []
    while (time.time()-start_time) < cutoff_time:
        randomnum = random.randint(0,dms-1)
        #print randomnum, 'randomnum'
        visited_dfs = dfs_it(graph_dfs_n, randomnum)
        visited_dfs.append(visited_dfs[0])
        ###print visited_dfs
        #print visited_dfs_cycle
        ###print FullGraph_Double
        FullGraph_Double_Dict = {}
        for i in range(len(FullGraph_Double)):
            FullGraph_Double_Dict[str([FullGraph_Double[i][0], FullGraph_Double[i][1]])] = FullGraph_Double[i][2]
        ###print FullGraph_Double_Dict
        path_len = []
        for i in range(len(visited_dfs)-1):
            path_len.append(FullGraph_Double_Dict[str([visited_dfs[i], visited_dfs[i+1]])])
        path_len_int = [int(x) for x in path_len]
        path_len_total = sum(path_len_int)
        if path_len_total < TOTAL_PATH_WEIGHT:
            visited_dfs_temp = visited_dfs
            path_len_int_temp = path_len_int
            trace_line.append([(time.time()-start_time)*1000, path_len_total])
        TOTAL_PATH_WEIGHT = min(TOTAL_PATH_WEIGHT, path_len_total)

        
    #print path_len_int
    #print path_len_total
    
    
    

    output_f = 'Output/'+city+'_MST_'+str(cutoff_time)+'_'+str(randSeed)+'.sol'
    with open(output_f, 'w') as of:
        of.write(str(TOTAL_PATH_WEIGHT)+"\n")
        for i in range(len(visited_dfs_temp)-1):
            of.write(str(visited_dfs_temp[i])+' '+str(visited_dfs_temp[i+1])+' '+str(path_len_int_temp[i])+"\n")
    trace_f = 'Output/'+city+'_MST_'+str(cutoff_time)+'_'+str(randSeed)+'.trace'
    with open(trace_f, 'w') as trace_of:
        for tl in trace_line:
            trace_of.write(str("%.2f" % tl[0])+'ms,'+' '+str(tl[1])+' '+'\n')





if __name__ == "__main__":
    #input_f = ['Atlanta', 'Boston', 'Champaign', 'Cincinnati', 'Denver', 'NYC', 'Philadelphia', 'Roanoke', 'SanFrancisco', 'Toronto', 'UKansasState', 'UMissouri']
    input_f = ['Atlanta']
    if not os.path.isdir("./Output"):
        os.makedirs('Output')
    for ipf in input_f:
        main_work(data_file,ipf, cutoff_time=1, randSeed=0)
