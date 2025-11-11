import math
import re
import sys 
import networkx as nx
import random

class GraphTolls:
    
    
    def Read_Graph(self, Path):
        
        if Path[len(Path)-3: ] == 'txt' or Path[len(Path)-3: ] == 'dat':
            Graph = nx.read_edgelist(Path, nodetype = int)
        elif Path[len(Path)-3: ] == 'gml':
            Graph = nx.read_gml(Path,label = 'id')
        else :
            raise TypeError (" the type of graph is not suportable or not no such file or directory")

        return Graph
    
    def Reve(self,x):
        sa = x.split()[::-1]
        l = []
        for i in sa:
            l.append(i)

        l=('  '.join(l))
        return l

    def Remove_Revers(self,path):
        with open(path, "r") as file:
            lines = file.readlines()
            result=[]
            for xa in lines:
                xa=re.sub(r'\s','  ',xa)
                if self.reve(xa) not in result:
                   result.append(xa.strip())   

        return(result)

    def Remove_Dublicate (self,path):
        pathw = path[ : -3]+'txt' 
        lines = self.remove_revers(path)
        with open(pathw, 'w') as f:
            for line in lines:
               f.write(line)
               f.write('\n')


    def Read_GroundTruth(self,path):
        with open(path, "r") as file:
            lines = file.readlines()
            result = []
            for x in lines:
                x = x.rstrip()
                result.append(x.split()[1])

        true_partion = [int(x)for x in result]
        return true_partion
    
    def Modularity(self, G, clusters):
        m = G.number_of_edges()
        n= G.number_of_nodes()
        label = sorted([i for i in G.nodes()])
        node_list = sorted([i for i in G.nodes()])
        for index,no in enumerate (label):
            for i in range(len(clusters)):
               if no in clusters[i]:
                    label[index] = i
 
        Q = 0
        for index,u in enumerate(node_list):
            if label[index] == None:
                continue
            for ind,v in enumerate(node_list):
                if label[ind] == None:
                    continue
                
                if v in list(nx.all_neighbors(G,u)):
                    Auv = 1
                else:
                    Auv = 0
                
                if  label[index] == label[ind] :
                    Q +=  1/(2*m) * (Auv - G.degree(u)*G.degree(v) / (2*m))
    

        return Q 

    def Is_Intersiction(self,communities):
        dupes = []
        flat = [item for sublist in communities for item in sublist]
        for f in flat:
            if flat.count(f) > 1:
                if f not in dupes:
                    dupes.append(f)

        if dupes:
            return True
        else:
            return False   
        
    def sum(self,arg):
        if len(arg) < 1:
            return None
        else:
            return sum(arg)

    def count(self,arg):
        return len(arg)
  
    def min(self,arg):
        if len(arg) < 1:
            return None
        else:
            return min(arg)
  
    def max(self,arg):
        if len(arg) < 1:
            return None
        else:
            return max(arg)
  
    def avg(self,arg):
        if len(arg) < 1:
            return None
        else:
            return sum(arg) / len(arg)   
  
    def median(self,arg):
        if len(arg) < 1:
            return None
        else:
            arg.sort()
            return  arg[len(arg) // 2]
  
    def stdev(self,arg):
        if len(arg) < 1 or len(arg) == 1:
            return None
        else:
            avg = self.avg(arg)
            sdsq = sum([(i - avg) ** 2 for i in arg])
            stdev = (sdsq / (len(arg) - 1)) ** .5
            return stdev
  
    def percentile(self, arg):
        if len(arg) < 1:
            value = None
        elif (arg >= 100):
            sys.stderr.write('ERROR: percentile must be < 100.  you supplied: %s\n'% arg)
            value = None
        else:
            element_idx = int(len(arg) * (arg / 100.0))
            self.arg.sort()
            value = self.arg[element_idx]
        return value  
    
    def select_edge_betw(self,g,vert,commu):
        Edg_betw = []
        for i in list(g.neighbors(vert)):
            if i in commu :
                Edg_betw.append(i)
        k = len(Edg_betw)
        
        return k 
   
