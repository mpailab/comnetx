import random
from collections import deque,Counter
import copy
import time 
import sys
import math
from scipy.stats import expon
from sklearn.metrics.cluster import normalized_mutual_info_score
from GraphTools import GraphTolls




class Fast_local_Move_IG(GraphTolls) :
    def __init__( self, Nb, Beta,path):    
        self.Nb = Nb
        self.Beta = Beta
        self.Mod_val = 0
        super(Fast_local_Move_IG, self).__init__(path)

    def expon (self , value , teta):
        x = (1/teta)*(value)
        p =float(math.exp(x))
        return p      
    

                         
    def GCH( self):
         
        vertex_list = list(self.graph.nodes())
        node = random.choice(vertex_list)
        #print("nnnn",node)
        com_id = 0
        self.membership[node] = com_id
        self.DegCom[com_id] = self.Degree[node]
        #print(self.Degree)
        vertex_list.remove(node)
        for node in  vertex_list:    
            comm_ngh = super().neigh_comm( node)
            MAX_Q = 0
            pos = -1
            for com, Kbv in comm_ngh.items():    
                db = self.DegCom[com]
                delta_Q =  Kbv - self.Degree[node]/(2.*self.m)*db
                #print(delta_Q)
                if delta_Q > MAX_Q:
                    MAX_Q = delta_Q
                    pos = com
                else :
                    delta_Q = 0
            
            if MAX_Q > 0:
                super().insert_node( node, pos, comm_ngh.get( pos , 0))
            else:
                com_id = com_id + 1
                super().insert_node(node , com_id, comm_ngh.get(com_id, 0))
                                     
        
        return  self.membership 

    def Destruction( self):       
        drop_node = []
        merg_node = []
        cut_len = int(len(self.membership)* float(self.Beta)) 
        index_community = random.sample( list(self.membership.keys()), cut_len )
        #print("list", cut_len, index_community)
        #print("degcomunity", self.DegCom)
        #print("mm",self.membership)
        for al in index_community:
            com_id = self.membership[al]
            wgh = super().neigh_comm(al)    
            super().delet_node(al, com_id, wgh.get(com_id, 0.))
            if self.internal[com_id] == 0.:
                del self.internal[com_id]            
                      
        #merg_node = [ nod for nod in self.Node_list if nod not in index_community] 
        return  self.membership, index_community
    
    def Reconstruction( self, drop_node):
        self.__affect_node(drop_node)
        self.__randomcom()
        self.__merge_community()                 
                                                                          
        return  self.membership


    def FL_move(self ):
        Qv = deque([ i for i in self.graph.nodes()])
        random.shuffle(Qv)
        while Qv:
            vsele = Qv.popleft()
            degree = self.Degree[vsele]
            #qum = 0
            com_befor = self.membership[vsele]
            ngh_com = super().neigh_comm(vsele)
            #print("combefor", com_befor,"nghhhh", ngh_com)
            dvc = super().ngh_node(vsele, com_befor)
            devc = self.DegCom[com_befor]
            maxq = 0
            #print("vsele", vsele)
            #print(self.m)
            #dd = (degree) / ( self.m * 2.)
            m_com = com_befor             
            for com, dvcp in ngh_com.items() : 
                #dvcp = ngh_com[com]
                #if com != com_befor:
                devcp = self.DegCom[com]
                deq = (1/self.m) * ( dvcp-dvc )- ((degree)/(2.*self.m**2.)) *( devcp-devc+degree )
                #deq = dvcp - (devcp)* dd
                #print("deq",deq) 
                if deq > maxq :
                    maxq = deq
                    m_com = com
                        
            super().delet_node( vsele, com_befor, ngh_com.get( com_befor, 0.))            
            super().insert_node( vsele, m_com, ngh_com.get( m_com, 0.))            
            if m_com != com_befor:                                       
                for veg in self.graph[vsele]:
                    if self.membership[veg] != m_com and veg not in Qv:    
                        Qv.append(veg)       
                                      
        return self.membership
    
    def __affect_node(self, drop_node = None):
        random.shuffle(drop_node)
        for node in  drop_node:    
            comm_ngh = super().neigh_comm( node)
            MAX_Q = 0
            pos = -1
            for com, Kbv in comm_ngh.items():    
                db = self.DegCom[com]
                delta_Q =  Kbv - self.Degree[node]/(2.*self.m)*db
                if delta_Q > MAX_Q:
                    MAX_Q = delta_Q
                    pos = com
                else :
                    delta_Q = 0
            
            if MAX_Q > 0:
                super().insert_node( node, pos, comm_ngh.get( pos, 0.))
            else:
                com_id = super().generate_random_not_in_list(set(self.membership.values()))
                super().insert_node( node, com_id, comm_ngh.get( com_id, 0.))
                
    def __randomcom ( self):
        
        for vsele in self.graph.nodes():
            degree = self.Degree[vsele]
            qum={}
            com_befor = self.membership[vsele]
            ngh_com = super().neigh_comm( vsele)
            #print( vsele, com_befor, ngh_com)
            dvc = super().ngh_node(vsele, com_befor)
            devc = self.DegCom[com_befor]              
            for com  in ngh_com :
                if com != com_befor:
                    dvcp = ngh_com[com]
                    devcp = self.DegCom[com]
                    deq = (1/self.m)*(dvcp-dvc)-(degree/(2.*self.m**2.))*(devcp-devc+degree)
                    if deq > 0: 
                        qum[com] = deq
            
            #print("q in reconstruction", qum)
            if len(qum) > 0:  
                prb = [ self.expon( i, 0.1) for k,i in qum.items() ]
                com_n = super().weighted_choice( list(qum.keys()), prb)
                #print(com_n , ngh_com)
                super().delet_node( vsele, com_befor, ngh_com.get( com_befor, 0.))
                super().insert_node( vsele, com_n, ngh_com.get( com_n, 0.))        
        
    def __merge_community( self):
        modified = True
        while modified :
            modified = False         
            community = set(self.membership.values())
            #print(community, self.membership, self.DegCom)
            visited = { i : False for i in community }
            for com1 in community:
                if visited[com1] == False:
                    maxqq = 0
                    poss = -1
                    ngh_comm = super().com_ngh_com(com1)
                    #print(com1 , ngh_comm)
                    for comm ,Kbv in ngh_comm.items():
                        delta_Qq =  Kbv  - ( self.DegCom[com1] * self.DegCom[comm] ) / ( 2. * self.m )
                    #print("delta",delta_Q)
                        if delta_Qq > maxqq :
                            maxqq= delta_Qq
                            poss = comm
                            
                    if maxqq > 0:
                        modified = True
                        self.merge_com( com1, poss)              
                        self.DegCom[com1] = self.DegCom[com1] + self.DegCom[poss]
                        self.internal[com1] = self.internal[com1] + self.internal[poss]+ ngh_comm.get(poss, 0.)
                        del self.DegCom[poss]
                        del self.internal[poss]
                        visited[poss] = True
  
    def Run_FMLIG (self):
        start = time.time()
        soltion = self.GCH()
        #print("solution", self.membership)
        #print(self.DegCom)
        soltion = self.FL_move()
        best_solution = copy.copy(soltion)
        #print(" internal ", self.internal)
        Q_best = super().modularity()
        #print("qqqqqqqqqqqq",Q_best)
        T_init = 0.025 * Q_best
        T = T_init
        nb_iter = 0
        status_list = []
        while nb_iter < self.Nb:
            Q1 = super().modularity()
            #print("q1q1qqqqqqq1", Q1)
            incumbent_solution = copy.copy(soltion)
            #print(self.membership)
            #soltion = self.FL_move( weight = 'weight')
            #print(super().modularity( weight ='weight'))
            #print("after internal",self.membership, self.internal)
            soltion, drop_nodes = self.Destruction()
            #print("after destruction", self.membership, drop_nodes , self.internal)
            soltion = self.Reconstruction( drop_nodes)
            #print("after reconstruction",super().modularity())
              
            soltion = self.FL_move()   
            #print("after fast", soltion)
            #print("after the fast loacl")                      
            Q2 = super().modularity()
            #print("Q2", Q2)
            #for data in c_graph.edges( data= True):
            #    print(data)
                
            if Q2 > Q_best:
                best_solution = copy.copy(soltion)
                Q_best = Q2    
            
            P = random.random()
            if Q2 < Q1 and P > math.exp((Q2 - Q1)//T):
                soltion = copy.copy(incumbent_solution)
                T = T_init

            elif Q2  >=  Q1:
                T = T_init
              
            else:
                T = T*0.9
            
            nb_iter = nb_iter + 1
        
        #print(status_list)
        end = time.time()
        t = end - start
        #partition = status_list[0].copy()
        #for index in range( 1, len(status_list)-1):
            #for node, community in partition.items():
                #partition[node] = status_list[index][community]
 
        return Q_best, best_solution, t
        

def de_main():
    path = sys.argv[1]
    Number_iter = int(sys.argv[2])
    Beta = sys.argv[3]
    data = GraphTolls(path)
    graph = data.Read_Graph()
    NMI_list = []
    Time_list = [] 
    Q_list = []
    nb_run = 0
    while nb_run < int(sys.argv[5]) :
        print("rb",nb_run)
        communities = Fast_local_Move_IG( Number_iter, Beta, path)
        mod,community,tim = communities.Run_FMLIG() 
        #print(community)
        Q_list.append(mod)
        Time_list.append(tim)
        #label = communities.lebel_node(community)  
        if sys.argv[4]!= 'None':
            True_partition = data.Read_GroundTruth(sys.argv[4])
            NMI = normalized_mutual_info_score(True_partition,community)
            NMI_list.append(NMI)
            
            
        elif sys.argv[4] == 'None':
            pass            
      
        nb_run = nb_run +1
            
    if sys.argv[4] != 'None':
        Q_avg = communities.avg(Q_list)
        Q_max = communities.max(Q_list)
        Q_std = communities.stdev(Q_list)
        NMI_max = communities.max(NMI_list)
        time_run = communities.avg(Time_list)
        return NMI_max, Q_max, Q_avg, Q_std, time_run     
    elif sys.argv[4] == 'None':
        Q_avg = communities.avg(Q_list)
        Q_max = communities.max(Q_list)
        Q_std = communities.stdev(Q_list)
        time_run = communities.avg(Time_list)
        return Q_max, Q_avg, Q_std, time_run

if __name__ == '__main__':

   
    if  sys.argv[4] != 'None':
        NMI_max,Q_Max,Q_avg ,Q_std,time_run = de_main()
        print("NMI_max",NMI_max)
        print("the value of Q_max",Q_Max)
        print("the value of Q_avg",Q_avg)
        print("the value of Q_std",Q_std)
        print("time ",time_run)
    elif sys.argv[4] == 'None' :
        Q_max, Q_avg, Q_std,time_run = de_main()
        print("the value of Q_max",Q_max)
        print("the value of Q_avg",Q_avg)
        print("the value of Q_std",Q_std)
        print("the value of time ",time_run) 
     
