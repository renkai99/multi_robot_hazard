import numpy as np
from Set import Set

class S_set(Set):
    def __init__(self,path_planner):
        self.Q=path_planner.Q
        self.QX=path_planner.QX
        self.Tau_Q=path_planner.Tau_Q
        self.illegals=self.generate_illegals()
        self.s_H='s_H'
        if self._is_goal_collection(self.Q.goal):
            self.s_G=[(set(self.Q.targets),goal_cell) for goal_cell in self.Q.goal]
        else:
            self.s_G=(set(self.Q.targets),self.Q.goal)
        S=self.generate_S()
        super(S_set,self).__init__(S)

    def _is_goal_collection(self,goal):
        if isinstance(goal,(list,set)):
            return len(goal)>0 and all(isinstance(e,tuple) and len(e)==2 for e in goal)
        if isinstance(goal,tuple):
            if len(goal)==2 and all(isinstance(e,(int,np.integer)) for e in goal):
                return False
        return False

    def generate_illegals(self):
        illegals=[]
        for q in self.Q:
            for x in self.Q.targets:
                if x not in q:
                    qx=(q,x)
                    illegals=illegals+[self.QX.index(qx)]
        return illegals

    def generate_S(self):
        QX_legal=[e for i,e in enumerate(self.QX) if i not in self.illegals]
        S=[self.s_H]+QX_legal
        return S 

    def get_QX_mask(self):
        QX_inds=np.ones(len(self),dtype=bool)
        QX_inds[self.index(self.s_H)]=False
        return QX_inds

    def get_H_mask(self):
        H_inds=np.zeros(len(self),dtype=bool)
        H_inds[self.index(self.s_H)]=True
        return H_inds

    def get_G_mask(self):
        G_inds=np.zeros(len(self),dtype=bool)
        if isinstance(self.s_G,list):
            for s_G_i in self.s_G:
                G_inds[self.index(s_G_i)]=True
        else:
            G_inds[self.index(self.s_G)]=True
        return G_inds
