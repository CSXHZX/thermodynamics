
# coding: utf-8

# In[ ]:

groups = { # main: {secondary i:[molecule,R,Q],secondary i+1:[molécula,R,Q],etc}
    1: {1:['CH3',0.9011,0.848],2:['CH2',0.6744,0.540],3:['CH',0.4469,0.228],4:['C',0.2195,0]},
    2: {5:['CH2CH',1.3454,1.176],6:['CHCH',1.1167,0.867],7:['CH2C',1.1173,0.988],8:['CHC',0.8886,0.676],70:['CC',0.6605,0.485]},
    3: {9:['ACH',0.5313,0.400],10:['AC',0.3652,0.120]},
    4: {11:['ACCH3',1.2663,0.968],12:['ACCH2',1.0396,0.660],13:['ACCH',0.8121,0.348]},
    5: {14:['OH',1.000,1.200]},
    6: {15:['CH3OH',1.4311,1.432]},
    7: {16:['H2O',0.9200,1.400]},
    8: {17:['ACOH',0.8952,0.680]},
    9: {18:['CH3CO',1.6724,1.488],19:['CH2CO',1.4457,1.180]},
    10: {20:['CHO',0.9980,0.948]},
    11: {21:['CH3COO',1.9031,1.728],22:['CH2COO',1.6764,1.420]},
    12: {23:['HCOO',1.2420,1.188]}
}
g_groups = {}
for i in groups.values():
    g_groups.update(i)

global g_groups
    
a_mn = pd.read_csv('a-mn.csv', sep=';', index_col=0)
a_mn.head()

class molecule():
    def __init__(self,molGroups):
        self.g = molGroups.keys()
        self.v = molGroups.values()
        self.groups = molGroups        
        
    def rql(self):
        z,r,q = 10,0,0        
        num = len(self.g)
        if num > 1:
            for i,k in zip(self.g,self.v):
                r += g_groups[i][1] * k
                q += g_groups[i][2] * k
        else:
            r = g_groups[tuple(self.g)[0]][1] * tuple(self.v)[0]
            q = g_groups[tuple(self.g)[0]][2] * tuple(self.v)[0]
        self.l = 0.5 * z * (r - q) - (r - 1)
        self.r,self.q = r,q
        return(self.r,self.q,self.l)
    
    def phi_theta(self, fluid):        
        self.x, fluid.x = x, (1-x)
        phi,theta = {},{}       
        phi[self] = self.r * x / (self.r * self.x + fluid.r * fluid.x)
        phi[fluid] = fluid.r * (1 - x) / (self.r * self.x + fluid.r * fluid.x)
        theta[self] = self.q * x / (self.q * x + fluid.q * (1 - x))
        theta[fluid] = fluid.q * (1 - x) / (self.q * self.x + fluid.q * fluid.x)
        self.phi, self.theta = phi[self], theta[self]
        fluid.phi, fluid.theta = phi[fluid], theta[fluid]
        return(self.phi, self.theta, fluid.phi, fluid.theta)
    
    def combinatorial(self, fluid):
        self.phi_theta(fluid)
        self.x, fluid.x = x, (1-x)
        self.gamma_c_ln = np.log(self.phi / self.x) + 5 * self.q * np.log(self.theta / self.phi) +                     self.l - (self.phi / self.x) * ( fluid.x * fluid.l + self.x * self.l )
        fluid.gamma_c_ln = np.log(fluid.phi / fluid.x) + 5 * fluid.q * np.log(fluid.theta / fluid.phi) +                     fluid.l - (fluid.phi / fluid.x) * ( self.x * self.l + fluid.x * fluid.l )
        self.gamma = np.exp(self.gamma_c_ln)
        fluid.gamma = np.exp(fluid.gamma_c_ln)
        return(self.gamma_c_ln, fluid.gamma_c_ln)    
        
    def psi_mod(self, fluid, pairs):
        self.p = tuple([(i,j) for i in pairs for j in pairs])
        a, psi = {}, {}
        for i in self.p:
            j = (i[0]-1,i[1]-1)
            a[i] = a_mn.iat[j]
            psi[i] = np.exp(-a_mn.iat[j] / T)
        self.a_k, self.a_v, self.a = a.keys(), a.values(), a
        self.psi_k, self.psi_v, self.psi = psi.keys(), psi.values(), psi
        self.psi_matrix = np.reshape(np.array([self.psi[i] for i in self.p]),(len(pairs),len(pairs)))
        return(self.a_k, self.a_v, self.a)
        return(self.psi_k, self.psi_v, self.psi)
        return(self.p)
        return(self.psi_matrix)
    
    def X_mod(self, fluid):            
        self.x, fluid.x = x, (1-x)
        den = sum([self.groups[k] for k in list(self.g)]) * self.x +               sum([fluid.groups[k] for k in list(fluid.g)]) * fluid.x
        X = []
        g_list = list(set([i for j in [list(self.g),list(fluid.g)] for i in j]))
        for k in sorted(g_list):
            nom = 0
            if k in list(self.g):
                nom += self.groups[k] * self.x
            if k in list(fluid.g):
                nom += fluid.groups[k] * fluid.x
            X.append(nom / den)      
        self.X = X       
        return(self.X)
    
    def X_mod_i(self, fluid):   
        v1, v2 = list(self.v), list(fluid.v)
        Xi = np.array(v1) / sum(v1), np.array(v2) / sum(v2)
        self.Xi = Xi
        return(self.Xi)
    
    def Theta_mod(self, fluid):
        Xx = self.X_mod(fluid)
        g_list = list(set([i for j in [list(self.g),list(fluid.g)] for i in j]))
        nom = np.multiply(np.transpose(Xx) ,np.array([g_groups[k][2] for k in sorted(g_list)]))
        den = sum(nom)
        self.Theta = np.divide(nom, den)
        return(self.Theta)
    
    def Theta_mod_i(self, fluid):
        X_i = self.X_mod_i(fluid)
        g1, g2 = list(self.g), list(fluid.g)
        Q1, Q2 = [g_groups[i][2] for i in g1], [g_groups[i][2] for i in g2]
        #print(X_i,'\n',Q1,'\n',Q2)
        nom1, nom2 = np.multiply(X_i[0],Q1), np.multiply(X_i[1],Q2)
        den1, den2 = sum(nom1), sum(nom2)
        self.Theta1, self.Theta2 = nom1 / den1, nom2 / den2
        return(self.Theta1, self.Theta2)
    
    def Gamma_k(self, fluid):        
        t = self.Theta_mod(fluid)
        g_list = list(set([i for j in [list(self.g),list(fluid.g)] for i in j]))
        Q = [g_groups[k][2] for k in sorted(g_list)]
        YY = self.psi_matrix
        nTheta = np.reshape(t,(len(YY),1))
        pt = np.multiply(YY,nTheta)
        pt2 = np.multiply(YY,t)
        denom = np.sum(pt,axis=0)
        gam = 1 - np.log(denom) - np.sum(np.divide([i for i in pt2],denom),axis=1)
        self.gamma_k_ln = np.multiply(Q,gam)
        return(self.gamma_k_ln)
    
    def Gamma_k_i(self, fluid):
        t1, t2 = self.Theta_mod_i(fluid)    
        g1, g2 = list(self.g), list(fluid.g)
        Q1, Q2 = [g_groups[i][2] for i in g1], [g_groups[i][2] for i in g2]
        YY = self.psi_matrix
        if len(YY) == 3:
            a, b = np.multiply([YY[0,0],YY[2,0]], t1), np.multiply([YY[0,2],YY[2,2]], t1)
            c, d = np.multiply([YY[0,0],YY[1,0]], t2), np.multiply([YY[0,1],YY[1,1]], t2)
            e, f = np.multiply([YY[0,0],YY[0,2]], t1), np.multiply([YY[2,0],YY[2,2]], t1)
            g, h = np.multiply([YY[0,0],YY[0,1]], t1), np.multiply([YY[1,0],YY[1,1]], t2)
            gamma_1 = Q1[0] * (1 - np.log(sum(a)) - e[0] / sum(a) - e[1] / sum(b) )
            gamma_2 = Q1[1] * (1 - np.log(sum(b)) - f[0] / sum(a) - f[1] / sum(b) )
            gamma_3 = Q2[0] * (1 - np.log(sum(c)) - g[0] / sum(c) - g[1] / sum(d) )
            gamma_4 = Q2[1] * (1 - np.log(sum(d)) - h[0] / sum(c) - h[1] / sum(d) ) 
        else:
            a, b = np.multiply([YY[0,0],YY[1,0]], t1), np.multiply([YY[0,1],YY[1,1]], t1)
            c, d = np.multiply([YY[0,0],YY[1,0]], t2), np.multiply([YY[0,1],YY[1,1]], t2)
            e, f = np.multiply([YY[0,0],YY[0,1]], t1), np.multiply([YY[1,0],YY[1,1]], t1)
            g, h = np.multiply([YY[0,0],YY[0,1]], t1), np.multiply([YY[1,0],YY[1,1]], t2)
            gamma_1 = Q1[0] * (1 - np.log(sum(a)) - e[0] / sum(a) - e[1] / sum(b) )
            gamma_2 = Q1[1] * (1 - np.log(sum(b)) - f[0] / sum(a) - f[1] / sum(b) )
            gamma_3 = Q2[0] * (1 - np.log(sum(c)) - g[0] / sum(c) - g[1] / sum(d) )
            gamma_4 = Q2[1] * (1 - np.log(sum(d)) - h[0] / sum(c) - h[1] / sum(d) )
        return(gamma_1,gamma_2,gamma_3,gamma_4)
    
    def residual(self, fluid):
        v1, v2 = list(self.v), list(fluid.v)
        k_i = self.Gamma_k_i(fluid)
        g2 = np.array([self.Gamma_k(fluid)[0],self.Gamma_k(fluid)[1]])
        if len(self.Gamma_k(fluid)) == 3:
            g1 = np.array([self.Gamma_k(fluid)[0],self.Gamma_k(fluid)[2]])
        else:
            g1 = g2        
        gg1 = g1-k_i[:2]
        gg2 = g2-k_i[2:]
        gamma_r_ln_1 = sum(np.multiply(v1,gg1))
        gamma_r_ln_2 = sum(np.multiply(v2,gg2))
        gamma_r_1, gamma_r_2 = np.exp(gamma_r_ln_1), np.exp(gamma_r_ln_2)
        return(gamma_r_ln_1, gamma_r_ln_2)
    
    def Gamma(self, fluid):
        yr = self.residual(fluid)
        #print(x,'\n',yr,'\n')
        yc = self.combinatorial(fluid)
        #print(yc,'\n')
        y_ln = np.add(yr, yc)
        #print(y_ln)
        self.y = np.exp(y_ln)
        return(self.y)

