import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.cluster.hierarchy import DisjointSet
from scipy.stats import gaussian_kde
import networkx as nx
import os
import pickle

#Quelques constantes physiques
m_e = 9.10938356e-31 #en kg
hbar = 1.0545718e-34 #en J.s
e = 1.602176634e-19 #en C
kB = 1.380649e-23 #en J/K
T = 300 #Température en K

#Constantes de la simulation
phi = 2*e # Taille de la barrière, en eV
a = 1e-9 # Taille de chaque maille élémentaire, en m

class Simulation:

    def __init__(self,N=30,sigma=0.3,n_el_initial=100,N_iterations=500,prob_deb=0.2,dropout=0.,coupe=0.5,sigma_elec=1.4, mode="normal"):
        """Initialise les paramètres de la simulation, puis la grille complète."""
        self.N = N
        self.sigma = sigma
        self.n_el_initial = n_el_initial
        self.N_iterations = N_iterations
        self.prob_deb = prob_deb
        self.dropout = dropout
        self.coupe = coupe
        self.sigma_elec = sigma_elec
        self.n_el = n_el_initial
        self.mode = mode
        self.li_va_globales = ['N','a','sigma','n_el_initial','N_iterations','prob_deb','dropout','coupe','sigma_elec', 'phi']

        self.z = np.zeros((self.N,self.N,2)) # Deux grilles : l'une suit les abscisses, l'autre les ordonnées
        self.x = np.random.normal(0,self.sigma/2,size = (self.N,self.N))
        self.y = np.random.normal(0,self.sigma/2,size = (self.N,self.N))
        self.x = np.clip(self.x,-self.coupe,self.coupe) #On empêche les points de sortir de leur case
        self.y = np.clip(self.y,-self.coupe,self.coupe) #Idem
        bruit = np.stack((self.x,self.y),axis=-1) 
        self.z += bruit #Génère le bruit gaussien

        self.electrons = np.full((self.N,self.N),fill_value=0) #Trace de la position des électrons à tout instant sur la grille
        self.traces = {i : [] for i in range(self.n_el_initial)} #Dictionnaire des chemins des électrons à une phase donnée
        self.sites = np.random.choice([0, 1], size=(self.N, self.N), p=[self.dropout, 1 - self.dropout])
        if self.dropout < 1e-11: #Pour être sûr que l'absence de dropout soit pris en compte
            self.dropout = 0. #Pour que cela paraisse bien dans le fichier texte généré par la simulation
            self.sites = np.ones((self.N, self.N), dtype = float) #Pas de dropout pour les tests

        self.proj  = np.zeros(self.N,dtype = float) #Liste des projections
        for i in range(N):
            if self.sites[i,0]==1:
                self.proj[i] = abs(self.y[i,0])
        s = np.sum(self.proj)
        if s > 0:
            self.proj /= s

    
    """
    =====================================================================================================================
                                                Initialisation de la grille
    =====================================================================================================================
    """

    def reinitialise(self):
        """
        Réinitialise la grille, les électrons et les traces.
        """
        N=self.N
        sigma = self.sigma
        n_el_initial = self.n_el_initial 
        N_iterations = self.N_iterations
        prob_deb = self.prob_deb
        dropout = self.dropout
        coupe = self.coupe
        sigma_elec = self.sigma_elec
        n_el_initial = self.n_el_initial
        mode = self.mode
        self.__init__(N,sigma,n_el_initial,N_iterations,prob_deb,dropout,coupe,sigma_elec,mode)


    def first_initialisation(self):
        """
        Réalise la toute première initialisation de la grille.
        """
        pos = np.random.choice(np.arange(self.N), size =self.n_el_initial, p = self.proj)
        for electron in range(self.n_el_initial):
            position = pos[electron]
            self.electrons[position,0] += 1
            self.traces[electron] = [(position,0)]
        
    def initialise(self,chemins):
        """
        Initialise la position des électrons lors des passes suivantes. 
        chemins : list : liste des chemins existants, récupérés de la dernière passe.
        """
        for elec in range(self.n_el_initial):
            t = np.random.choice(["Début", "Fin"], p = [self.prob_deb, 1 - self.prob_deb])
            if len(chemins)==0 or t=="Début":
                position = np.random.choice(np.arange(self.N), p = self.proj)
                self.electrons[position,0] += 1
                self.traces[elec] = [(position,0)]
            else:
                chem = np.random.choice(np.arange(len(chemins)))
                pos_chem = np.random.choice(np.arange(len(chemins[chem])))
                position = chemins[chem][pos_chem]
                self.electrons[position] += 1
                self.traces[elec] = chemins[chem][:pos_chem+1]

    """
    =====================================================================================================================
                                                Calcul de la matrice de transition
    =====================================================================================================================
    """

    def position_into_reality(self,i,j):
        """
        Transforme les indices de la matrice en coordonnées réelles.
        i : int : indice de la ligne.
        j : int : indice de la colonne.
        """
        return a*(i+self.z[i,j,0]),a*(j+self.z[i,j,1])
    
    def distance(self,i,j,k,l):
        """
        Calcule la distance entre deux points dans la grille.
        i,j : int : indices du premier point
        k,l : int : indices du deuxième point
        """
        xi,yi = self.position_into_reality(i,j)
        xj,yj = self.position_into_reality(k,l)
        return np.sqrt((xi-xj)**2 + (yi-yj)**2)

    def tension(self,V,i,j):
        """
        Calcule le potentiel en un point de la grille
        V : float : tension appliquée 
        i,j : int : indices du point
        """
        return (1-(j/self.N))*V*e
    
    def prob(self,V,i,j,k,l):
        """
        Renvoie la probabilité de passer d'une case à une autre
        V : float : tension appliquée
        i,j : int : indices de la case de départ
        k,l : int : indices de la case d'arrivée
        """
        Delta = self.distance(i,j,k,l)
        V_i = self.tension(V,i,j)
        V_f = self.tension(V,k,l)
        V = max(V_i,V_f) + phi
        if self.mode == "uniform":
            E = np.random.random()*(V-V_i) + V_i
        else:
            E = np.tanh(np.random.normal(0,self.sigma_elec))*(V-V_i)/2 + ((V+V_i)/2)
        if E<V_f:
            return 0.
        else:
            k = np.sqrt(2*m_e*(E-V_i))/hbar
            K = np.sqrt(2*m_e*(V-E))/hbar
            kappa = np.sqrt(2*m_e*(E-V_f))/hbar
            t = 4*np.exp(-kappa*Delta*1j) * (2*(1+kappa/k)*np.cosh(K*Delta) -2*(1j*kappa/K + K/(1j*k))*np.sinh(K*Delta))**(-1)
            return abs(t)**2
        
    def prob(self,V,i,j,k,l):
        """
        Renvoie la probabilité de passer d'une case à une autre
        V : float : tension appliquée
        i,j : int : indices de la case de départ
        k,l : int : indices de la case d'arrivée
        """
        Delta = self.distance(i,j,k,l)
        V_i = self.tension(V,i,j)
        V_f = self.tension(V,k,l)
        k = np.sqrt(2*m_e*phi)/hbar
        return np.exp(-k*Delta)*np.exp(e*(V_i-V_f)/(kB*T))

    def calcul_probas(self,V,i,j):
        """
        Calcule les probabilités de transition entre la case (i,j) et ses voisines.
        V : float : tension appliquée
        i,j : int : indices de la case
        """
        droite,gauche,haut,bas = 0.,0.,0.,0.
        phi = 2*1.60217662e-19
        if j<self.N-1:
            if j+1<self.N and self.sites[i,j+1] == 1:
                haut = self.prob(V,i,j,i,j+1)
            if j>0 and self.sites[i,j-1] == 1:
                bas = self.prob(V,i,j,i,j-1)
            if i+1<self.N and self.sites[i+1,j] == 1:
                droite = self.prob(V,i,j,i+1,j)
            if i>0 and self.sites[i-1,j] == 1:
                gauche = self.prob(V,i,j,i-1,j)
            sum = droite + gauche + haut + bas
            if sum==0: #Cas en théorie impossible mais qui pourrait advenir si self.dropout est trop élevé
                return 0.,0.,0.,0.
            else:
                return droite/sum,gauche/sum,haut/sum,bas/sum
        else:
            return 0.,0.,0.,0.
    
    def pp_probas(self,V,case):
        """
        Affiche les probabilités de transition entre la case (i,j) et ses voisines, utile pour déboguer les probabilités.
        V : float : tension appliquée
        case : tuple : indices de la case
        """
        droite,gauche,haut,bas = self.calcul_probas(V,case[0],case[1])
        print(f"Case ({case[0]}, {case[1]}):")
        print(f"\t Probabilité d'aller à droite : {droite}")
        print(f"\t Probabilité d'aller à gauche : {gauche}")
        print(f"\t Probabilité d'aller en haut : {haut}")
        print(f"\t Probabilité d'aller en bas : {bas}")
    
    def dico_probas(self,V,seuil=1e-10):
        """
        Renvoie un dictionnaire contenant les probabilités de chaque case
        V : float : tension appliquée
        seuil : float : seuil de probabilité pour considérer une transition comme significative, permet d'éviter les erreurs numériques
        """
        dico = {}
        for i in range(self.N):
            for j in range(self.N):
                dico[(i,j)] = {}
                if self.sites[i,j]==1:
                    droite,gauche,haut,bas = self.calcul_probas(V,i,j)
                    if droite>seuil:
                        dico[(i,j)][(i+1,j)] = droite
                    if gauche>seuil:
                        dico[(i,j)][(i-1,j)] = gauche
                    if haut>seuil:
                        dico[(i,j)][(i,j+1)] = haut
                    if bas>seuil:
                        dico[(i,j)][(i,j-1)] = bas
        return dico

    """
    =====================================================================================================================
                                                Réalisation d'une étape
    =====================================================================================================================
    """
    def calcule_intensite(self,V,chem):
        """Calcule l'intensité de la grille à partir des chemins des électrons.
        V : float : tension appliquée
        chem : list[list] : liste des chemins des électrons.
        """
        li = np.array([len(chemin) for chemin in chem])
        res = 1/(np.sum(1/li))
        return V/res

    def deplacement(self,V,elec,probs):
        """
        Déplace un électron d'une case à une autre selon les probabilités de transition.
        V : float : tension appliquée
        elec : int : indice de l'électron à déplacer
        probs : dict : dictionnaire des probabilités de transition entre les cases
        """
        i,j = self.traces[elec][-1]
        clefs = list(probs[(i,j)].keys())
        if len(clefs) != 0:
            valeurs = list(probs[(i,j)].values())
            indice = np.random.choice(np.arange(len(clefs)), p = valeurs)
            new = clefs[indice]
            self.traces[elec].append(new)
            self.electrons[i,j] -= 1
            self.electrons[new[0],new[1]] += 1

    def transition(self,V, probas):
        """
        Effectue une transition de tous les électrons sur la grille.
        V : float : tension appliquée
        probas : dict : dictionnaire des probabilités de transition entre les cases
        Renvoie :
        - l'intensité calculée
        - liste : liste des chemins des électrons qui ont atteint la fin de la grille
        - probas : dict : dictionnaire des probabilités de transition entre les cases
        Pour utilisation à l'étape suivante
        """
        for k in range(self.n_el_initial):
            self.deplacement(V,k,probas)
    
    def etape(self,V, chem, first = False):
        if first:
            self.first_initialisation()
        else:
            self.initialise(chem)
        for i in range(self.N_iterations):
            probas = self.dico_probas(V)
            self.transition(V,probas)
        liste = [chemin for chemin in self.traces.values() if chemin[-1][1]==N-1]
        return self.calcule_intensite(V,liste),liste, probas
    
    """
    =====================================================================================================================
                                                Méthodes d'affichage
    =====================================================================================================================
    """
        
    def carte_electrons(self, dossier="",titre = "Carte des électrons", fig_size = (10,10), affiche = True):
        """Affiche la carte de chaleur des électrons sur la grille.
        dossier : str : chemin du dossier où enregistrer la figure. Par défaut, la figure n'est pas enregistrée.
        titre : str : titre de la figure.
        fig_size : tuple : taille de la figure. Par défaut, (10, 10).
        affiche : bool : si True, la figure est affichée. Par défaut, elle est affichée.
        """
        plt.figure(figsize=fig_size)
        plt.xticks(ticks=np.arange(self.electrons.shape[1]), labels=np.arange(self.electrons.shape[1]))
        plt.yticks(ticks=np.arange(self.electrons.shape[0]), labels=np.arange(self.electrons.shape[0]))

        plt.imshow(self.electrons, cmap=plt.cm.Blues, vmin=1)  # vmin=1 pour que les valeurs 0 soient blanches
        plt.colorbar()
        plt.title(titre)
        if dossier != "":
            chemin = os.path.join(dossier,titre+".png")
            plt.savefig(chemin)
        if affiche:
            plt.show()
    
    def affiche_graphe(self,probabilites, titre="Schéma du memristor", dossier="", fig_size=(30, 30), affiche=True):
        """Affiche le graphe des probabilités de transition entre les cases.
        probabilites : dict : dictionnaire des probabilités de transition entre les cases.
        titre : str : titre du graphe.
        dossier : str : chemin du dossier où enregistrer la figure. Par défaut, la figure n'est pas enregistrée.
        fig_size : tuple : taille de la figure. Par défaut, (30, 30).
        affiche : bool : si True, la figure est affichée. Par défaut, elle est affichée.
        """
        G = nx.DiGraph()

        for case, voisines in probabilites.items():
            for voisine, prob in voisines.items():
                G.add_edge(case, voisine, weight=prob)

        pos = {(i, j): (j + self.x[j, i], self.N - i - self.y[j, i]) for i in range(self.N) for j in range(self.N)}

        fig, ax = plt.subplots(figsize=fig_size)

        cmap_nodes = plt.get_cmap('coolwarm')

        nx.draw(
            G, pos, with_labels=False, node_size=100, ax=ax, arrows=False, edgelist=[]
        )

        edges = G.edges(data=True)
        weights = [edge[2]['weight'] for edge in edges]
        max_weight = max(weights) if weights else 1
        cmap_edges = plt.get_cmap('coolwarm')

        nx.draw_networkx_edges(
            G, pos, edge_color=weights, edge_cmap=cmap_edges, edge_vmin=0, edge_vmax=max_weight,
            connectionstyle="arc3,rad=0.3", arrows=True, ax=ax
        )

        sm_edges = plt.cm.ScalarMappable(cmap=cmap_edges, norm=plt.Normalize(0, max_weight))
        sm_edges.set_array([])
        cbar_edges = plt.colorbar(sm_edges, ax=ax, label='Probabilité')

        ax.plot([-1, -1], [0, self.N], color='black', linewidth=3, label = "Electrode")
        ax.plot([self.N, self.N], [0, self.N], color='black', linewidth=3)

        plt.title(titre)
        plt.axis('equal')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.xticks(np.arange(0, self.N, 1))
        plt.yticks(np.arange(0, self.N, 1))
        plt.legend(loc='upper right')
        if dossier != "":
            chemin = os.path.join(dossier, titre + ".png")
            plt.savefig(chemin)
        if affiche:
            plt.show()
    
    def affiche_chemins(self,chemins, head=5, titre="Graphe des chemins", dossier="",fig_size=(30, 30), affiche = True):
        """
        Affiche les chemins des électrons sur la grille.
        chemins : list[list] : liste des chemins des électrons.
        head : int : nombre de chemins à afficher, prend les chemins les plus longs. Par défaut, 5.
        titre : str : titre du graphe.
        dossier : str : chemin du dossier où enregistrer la figure. Par défaut, la figure n'est pas enregistrée.
        fig_size : tuple : taille de la figure. Par défaut, (30, 30).
        affiche : bool : si True, la figure est affichée. Par défaut, elle est affichée.
        """
        G = nx.DiGraph()
        for i in range(self.N):
            for j in range(self.N):
                if self.sites[i,j]==1:
                    G.add_node((i,j), weight=self.electrons[i,j])

        pos = {(i, j): (j + self.x[j, i], self.N - i - self.y[j, i]) for i in range(self.N) for j in range(self.N)}

        fig, ax = plt.subplots(figsize=fig_size)

        nx.draw(
            G, pos, with_labels=False, node_size=100, ax=ax, arrows=False, edgelist=[]
        )


        if len(chemins) > 0:
            chems = [(chemin,len(chemin)) for chemin in chemins]
            chems.sort(key = lambda x : x[1], reverse = True)
            chems_select = [chems[i][0] for i in range(min(head,len(chems)))]
            chemin_colors = plt.cm.tab10(np.linspace(0, 1, head))  # Utiliser une colormap pour les chemins
            for (i, chemin) in enumerate(chems_select):
                chemin_positions = [pos[node] for node in chemin]
                chemin_x, chemin_y = zip(*chemin_positions)
                chemin_x = list(chemin_x)
                chemin_y = list(chemin_y)
                if len(chemin_x) > 0:
                    chemin_x.insert(0,-1)
                    chemin_x.append(self.N)
                    chemin_y.insert(0,chemin_y[0])
                    chemin_y.append(chemin_y[-1])
                ax.plot(chemin_x, chemin_y, color=chemin_colors[i], linewidth=5, alpha=0.7, label=f"Chemin {i+1}")

        ax.plot([-1, -1], [0, self.N], color='black', linewidth=3)
        ax.plot([self.N, self.N], [0, self.N], color='black', linewidth=3)

        plt.title(titre)
        plt.axis('equal')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.xticks(np.arange(0, self.N, 1))
        plt.yticks(np.arange(0, self.N, 1))
        plt.legend(loc='upper right')
        if dossier != "":
            chemin = os.path.join(dossier, titre + ".png")
            plt.savefig(chemin)
        if affiche:
            plt.show()

    def affiche_graphes_colormap(self,probabilites, chemins, head=5, titre="Graphe des électrons", dossier="",fig_size=(30, 30), affiche = True):
        """
        Affiche toutes les informations sur le memristor.
        probabilites : dict : dictionnaire des probabilités de transition entre les cases.
        chemins : list[list] : liste des chemins des électrons.
        head : int : nombre de chemins à afficher, prend les chemins les plus longs. Par défaut, 5.
        titre : str : titre du graphe.
        dossier : str : chemin du dossier où enregistrer la figure. Par défaut, la figure n'est pas enregistrée.
        fig_size : tuple : taille de la figure. Par défaut, (30, 30).
        affiche : bool : si True, la figure est affichée. Par défaut, elle est affichée.
        """
        G = nx.DiGraph()

        for case, voisines in probabilites.items():
            for voisine, prob in voisines.items():
                G.add_edge(case, voisine, weight=prob)

        pos = {(i, j): (j + self.x[j, i], self.N - i - self.y[j, i]) for i in range(self.N) for j in range(self.N)}

        fig, ax = plt.subplots(figsize=(30, 30))

        cmap_nodes = plt.get_cmap('coolwarm')

        node_values = [self.electrons[node[0], node[1]] for node in G.nodes()]
        nx.draw(
            G, pos, with_labels=False, node_size=100, node_color=node_values,
            cmap=cmap_nodes, ax=ax, arrows=False, edgelist=[]
        )

        edges = G.edges(data=True)
        weights = [edge[2]['weight'] for edge in edges]
        max_weight = max(weights) if weights else 1
        cmap_edges = plt.get_cmap('coolwarm')

        nx.draw_networkx_edges(
            G, pos, edge_color=weights, edge_cmap=cmap_edges, edge_vmin=0, edge_vmax=max_weight,
            connectionstyle="arc3,rad=0.3", arrows=True, ax=ax
        )

        if len(chemins) > 0:
            chems = [(chemin,len(chemin)) for chemin in chemins]
            chems.sort(key = lambda x : x[1], reverse = True)
            chems_select = [chems[i][0] for i in range(min(head,len(chems)))]
            chemin_colors = plt.cm.tab10(np.linspace(0, 1, head))  # Utiliser une colormap pour les chemins
            for (i, chemin) in enumerate(chems_select):
                chemin_positions = [pos[node] for node in chemin]
                chemin_x, chemin_y = zip(*chemin_positions)
                chemin_x = list(chemin_x)
                chemin_y = list(chemin_y)
                if len(chemin_x) > 0:
                    chemin_x.insert(0,-1)
                    chemin_x.append(self.N)
                    chemin_y.insert(0,chemin_y[0])
                    chemin_y.append(chemin_y[-1])
                ax.plot(chemin_x, chemin_y, color=chemin_colors[i], linewidth=5, alpha=0.7, label=f"Chemin {i+1}")

        sm_nodes = plt.cm.ScalarMappable(cmap=cmap_nodes, norm=plt.Normalize(0, max(node_values)))
        sm_nodes.set_array([])
        cbar_nodes = plt.colorbar(sm_nodes, ax=ax, label='Valeur des électrons')

        sm_edges = plt.cm.ScalarMappable(cmap=cmap_edges, norm=plt.Normalize(0, max_weight))
        sm_edges.set_array([])
        cbar_edges = plt.colorbar(sm_edges, ax=ax, label='Probabilité')

        # Décommenter pour ajouter les axes
        #plt.annotate('', xy=(N+1, 0), xytext=(-1, 0),
        #        arrowprops=dict(arrowstyle='->', color='grey', lw=2.5))
        #plt.annotate('', xy=(-1, N+1), xytext=(-1, 0),
        #        arrowprops=dict(arrowstyle='->', color='grey', lw=2.5))
        
        ax.plot([-1, -1], [0, self.N], color='black', linewidth=3)
        ax.plot([self.N, self.N], [0, self.N], color='black', linewidth=3)


        plt.title(titre)
        plt.axis('equal')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.xticks(np.arange(0, self.N, 1))
        plt.yticks(np.arange(0, self.N, 1))
        plt.legend(loc='upper right')
        if dossier != "":
            chemin = os.path.join(dossier, titre + ".png")
            plt.savefig(chemin)
        if affiche:
            plt.show()
        
    
    def print_distrib(self,chem, titre="Distribution des chemins", dense = False, dossier = "",fig_size=(10, 5), affiche = True):
        """Affiche la distribution des chemins des électrons avec densité
        chem : list[list] : liste des chemins des électrons.
        titre : str : titre de la figure.
        dense : bool : si True, la densité est affichée. Par défaut, False.
        dossier : str : chemin du dossier où enregistrer la figure. Par défaut, la figure n'est pas enregistrée.
        fig_size : tuple : taille de la figure. Par défaut, (10, 5).
        affiche : bool : si True, la figure est affichée. Par défaut, elle est affichée.
        """
        chemin_lengths = [len(chem[i]) for i in range(len(chem))]
        if len(chemin_lengths)>1:
            dense = False
        if len(chem)>0:
            xs = np.linspace(min(chemin_lengths), max(chemin_lengths), 200)
        else:
            xs = np.linspace(0, 1, 10)
        plt.figure(figsize=(10, 5))
        if dense:
            plt.plot(xs, gaussian_kde(chemin_lengths)(xs), color='red', label="Densité")
        else:
            plt.hist(chemin_lengths, bins='auto', color='blue', alpha=0.7, density=dense, label=f"Nombre de chemins : {len(chemin_lengths)}")
        plt.title(titre)
        plt.xlabel('Longueur du chemin')
        if dense:
            density = gaussian_kde(chemin_lengths)
            density_values = density(xs)
            plt.plot(xs, density_values, color='red', label="Densité")
            plt.ylabel("Densité")
        else:
            plt.ylabel("Nombre d'électrons")
        plt.grid()
        plt.legend()
        if dossier != "":
            chemin = os.path.join(dossier,titre+".png")
            plt.savefig(chemin)
        if affiche:
            plt.show()
    
    def plot_deux_couleurs(x_points,y_points,dossier,titre, sep = 10,xlabel = "Tension (V)",ylabel= "Intensité (A)", affiche = True):
        """
        Trace les graphiques en deux couleurs : rouge pour la phase de montée et bleu pour la phase de descente.
        x_points : list : liste des points sur l'axe des abscisses.
        y_points : list : liste des points sur l'axe des ordonnées.
        dossier : str : chemin du dossier où enregistrer la figure. Par défaut, la figure n'est pas enregistrée.
        titre : str : titre de la figure.
        sep : int : séparation entre les deux phases. Par défaut, 10.
        xlabel : str : titre de l'axe des abscisses. Par défaut, "Tension (V)".
        ylabel : str : titre de l'axe des ordonnées. Par défaut, "Intensité (A)".
        affiche : bool : si True, la figure est affichée. Par défaut, elle est affichée.
        """
        plt.scatter(x_points[:10], y_points[:10], color='red', label='Phase de montée', marker='x')
        plt.scatter(x_points[10:], y_points[10:], color='blue', label='Phase de descente', marker='x')

        plt.legend()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(titre)
        chemin_fichier = os.path.join(dossier,titre+".png")
        plt.grid()   
        plt.savefig(chemin_fichier)
        if affiche:
            plt.show()    
    
    """
    =====================================================================================================================
                                                Simulation complète
    =====================================================================================================================
    """
    def simulation(self,array_V, id = "",affiche = True, trace_carac = True, snapshot = [0,10,20],sep = None):
        """
        Effectue la simulation complète du memristor.
        array_V : list : liste des tensions appliquées.
        id : str : identifiant de la simulation, nom du dossier d'enregistrement. Par défaut, pas d'enregistrement.
        affiche : bool : si True, la figure est affichée. Par défaut, elle est affichée.
        trace_carac : bool : si True, les graphiques caractéristiques sont tracés. Par défaut, elle est tracée.
        snapshot : list : liste des étapes où les schémas complets sont affichés. Par défaut, [0,10,20].
        sep : int : séparation entre les deux phases. Par défaut, il est calculé à la moitié de la liste.
        Enregistre tous les graphes affichés, y ajoute les listes des intensités, des longueurs de chemins et de tensions en pickle,
        ainsi qu'un fichier texte contenant les principales variables globales.
        """
        self.reinitialise()
        chem = []
        intense = []
        nb_chem = []
        moy_chem = []
        var_chem = []
        if id != "" and not os.path.exists(id):
            os.makedirs(id)
        for (i,V) in enumerate(array_V):
            intensite, chem, probas = self.etape(V,chem,(i==0))
            nb_chem.append(len(chem))
            intense.append(intensite)
            chemins = np.array([len(chemin) for chemin in chem])
            moy_chem.append(np.mean(chemins))
            var_chem.append(np.var(chemins))
            print(f"Etape {i}, Intensité : {intensite}")
            if affiche:
                if i in snapshot:
                    self.affiche_graphe(probas, titre = f"Schéma du memristor à V = {V}, étape {i}", dossier = id, affiche = affiche)
                    self.affiche_chemins(chem, titre = f"Chemins des électrons à V = {V}, étape {i}", dossier = id, affiche = affiche)
                    self.affiche_graphes_colormap(probas, [chemin for chemin in chem if len(chemin)>4], titre = f"Schéma complet du memristor à V = {V}, étape {i}", dossier = id, affiche = affiche)
                self.heatmap(titre = f"Position des électrons à V = {V}, étape {i}", dossier = id,affiche = affiche)
                self.print_distrib(chem, titre = f"Distribution des chemins à V = {V}, étape {i}", dossier = id, affiche = affiche)
        if trace_carac:
            if sep is None:
                sep = int(len(array_V)/2)
            self.plot_deux_couleurs(array_V,intense,id,f"Caractéristique I(V)", sep=sep, ylabel = "Intensité (A)")
            self.plot_deux_couleurs(array_V,nb_chem,id, f"Evolution du nombre de chemins", sep = sep, ylabel = "Nombre de chemins")
            self.plot_deux_couleurs(array_V,moy_chem,id, f"Longueur moyenne des chemins", sep = sep, ylabel = "Longueur moyenne des chemins")
            self.plot_deux_couleurs(array_V,var_chem,id, f"Variance des longueurs des chemins", sep = sep, ylabel = "Variance des longueurs des chemins")

        if id != "":      
            variables_globales = globals()
            chemin_fichier = os.path.join(id, "variables_globales.txt")
            with open(chemin_fichier,"w") as f:
                for nom in self.li_va_globales:
                    f.write(f"{nom} : {variables_globales[nom]}\n")
            
            chemin_fichier = os.path.join(id,"intensite.pkl")
            with open(chemin_fichier,"wb") as f:
                pickle.dump(intense,f)
            chemin_fichier = os.path.join(id,"tension.pkl")
            with open(chemin_fichier,"wb") as f:
                pickle.dump(array_V,f)
            chemin_fichier = os.path.join(id,"nb_chem.pkl")
            with open(chemin_fichier,"wb") as f:
                pickle.dump(nb_chem,f)
        return intense 

if __name__ == "main":
    array_V = list(range(0,10))
    array_V += list(range(10,-1,-1))
    array_V = 0.1 * np.array(array_V)
    simulation = Simulation()
    simulation.simulation(array_V, id = "test", affiche = True, trace_carac = True, snapshot = [0,10,20],sep = 10)