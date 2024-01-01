import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def back_substitution(U,Y):
    '''
    back_substitution(U,Y)
    
    back_substitution은 역치환을 수행하여 UX = Y의 해를 구합니다. 
    U가 전체 순위인지 확인하기 위해 오류를 검사하지 않습니다.

    Parameters
    ----------
    U : NumPy array object of dimension mxm
    Y : NumPy array object of dimension mx1

    Returns
    -------
    X : NumPy array object of dimension mx1
    '''

    m = U.shape[0]  # m은 $U$의 행과 열 수입니다.
    X = np.zeros((m,1))
    
    for i in range(m-1,-1,-1):  # m-1에서 0으로 역방향으로 항목을 계산합니다.
        X[i] = Y[i]
        for j in range(i+1,m):
            X[i] -= U[i][j]*X[j]
        if (U[i][i] != 0):
            X[i] /= U[i][i]
        else:
            print("Zero entry found in U pivot position",i,".")
    return X

def determinant_iteration(A):
    ''' 
    determinant_iteration(A)
    
    행렬식을 계산 합니다.
    선호하진 않지만 재귀를 활용합니다.

    Parameters
    ----------
    A: NumPy array object of dimension nxn
    
    Returns
    -------
    D: int
    '''

    # Check shape of A
    if (A.shape[0] != A.shape[1]):
        print("Determinant only defined for square arrays.")
        return None
    n = A.shape[0]  # n is number of rows and columns in A
    
    size = A.shape[0]
    if size == 2:
        return A[0,0]*A[1,1]-A[0,1]*A[1,0]
    
    else:
        m=0  # Determinant expansion along row 0
        D=0. # Set determinant to zero and add contributions

        for n in range(size):
            minor = []
            k=-1
            # Construct (m,n) minor array (row m, column n deleted)
            for i in range(size):
                if(i != m): 
                    minor.append([])    
                    k += 1
                    for j in range(size):
                        if(j != n):
                            minor[k].append(A[i,j])
            Minor_array = np.array(minor)
            cofactor = (-1)**(m+n)*determinant_iteration(Minor_array)
            D += cofactor*A[m,n]
        return D

def draw_graph(A, pos = None):
    '''
    draw_graph(A, pos = None)
    
    인정햅렬(adjacency matrix)을 기반으로 한 방향 그래프를 그립니다.

    Parameters
    ----------
    A : NumPy array object.
    pos: Optional dictionary to specify node coordinates
    
    Returns
    -------
    pos: Dictionary of node coordinates used to draw graph

    '''

    plt.figure(figsize=(6,6))
    G = nx.DiGraph()
    
    N = A.shape[0]
    edge_list = []
    
    for i in range(N):
        for j in range(N):
            if(A[i,j] == 1):
                edge_list.append((i,j))
    
    G.add_edges_from(edge_list)
    if (pos == None):
        pos = nx.spring_layout(G)
    
    options = {"node_size" : 500, "with_labels": True,"font_size":20}
    nx.draw(G, pos,connectionstyle='arc3, rad = 0.1',arrowsize=30,**options)
    return pos

def full_row_reduction(A, tol = 1e-14):
    ''' 
    full_row_reduction(A, tol = 1e-14)
    
    모든 형태의 행렬에 대해 RREF를 생성합니다.  
    피벗 전략이 구현되지 않았습니다.
    
    Parameters
    ----------
    A : NumPy array object of dimension mxn
    tol: optional float

    Returns
    -------
    B: NumPy array object of dimension mxn
    '''
    
    m = A.shape[0]  # m is number of rows in A
    n = A.shape[1]  # n is number of columns in A

    B = np.copy(A).astype('float64')

    # Set initial pivot search position
    pivot_row = 0
    pivot_col = 0
    
    # Continue steps of elimination while possible pivot positions are 
    # within bounds of the array.
    
    while(pivot_row < m and pivot_col < n):

        # Set pivot value to current pivot position
        pivot = B[pivot_row,pivot_col]
        
        # If pivot is zero, search down current column, and then subsequent
        # columns (at or beyond pivot_row) for the next nonzero entry in the 
        # array is found, or the last entry is reached.

        row_search = pivot_row
        col_search = pivot_col
        search_end = False

        while(pivot == 0 and not search_end):
            if(row_search < m-1):
                row_search += 1
                pivot = B[row_search,col_search]
            else:
                if(col_search < n-1):
                    row_search = pivot_row
                    col_search += 1
                    pivot = B[row_search,col_search]
                else:  
                    # col_search = n-1 and row_search = m-1
                    search_end = True
                        
        # Swap row if needed to bring pivot to position for rref
        if (pivot != 0 and pivot_row != row_search):
            B = row_swap(B,pivot_row,row_search)
            pivot_row, row_search = row_search, pivot_row
            
        # Set pivot position to search position
        pivot_row = row_search
        pivot_col = col_search
            
        # If pivot is nonzero, carry on with elimination in pivot column 
        if (pivot != 0):
            
            # Set pivot entry to one
            B = row_scale(B,pivot_row,1./B[pivot_row,pivot_col])

            # Create zeros above pivot
            for i in range(pivot_row):    
                B = row_add(B,pivot_row,i,-B[i][pivot_col])
                # Force known zeros
                B[i,pivot_col] = 0

            # Create zeros below pivot
            for i in range(pivot_row+1,m):    
                B = row_add(B,pivot_row,i,-B[i][pivot_col])
                # Force known zeros
                B[i,pivot_col] = 0

            # Force small numbers to zero to account for roundoff error
            for i in range(m):
                for j in range(n):
                    if abs(B[i,j])< tol :
                        B[i,j] = 0

        # Advance to next possible pivot position
        pivot_row += 1
        pivot_col += 1
        
    return B

def highlight_subgraph(A,pos,subgraph):
    '''
    highlight_subgraph(A,pos,subgraph)

    Parameters
    ----------
    A : NumPy array object
    
    pos : dictionary of node positions
    
    nodelist : list of ints representing the nodes in the subgraph

    Returns
    -------
    None.

    '''
    plt.figure(figsize=(6,6))
    G = nx.DiGraph()
    
    N = A.shape[0]
    edge_list = []
    subgraph_edges = []
    
    for i in range(N):
        for j in range(N):
            if(A[i,j] == 1):
                edge_list.append((i,j))
 
    for edge in edge_list:           
        if (edge[0] in subgraph and edge[1] in subgraph):
                    subgraph_edges.append(edge)
    

    G.add_edges_from(edge_list)
    
    options = {"with_labels": True,"font_size":20}
    nx.draw(G, pos,connectionstyle='arc3, rad = 0.1',arrowsize=30,**options)
    
    node_options = {"node_color":'r',"node_size" : 400}
    nx.draw_networkx_nodes(G, pos, nodelist=subgraph, **node_options)

    
    edge_options = {"width" : 8,"alpha" : 0.5, "edge_color" : 'r',
                    "connectionstyle":"arc3, rad=0.1"}    
    nx.draw_networkx_edges(G,pos,edgelist=subgraph_edges, **edge_options)

def inverse(A):
    '''
    inverse(A)
    
    Parameters
    ----------
    A: NumPy array object of dimension nxn
    
    Returns
    -------
    Inverse: NumPy array object of dimension nxn
    '''

    # Check shape of A
    if (A.shape[0] != A.shape[1]):
        print("Inverse accepts only square arrays.")
        return
    n = A.shape[0]  # n is number of rows and columns in A

    I = np.eye(n)
    
    # The augmented matrix is A together with all the columns of I.  RowReduction is
    # carried out simultaneously for all n systems.
    A_augmented = np.hstack((A,I))
    R = row_reduction(A_augmented)
    
    Inverse = np.zeros((n,n))
    
    # Now BackSubstitution is carried out for each column and the result is stored 
    # in the corresponding column of Inverse.
    A_reduced = R[:,0:n]
    for i in range(0,n):
        B_reduced = R[:,n+i:n+i+1]
        Inverse[:,i:i+1] = back_substitution(A_reduced,B_reduced)
    
    return(Inverse)

def row_swap(A,k,l):
    ''' 
    row_swap(A,k,l)
    
    Parameters
    ----------
    A : NumPy array object of dimension mxn
    k : int
    l : int
    scale : float
    
    Returns
    -------
    B: NumPy array object of dimension mxn
    '''

    m = A.shape[0]  # m is number of rows in A
    n = A.shape[1]  # n is number of columns in A
    
    B = np.copy(A).astype('float64')

    for j in range(n):
        temp = B[k][j]
        B[k][j] = B[l][j]
        B[l][j] = temp
        
    return B

def row_scale(A,k,scale):
    ''' 
    row_scale(A,k,scale)
    
    Parameters
    ----------
    A : NumPy array object of dimension mxn
    k : int
    scale : float
    
    Returns
    -------
    B: NumPy array object of dimension mxn
    '''
    
    m = A.shape[0]  # m is number of rows in A
    n = A.shape[1]  # n is number of columns in A
    
    B = np.copy(A).astype('float64')

    for j in range(n):
        B[k][j] *= scale
        
    return B

def row_add(A,k,l,scale):
    ''' 
    row_add(A,k,l,scale)
    
    Parameters
    ----------
    A : NumPy array object of dimension mxn
    k : int
    l : int
    scale : float
    
    Returns
    -------
    B: NumPy array object of dimension mxn
    '''

    m = A.shape[0]  # m is number of rows in A
    n = A.shape[1]  # n is number of columns in A
    
    B = np.copy(A).astype('float64')
        
    for j in range(n):
        B[l][j] += B[k][j]*scale
        
    return B

def row_reduction(A):
    ''' 
    row_reduction(A)
    
    Parameters
    ----------
    A : NumPy array object of dimension mxn
    
    Returns
    -------
    B: NumPy array object of dimension mxn
    '''
    
    m = A.shape[0]  # A has m rows 
    n = A.shape[1]  # It is assumed that A has m+1 columns
    
    B = np.copy(A).astype('float64')

    # For each step of elimination, we find a suitable pivot, move it into
    # position and create zeros for all entries below.
    
    for k in range(m):
        # Set pivot as (k,k) entry
        pivot = B[k][k]
        pivot_row = k
        
        # Find a suitable pivot if the (k,k) entry is zero
        while(pivot == 0 and pivot_row < m-1):
            pivot_row += 1
            pivot = B[pivot_row][k]
            
        # Swap row if needed
        if (pivot_row != k):
            B = row_swap(B,k,pivot_row)
            
        # If pivot is nonzero, carry on with elimination in column k
        if (pivot != 0):
            B = row_scale(B,k,1./B[k][k])
            for i in range(k+1,m):    
                B = row_add(B,k,i,-B[i][k])
        else:
            print("Pivot could not be found in column",k,".")
            
    return B

def solve_system(A,B):
    ''' 
    solve_system(A,B)
    
    Parameters
    ----------
    A : NumPy array object of dimension nxn
    B : NumPy array object of dimension nx1
    
    Returns
    -------
    X: NumPy array object of dimension nx1
    '''
    # Check shape of A
    if (A.shape[0] != A.shape[1]):
        print("SolveSystem accepts only square arrays.")
        return None
    n = A.shape[0]  # n is number of rows and columns in A
    B.shape = (n,1)
    
    # Join A and B to make the augmented matrix
    A_augmented = np.hstack((A,B))
    
    # Carry out elimination    
    R = row_reduction(A_augmented)

    # Split R back into nxn piece and nx1 piece
    B_reduced = R[:,n:n+1]
    A_reduced = R[:,0:n]

    # Do back substitution
    X = back_substitution(A_reduced,B_reduced)
    
    return X