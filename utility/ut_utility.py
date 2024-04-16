import numpy as np
import math
import matplotlib.path as mpath
import gc


MAXSIZE = 1000000
def CreatePolygon(x,y):
    x,y = np.append(x,x[-1]), np.append(y,y[-1])
    Path = mpath.Path
    codes = np.ones(x.size)*Path.LINETO
    codes[0] = Path.MOVETO
    codes[-1] = Path.CLOSEPOLY
    verts = list(zip(x,y))
    path = mpath.Path(verts, codes)
    return path

def chunk_counting(a):
    #chunk counting (a=[3,2,5] ---> returns [0,1,2, 0,1, 0,1,2,3,4]) 

    csum = np.cumsum(a)[0:-1]
    csum = np.insert(csum, 0, 0)

    r_csum = np.repeat(csum, a)

    return np.arange(r_csum.size) - r_csum


def find_a_in_b(a, b):
    print "this will only work if every element in a is in b at least once"
    #pre condition the b vector
    ii = np.array( np.where( (b>(a.min()-1)) & (b<(a.max()+1)) ) )[0,:]
    b = b[ii]
    orig_indices = b.argsort() #this is the most expensive part, especially if b is very large
    b = b[orig_indices]
    
    #now the actual searching

    #MAXSIZE = 100 * a.size # in this way if the b vector is much larger than a we can split it and most of 
                          # the parts will be empty
    if (b.size > MAXSIZE): #if the b array is still to large, we need to split it into several parts
      size = b.size
      breaks = np.arange(0,size,MAXSIZE)
      if (breaks[-1] != size):
        breaks = np.append(breaks, size)

      ind2 = np.array([])

      for i in np.arange(breaks.size-1):
        b_part_i = b[breaks[i]:breaks[i+1]]

        wh = np.array( np.where( (a>=b_part_i[0]) & (a<=b_part_i[-1]) ) )[0,:]

        if (wh.size != 0):
          unique_b, index_start, count = np.unique(b_part_i, return_counts=True, return_index=True)
          ind = np.searchsorted(unique_b, a[wh])

          ind_start = index_start[ind]
          ind_count = count[ind]

          indi = np.repeat(ind_start, ind_count) + chunk_counting(ind_count)

          indi += breaks[i]

          ind2 = np.append(ind2, indi)

      ind2 = ind2.astype(int)
      print "done with loops ", i
    else:    
      unique_b, index_start, count = np.unique(b, return_counts=True, return_index=True)
  
      ind = np.searchsorted(unique_b, a)
  
      ind_start = index_start[ind]
      ind_count = count[ind]
  
      ind2 = np.repeat(ind_start, ind_count) + chunk_counting(ind_count)
  

    ind2 = orig_indices[ind2]
  
    return ii[ind2]


