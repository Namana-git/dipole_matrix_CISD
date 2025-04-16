import numpy as np
import math


# There are 8 files corresponding to each eigenvector
# uuuu.dat dddd.dat udud.dat dudu.dat uddu.dat duud.dat ud.dat du.dat

# define float arrays for each of then
nc = 10
nv = 5
s= nc*(nc-1)*nv*(nv-1)/4
p = nc*(nc-1)*nv/2
q = nv*(nv-1)*nc/2
m = 2*s + nc*nv + p + q
n = 2*s + p
o = 2*s + q

"""
np.loadtxt() default usage converts entire text into an array with array[line][words] 


"""
uuuu1 = np.loadtxt("uuuu1.dat",dtype=[('i',int),('j',int),('a',int),('b',int),('w',float)])
dddd1 = np.loadtxt("dddd1.dat",dtype=[('i',int),('j',int),('a',int),('b',int),('w',float)])
udud1 = np.loadtxt("udud1.dat",dtype=[('i',int),('j',int),('a',int),('b',int),('w',float)])
dudu1 = np.loadtxt("dudu1.dat",dtype=[('i',int),('j',int),('a',int),('b',int),('w',float)])
uddu1 = np.loadtxt("uddu1.dat",dtype=[('i',int),('j',int),('a',int),('b',int),('w',float)])
duud1 = np.loadtxt("duud1.dat",dtype=[('i',int),('j',int),('a',int),('b',int),('w',float)])
ud1 = np.loadtxt("ud1.dat",dtype=[('i',int),('a',int),('w',float)])
du1 = np.loadtxt("du1.dat",dtype=[('i',int),('a',int),('w',float)])

uuuu2 = np.loadtxt("uuuu2.dat",dtype=[('i',int),('j',int),('a',int),('b',int),('w',float)])
dddd2 = np.loadtxt("dddd2.dat",dtype=[('i',int),('j',int),('a',int),('b',int),('w',float)])
udud2 = np.loadtxt("udud2.dat",dtype=[('i',int),('j',int),('a',int),('b',int),('w',float)])
dudu2 = np.loadtxt("dudu2.dat",dtype=[('i',int),('j',int),('a',int),('b',int),('w',float)])
uddu2 = np.loadtxt("uddu2.dat",dtype=[('i',int),('j',int),('a',int),('b',int),('w',float)])
duud2 = np.loadtxt("duud2.dat",dtype=[('i',int),('j',int),('a',int),('b',int),('w',float)])
ud2 = np.loadtxt("ud2.dat",dtype=[('i',int),('a',int),('w',float)])
du2 = np.loadtxt("du2.dat",dtype=[('i',int),('a',int),('w',float)])







# read the mu_matrix

col1, col2, col3, col4, col5 = [], [], [], [], []

with open("dipole.dat", "r") as f:
    lines = f.readlines()
mat = np.zeros((int(math.sqrt(len(lines)//2)),int(math.sqrt(len(lines)//2)),3),dtype=float)
# Iterate in steps of 2 lines
for i in range(0, len(lines), 2):
    line1 = lines[i].strip()
    line2 = lines[i + 1].strip()

    # Parse first line: int, int, float, float
    i1, i2, f1, f2 = line1.split()
    
    # Parse second line: float
    f3 = line2
    i1 = int(i1)
    i2 = int(i2)
    f1 = float(f1)
    f2 = float(f2)
    f3 = float(f3)
    
    mat[i1-1,i2-1,0] = float(f1)
    mat[i1-1,i2-1,1] = float(f2)
    mat[i1-1,i2-1,2] = float(f3)
print(mat[172,172,:])
    


nf = 172
num_start = nf +1 
num_end = nf + nv + nc
mu_matrix = np.zeros((num_end-num_start+1,num_end-num_start+1,3),dtype=float) 
#mu_matrix[:,:,0] = mat[num_start-1:num_end,num_start-1:num_end,0]
#mu_matrix[:,:,1] = mat[num_start-1:num_end,num_start-1:num_end,1]
#mu_matrix[:,:,2] = mat[num_start-1:num_end,num_start-1:num_end,2]
mu_matrix[:,:,:] = mat[172:187,172:187,:]
print(np.shape(mu_matrix))
# verify this   
print(mu_matrix[0,5,:])

def fermionic_sign(occ_det1,occ_det2):
    occ1 = np.where(occ_det1 == 1)[0]
    occ2 = np.where(occ_det1 == 1)[0]
    # Compare permutations
    for i in range(len(occ1)):
        if occ1[i] != occ2[i]:
            return (-1)**(i)
    return 1


def calculate_dip_between_det1_det2(det1_ind,det2_ind,det1_spin,det2_spin,mu_matrix):
   """
   Calculates the dipole matrix elements between det1 and det2 

   Reads the whether det1 and det2 are single excitations 

   The creates the occupation array for each of the slater determinants

   Then these occupation arrays of two lists are called into another function that calculate the dipole mo between these determinants according to Slater-Condon rules

   The ordering is energetic and down first and up next 

   Parmeters:
   det1_ind --> [ijab]/[ia] slater determinant represented as in electron hole state on top of the GS slater determinants of det1 
   det2_ind --> [mn\gama\eta]/[m\gama] slater determinant represented as in electron hole state on top of the GS slater determinants of det1
   det1_spin --> spins of  [ijab]/[ia]
   det2_spin --> spins of [mn\gama\eta]/[m\gama]
 
   return:
   
   <det2|p|det1>
   
   """
   occ_det1 = np.zeros(2*(nc+nv),dtype=int)
   occ_det2 = np.zeros(2*(nc+nv),dtype=int)
   occ_det1[:2 * nv] = 1     #initialize to gs SD
   occ_det2[:2 * nv] = 1     #initialize to gs SD
   sign_det1 = 2
   sign_det2 = 2 

   if(len(det1_ind) == 4):
      occ_det1[2*(det1_ind[0])-det1_spin[0] -1] = 0   
      occ_det1[2*det1_ind[1]-det1_spin[1]-1] = 0
      occ_det1[2*det1_ind[2]-det1_spin[2]-1] =  1
      occ_det1[2*det1_ind[3]-det1_spin[3]-1] = 1
      sign_det1 = (-1)^(det1_spin[0])*(-1)^(det1_spin[1])    

   if(len(det2_ind) == 4):
      occ_det2[2*det2_ind[0]-det2_spin[0]-1] = 0   
      occ_det2[2*det2_ind[1]-det2_spin[1]-1] = 0
      occ_det2[2*det2_ind[2]-det2_spin[2]-1] =  1
      occ_det2[2*det2_ind[3] -det2_spin[3]-1] = 1
      sign_det2 = (-1)^(det2_spin[0])*(-1)^(det2_spin[1]) 

   if(len(det1_ind) == 2):
        occ_det1[2*det1_ind[0]-det1_spin[0]-1] = 0   
        occ_det1[2*det1_ind[1]-det1_spin[1]-1] = 1
        sign_det1 = -1^(det1_spin[0])
   if(len(det2_ind) == 2):
        occ_det2[2*det2_ind[0]-det2_spin[0]-1] = 0   
        occ_det2[2*det2_ind[1]-det2_spin[1]-1] = 1
        sign_det2 = -1^(det2_spin[0])

   diff = np.nonzero(occ_det1 != occ_det2)[0]   # return the indices where the occupation no is no the same      
   ndiff = len(diff)

   if ndiff==0:
       return np.sum([mu_matrix[i//2,i//2,:] for i in np.where(occ_det1==1)[0]],axis=0)
   #dip_mo = dip_det1_det2(occ_det1,occ_det2)

   elif ndiff == 2:    # if only two places have this, 1 goes to zero and zero goes to 1 
        i, a = diff      
        # Determine which determinant has which orbital
        if occ_det1[i] == 1 and occ_det2[a] == 1:
            sign = fermionic_sign(occ_det1, occ_det2)
            
            print(occ_det1)
            print(occ_det2)
            print(i,a)
            print( sign * mu_matrix[i//2, a//2, :])
            return sign * mu_matrix[i//2, a//2, :]
        elif occ_det1[a] == 1 and occ_det2[i] == 1:
            sign = fermionic_sign(occ_det1, occ_det2)
           
            print(occ_det1)
            print(occ_det2)
            print(i,a)
            print( sign * mu_matrix[i//2, a//2, :])
            return sign * mu_matrix[a//2, i//2, :]
        else:
            return np.zeros(3)

   else:
        # More than 2 differences: dipole element is zero
        return np.zeros(3)

 



def calculate_dip_between_eig1_eig2(uuuu1,uuuu2,dddd1,dddd2,udud1,udud2,dudu1,dudu2,uddu1,uddu2,duud1,duud2,ud1,ud2,du1,du2,mu_matrix):
    sum = np.array([0.0,0.0,0.0])
    sum1 = np.array([0.0,0.0,0.0])
    #det2_ind=np.array([5,4,6,7])
    #det2_spin=np.array([0,0,0,0]) 
    for  x1 in range(8):
        if(x1==0):
            spin_array1= uuuu1
            det1_spin = np.array([0,0,0,0])
            det1_sign = 0
        if(x1==1):
             spin_array1= dddd1
             det1_spin = np.array([1,1,1,1])
             det1_sign = 0
        if(x1==2):
             spin_array1= udud1
             det1_spin = np.array([1,0,1,0])
             det1_sign = 1
        if(x1==3):
             spin_array1= dudu1
             det1_spin = np.array([0,1,0,1])
             det1_sign = 1
        if(x1==4):
             spin_array1= uddu1
             det1_spin = np.array([0,1,1,0])
             det1_sign = 1
        if(x1==5):
             spin_array1= duud1
             det1_spin = np.array([1,0,0,1])
             det1_sign = 1
        if(x1==6):
             spin_array1= ud1
             det1_spin = np.array([0,0])
             det1_sign = 1
        if(x1==7):
             spin_array1= du1
             det1_spin = np.array([1,1])
             det1_sign = 0
        for x2 in range(8):
              if(x2==0):
                 spin_array2= uuuu2
                 det2_spin = np.array([0,0,0,0])
                 det2_sign = 0
              if(x2==1):
                 spin_array2= dddd2
                 det2_spin = np.array([1,1,1,1])
                 det2_sign = 0
              if(x2==2):
                 spin_array2= udud2
                 det2_spin = np.array([1,0,1,0])
                 det2_sign = 1
              if(x2==3):
                 spin_array2= dudu2
                 det2_spin = np.array([0,1,0,1])
                 det2_sign = 1
              if(x2==4):
                 spin_array2= uddu2
                 det2_spin = np.array([0,1,1,0])
                 det2_sign = 1
              if(x2==5):
                 spin_array2= duud2
                 det2_spin = np.array([1,0,0,1])
                 det2_sign = 1
              if(x2==6):
                 spin_array2= ud2
                 det2_spin = np.array([0,0])
                 det2_sign = 1
              if(x2==7):
                 spin_array2= du2
                 det2_spin = np.array([1,1])
                 det2_sign = 0 
              for z1 in range(len(spin_array1[:]['i'])):
                  if(x1 < 6):
                        det1_ind=np.array([(nv+1-spin_array1[z1]['b']),(nv+1-spin_array1[z1]['a']),(nv+spin_array1[z1]['j']),(nv+spin_array1[z1]['i'])])
                  if(x1 > 5):
                        det1_ind=np.array([(nv+1-spin_array1[z1]['a']),(nv+spin_array1[z1]['i'])])
                  #print(det1_ind)
                  #print(det1_spin)
                  #occ1,occ2 = calculate_dip_between_det1_det2(det1_ind,det2_ind,det1_spin,det2_spin,mu_matrix)
                  for z2 in range(len(spin_array2[:]['i'])):
                        if(x2 < 6):
                           det2_ind=np.array([(nv+1-spin_array2[z2]['b']),(nv+1-spin_array2[z2]['a']),(nv+spin_array2[z2]['j']),(nv+spin_array2[z2]['i'])])
                        if(x2 > 5):
                           det2_ind=np.array([(nv+1-spin_array2[z2]['a']),(nv+spin_array2[z2]['i'])])
                        
                        #occ1,occ2 = calculate_dip_between_det1_det2(det1_ind,det2_ind,det1_spin,det2_spin,mu_matrix)
                        #if(x1==1):
                         # if(x2==0):
                          #  print(det1_ind,det1_spin)
                           # print(det2_ind,det2_spin)
                            #print(occ1)
                            #print(occ2)

                        dip_mo = calculate_dip_between_det1_det2(det1_ind,det2_ind,det1_spin,det2_spin,mu_matrix)
                        #print*,dip_mo
                        sum = sum + spin_array1[z1]['w']*spin_array2[z2]['w']*dip_mo
                        sum1 = sum1 +  spin_array1[z1]['w']*spin_array2[z2]['w']*dip_mo*(-1)**(det1_sign)*(-1)**(det2_sign)
                 
    return sum,sum1

dipole,dipole1 = calculate_dip_between_eig1_eig2(uuuu1,uuuu2,dddd1,dddd2,udud1,udud2,dudu1,dudu2,uddu1,uddu2,duud1,duud2,ud1,ud2,du1,du2,mu_matrix)  
print("dipole",dipole)
print("dipole1",dipole1)        
"""
    for z in range(len(uuuu1[:]['i'])):
        det1_ind=np.array[(nv+1-uuuu1[z]['b']),(nv+1-uuuu1[z]['a']),(nv+uuuu1[z]['j']),(nv+uuuu1[z]['i'])]
        det2_ind=np.array[(nv+1-uuuu2[z]['b']),(nv+1-uuuu2[z]['a']),(nv+uuuu2[z]['j']),(nv+uuuu2[z]['i'])]
        det1_spin = np.array[0,0,0,0]
        det2_spin = np.array[0,0,0,0]

        dip_mo =  calculate_dip_between_det1_det2(det1_ind,det2_ind,det1_spin,det2_spin)
        sum = sum + uuuu1[z]['w']*uuuu2[z]['w']*dip_mo

    for z in range(len(dddd1[:]['i'])):
        det1_ind=np.array[(nv+1-dddd1[z]['b']),(nv+1-dddd1[z]['a']),(nv+dddd1[z]['j']),(nv+dddd1[z]['i'])]
        det2_ind=np.array[(nv+1-dddd2[z]['b']),(nv+1-dddd2[z]['a']),(nv+dddd2[z]['j']),(nv+dddd2[z]['i'])]
        det1_spin = np.array[1,1,1,1]
        det2_spin = np.array[1,1,1,1]

        dip_mo =  calculate_dip_between_det1_det2(det1_ind,det2_ind,det1_spin,det2_spin)
        sum = sum + dddd1[z]['w']*dddd2[z]['w']*dip_mo 

    for z in range(len(udud1[:]['i'])):
        det1_ind=np.array[(nv+1-udud1[z]['b']),(nv+1-udud1[z]['a']),(nv+udud1[z]['j']),(nv+uddu1[z]['i'])]
        det2_ind=np.array[(nv+1-udud2[z]['b']),(nv+1-udud2[z]['a']),(nv+udud2[z]['j']),(nv+udud2[z]['i'])]
        det1_spin = np.array[1,0,1,0] 
        det2_spin = np.array[1,0,1,0]

        dip_mo =  calculate_dip_between_det1_det2(det1_ind,det2_ind,det1_spin,det2_spin)
        sum = sum + udud1[z]['w']*udud2[z]['w']*dip_mo

    for z in range(len(dudu1[:]['i'])):
        det1_ind=np.array[(nv+1-dudu1[z]['b']),(nv+1-dudu1[z]['a']),(nv+dudu1[z]['j']),(nv+dudu1[z]['i'])]
        det2_ind=np.array[(nv+1-dudu2[z]['b']),(nv+1-dudu2[z]['a']),(nv+dudu2[z]['j']),(nv+dudu2[z]['i'])]
        det1_spin = np.array[0,1,0,1]
        det2_spin = np.array[0,1,0,1]

        dip_mo =  calculate_dip_between_det1_det2(det1_ind,det2_ind,det1_spin,det2_spin)
        sum = sum + dudu1[z]['w']*dudu2[z]['w']*dip_mo

    for z in range(len(uddu1[:]['i'])):
        det1_ind=np.array[(nv+1-uddu1[z]['b']),(nv+1-uddu1[z]['a']),(nv+uddu1[z]['j']),(nv+uddu1[z]['i'])]
        det2_ind=np.array[(nv+1-uddu2[z]['b']),(nv+1-uddu2[z]['a']),(nv+uddu2[z]['j']),(nv+uddu2[z]['i'])]
        det1_spin = np.array[0,1,1,0]
        det2_spin = np.array[0,1,1,0]

        dip_mo =  calculate_dip_between_det1_det2(det1_ind,det2_ind,det1_spin,det2_spin)
        sum = sum + uddu1[z]['w']*uddu2[z]['w']*dip_mo

    for z in range(len(duud1[:]['i'])):
        det1_ind=np.array[(nv+1-duud1[z]['b']),(nv+1-duud1[z]['a']),(nv+duud1[z]['j']),(nv+duud1[z]['i'])]
        det2_ind=np.array[(nv+1-duud2[z]['b']),(nv+1-duud2[z]['a']),(nv+duud2[z]['j']),(nv+duud2[z]['i'])]
        det1_spin = np.array[1,0,0,1]
        det2_spin = np.array[1,0,0,1]

        dip_mo =  calculate_dip_between_det1_det2(det1_ind,det2_ind,det1_spin,det2_spin)
        sum = sum + duud1[z]['w']*duud2[z]['w']*dip_mo

    for z in range(len(ud1[:]['i'])):
        det1_ind=np.array[(nv+1-ud1[z]['a']),(nv+ud1[z]['i'])]
        det2_ind=np.array[(nv+1-ud2[z]['a']),(nv+ud2[z]['i'])]
        det1_spin = np.array[1,0]
        det2_spin = np.array[1,0]

        dip_mo =  calculate_dip_between_det1_det2(det1_ind,det2_ind,det1_spin,det2_spin)
        sum = sum + ud1[z]['w']*ud2[z]['w']*dip_mo

    for z in range(len(du1[:]['i'])):
        det1_ind=np.array[(nv+1-du1[z]['a']),(nv+du1[z]['i'])]
        det2_ind=np.array[(nv+1-du2[z]['a']),(nv+du2[z]['i'])]
        det1_spin = np.array[0,1]
        det2_spin = np.array[0,1]

        dip_mo =  calculate_dip_between_det1_det2(det1_ind,det2_ind,det1_spin,det2_spin)
        sum = sum + du1[z]['w']*du2[z]['w']*dip_mo                 

    """




"""      
def dip_det1_det2(occ_det1,occ_det2,mu_matrix):
    
    given the occupation arrays of det calculate the dipole moment

    parameters:
    occ_det1: occupation array of det1
    occ_det2: occupation array of det2
    return:
    dip_mo =  <det2|p|det1>
 
    


    diff = np.where(det1 != det2)[0]

    if len(diff) == 0:
        # Determinants are the same: expectation value
        occ_indices = np.where(det1 == 1)[0]
        return np.sum(mu_matrix[occ_indices[:, None], occ_indices])
    
    if(len(diff) ==2):
         i, a = diff

         if det1[i] == 1 and det2[a] == 1 and det1[a] == 0 and det2[i] == 0:
            # Count number of 1s (occupied orbitals) between i and a in det1
            if i < a:
                sign = (-1) ** np.sum(det1[i+1:a])
            else:
                sign = (-1) ** np.sum(det1[a+1:i])
            return sign * mu_matrix[i, a]

        # The other way around: a occupied in det1, i in det2
         elif det1[a] == 1 and det2[i] == 1 and det1[i] == 0 and det2[a] == 0:
            if a < i:
                sign = (-1) ** np.sum(det1[a+1:i])
            else:
                sign = (-1) ** np.sum(det1[i+1:a])
            return sign * mu_matrix[a, i]

         else:
            return 0.0

 """
    




