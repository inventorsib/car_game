from   matplotlib import pyplot as plt
import numpy as np


def plotTraj(a, b, d, R, isPositive):
  #first arc
  x = [0]
  y = [0]

  if a < b:
    ang = np.arange( a, b, 0.001)
    x = np.append(x, np.cos(ang)*R - np.cos(ang[0])*R )
    y = np.append(y, np.sin(ang)*R - np.sin(ang[0])*R )

  #second arc
  ang = np.arange( b-np.pi, np.pi/2, -0.001)

  x = np.append(x, x[-1]     + np.cos(ang)*R - np.cos(ang[0])*R)
  y = np.append(y, y[-1] + d + np.sin(ang)*R - np.sin(ang[0])*R)

  #line should be 0 
  if isPositive:
    y = -y
  y = y - y[-1] 
  plt.plot(x,y)
  plt.grid()

  apts = np.array([x,y]) # Make it a numpy array
  lengths = np.sqrt(np.sum(np.diff(apts, axis=1)**2, axis=0)) # Length between corners
  total_length = np.sum(lengths)
  #print('path_length:', total_length)
  return x[-1]

def plotDubinsTraj(xte, hte, R):

  #assume we have down left quarted
  A =  hte
  D = abs(xte)
  
  #mean we have apper left quarter
  if xte > 0:
    A = -hte  

  if A >= 90 or  A <= -90:
    print('Do not support A>=90')
  
  if A==0 and D == 0:
    print('Already on line')
    return 0

  a = np.deg2rad(A)
  b = 0
  dline = 0
  one_arc_dist = R*(1 - np.cos(a)) #distance to use onlt 1 arc to go to line
  if D < one_arc_dist and a > 0: #cross line 
    D = one_arc_dist - D  #find max line distance
    b = np.arcsin(1 - D/(2*R))
    a = -a
    xte = 0 #(not to swap data)
  else : # do not cross line
    l1 = R*(1 + np.cos(a)) # maximum distance to move without line interval (only arc1+arc2)
    dline = D - l1
    if l1 > D:
      b = np.arcsin((l1-D)/(2*R))
      dline = 0
  
  maxx = plotTraj(1.5*np.pi+a , 2*np.pi-b, dline, R, xte > 0)
  
  arc1 = abs( (2*np.pi-b) - (1.5*np.pi+a) )
  arc2 = abs( np.pi/2 -b)
  
  d = R*(arc1 + arc2) + dline
  #print('calculated length', d)
  return maxx



def main():

  plt.figure(figsize=(15, 10))

  plt.subplot( 2,  3, 1)
  plotDubinsTraj(10, 15,  7)

  plt.subplot( 2,  3, 2)
  plotDubinsTraj(2, 70,  7)

  plt.subplot( 2,  3, 3)
  plotDubinsTraj(10, 15,  7)

  plt.subplot( 2,  3, 4)
  plotDubinsTraj(3.5, 60,  7)

  plt.subplot( 2,  3, 5)
  plotDubinsTraj(0, 0, 0)

  plt.grid()
  plt.show()

if __name__ == "__main__":
  main()
