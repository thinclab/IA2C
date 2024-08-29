import os, sys
os.chdir('log/')
import pandas as pd
HVT_SIZE = 0.106
fn = sys.argv[1]
pdffilepath = sys.argv[2]
print(fn, pdffilepath)
#df = pd.read_csv('logfile_State.csv')
df = pd.read_csv(fn)

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages

#classes = ['inside_TSR_intruder','inside_TSR_defender','inside_intruder_sense_intruder','inside_intruder_sense_defender','outside_TSR_intruder','outside_TSR_defender']

def showfig(data_i,point1,point2,inside_TSR_intruder,inside_TSR_defender,inside_intruder_sense_intruder,inside_intruder_sense_defender,outside_TSR_intruder,outside_TSR_defender, pdf):

  # "red","green","blue","yellow","pink","black","orange","purple","beige","brown"
  epsoid1=data_i

  epsoid1_0 = epsoid1.query("Agent==0")
  epsoid1_1 = epsoid1.query("Agent==1")

  plt.plot(epsoid1_0['X'],epsoid1_0['Y'],color ='tab:blue',label = "intruder")
  plt.plot(epsoid1_1['X'],epsoid1_1['Y'],color ='tab:orange',label = "defener")

  # plt.scatter(*point1,c="black",label="start")
  # plt.scatter(*point2,c="black",label='start')
  circle1 = plt.Circle((0, 0), HVT_SIZE, fill = False,color='r') #BiB changed from 0.25
  circle2 = plt.Circle((0,0), 0.5,fill = False,color='blue')

  # inside_TSR_intruder,inside_TSR_defender,inside_intruder_sense_intruder,inside_intruder_sense_defender,outside_TSR_intruder,outside_TSR_defender
  if len(inside_TSR_intruder)>0:
    x1=[]
    y1=[]
    x2=[]
    y2=[]
    for k in range(len(inside_TSR_intruder)):
      x1.append(inside_TSR_intruder[k][0])
      y1.append(inside_TSR_intruder[k][1])
      x2.append(inside_TSR_defender[k][0])
      y2.append(inside_TSR_defender[k][1])


    plt.scatter(x1,y1,c="purple",label="inside_TSR_intruder")
    plt.scatter(x2,y2,c="purple",label="inside_TSR_defender")

  if len(inside_intruder_sense_intruder)>0:
    x3=[]
    y3=[]
    x4=[]
    y4=[]
    for k in range(len(inside_intruder_sense_intruder)):
      x3.append(inside_intruder_sense_intruder[k][0])
      y3.append(inside_intruder_sense_intruder[k][1])
      x4.append(inside_intruder_sense_defender[k][0])
      y4.append(inside_intruder_sense_defender[k][1])

    plt.scatter(x3,y3,c="r",label="inside_intruder_sense_intruder")
    plt.scatter(x4,y4,c='r',label="inside_intruder_defender")


  if len(outside_TSR_intruder)>0:
    x5=[]
    y5=[]
    x6=[]
    y6=[]
    for k in range(len(outside_TSR_intruder)):
      x5.append(outside_TSR_intruder[k][0])
      y5.append(outside_TSR_intruder[k][1])
      x6.append(outside_TSR_defender[k][0])
      y6.append(outside_TSR_defender[k][1])

    plt.scatter(x5,y5,c="green",label="outside_TSR_intruder")
    plt.scatter(x6,y6,c="green",label="outside_TSR_defender")
  # if len(outside_TSR_intruder)>0:
  #   plt.scatter(Extract(outside_TSR_intruder,0),Extract(outside_TSR_intruder,0),c='g',label="outside_TSR_intruder")
  #   plt.scatter(Extract(outside_TSR_defender,0),Extract(outside_TSR_defender,0),c='g',label="outside_TSR_defender")


  circle3 = plt.Circle(point1, 0.1, fill = False,color='green')
  plt.scatter(*point1,c="black",label="start")
  plt.scatter(*point2,c="black",label='start')

  fig = plt.gcf()
  ax = fig.gca()
  ax.add_patch(circle1)
  ax.add_patch(circle2)
  ax.add_patch(circle3)

  ax.set_aspect('equal')
  plt.legend()

  plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
  plt.tight_layout()
  if pdf:
    pdf.savefig(fig)
  plt.show()

intruder_sense = 0.1 #06
TSR = 0.5
start_episode = df['Episode'].min()
if pdffilepath:
  pdf = PdfPages(pdffilepath)
else:
  pdf = None
for i in range(start_episode, start_episode+5):

  data_i=df.query("Episode==@i")
  len_ep=len(data_i)# get the length of this episode

  epsoid1_0 = data_i.query("Agent==0")
  epsoid1_1 = data_i.query("Agent==1")
  index1=data_i.index[0]
  curent_start_point1=(df.iloc[index1]['X'],df.iloc[index1]['Y'])
  curent_start_point2=(df.iloc[index1+1]['X'],df.iloc[index1+1]['Y'])

  inside_TSR_intruder=[]
  inside_TSR_defender=[]

  inside_intruder_sense_intruder=[]
  inside_intruder_sense_defender=[]

  outside_TSR_intruder=[]
  outside_TSR_defender=[]

  # intruder_sense = 0.106
  # TSR = 0.5
  print("len epsoid1_0:",len(epsoid1_0))
  for inde in range(len(epsoid1_0)):
    if (epsoid1_0.iloc[inde]['X'])**2+(epsoid1_0.iloc[inde]['Y'])**2 > (TSR**2):
      outside_TSR_intruder.append((epsoid1_0.iloc[inde]['X'],epsoid1_0.iloc[inde]['Y']))
      outside_TSR_defender.append((epsoid1_1.iloc[inde]['X'],epsoid1_1.iloc[inde]['Y']))

    if (epsoid1_0.iloc[inde]['X'])**2+(epsoid1_0.iloc[inde]['Y'])**2 <= float(TSR*TSR):
      inside_TSR_intruder.append((epsoid1_0.iloc[inde]['X'],epsoid1_0.iloc[inde]['Y']))
      inside_TSR_defender.append((epsoid1_1.iloc[inde]['X'],epsoid1_1.iloc[inde]['Y']))

    if (epsoid1_0.iloc[inde]['X']-epsoid1_1.iloc[inde]['X'])**2 + (epsoid1_0.iloc[inde]['Y']-epsoid1_1.iloc[inde]['Y'])**2 <= float(intruder_sense*intruder_sense):
      inside_intruder_sense_intruder.append((epsoid1_0.iloc[inde]['X'],epsoid1_0.iloc[inde]['Y']))
      inside_intruder_sense_defender.append((epsoid1_1.iloc[inde]['X'],epsoid1_1.iloc[inde]['Y']))

  showfig(data_i,curent_start_point1,curent_start_point2,inside_TSR_intruder,inside_TSR_defender,inside_intruder_sense_intruder,inside_intruder_sense_defender,outside_TSR_intruder,outside_TSR_defender, pdf)

# remember to close the object to ensure writing multiple plots
if pdf:
  pdf.close()
