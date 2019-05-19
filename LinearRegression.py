import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


def minMaxValue():
    print("Linear Regression:")
    minX=float(input("Min x:"))
    maxX=float(input("Max x:"))

    minY=float(input("Min y:"))
    maxY=float(input("Max y:"))
    return minX,maxX,minY,maxY
def createList(display):
    liste=[]
    key=False
    i=0
    print("To stop : 'n'")
    while(key==False):
        print(display,i,":",end='')
        thisX=str(input(""))
        if (thisX!="n"):
            liste.append(float(thisX))
            i+=1
        else:
            key=True
    return liste
   
value=minMaxValue()
print("X=[",value[0],",",value[1],"] and Y=[",value[2],",",value[3],"]")
question=str(input("It's ok ? (o/n)"))
if(question=="n"):
    display()
x=createList("x");
print("x: ",x)
print()
y=createList("y");
print("y: ",y)

if(len(x)!=len(y)):
    print("x and y must be of the same length")
    exit()

x = np.array(x).reshape((-1, 1))
y = np.array(y)

model = LinearRegression()
model.fit(x, y)
r_sq = model.score(x, y)
print('a:', r_sq)

plt.plot(x, y, 'ro')
plt.plot([value[0], value[1]],[0.0, r_sq*value[1]], 'r-', lw=2)
plt.axis([value[0], value[1],value[2], value[3]])

plt.show()