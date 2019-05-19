
# example of training a final classification model
from sklearn.linear_model import LogisticRegression
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# generate 2d classification dataset
X, y = make_blobs(n_samples=100, n_features=2, random_state=3)
print(y)

myx=[]
myy=[]
myz=y
for i in range(len(X)):
    myx.append(X[i][0])
    myy.append(X[i][1])
# fit final model
model = LogisticRegression()
model.fit(X, y)
# # new instances where we do not know the answer
Xnew, _ = make_blobs(n_samples=15, centers=2, n_features=2, random_state=3)

print(X[0][0])
print("=================")
# make a prediction
ynew = model.predict(Xnew)
# show the inputs and predicted outputs
newx=[]
newy=[]
newz=ynew
for i in range(len(Xnew)):
    newx.append(Xnew[i][0])
    newy.append(Xnew[i][1])

for i in range(len(Xnew)):
	print("X=%s, Predicted=%s" % (Xnew[i], ynew[i]))


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(myx, myy, myz, c='r', marker='o')
ax.scatter(newx, newy, newz, c='b', marker='o')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()