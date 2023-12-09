import torch
import torch.nn as nn
import numpy as np
import math
import scipy.io
from torch.autograd import Variable
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import torch.nn.functional as F

# uA and vA are sediment phase x and y velocities
# uB and vB are fluid phae x and y velocities
#Alpha is sediment phase concentration
#P is pressure
# SedFoam outputs were saved at 0.1 seconds.

#use GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# CFDData.mat contains the CFD results. It contains 5 columns, U contains (uA, vA, uB, vB),  AlphaA contains AlphaA, P contains Pressure, and t is time. The dimensions of the array
# are shown below.
data = scipy.io.loadmat('CFDdata.mat')

UTemp1 = data['U']  # N x 4 x T
alphaAOLD = data['alphaA']  # N x T
timeTemp1 = data['t']  # T x 1
CoordData = data['X']  # N x 2
Pressure = data['P']  # N x 2

N = CoordData.shape[0]
T = timeTemp1.shape[0]

x_test = CoordData[:, 0:1]
y_test = CoordData[:, 1:2]

P_test = Pressure[:, 0:1]
uA_test = UTemp1[:, 0:1, 0]
uV_test = UTemp1[:, 1:2, 0]
uB_test = UTemp1[:, 2:3, 0]
vB_test = UTemp1[:, 3:4, 0]
t_test = np.ones((x_test.shape[0], x_test.shape[1]))

# Rearrange Data
XX = np.tile(CoordData[:, 0:1], (1, T))  # N x T
YY = np.tile(CoordData[:, 1:2], (1, T))  # N x T
TT = np.tile(timeTemp1, (1, N)).T  # N x T

UUA = UTemp1[:, 0, :]  # N x T
VVA = UTemp1[:, 1, :]  # N x T
UUB = UTemp1[:, 2, :]  # N x T
VVB = UTemp1[:, 3, :]  # N x T
PP = alphaAOLD  # N x T
PQ = Pressure

x = XX.flatten()[:, None]  # NT x 1
y = YY.flatten()[:, None]  # NT x 1
t = TT.flatten()[:, None]  # NT x 1
uA = UUA.flatten()[:, None]  # NT x 1
vA = VVA.flatten()[:, None]  # NT x 1
uB = UUB.flatten()[:, None]  # NT x 1
vB = VVB.flatten()[:, None]  # NT x 1
alphaA = PP.flatten()[:, None]  # NT x 1
P = PQ.flatten()[:, None]  # NT x 1
N_train = 180 # trained only on the first 180 seconds of the data


# Following part for creating training data
idx = np.random.choice(N * T, N_train, replace=False)
x_train1 = x[idx, :]
y_train1 = y[idx, :]
t_train1 = t[idx, :]
uA_train1 = uA[idx, :]
vA_train1 = vA[idx, :]
uB_train1 = uB[idx, :]
vB_train1 = vB[idx, :]
P_train1 = P[idx, :]
alphaA_train = alphaA[idx, :]
# Boundary constraints
inletIndicies = np.where(x == -0.75)
xInlet = x[inletIndicies[0]]
yInlet = y[inletIndicies[0]]
tInlet = t[inletIndicies[0]]
uAInlet = uA[inletIndicies[0]]
vAInlet = vA[inletIndicies[0]]
uBInlet = uB[inletIndicies[0]]
vBInlet = vB[inletIndicies[0]]
PInlet = P[inletIndicies[0]]
alphaAInlet = alphaA[inletIndicies[0]]

outletIndicies = np.where(x == 1)
xOutlet = x[outletIndicies[0]]
yOutlet = y[outletIndicies[0]]
tOutlet = t[outletIndicies[0]]
uAOutlet = uA[outletIndicies[0]]
vAOutlet = vA[outletIndicies[0]]
uBOutlet = uB[outletIndicies[0]]
vBOutlet = vB[outletIndicies[0]]
POutlet = P[outletIndicies[0]]
alphaAOutlet = alphaA[outletIndicies[0]]





# Create training data including boundary conditions
x_train = np.vstack((xInlet, xOutlet, x_train1))
y_train = np.vstack((yInlet, yOutlet,  y_train1))
t_train = np.vstack((tInlet, tOutlet,  t_train1))
uA_train = np.vstack((uAInlet, uAOutlet,  uA_train1))
vA_train = np.vstack((vAInlet, vAOutlet,  vA_train1))
uB_train = np.vstack((uBInlet, uBOutlet,  uB_train1))
vB_train = np.vstack((vBInlet, vBOutlet,  vB_train1))
P_train = np.vstack((PInlet, POutlet,  P_train1))
alphaA_train = np.vstack((alphaAInlet, alphaAOutlet, alphaA_train))

# Plot Training Points
plt.scatter(x_train, y_train)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Training Points')
plt.savefig('TrainPoints.png')

# convert to pytorch tensor
x_train = Variable(torch.from_numpy(x_train).float(), requires_grad=True).to(device)
y_train = Variable(torch.from_numpy(y_train).float(), requires_grad=True).to(device)
t_train = Variable(torch.from_numpy(t_train).float(), requires_grad=True).to(device)
uA_train = Variable(torch.from_numpy(uA_train).float(), requires_grad=True).to(device)
vA_train = Variable(torch.from_numpy(vA_train).float(), requires_grad=True).to(device)
uB_train = Variable(torch.from_numpy(uB_train).float(), requires_grad=True).to(device)
vB_train = Variable(torch.from_numpy(vB_train).float(), requires_grad=True).to(device)
P_train = Variable(torch.from_numpy(P_train).float(), requires_grad=True).to(device)
alphaA_train = Variable(torch.from_numpy(alphaA_train).float(), requires_grad=True).to(device)

#Create a NN model

class PINN(nn.Module):
    def __init__(self):
            super().__init__()
            self.hidden_layer1 = nn.Linear(3,20)
            self.hidden_layer2 = nn.Linear(20,20)
            self.hidden_layer3 = nn.Linear(20,20)
            self.hidden_layer3 = nn.Linear(20,20)
            self.hidden_layer4 = nn.Linear(20,20)
            self.hidden_layer5 = nn.Linear(20,20)
            self.hidden_layer6 = nn.Linear(20,20)
            self.hidden_layer7 = nn.Linear(20,20)
            self.hidden_layer8 = nn.Linear(20,20)
            self.output_layer = nn.Linear(20,6)
            self.activ = nn.ReLU()
    def forward(self,x,y,t):
            inputs = torch.cat([x.reshape(-1, 1), y.reshape(-1, 1), t.reshape(-1, 1)], axis=1)
            layer1_out = self.activ(self.hidden_layer1(inputs))
            layer2_out = self.activ(self.hidden_layer2(layer1_out))
            layer3_out = self.activ(self.hidden_layer3(layer2_out))
            layer4_out = self.activ(self.hidden_layer4(layer3_out))
            layer5_out = self.activ(self.hidden_layer5(layer4_out))
            layer6_out = self.activ(self.hidden_layer6(layer5_out))
            layer7_out = self.activ(self.hidden_layer7(layer6_out))
            layer8_out = self.activ(self.hidden_layer8(layer7_out))
            output = self.output_layer((layer8_out))

            return output

net = PINN()
net = net.to(device)
mse_cost_function = torch.nn.MSELoss()
# Adam Optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=0.00001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False, foreach=None, maximize=False, capturable=False, differentiable=False, fused=None)

# Define loss functions
def function (x, y, t, net):
    res = net(x, y, t)

    uATemp, vATemp, ubTemp, vbTemp, alphaA, PTemp = res[:, 0:1], res[:, 1:2],res[:, 2:3], res[:, 3:4], res[:, 4:5],  res[:, 5:6]
    nuF = 0.00001
    nuS = 0.001
    rhoF = 1000
    rhoS = 2650
    Beta = (1-alphaA)
         
    uA = uATemp.clone().requires_grad_(True)
    vA = vATemp.clone().requires_grad_(True)
    uB = ubTemp.clone().requires_grad_(True)
    vB = vbTemp.clone().requires_grad_(True)
    P = PTemp.clone().requires_grad_(True)
    alphaA = alphaA.clone().requires_grad_(True)

    # solid phase concentration constrained to be positive and less than maximum packing fraction (0.63)
    alphaA[alphaA < 0] = 0
    alphaA[alphaA > 0.63] = 0.63
    
    
    #_x represents the gradient of variable with respect to x.
    uA_x = torch.autograd.grad(uA, x, grad_outputs=torch.ones_like(uA), create_graph=True)[0]
    uA_y = torch.autograd.grad(uA, y, grad_outputs=torch.ones_like(uA), create_graph=True)[0]

    vA_x = torch.autograd.grad(vA, x, grad_outputs=torch.ones_like(vA), create_graph=True)[0]
    vA_y = torch.autograd.grad(vA, y, grad_outputs=torch.ones_like(vA), create_graph=True)[0]

    uAalphaA_x = torch.autograd.grad(uA*alphaA, x, grad_outputs=torch.ones_like(uA), create_graph=True)[0]
    vAalphaA_y = torch.autograd.grad(vA*alphaA, y, grad_outputs=torch.ones_like(uA), create_graph=True)[0]

    uAuA_x = torch.autograd.grad(uA*uA*rhoS*alphaA, x, grad_outputs=torch.ones_like(uA), create_graph=True)[0]
    uAvA_x = torch.autograd.grad(uA*vA*rhoS*alphaA, x, grad_outputs=torch.ones_like(uA), create_graph=True)[0]
    uA_t = torch.autograd.grad(uA*alphaA*rhoS, t, grad_outputs=torch.ones_like(uA), create_graph=True)[0]

    uAvA_x = torch.autograd.grad(vA*uA*rhoS*alphaA, x, grad_outputs=torch.ones_like(vA), create_graph=True)[0]
    vAvA_y = torch.autograd.grad(vA*vA*rhoS*alphaA, x, grad_outputs=torch.ones_like(vA), create_graph=True)[0]
    vA_t = torch.autograd.grad(vA*alphaA*rhoS, t, grad_outputs=torch.ones_like(vA), create_graph=True)[0]
    
    alphaA_x = torch.autograd.grad(alphaA*uA, x, grad_outputs=torch.ones_like(uA), create_graph=True)[0]
    alphaA_y = torch.autograd.grad(alphaA*vA, y, grad_outputs=torch.ones_like(uA), create_graph=True)[0]
    alphaA_t = torch.autograd.grad(alphaA, t, grad_outputs=torch.ones_like(uA), create_graph=True)[0]

    TxxS = rhoS*alphaA*nuS*(2*uA_x)
    TxyS = rhoS*alphaA*nuS*(uA_y+vA_x)
    TyyS = rhoS*alphaA*nuS*(2*vA_y)

    PxS = (TxxS + TxyS + P )
    PyS = (TxyS + TyyS + P )

    # Continuity Equations for fluid phase

    Cont_SX = alphaA_t + uAalphaA_x
    Cont_SY = alphaA_t + vAalphaA_y

    PxS_x = torch.autograd.grad(PxS, x, grad_outputs=torch.ones_like(PxS), create_graph=True)[0]
    PxS_y = torch.autograd.grad(PyS, y, grad_outputs=torch.ones_like(PyS), create_graph=True)[0]

    # fluid phase momentum equations

    PDE_S_X = uA_t + (uAuA_x + uAvA_x) + (alphaA) *  PxS_x + PxS * (1-alphaA) + alphaA*(1-alphaA) * (uA-uA) * torch.norm(uA-uA, p='fro')*0.44*(rhoS/0.002) - alphaA_x
    PDE_S_Y = vA_t + (uAvA_x + vAvA_y) + (alphaA) *  PxS_y + PyS * (1-alphaA) - 9.81 * rhoS * (1-alphaA) + alphaA*(1-alphaA) * (uA-uA) * torch.norm(uA-uA, p='fro')*0.44*(rhoS/0.002) -  alphaA_y

    # Eqn below belongs to fluid phase

    uB_x = torch.autograd.grad(uB, x, grad_outputs=torch.ones_like(uB), create_graph=True)[0]
    uB_y = torch.autograd.grad(uB, y, grad_outputs=torch.ones_like(uB), create_graph=True)[0]

    vB_x = torch.autograd.grad(vB, x, grad_outputs=torch.ones_like(vB), create_graph=True)[0]
    vB_y = torch.autograd.grad(vB, y, grad_outputs=torch.ones_like(vB), create_graph=True)[0]

    uBBeta_x = torch.autograd.grad(uB*Beta, x, grad_outputs=torch.ones_like(uB), create_graph=True)[0]
    vBBeta_y = torch.autograd.grad(vB*Beta, y, grad_outputs=torch.ones_like(uB), create_graph=True)[0]

    uBuB_x = torch.autograd.grad(uB*uB*rhoF*Beta, x, grad_outputs=torch.ones_like(uB), create_graph=True)[0]
    uBvB_x = torch.autograd.grad(uB*vB*rhoF*Beta, x, grad_outputs=torch.ones_like(uB), create_graph=True)[0]
    uB_t = torch.autograd.grad(uB*Beta*rhoF, t, grad_outputs=torch.ones_like(uB), create_graph=True)[0]

    uBvB_x = torch.autograd.grad(vB*uB*rhoF*Beta, x, grad_outputs=torch.ones_like(vB), create_graph=True)[0]
    vBvB_y = torch.autograd.grad(vB*vB*rhoF*Beta, x, grad_outputs=torch.ones_like(vB), create_graph=True)[0]
    vB_t = torch.autograd.grad(vB*Beta*rhoF, t, grad_outputs=torch.ones_like(vB), create_graph=True)[0]
    
    Beta_x = torch.autograd.grad(Beta*uB, x, grad_outputs=torch.ones_like(uB), create_graph=True)[0]
    Beta_y = torch.autograd.grad(Beta*vB, y, grad_outputs=torch.ones_like(uB), create_graph=True)[0]
    Beta_t = torch.autograd.grad(Beta, t, grad_outputs=torch.ones_like(uB), create_graph=True)[0]

    TxxF = rhoS*Beta*nuF*(2*uB_x)
    TxyF = rhoS*Beta*nuF*(uB_y+vB_x)
    TyyF = rhoS*Beta*nuF*(2*vB_y)

    PxF = (TxxF + TxyF +P )
    PyF = (TxyF + TyyF +P )

    # Continuity Equations for fluid phase

    Cont_FX = Beta_t + uBBeta_x
    Cont_FY = Beta_t + vBBeta_y

    PxF_x = torch.autograd.grad(PxF, x, grad_outputs=torch.ones_like(PxF), create_graph=True)[0]
    PxF_y = torch.autograd.grad(PyF, y, grad_outputs=torch.ones_like(PyF), create_graph=True)[0]

    # fluid phase momentum equations

    PDE_F_X = uB_t + (uBuB_x + uBvB_x) + (Beta) *  PxF_x +  (1-alphaA) + alphaA*(1-alphaA) * (uB-uA) * torch.norm(uB-uA, p='fro')*0.44*(rhoF/0.002) - Beta_x
    PDE_F_Y = vB_t + (uBvB_x + vBvB_y) + (Beta) *  PxF_y +  (1-alphaA) - 9.81 * rhoS * (1-alphaA) + alphaA*(1-alphaA) * (uB-uA) * torch.norm(uB-uA, p='fro')*0.44*(rhoF/0.002) -  Beta_y

    return uA, vA, uB, vB, alphaA, PDE_S_X, PDE_S_Y,  PDE_F_X, PDE_F_Y, Cont_SX, Cont_SY, Cont_FX, Cont_FY      

zeros_train = np.zeros((40180,1))
zeros_train = torch.from_numpy(zeros_train).float().to(device)
zeros_train.requires_grad = True

#Train model

losses = []
# Train model
iterations = 400000
for epoch in range(iterations):
    optimizer.zero_grad()
    uA_out, vA_out, uB_out, vB_out, AlphaA_out, PDE_S_X_out, PDE_S_Y_out, PDE_F_X_out, PDE_F_Y_out, Cont_SX, Cont_SY, Cont_FX, Cont_FY = function(x_train, y_train, t_train, net)
    
    # cost function for the data
    mse_uA = mse_cost_function(uA_out, uA_train)
    mse_vA = mse_cost_function(vA_out, vA_train)
    mse_uB = mse_cost_function(uB_out, uB_train)
    mse_vB = mse_cost_function(vB_out, vB_train)
    mse_alphaA = mse_cost_function(AlphaA_out, alphaA_train)
    
    # Cost function for momentum equation
    mse_PDE_S_X = mse_cost_function(PDE_S_X_out, zeros_train)
    mse_PDE_S_Y = mse_cost_function(PDE_S_Y_out, zeros_train)
    mse_PDE_F_X = mse_cost_function(PDE_F_X_out, zeros_train)
    mse_PDE_F_Y = mse_cost_function(PDE_F_Y_out, zeros_train)

    # Cost function for continuity equation
    mse_Cont_SX = mse_cost_function(Cont_SX, zeros_train)
    mse_Cont_SY = mse_cost_function(Cont_SY, zeros_train)
    mse_Cont_FX = mse_cost_function(Cont_FX, zeros_train)
    mse_Cont_FY = mse_cost_function(Cont_FY, zeros_train)
    
    loss = mse_uA + mse_vA + mse_uB + mse_vB + mse_PDE_S_X + mse_PDE_S_Y + mse_PDE_F_X + mse_PDE_F_Y + mse_Cont_SX + mse_Cont_SY + mse_Cont_FX + mse_Cont_FY + mse_alphaA
    loss.backward()
    optimizer.step()
    with torch.autograd.no_grad():
        print(f'Iteration {epoch}/{iterations}: Loss = {loss.item()}')
        losses.append((epoch, loss.item()))
with open("training_loss.txt", "w") as file:
    for iteration, loss in losses:
        file.write(f"{iteration};{loss}\n")

torch.save(net.state_dict(), 'SedFoamPINN.pt')
