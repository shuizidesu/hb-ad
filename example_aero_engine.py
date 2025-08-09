import torch
import time
import matplotlib.pyplot as plt
from torch.func import jacrev
from scipy.io import loadmat
import numpy as np

torch.set_default_dtype(torch.double)

'''Config and time-invariant parameter'''
device = 'cuda'
incremental_tolerance = 1e-12
residual_tolerance = 1e-12
max_epoch = 500
solution_step = 1000

'''Define the sample time series for the FFT'''
time_series = torch.linspace(0, 10 * torch.pi, 2**7, device=device)
num_time_steps = time_series.shape[0]
num_dof= 284

harmonic_frequency = torch.tensor([1.0, 1.2], device=device)
num_harmonic = harmonic_frequency.shape[0]
num_harmonic_term = num_harmonic * 2 + 1
frequency_index = [5, 6]

'''Harmonic basis functions and their derivatives'''
hb_term_x = torch.ones([num_harmonic_term, num_time_steps], device=device)
hb_term_dx = torch.zeros([num_harmonic_term, num_time_steps], device=device)
hb_term_ddx = torch.zeros([num_harmonic_term, num_time_steps], device=device)

hb_term_x[1:num_harmonic+1, :] = torch.cos(harmonic_frequency.unsqueeze(1) * time_series)
hb_term_x[num_harmonic+1:num_harmonic_term, :] = torch.sin(harmonic_frequency.unsqueeze(1) * time_series)

hb_term_dx[1:num_harmonic+1, :] = (-harmonic_frequency.unsqueeze(1) *
                                   torch.sin(harmonic_frequency.unsqueeze(1) * time_series))
hb_term_dx[num_harmonic+1:num_harmonic_term, :] = (harmonic_frequency.unsqueeze(1) *
                                                   torch.cos(harmonic_frequency.unsqueeze(1) * time_series))

hb_term_ddx[1:num_harmonic+1, :] = (-harmonic_frequency.unsqueeze(1) ** 2 *
                                    torch.cos(harmonic_frequency.unsqueeze(1) * time_series))
hb_term_ddx[num_harmonic+1:num_harmonic_term, :] = (-harmonic_frequency.unsqueeze(1) ** 2 *
                                                    torch.sin(harmonic_frequency.unsqueeze(1) * time_series))


'''Load matrices from MATLAB and convert to torch tensor'''
def load_matlab_matrices(mat_path):
    mat_data = loadmat(mat_path)
    M_np = mat_data['M'].astype(np.float64)
    K_np = mat_data['K'].astype(np.float64)
    C_np = mat_data['C'].astype(np.float64)
    J_np = mat_data['J'].astype(np.float64)

    M = torch.tensor(M_np, device=device, dtype=torch.double)
    K = torch.tensor(K_np, device=device, dtype=torch.double)
    C = torch.tensor(C_np, device=device, dtype=torch.double)
    J = torch.tensor(J_np, device=device, dtype=torch.double)

    return M, K, C, J


'''Define the system equation'''
class AeroEngineSystem(torch.nn.Module):
    def __init__(self, mat_path="./aero_engine_system_parameter_matrix.mat"):
        super().__init__()
        self.M, self.K, self.C, self.J = load_matlab_matrices(mat_path)
        super(AeroEngineSystem, self).__init__()
        self.lmda = torch.tensor(1.2, device=device, dtype=torch.double)
        self.Lp_Disk_e = torch.tensor([20e-6,20e-6,30e-6], device=device, dtype=torch.double)
        self.Hp_Disk_e = torch.tensor([0e-6, 100e-6, 0e-6, 0e-6, 0e-6, 0e-6, 134e-6, 10e-6], device=device,
                                dtype=torch.double)
        self.Lp_Disk_m = torch.tensor([50.292, 51.1704, 73.115], device=device, dtype=torch.double)
        self.Hp_Disk_m  = torch.tensor([38.2962, 12.1848, 9.1364, 8.6804,8.2476,8.2278,9.4462,78.436], device=device,
                                 dtype=torch.double)
        self.Lp_Disk_Loc = torch.tensor([2, 3, 19], device=device, dtype=torch.long)
        self.Hp_Disk_Loc = torch.tensor(list(range(23, 30)) + [33], device=device, dtype=torch.long)

        self.BearingPara_NodeI = 18
        self.BearingPara_NodeO = 35
        self.BearingPara_Di = 118.94e-3
        self.BearingPara_Do = 164.064e-3
        self.BearingPara_Nb = 28
        self.BearingPara_Deltab = 2e-6
        self.BearingPara_Kb = 2.5e8

    def initialize_parameters(self, omega1):
        self.omega1 = omega1
        self.omega2 = omega1 * self.lmda
        self.omega_c = ((self.BearingPara_Di + self.lmda * self.BearingPara_Do) /
                        ( self.BearingPara_Di + self.BearingPara_Do))

    def force(self):
        FinbL = torch.zeros([num_dof, num_time_steps], device=device)
        FinbH = torch.zeros([num_dof, num_time_steps], device=device)
        Lp_Me = self.Lp_Disk_e*self.Lp_Disk_m*self.omega1**2
        Hp_Me = self.Hp_Disk_e * self.Hp_Disk_m*self.omega1**2
        FinbL[2 * self.Lp_Disk_Loc - 1-1, :] = Lp_Me.unsqueeze(1) *torch.cos(time_series)
        FinbL[2 * self.Lp_Disk_Loc - 1-1 + num_dof//2, :] = Lp_Me.unsqueeze(1) * torch.sin(time_series)
        FinbH[2 * self.Hp_Disk_Loc - 1-1, :] = self.lmda**2*Hp_Me.unsqueeze(1)* torch.cos(self.lmda*time_series)
        FinbH[2 * self.Hp_Disk_Loc - 1-1 + num_dof//2, :] = (self.lmda**2*Hp_Me.unsqueeze(1) *
                                                             torch.sin(self.lmda*time_series))
        Finb = FinbL + FinbH

        return  Finb

    def nonlinearity(self, x):
        nonlinear = torch.zeros([num_dof, num_time_steps], device=device)

        theta = (2*torch.pi/
                 self.BearingPara_Nb*torch.linspace(0,  self.BearingPara_Nb-1,  self.BearingPara_Nb, device=device, dtype=torch.double).unsqueeze(1)+
                 self.omega_c*time_series)
        delta = ((x[2*self.BearingPara_NodeI-1-1] - x[2*self.BearingPara_NodeO-1-1]).unsqueeze(0) * torch.cos(theta) +
                 (x[2*self.BearingPara_NodeI-1-1 + num_dof//2] - x[2*self.BearingPara_NodeO-1-1 + num_dof//2]) .unsqueeze(0) * torch.sin(theta) - self.BearingPara_Deltab)

        delta = torch.where(delta < 0.0, torch.tensor(0.0, device=device, dtype=torch.double), delta)

        FX = (self.BearingPara_Kb * delta ** (3 / 2) * torch.cos(theta)).sum(0)
        FY = (self.BearingPara_Kb * delta ** (3 / 2) * torch.sin(theta)).sum(0)

        nonlinear[2*self.BearingPara_NodeI-1-1, :] = FX
        nonlinear[2*self.BearingPara_NodeI-1-1 + num_dof//2, :] = FY
        nonlinear[2*self.BearingPara_NodeO-1-1, :] = -FX
        nonlinear[2*self.BearingPara_NodeO-1-1+ num_dof//2, :] = -FY

        return nonlinear

    def calculate_residual(self, hb_coeff):
        hb_coeff = hb_coeff.view(num_dof, num_harmonic_term)

        x = hb_coeff @ hb_term_x
        dx = hb_coeff @ hb_term_dx
        ddx = hb_coeff @ hb_term_ddx

        Finb = self.force()

        residual_vector = torch.zeros([num_dof, num_harmonic_term], device=device)
        residual_fft = (torch.fft.rfft(self.omega1**2*self.M @ ddx + self.omega1*(self.C + self.omega1*self.J)@ dx + self.K @ x + self.nonlinearity(x) - Finb, dim=1)
                        * 2 / num_time_steps)
        residual_vector[:, 0] = torch.real(residual_fft[:, 0] / 2)
        residual_vector[:, 1:num_harmonic+1] = torch.real(residual_fft[:, frequency_index])
        residual_vector[:, num_harmonic+1:num_harmonic_term] = -torch.imag(residual_fft[:, frequency_index])

        residual_vector = residual_vector.view(num_dof * num_harmonic_term)
        return residual_vector

    def calculate_residual_lambda(self, Y):
        hb_coeff = Y[:-1]
        omega1 = Y[-1]

        self.initialize_parameters(omega1)

        hb_coeff = hb_coeff.view(num_dof, num_harmonic_term)

        x = hb_coeff @ hb_term_x
        dx = hb_coeff @ hb_term_dx
        ddx = hb_coeff @ hb_term_ddx

        Finb = self.force()

        residual_vector = torch.zeros([num_dof, num_harmonic_term], device=device)
        residual_fft = (torch.fft.rfft(self.omega1**2*self.M @ ddx + self.omega1*(self.C + self.omega1*self.J)@ dx + self.K @ x + self.nonlinearity(x) - Finb, dim=1)
                        * 2 / num_time_steps)
        residual_vector[:, 0] = torch.real(residual_fft[:, 0] / 2)
        residual_vector[:, 1:num_harmonic+1] = torch.real(residual_fft[:, frequency_index])
        residual_vector[:, num_harmonic+1:num_harmonic_term] = -torch.imag(residual_fft[:, frequency_index])

        residual_vector = residual_vector.view(num_dof * num_harmonic_term)
        return residual_vector


duffing_rotor = AeroEngineSystem()

s = 0.1
# Initial rotation speed
omega1 = torch.tensor(140.0, device=device)
duffing_rotor.initialize_parameters(omega1)
hb_coeff = torch.rand([num_dof*num_harmonic_term], device=device) * 1e-5

'''Initial solve'''
epoch = 0
incremental = torch.inf
residual = torch.inf
while not ((epoch > max_epoch) or (incremental < incremental_tolerance) or (residual < residual_tolerance)):
    jacobian = jacrev(duffing_rotor.calculate_residual)(hb_coeff)
    with torch.no_grad():
        residual_equation = duffing_rotor.calculate_residual(hb_coeff)
        delta_hb_coeff = torch.linalg.solve(jacobian, residual_equation)
        hb_coeff = hb_coeff - delta_hb_coeff

    incremental = torch.norm(delta_hb_coeff)
    residual = torch.norm(residual_equation)
    epoch += 1

    print(f'Initial solve: epoch={epoch}, incremental={incremental:.5e}, residual={residual:.5e}')

Y0 = torch.concat([hb_coeff, omega1.unsqueeze(0)], dim=0)
residual_v = torch.concat([torch.zeros([num_dof, num_harmonic_term], device=device).view(-1),
                          torch.tensor(1.0, device=device).unsqueeze(0)], dim=0)
start_time = time.time()

jacobian_arc = jacrev(duffing_rotor.calculate_residual_lambda)(Y0)

# Null space of the Jacobian
U, S, Vh = torch.linalg.svd(jacobian_arc)
v = Vh[-1, :]

jacobian_arc = torch.empty((jacobian_arc.shape[0] + 1, jacobian_arc.shape[1]), device=jacobian_arc.device, dtype=jacobian_arc.dtype)

'''Iterative solution process with arc-length continuation'''
for step in range(1, solution_step):
    epoch = 0

    Y = Y0 + s * v

    incremental = torch.inf
    residual = torch.inf

    jacobian_arc[-1] = v

    while not ((epoch > max_epoch) or (incremental < incremental_tolerance) or (residual < residual_tolerance)):
        jacobian_arc[:-1] = jacrev(duffing_rotor.calculate_residual_lambda)(Y)

        with torch.no_grad():
            residual_equation = torch.concat([duffing_rotor.calculate_residual_lambda(Y),
                                              ((Y-Y0) @ v - s).unsqueeze(0)], dim=0)
            delta_Y = torch.linalg.solve(jacobian_arc, residual_equation)
            Y = Y - delta_Y

        hb_coeff = Y[:-1]
        omega1 = Y[-1]

        incremental = torch.norm(delta_Y)
        residual = torch.norm(residual_equation)
        epoch += 1

        '''Adaptive arc-length parameter adjustment, if the solution is not converging'''
        if epoch > 10:
            s = max(0.5 * s, 1e-9)
            Y = Y0 + s * v
            epoch = 0

    Y0 = Y

    v1 = torch.linalg.solve(jacobian_arc, residual_v)
    v1 = v1 / torch.norm(v1)
    if v @ v1 > 0:
        v = v1
    else:
        v = -v1

    '''Adaptive arc-length parameter adjustment, if the solution is converging'''
    if epoch < 7:
        s = min(2.0 * s, 0.3)

    print(f'Solution step={step}, epoch={epoch}, incremental={incremental:.5e}, '
          f'residual={residual:.5e}, omega={omega1:.5e}, s={s:.5e}')

end_time = time.time()
print(f'Time cost: {end_time - start_time:.5f}s')
