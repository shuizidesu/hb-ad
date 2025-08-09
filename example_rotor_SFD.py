import torch
import time
import matplotlib.pyplot as plt
from torch.func import jacrev
from scipy.io import loadmat
import numpy as np
start = time.time()

torch.set_default_dtype(torch.double)

'''Config and time-invariant parameter'''
device = 'cuda'
incremental_tolerance = 1e-14
residual_tolerance = 1e-12
max_epoch = 500
solution_step = 1000

time_series = torch.linspace(0, 2 * torch.pi, 2**8, device=device)
num_time_steps = time_series.shape[0]
num_dof= 4

harmonic_frequency = torch.linspace(1, 10, steps=10, device=device)
num_harmonic = harmonic_frequency.shape[0]
num_harmonic_term = num_harmonic * 2 + 1
frequency_index = list(range(1, 11))

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


'''Load the system matrices from a MATLAB file and convert them to PyTorch tensors'''
def load_matlab_matrices(mat_path):
    mat_data = loadmat(mat_path)
    M_np = mat_data['M'].astype(np.float64)
    K_np = mat_data['K'].astype(np.float64)
    C_np = mat_data['C'].astype(np.float64)
    J_np = mat_data['J'].astype(np.float64)
    Fg_np = mat_data['Fg'].astype(np.float64)

    M = torch.tensor(M_np, device=device, dtype=torch.double)
    K = torch.tensor(K_np, device=device, dtype=torch.double)
    C = torch.tensor(C_np, device=device, dtype=torch.double)
    J = torch.tensor(J_np, device=device, dtype=torch.double)
    Fg = torch.tensor(Fg_np, device=device, dtype=torch.double)

    return M, K, C, J, Fg


'''Define the system equation'''
class RotorSFD(torch.nn.Module):
    def __init__(self, mat_path = "./rotor_SFD_system_parameter_matrix.mat"):
        super().__init__()
        self.M, self.K, self.C, self.J, self.Fg = load_matlab_matrices(mat_path)
        super(RotorSFD, self).__init__()
        self.Oil_film_clearance = torch.tensor(5.0e-4, device=device, dtype=torch.double)
        self.Mu = torch.tensor(6.76e-3, device=device, dtype=torch.double)
        self.Journal_radius = torch.tensor(39.15e-3, device=device, dtype=torch.double)
        self.Damper_length = torch.tensor(15e-3, device=device, dtype=torch.double)
        self.l1 = torch.tensor(0.894, device=device, dtype=torch.double)
        self.M_e = torch.tensor([65.08e-5], device=device, dtype=torch.double)

    def initialize_parameters(self, omega1):
        self.omega1 = omega1

    def force(self):

        Finb = torch.zeros([num_dof, num_time_steps], device=device)
        Disk_Me = self.M_e * self.omega1 ** 2
        Finb[0, :] = Disk_Me.unsqueeze(1) * torch.cos(time_series)
        Finb[1, :] = Disk_Me.unsqueeze(1) * torch.sin(time_series)

        return  Finb

    def nonlinearity(self, x, dx):
        nonlinear = torch.zeros([num_dof, num_time_steps], device=device)
        '''Calculate the nonlinear force due to the oil film damping in the SFD'''
        x1 = (x[0] + x[3] * self.l1).unsqueeze(0)
        y1 = (x[1] - x[2] * self.l1).unsqueeze(0)
        dx1 = (dx[0] + dx[3] * self.l1).unsqueeze(0)
        dy1 = (dx[1] - dx[2] * self.l1).unsqueeze(0)
        ee = (x1 ** 2 + y1 ** 2) ** (1 / 2)
        rr = ee / self.Oil_film_clearance
        drr = (x1 * dx1 + y1 * dy1) / (self.Oil_film_clearance * ee )
        dpsi = (x1 * dy1 - y1 * dx1) / (x1 ** 2 + y1 ** 2)
        theta1 = torch.atan(-drr / (rr * dpsi))
        theta2 = theta1 + torch.pi
        '''Using Gauss quadrature for numerical integration to handle the Sommerfeld integral'''
        pnodes = [-0.991455371120813,
                  -0.949107912342759,
                  -0.864864423359769,
                  -0.741531185599394,
                  -0.586087235467691,
                  -0.405845151377397,
                  -0.207784955007899,
                  0,
                  0.207784955007899,
                  0.405845151377397,
                  0.586087235467691,
                  0.741531185599394,
                  0.864864423359769,
                  0.949107912342759,
                  0.991455371120813]

        pwt = [0.0229353220105292,
               0.0630920926299785,
               0.104790010322250,
               0.140653259715526,
               0.169004726639268,
               0.190350578064785,
               0.204432940075299,
               0.209482141084728,
               0.204432940075299,
               0.190350578064785,
               0.169004726639268,
               0.140653259715526,
               0.104790010322250,
               0.0630920926299785,
               0.0229353220105292]

        I11, I02, I20 = 0.0, 0.0, 0.0
        '''Calculate the integral terms using Gauss quadrature'''
        for i in range(15):
            cita = (theta1 + theta2) / 2 + (theta2 - theta1) * pnodes[i] / 2
            denominator = (1 + rr * torch.cos(cita)) ** 3
            I11 += (torch.pi / 2) * (pwt[i] * torch.sin(cita) * torch.cos(cita)) / denominator
            I02 += (torch.pi / 2) * (pwt[i] * torch.cos(cita) ** 2) / denominator
            I20 += (torch.pi / 2) * (pwt[i] * torch.sin(cita) ** 2) / denominator
        Fr = (self.Mu * self.Journal_radius * self.Damper_length ** 3) * (I11 * dpsi * rr + I02 * drr) / (self.Oil_film_clearance ** 2)
        Ft = (self.Mu * self.Journal_radius * self.Damper_length ** 3) * (I20 * dpsi * rr + I11 * drr) / (self.Oil_film_clearance ** 2)
        Fx = (Fr * x1 - Ft * y1) / self.Oil_film_clearance
        Fy = (Fr * y1 + Ft * x1) / self.Oil_film_clearance

        nonlinear[0, :] = -Fx.squeeze()
        nonlinear[1, :] = -Fy.squeeze()
        nonlinear[2, :] = (Fy * self.l1).squeeze()
        nonlinear[3, :] = (-Fx * self.l1).squeeze()

        return nonlinear

    def calculate_residual(self, hb_coeff):
        hb_coeff = hb_coeff.view(num_dof, num_harmonic_term)

        x = hb_coeff @ hb_term_x
        dx = hb_coeff @ hb_term_dx
        ddx = hb_coeff @ hb_term_ddx

        Finb = self.force()

        residual_vector = torch.zeros([num_dof, num_harmonic_term], device=device)
        residual_fft = (torch.fft.rfft(self.omega1 ** 2 * self.M @ ddx + self.omega1*(self.C + self.omega1*self.J) @ dx + self.K @ x - self.nonlinearity(x,dx) - Finb, dim=1)
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
        residual_fft = (torch.fft.rfft(self.omega1**2*self.M @ ddx + self.omega1*(self.C + self.omega1*self.J)@ dx + self.K @ x - self.nonlinearity(x,dx) - Finb, dim=1)
                        * 2 / num_time_steps)
        residual_vector[:, 0] = torch.real(residual_fft[:, 0] / 2)
        residual_vector[:, 1:num_harmonic+1] = torch.real(residual_fft[:, frequency_index])
        residual_vector[:, num_harmonic+1:num_harmonic_term] = -torch.imag(residual_fft[:, frequency_index])

        residual_vector = residual_vector.view(num_dof * num_harmonic_term)
        return residual_vector


duffing_rotor = RotorSFD()

s = 0.1
omega1 = torch.tensor(350.0, device=device)
duffing_rotor.initialize_parameters(omega1)
hb_coeff = torch.rand([num_dof*num_harmonic_term], device=device) * 1e-5

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

omega_list = []
y_list = []

jacobian_arc = torch.empty((jacobian_arc.shape[0] + 1, jacobian_arc.shape[1]), device=jacobian_arc.device, dtype=jacobian_arc.dtype)

for step in range(1, solution_step):
    epoch = 0

    Y = Y0 + s * v

    incremental = torch.inf
    residual = torch.inf

    jacobian_arc[-1] = v

    while not ((epoch > max_epoch) or (incremental < incremental_tolerance) or (residual < residual_tolerance)):
        jacobian_arc[:-1] = jacrev(duffing_rotor.calculate_residual_lambda)(Y)
        # jacobian_arc = torch.concat([jacobian_arc, v.unsqueeze(0)], dim=0)
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

        if epoch > 8:
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

    if epoch < 4:
        s = min(2.0 * s, 0.3)

    print(f'Solution step={step}, epoch={epoch}, incremental={incremental:.5e}, '
          f'residual={residual:.5e}, omega={omega1:.5e}, s={s:.5e}')

end_time = time.time()
print(f'Time cost: {end_time - start_time:.5f}s')
