import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from itertools import product

class Box:
    """variable range"""
    def __init__(self, lmin, lmax, N):
        self.lmin, self.lmax = lmin, lmax
        self.N = N
        self.dt = (lmax - lmin) / (N-1)
    
    def value(self, idx):
        assert idx < self.N
        return self.lmin + self.dt * idx
    
    def values(self):
        return np.linspace(self.lmin, self.lmax, self.N)
    
    def index(self, value):
        return int(np.round((value - self.lmin) / self.dt))

class Solver:
    def __init__(
        self, 
        sigma,
        delta,
        A,
        tbox, 
        alpha_boxes, 
        gamma, 
        normalize = False
    ):
        """Solves the problem with exponential utility

        Parameters
        ----------
        sigma : np.array
            square root of volatility 
        delta : np.array
            delta matrix
        A : np.array
            log price weights
        tbox : Box
            t range
        alpha_boxes : List[Box]
            list of alpha ranges
        gamma : float
            gamma value in the expectation
        normalize : bool, optional
            if true, policy is normalized by the maximum sum |pi_i| value, by default False
        """
        self.sigma = sigma
        self.delta = delta
        self.A = A
        self.tbox = tbox
        self.alpha_boxes = alpha_boxes
        self.omega = sigma @ sigma.T
        self.N = len(self.omega)
        self.M = len(self.alpha_boxes)
        self.boxes = [self.tbox] + self.alpha_boxes
        self.norm = normalize
        
        self.partial1 = np.ones(self.M + 1)
        for m in range(self.M):
            self.partial1[m+1] = -self.A[:, m].T @ np.diag(self.omega) / 2.0        
        
        self.partial2 = np.zeros((self.M, self.M))
        for m in range(self.M):
            for l in range(self.M):
                self.partial2[m, l] = self.A[:, m].T @ self.omega @ self.A[:, l] / 2.0

        self.partial3 = delta.T @ np.linalg.inv(self.omega) @ delta

        self.idxmult = [1] * len(self.boxes)
        m = 1
        for i in range(len(self.boxes) - 2, -1, -1):
            m *= self.boxes[i+1].N 
            self.idxmult[i] = m

        self.gamma = gamma
        self.sizes = [x.N for x in self.boxes]
        self.policy_mult = np.linalg.inv(self.omega) @ delta


    def idx(self, *args):
        return sum([self.idxmult[i] * args[i] for i in range(len(self.idxmult))])
    
    def next(self, curr_idx, at):
        return curr_idx + self.idxmult[at]
        
    def prev(self, curr_idx, at):
        return curr_idx - self.idxmult[at]
    
    def get_next(self, idxs, curr_idx, at):
        if idxs[at] == self.boxes[at].N - 1:
            return curr_idx
        return self.next(curr_idx, at)
    
    def get_prev(self, idxs, curr_idx, at):
        if idxs[at] == 0:
            return curr_idx
        return self.next(curr_idx, at)
    
    def handle_idxs(self, idxs):
        curr_idx = self.idx(*idxs)
        c = self.counter

        # partial_k
        for k in range(len(idxs)):
            if idxs[k] == self.boxes[k].N - 1:
                self.X[c, self.prev(curr_idx, k)] -= self.partial1[k] / self.boxes[k].dt
                self.X[c, curr_idx] += self.partial1[k] / self.boxes[k].dt
            else:
                self.X[c, self.next(curr_idx, k)] += self.partial1[k] / self.boxes[k].dt
                self.X[c, curr_idx] -= self.partial1[k] / self.boxes[k].dt

        # partial_kl
        for k in range(1, len(idxs)):
            for l in range(1, len(idxs)):
                v = self.partial2[k-1, l-1] / self.boxes[k].dt / self.boxes[l].dt
                if k == l: 
                    self.X[c, self.get_next(idxs, curr_idx, k)] += v
                    self.X[c, curr_idx] -= 2 * v
                    self.X[c, self.get_prev(idxs, curr_idx, k)] += v
                else:
                    v /= 4
                    n = self.get_next(idxs, self.get_next(idxs, curr_idx, k), l)
                    self.X[c, n] += v
                    n = self.get_next(idxs, self.get_prev(idxs, curr_idx, k), l)
                    self.X[c, n] -= v
                    n = self.get_prev(idxs, self.get_next(idxs, curr_idx, k), l)
                    self.X[c, n] -= v
                    n = self.get_prev(idxs, self.get_prev(idxs, curr_idx, k), l)
                    self.X[c, n] += v

        
        alpha = np.array([self.alpha_boxes[k].value(idxs[k+1]) for k in range(len(self.alpha_boxes))])
        self.b[c] -= alpha.T @ self.partial3 @ alpha / 2 / self.gamma
        self.counter += 1

    def iterate(self, k, idxs, func):
        if k == len(self.boxes):
            func(idxs)
        else:
            M = self.boxes[k].N
            if k == 0: M -= 1
            for i in range(M):
                idxs[k] = i
                self.iterate(k+1, idxs, func)

    def terminal(self, idxs):
        curr_idx = self.idx(*idxs)
        self.X[self.counter, curr_idx] = 1
        self.b[self.counter] = 0
        self.counter += 1

    def set_terminal(self, idxs, k=1):
        if k == len(self.boxes):
            self.terminal(idxs)
        else:
            for i in range(self.boxes[k].N):
                idxs[k] = i
                self.set_terminal(idxs, k+1)

    def _calc_policy(self, idxs):
        alpha = np.array([self.alpha_boxes[k].value(idxs[k+1]) for k in range(len(self.alpha_boxes))])
        p = self.policy_mult @ alpha / self.gamma
        curr_idx = self.idx(*idxs)
        for m in range(len(self.alpha_boxes)):
            if idxs[m+1] == self.alpha_boxes[m].N - 1:
                dalpha = (self.solution[curr_idx] - self.solution[self.prev(curr_idx, m+1)])/self.alpha_boxes[m].dt
            elif idxs[m+1] == 0:
                dalpha = (self.solution[self.next(curr_idx, m+1)] - self.solution[curr_idx])/self.alpha_boxes[m].dt
            else:
                dalpha = (self.solution[self.next(curr_idx, m+1)] - self.solution[self.prev(curr_idx, m+1)])/self.alpha_boxes[m].dt/2.0

            p -= self.A[:, m] * dalpha
    

        self.policy[curr_idx] = p


    def calc_policy(self):
        N = len(self.omega)
        self.policy = np.zeros((self.solution.shape[0], N))
        self.iterate(0, [0] * len(self.boxes), self._calc_policy)
        if self.norm:
            smax = np.max(np.sum(np.abs(self.policy), axis=-1))
            self.policy /= smax
        self.policy = self.policy.reshape(self.sizes + [N])


    def solve(self):
        """entry point for calculating solution"""
        s = self.tbox.N * np.prod([x.N for x in self.alpha_boxes])
        self.X = lil_matrix((s, s), dtype=np.float32)
        self.b = np.zeros(s)
        self.counter = 0
        self.iterate(0, [0] * len(self.boxes), self.handle_idxs)
        self.set_terminal([self.tbox.N - 1] + [0] * len(self.alpha_boxes))
        self.solution = spsolve(self.X.tocsr(), self.b)
        self.calc_policy()
        self.solution = self.solution.reshape(self.sizes)

    def get_policy(self, v):
        """query calculated policy"""
        idx = [x.index(v[i]) for i, x in enumerate(self.boxes)]
        for i, j in enumerate(idx):
            if not (0 <= j < self.boxes[i].N):
                print(f"value {v[i]} outside range [{self.boxes[i].lmin}, {self.boxes[i].lmax}]")
                idx[i] = max(0, min(j, self.boxes[i].N-1))
        return self.policy[tuple(idx)]


class SolverLinear(Solver):
    def __init__(self, sigma, delta, A, tbox, alpha_boxes, gamma, divers=2):
        """Linear model solution

        See utils.solver.Solver()

        Parameters
        ----------
        divers : int, optional
            Number of assets to invest in, by default 2
        """
        super().__init__(sigma, delta, A, tbox, alpha_boxes, gamma)
        for i in range(len(self.alpha_boxes)):
            self.partial3[i] = A[:, i].T @ delta

        self.divers = divers

    def handle_idxs(self, idxs):
        curr_idx = self.idx(*idxs)
        c = self.counter
        alpha = np.array([self.alpha_boxes[k].value(idxs[k+1]) for k in range(len(self.alpha_boxes))])

        # partial_k
        for k in range(len(idxs)):
            v = self.partial1[k]
            if k > 0: v += self.partial3[k-1] @ alpha
            if idxs[k] == self.boxes[k].N - 1:
                self.X[c, self.prev(curr_idx, k)] -= v / self.boxes[k].dt
                self.X[c, curr_idx] += v / self.boxes[k].dt
            else:
                self.X[c, self.next(curr_idx, k)] += v / self.boxes[k].dt
                self.X[c, curr_idx] -= v / self.boxes[k].dt

        # partial_kl
        for k in range(1, len(idxs)):
            for l in range(1, len(idxs)):
                v = self.partial2[k-1, l-1] / self.boxes[k].dt / self.boxes[l].dt
                if k == l: 
                    self.X[c, self.get_next(idxs, curr_idx, k)] += v
                    self.X[c, curr_idx] -= 2 * v
                    self.X[c, self.get_prev(idxs, curr_idx, k)] += v
                else:
                    v /= 4
                    n = self.get_next(idxs, self.get_next(idxs, curr_idx, k), l)
                    self.X[c, n] += v
                    n = self.get_next(idxs, self.get_prev(idxs, curr_idx, k), l)
                    self.X[c, n] -= v
                    n = self.get_prev(idxs, self.get_next(idxs, curr_idx, k), l)
                    self.X[c, n] -= v
                    n = self.get_prev(idxs, self.get_prev(idxs, curr_idx, k), l)
                    self.X[c, n] += v
        
        self.b[c] -= np.mean(np.sort(np.abs(self.delta @ alpha))[-self.divers:])
        self.counter += 1

    def _calc_policy(self, idxs):
        alpha = np.array([self.alpha_boxes[k].value(idxs[k+1]) for k in range(len(self.alpha_boxes))])
        curr_idx = self.idx(*idxs)
        v = self.delta @ alpha
        v = sorted(enumerate(v), key=lambda l: abs(l[1]), reverse=True)
        for i in range(self.divers):
            self.policy[curr_idx, v[i][0]] = np.sign(v[i][1]) / self.divers

class SolverLinearFees(Solver):
    def __init__(self, sigma, delta, A, tbox, alpha_box, gamma, qmax=4, fees=0.002):
        """Linear model with fees"""
        super().__init__(sigma, delta, A, tbox, [alpha_box], gamma)
        for i in range(len(self.alpha_boxes)):
            self.partial3[i] = A[:, i].T @ delta

        self.qmax = qmax
        self.alpha_box = alpha_box
        self.fees = fees
        self.Q = [np.array(x) for x in product([-1, 0, 1], repeat=self.N)]
        self.Q = [(q, np.sum(np.abs(q)) * self.fees) for q in self.Q]
        self.nq = 2 * self.qmax + 1


    def qidx_to_qvec(self, q_idx):
            qvec = np.zeros(self.N)
            for i in range(self.N):
                q_idx, qvec[i] = divmod(q_idx, self.nq)
                qvec[i] -= self.qmax
            return qvec

    def qvec_to_qidx(self, qvec):
        q_idx = 0
        for i in range(self.N-1, -1, -1):
            q_idx = q_idx * self.nq + qvec[i] + self.qmax
        return int(q_idx)

    def qvec_is_valid(self, qvec):
        return np.all((-self.qmax <= qvec) & (qvec <= self.qmax))


    def solve(self):
        s = np.prod([x.N for x in self.alpha_boxes])
        t = np.prod([self.nq for _ in range(self.N)])
        self.solution = np.zeros(s * t)
        self.policy = np.zeros((self.tbox.N, self.alpha_box.N, t), dtype=np.int16)

        for t_idx in range(self.tbox.N - 2, -1, -1):
            print(t_idx)
            next_solution = np.zeros_like(self.solution)
            for q_idx in range(t):
                qvec = self.qidx_to_qvec(q_idx)
                assert q_idx == self.qvec_to_qidx(qvec)

                X = lil_matrix((s, s), dtype=np.float32)
                b = np.zeros(s)
                counter = 0
                for i in range(self.alpha_box.N):
                    alpha = np.array([self.alpha_box.value(i)])
                    X[counter, i] -= 1/self.tbox.dt
                    v1 = (self.partial1[1] + self.partial3[0] @ alpha) / self.alpha_box.dt 
                    v2 = self.partial2[0, 0] / self.alpha_box.dt / self.alpha_box.dt
                    # print(v1, v2, self.partial1[1], self.partial3[0], alpha)

                    if i == 0:
                        X[counter, i] -= v1 + v2
                        X[counter, i+1] += v1 + v2
                    elif i == self.alpha_box.N - 1:
                        X[counter, i] += v1 + v2
                        X[counter, i-1] -= v1 + v2
                    else:
                        X[counter, i-1] += -v1 + v2
                        X[counter, i] -= 2 * v2
                        X[counter, i+1] += v1 + v2

                    supremum = -np.infty
                    for act_idx, (q, sm) in enumerate(self.Q):
                        qnew = qvec + q
                        if not self.qvec_is_valid(qnew): continue
                        qnew = self.qvec_to_qidx(qnew)
                        
                        value = self.solution[qnew * self.alpha_box.N + i] - self.solution[q_idx * self.alpha_box.N + i] - sm
                        if value > supremum: 
                            supremum = value
                            self.policy[t_idx, i, q_idx] = act_idx


                    b[counter] -= supremum
                    b[counter] -= qvec.T @ self.delta @ alpha
                    b[counter] -= self.solution[q_idx * self.alpha_box.N + i] / self.tbox.dt
                    counter += 1

                next_solution[q_idx * self.alpha_box.N:(q_idx+1) * self.alpha_box.N] = spsolve(X.tocsr(), b)

            self.solution = next_solution    

# import torch
# from torch import nn

# class SolverLinearFeesPINN(SolverLinearFees):
#     def __init__(self, sigma, delta, A, tbox, alpha_box, gamma, qmax=4, fees=0.002, net=None):
#         super().__init__(sigma, delta, A, tbox, alpha_box, gamma, qmax, fees)
#         self.net = net

#     def _get_output_for_potential(self, t, alpha, qvec):
#         qvec = np.array([qvec.copy() for _ in range(len(t))])
#         qvec = torch.Tensor(qvec)
#         net_input = torch.cat([t, alpha, qvec], 1)
#         return self.net(net_input)

#     def criterion(self, t, alpha, qvec):
#         t = torch.Tensor(t)
#         alpha = torch.Tensor(alpha)

#         t.requires_grad = True
#         alpha.requires_grad = True

#         output = self._get_output_for_potential(t, alpha, qvec)
#         output = output.view(len(output), -1)

#         outputs = []
#         for q, s in self.Q:
#             qnew = qvec + q
#             if not self.qvec_is_valid(qnew): continue
#             outputs.append(self._get_output_for_potential(t, alpha, qnew) - s)
        
#         outputs = torch.cat(outputs, 1)

#         H_t = torch.autograd.grad(output, t, grad_outputs=torch.ones_like(t), create_graph=True, only_inputs=True)[0]
#         H_x = torch.autograd.grad(output, alpha, grad_outputs=torch.ones_like(alpha), create_graph=True, only_inputs=True)[0] 
#         H_xx = torch.autograd.grad(H_x, alpha, grad_outputs=torch.ones_like(alpha), create_graph=True, only_inputs=True)[0]

#         const = qvec.T @ self.delta

#         loss = H_t + (self.partial1[1] + self.partial3[0, 0] * alpha) * H_x + self.partial2[0, 0] * H_xx - output + const[0] * alpha + torch.max(outputs, dim=1)[0].view(len(output), -1)
        
#         loss_f = torch.nn.MSELoss()
#         loss = loss_f(loss, torch.zeros_like(loss)) 

#         return loss
    
#     def boundary(self, alpha, qvec):
#         t = np.ones_like(alpha) * self.tbox.lmax
#         t = torch.Tensor(t)
#         alpha = torch.Tensor(alpha)
#         output = self._get_output_for_potential(t, alpha, qvec)
#         output = output.view(len(output), -1)

#         loss_f = torch.nn.MSELoss()
#         loss = loss_f(output, torch.zeros_like(output)) 

#         return loss
    

# class Net(nn.Module):
#         def __init__(self):
#             super().__init__()
#             self.main = nn.Sequential(
#                 nn.Linear(5,50),
#                 nn.Tanh(),
#                 nn.Linear(50, 50),
#                 nn.Tanh(),
#                 nn.Linear(50,20),
#                 nn.Tanh(),
#                 nn.Linear(20,1),
#             )

#         def forward(self,x):
#             output = self.main(x)
#             return  output



# a0 = np.array([[-36.08957674]])
# A = np.array([[ 1.0        ],
#  [ 2.8298206 ],
#  [-0.68570678]]).T

# delta = np.array([[-0.0009497 ],
#  [-0.00348416],
#  [-0.00187972]]).T

# sigma = np.linalg.cholesky([
#     [2.47464553e-05, 1.89780348e-05, 2.90502711e-05],
#     [1.89780348e-05, 2.32334001e-05, 2.49408923e-05],
#     [2.90502711e-05, 2.49408923e-05, 7.81131715e-05]
# ])




# sigma = np.array([
#     [0.2, 0, 0],
#     [0.0375, 0.1452, 0], 
#     [0.0250, 0.0039, 0.0967]
# ]) * .01
# delta = np.array([
#     [1.0, .9, 0], 
# ])
# A = np.array([
#     [-1.0, 1, 0], 
# ])
# print(delta)
# # print(A)
# solver = SolverLinearFeesPINN(
#     sigma, 
#     delta.T,
#     A.T,
#     Box(0, 24, 100),  
#     Box(-1, 1, 100),
#     1.5,
#     qmax=2,
#     fees=0.0005, 
#     net=Net()
# )

# import torch.optim as optim
# optimizer = optim.Adam(solver.net.parameters(), lr=1e-4, betas = (0.9,0.99), eps = 10**-15)
# losses = []
# t, alpha = [], []
# for s in np.linspace(0, 24, 100):
#     for a in np.linspace(-1, 1, 100):
#         t.append(s)
#         alpha.append(a)

# t, alpha = np.array(t).reshape(-1, 1), np.array(alpha).reshape(-1, 1)
# alpha2 = np.linspace(-1, 1, 100).reshape(-1, 1)

# for i in range(10000):
#     t=np.random.rand(1000).reshape(-1, 1) * 24
#     alpha=np.random.rand(1000).reshape(-1, 1) * -1 + 2
#     qvec = np.array([np.random.randint(-2, 3) for _ in range(3)])
#     loss = solver.criterion(t, alpha, qvec) + solver.boundary(alpha2, qvec)
#     loss.backward()
#     optimizer.step()
#     losses.append(loss.detach().numpy())
#     print(i, np.mean(losses[-100:]))



# import time
# t = time.time()
# solver.solve()
# print(time.time() - t)
