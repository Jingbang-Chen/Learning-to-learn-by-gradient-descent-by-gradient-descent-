import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
# from torch.autograd import Variable
import random
import multiprocessing
import os.path
import csv
import functools  # added for resolve the problem of optimizer parameter's list empty
from meta_module import *
import copy
from torchvision import datasets
import torchvision
from tqdm import tqdm
USE_CUDA = torch.cuda.is_available()


def w(v):
    if USE_CUDA:
        return v.cuda()
    return v


def detach_var(v):
    # var = w(Variable(v.data, requires_grad=True))
    var = w(v.clone().detach().float().requires_grad_(True))
    var.retain_grad()
    return var


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition(".")
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


# using wonder's beautiful simplification: https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-objects/31174427?noredirect=1#comment86638618_31174427


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))

def listminus(a,b):
    if not isinstance(a, list):
        a = list(a)
    if not isinstance(b, list):
        b = list(b)
    return [int(i) for i in a if not i in b or b.remove(i)]

def vec_match_mat(need_aug, tgt, augisweight=True):
    newdim = listminus(tgt.shape, need_aug.shape)
    assert len(newdim)==1, f"Only support dim(matrix) - dim(vector)=1 but get {len(tgt.shape)} - {len(need_aug.shape)}"
    newdim = newdim[0]
    repeatlist = []
    flag = True
    for x in tgt.shape:
        if x == newdim and flag:
            repeatlist.append(newdim)
            flag = False
        else:
            repeatlist.append(1)
    if augisweight:
        repeatlist = repeatlist + [1]
        return need_aug.unsqueeze(0).repeat(*repeatlist)
    else:
        repeatlist = [1] + repeatlist
        return need_aug.unsqueeze(1).repeat(*repeatlist)

def do_fit(
    opt_net,
    meta_opt,
    target_cls,
    target_to_opt,
    unroll,
    optim_it,
    n_epochs,
    out_mul,
    should_train=True,
):
    if should_train:
        opt_net.train()
    else:
        opt_net.eval()
        unroll = 1

    target = target_cls(training=should_train)
    optimizee = w(target_to_opt())
    n_params = 0
    for name, p in optimizee.all_named_parameters():
        n_params += int(np.prod(p.size()))
    # hidden_states = [
    #     w(Variable(torch.zeros(n_params, opt_net.hidden_sz))) for _ in range(2)
    # ]
    # cell_states = [
    #     w(Variable(torch.zeros(n_params, opt_net.hidden_sz))) for _ in range(2)
    # ]
    hidden_states = [
        w(torch.zeros(n_params, opt_net.hidden_sz)) for _ in range(2)
    ]
    cell_states = [
        w(torch.zeros(n_params, opt_net.hidden_sz)) for _ in range(2)
    ]
    all_losses_ever = []
    if should_train:
        meta_opt.zero_grad()
    all_losses = None
    for iteration in range(1, optim_it + 1):
        loss = optimizee(target)

        if all_losses is None:
            all_losses = loss
        else:
            all_losses += loss

        all_losses_ever.append(loss.data.cpu().numpy())
        loss.backward(retain_graph=should_train)

        offset = 0
        result_params = {}
        # hidden_states2 = [
        #     w(Variable(torch.zeros(n_params, opt_net.hidden_sz))) for _ in range(2)
        # ]
        # cell_states2 = [
        #     w(Variable(torch.zeros(n_params, opt_net.hidden_sz))) for _ in range(2)
        # ]
        hidden_states2 = [
            w(torch.zeros(n_params, opt_net.hidden_sz)) for _ in range(2)
        ]
        cell_states2 = [
            w(torch.zeros(n_params, opt_net.hidden_sz)) for _ in range(2)
        ]
        for name, p in optimizee.all_named_parameters():
            cur_sz = int(np.prod(p.size()))
            # We do this so the gradients are disconnected from the graph but we still get
            # gradients from the rest
            gradients = detach_var(p.grad.view(cur_sz, 1)) # the gradient is still in the tensor but the variable is free from the previous network
            updates, new_hidden, new_cell = opt_net(
                gradients,
                [h[offset : offset + cur_sz] for h in hidden_states], # hidden_states shape: [2, num_param, 20]
                [c[offset : offset + cur_sz] for c in cell_states], # same above
            )
            for i in range(len(new_hidden)):
                hidden_states2[i][offset : offset + cur_sz] = new_hidden[i]
                cell_states2[i][offset : offset + cur_sz] = new_cell[i]
            result_params[name] = p + updates.view(*p.size()) * out_mul
            result_params[name].retain_grad()

            offset += cur_sz

        if iteration % unroll == 0:
            if should_train:
                meta_opt.zero_grad()
                all_losses.backward()
                meta_opt.step()

            all_losses = None

            optimizee = w(target_to_opt())
            optimizee.load_state_dict(result_params)
            optimizee.zero_grad()
            hidden_states = [detach_var(v) for v in hidden_states2]
            cell_states = [detach_var(v) for v in cell_states2]

        else:
            for name, p in optimizee.all_named_parameters():
                rsetattr(optimizee, name, result_params[name])
            assert len(list(optimizee.all_named_parameters()))
            hidden_states = hidden_states2
            cell_states = cell_states2

    return all_losses_ever


def fit_optimizer(
    target_cls,
    target_to_opt,
    preproc=False,
    unroll=20,
    optim_it=100,
    n_epochs=20,
    n_tests=100,
    lr=0.001,
    out_mul=1.0,
):
    # do_fit(opt_net, meta_opt, target_cls, target_to_opt, unroll, optim_it, n_epochs, out_mul, should_train=True):
    opt_net = w(Optimizer(preproc=preproc))
    meta_opt = optim.Adam(opt_net.parameters(), lr=lr)

    best_net = None
    best_loss = 100000000000000000

    for _ in tqdm(range(n_epochs), "epochs"):
        for _ in tqdm(range(20), "iterations"):
            do_fit(
                opt_net,
                meta_opt,
                target_cls,
                target_to_opt,
                unroll,
                optim_it,
                n_epochs,
                out_mul,
                should_train=True,
            )

        loss = np.mean(
            [
                np.sum(
                    do_fit(
                        opt_net,
                        meta_opt,
                        target_cls,
                        target_to_opt,
                        unroll,
                        optim_it,
                        n_epochs,
                        out_mul,
                        should_train=False,
                    )
                )
                for _ in tqdm(range(n_tests), "tests")
            ]
        )
        print(loss)
        if loss < best_loss:
            print(best_loss, loss)
            best_loss = loss
            best_net = copy.deepcopy(opt_net.state_dict())

    return best_loss, best_net


class Optimizer(MetaModule):
    def __init__(self, preproc=False, hidden_sz=20, preproc_factor=10.0):
        super().__init__()
        self.hidden_sz = hidden_sz
        if preproc:
            self.recurs = nn.LSTMCell(2, hidden_sz)
        else:
            self.recurs = nn.LSTMCell(1, hidden_sz)
        self.recurs2 = nn.LSTMCell(hidden_sz, hidden_sz)
        self.output = MetaLinear(hidden_sz, 1)
        self.preproc = preproc
        self.preproc_factor = preproc_factor
        self.preproc_threshold = np.exp(-preproc_factor)

    def forward(self, inp, hidden, cell):
        if self.preproc:
            # Implement preproc described in Appendix A

            # Note: we do all this work on tensors, which means
            # the gradients won't propagate through inp. This
            # should be ok because the algorithm involves
            # making sure that inp is already detached.
            inp = inp.data
            inp2 = w(torch.zeros(inp.size()[0], 2))
            keep_grads = (torch.abs(inp) >= self.preproc_threshold).squeeze()
            inp2[:, 0][keep_grads] = (
                torch.log(torch.abs(inp[keep_grads]) + 1e-8) / self.preproc_factor
            ).squeeze()
            inp2[:, 1][keep_grads] = torch.sign(inp[keep_grads]).squeeze()

            inp2[:, 0][~keep_grads] = -1
            inp2[:, 1][~keep_grads] = (
                float(np.exp(self.preproc_factor)) * inp[~keep_grads]
            ).squeeze()
            # inp = w(Variable(inp2))
            inp = w(inp2)
        hidden0, cell0 = self.recurs(inp, (hidden[0], cell[0]))
        hidden1, cell1 = self.recurs2(hidden0, (hidden[1], cell[1]))
        return self.output(hidden1), (hidden0, hidden1), (cell0, cell1)


class MNISTLoss:
    def __init__(self, training=True):
        dataset = datasets.MNIST(
            "mnist",
            train=True,
            download=True,
            transform=torchvision.transforms.ToTensor(),
        )
        indices = list(range(len(dataset)))
        np.random.RandomState(10).shuffle(indices)
        if training:
            indices = indices[: len(indices) // 2]
        else:
            indices = indices[len(indices) // 2 :]

        self.loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=128,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices),
            num_workers=0,
        )

        self.batches = []
        self.cur_batch = 0

    def sample(self):
        if self.cur_batch >= len(self.batches):
            self.batches = []
            self.cur_batch = 0
            for b in self.loader:
                self.batches.append(b) # every b includes [data, label]
        batch = self.batches[self.cur_batch]
        self.cur_batch += 1
        return batch

class MNISTData:
    def __init__(self, training=True):
        dataset = datasets.MNIST(
            "mnist",
            train=training,
            download=True,
            transform=torchvision.transforms.ToTensor(),
        )
        # indices = list(range(len(dataset)))
        # np.random.RandomState(10).shuffle(indices)

        self.loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=128,
            shuffle=True,
            # sampler=torch.utils.data.sampler.SubsetRandomSampler(indices),
            num_workers=0,
        )
        self.datalist = list(self.loader)
        self.cur_batch = 0

    def sample(self):
        if self.cur_batch >= len(self.datalist):
            self.cur_batch = 0
        batch = self.datalist[self.cur_batch]
        return batch

class MNISTNet(MetaModule):
    def __init__(self, layer_size=20, n_layers=1, **kwargs):
        super().__init__()

        inp_size = 28 * 28
        self.layers = {}
        for i in range(n_layers):
            self.layers[f"mat_{i}"] = MetaLinear(inp_size, layer_size)
            inp_size = layer_size

        self.layers["final_mat"] = MetaLinear(inp_size, 10)
        self.layers = ModuleDict(self.layers)

        self.activation = nn.Sigmoid()
        self.loss = nn.NLLLoss()

    # Added method to resolve the problem of parameters empty
    def all_named_parameters(self):
        return [(k, v) for k, v in self.named_parameters()]

    def forward(self, loss):
        inp, out = loss.sample()
        # inp = w(Variable(inp.view(inp.size()[0], 28 * 28)))
        # out = w(Variable(out))
        inp = w(inp.view(inp.size()[0], 28 * 28))
        out = w(out)
        cur_layer = 0
        while f"mat_{cur_layer}" in self.layers:
            inp = self.activation(self.layers[f"mat_{cur_layer}"](inp))
            cur_layer += 1

        inp = F.log_softmax(self.layers["final_mat"](inp), dim=1)
        l = self.loss(inp, out)
        return l

class MNISTNet2(MetaModule):
    def __init__(self, layer_size=20, n_layers=1, **kwargs):
        super().__init__()

        inp_size = 28 * 28
        self.layers = {}
        self.n_layers = n_layers
        for i in range(n_layers):
            self.layers[f"mat_{i}"] = MetaLinear(inp_size, layer_size)
            inp_size = layer_size

        self.layers["final_mat"] = MetaLinear(inp_size, 10)
        self.layers = ModuleDict(self.layers)

        self.activation = nn.Sigmoid()
        self.loss = nn.NLLLoss()
        self.neruon_mat_buffer = None
        self.layeroutput_buffer = None
        self.train_data = MNISTData(training=True)
        self.eval_data = MNISTData(training=False)
        self.list_named_parameters = dict(self.named_parameters())
    # Added method to resolve the problem of parameters empty
    def all_named_parameters(self, output_list=True):
        return [(k,v) for k, v in self.named_parameters()] if output_list else {k:v for k, v in self.named_parameters()}
    
    def neuron_params(self, layer_name, neuron_idx):
        # layers.mat_0.bias
        # layers.mat_0.weight
        # layers.final_mat.bias
        # layers.final_mat.weight
        if neuron_idx == -1:
            return self.list_named_parameters[layer_name]
        else:
            return self.list_named_parameters[layer_name][neuron_idx]

    def get_neruon_mat(self, input, weight):
        aligned_input = vec_match_mat(input, weight, augisweight=False)
        aligned_weight = vec_match_mat(weight, input, augisweight=True)
        neruon_mat = aligned_input * aligned_weight
        return neruon_mat

    def forward_step(self, input, previous_layer=-1, neuron_idx=0):
        if previous_layer==-1:
            input, out = input.sample()
            input = w(input.view(input.size()[0], 28 * 28))
            out = w(out)
        if neuron_idx==-1:
            return self.layers[f"mat_{previous_layer+1}"].bias, self.layeroutput_buffer
        if neuron_idx==0:
            self.neruon_mat_buffer = self.get_neruon_mat(input, self.layers[f"mat_{previous_layer+1}"].weight)
            self.layeroutput_buffer = torch.sum(self.neruon_mat_buffer, axis=-1) + (self.layers[f"mat_{previous_layer+1}"].bias).unsqueeze(0).repeat(100,1)
        return self.neruon_mat_buffer[:,:,neuron_idx], self.layeroutput_buffer

    def get_layer_dimension(self):
        param_dict = self.all_named_parameters(output_list=False)
        dim_dict = {}
        for k, v in param_dict.items():
            dim_dict[k] = v.shape[-1]
        return dim_dict

    def forward(self, data):
        inp, out = data.sample()
        # inp = w(Variable(inp.view(inp.size()[0], 28 * 28)))
        # out = w(Variable(out))

        inp = inp.view(inp.size()[0], 28 * 28).cuda()
        out = out.cuda()
        cur_layer = 0
        while f"mat_{cur_layer}" in self.layers:
            inp = self.activation(self.layers[f"mat_{cur_layer}"](inp))
            cur_layer += 1

        inp = F.log_softmax(self.layers["final_mat"](inp), dim=1)
        l = self.loss(inp, out)
        return l

# loss, net = fit_optimizer(
#     MNISTData, MNISTNet, lr=1e-3, out_mul=0.1, preproc=True, n_tests=1, n_epochs=2
# )

def init_optimizers(net):
    optimizers = {}
    dims = net.get_layer_dimension()
    for k, v in dims.items():
        if "bias" not in k:
            optimizers[k] = torch.optim.Adam([net.neuron_params(k,-1).detach().cuda().requires_grad_(True)], lr=0.001)
        else:
            for i in range(v):
                optimizers[f"{k}_{i}"] = torch.optim.Adam([net.neuron_params(k, i).detach().cuda().requires_grad_(True)], lr=0.001)
    # optimizers = ModuleDict(optimizers)
    return optimizers

def zero_optimizers(optimizers):
    for k, v in optimizers.items():
        v = v.zero_grad()
    return optimizers
def step_optimizers(optimizers):
    for k, v in optimizers.items():
        v = v.step()
    return optimizers

def train_with_indep_opt(net, data, iteration=1000000):
    optimizers = init_optimizers(net)
    for i in tqdm(range(iteration)):
        optimizers = zero_optimizers(optimizers)
        loss = net(data)
        loss.backward(retain_graph=False)
        optimizers = step_optimizers(optimizers)
        if i%1000==0:
            print(f"{i}th iteration: {loss}")

# Contrast experinments
def init_optimizers2(net):
    optimizers = {}
    list_param = list(net.parameters())
    optimizers["x"] = torch.optim.Adam([x.detach().cuda().requires_grad_(True) for x in list_param], lr=0.001)
    return optimizers

def train_with_indep_opt2(net, data, iteration=1000000):
    optimizers = init_optimizers2(net)
    for i in tqdm(range(iteration)):
        optimizers = zero_optimizers(optimizers)
        loss = net(data)
        loss.backward(retain_graph=False)
        optimizers = step_optimizers(optimizers)
        if i%1000==0:
            print(f"{i}th iteration: {loss}")

net = MNISTNet2()
net.cuda()
train_data = MNISTData(training=True)
# print(net.list_named_parameters)
train_with_indep_opt(net, train_data, iteration=1000000)
# train_with_indep_opt2(net, train_data, iteration=1000000)