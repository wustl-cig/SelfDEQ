from torch.multiprocessing import Queue

global forward_res
global forward_itr
global backward_res
global backward_itr


def _init():

    global forward_res
    global forward_itr
    global backward_res
    global backward_itr

    forward_res = Queue()
    forward_itr = Queue()
    backward_res = Queue()
    backward_itr = Queue()


def set_forward(f_res, f_itr):
    forward_res.put(f_res)
    forward_itr.put(f_itr)


def set_backward(b_res, b_itr):
    backward_res.put(b_res)
    backward_itr.put(b_itr)


def get_forward():
    f_res, f_itr = [], []
    while not forward_res.empty():
        f_res.append(forward_res.get())
    while not forward_itr.empty():
        f_itr.append(forward_itr.get())

    return f_res, f_itr


def get_backward():
    b_res, b_itr = [], []
    while not backward_res.empty():
        b_res.append(backward_res.get())
    while not backward_itr.empty():
        b_itr.append(backward_itr.get())

    return b_res, b_itr
