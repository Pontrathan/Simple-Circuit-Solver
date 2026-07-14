from dataclasses import dataclass
import numpy as np

id_cnt = 1


@dataclass
class Node:
    name: str = ""

    def __init__(self, name):
        global id_cnt
        self.name = name
        if name == "V0" or name == "GND":
            self.id = 0
        else:
            self.id = id_cnt
            id_cnt += 1


@dataclass
class Component:
    name: str
    n0: Node
    n1: Node
    value: str | float | None = None


class Netlist:
    def __init__(self):
        self.c = {}
        self.n = {}

        self.v_num = 0

    def add(self, name: str, n0: str, n1: str, value: str | float | None = None):
        if n0 not in self.n:
            self.n[n0] = Node(n0)
        if n1 not in self.n:
            self.n[n1] = Node(n1)

        self.c[name] = Component(name, self.n[n0], self.n[n1], value)

        if name.startswith("V"):
            self.v_num += 1

    def get_nodes(self):
        return [n for n in self.n]


class Netsolver:
    def __init__(self, net):
        self.net = net

    def solve(self):
        N = len(self.net.n) - 1 + self.net.v_num
        A = np.zeros((N, N))
        I = np.zeros((N))

        v_idx = 0
        for c in self.net.c.values():
            if c.name.startswith("R"):
                g = 1.0 / c.value

                a = c.n0.id
                b = c.n1.id

                if a != 0:
                    A[a - 1, a - 1] += g
                if b != 0:
                    A[b - 1, b - 1] += g
                if a != 0 and b != 0:
                    A[a - 1, b - 1] -= g
                    A[b - 1, a - 1] -= g
            elif c.name.startswith("V"):
                idx = len(self.net.n)-1 + v_idx
                v_idx += 1

                I[idx] = c.value

                a = c.n0.id
                b = c.n1.id

                if a != 0:
                    A[idx, a - 1] = 1
                    A[a - 1, idx] = 1
                if b != 0:
                    A[idx, b - 1] = -1
                    A[b - 1, idx] = -1
            elif c.name.startswith("I"):
                a = c.n0.id
                b = c.n1.id

                if a != 0:
                    I[a-1] -= c.value

                if b != 0:
                    I[b-1] += c.value
            
        x = np.linalg.solve(A, I)
        node_count = len(self.net.n)-1
        voltage = x[:node_count]
        current = x[node_count:]

        print("nodes = " + str([entry.name for entry in self.net.n.values() if entry.name != "V0"]))
        print("voltage = " + str(voltage))
        print("current = " + str(current))

# test netlist

mynet = Netlist()
mynet.add("I1", "Vi", "V0", 2e-3)
#mynet.add("V1", "Vi", "V0", 5)
mynet.add("R1", "Vi", "V1", 1e3)
# mynet.add("R2", "V1", "V0", 1e3)
mynet.add("R3", "V1", "V0", 1e3)

# test solver

mysolve = Netsolver(mynet)
mysolve.solve()
