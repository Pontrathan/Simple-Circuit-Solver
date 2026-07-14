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

    def stamp_dc(self):
        raise NotImplementedError()

    def stamp_ac(self, omega):
        raise NotImplementedError()

    def stamp_trans(self, solver, dt):
        raise NotImplementedError()

class VoltageSource(Component):
    def __init__(self, name, n0, n1, value):
        self.name = name
        self.n0 = n0
        self.n1 = n1
        self.value = value

    def stamp_dc(self):
        return self.value

    def stamp_ac(self, omega):
        return self.value

    def stamp_trans(self, solver):
        raise NotImplementedError()

class CurrentSource(Component):
    def __init__(self, name, n0, n1, value):
        self.name = name
        self.n0 = n0
        self.n1 = n1
        self.value = value

    def stamp_dc(self):
        return self.value

    def stamp_ac(self, omega):
        return self.value

    def stamp_trans(self, solver):
        raise NotImplementedError()

class Resistor(Component):
    def __init__(self, name, n0, n1, value):
        self.name = name
        self.n0 = n0
        self.n1 = n1
        self.value = value

    def stamp_dc(self):
        return 1 / self.value

    def stamp_ac(self, omega):
        return 1 / self.value

    def stamp_trans(self, solver):
        raise NotImplementedError()


class Capacitor(Component):
    def __init__(self, name, n0, n1, value):
        self.name = name
        self.n0 = n0
        self.n1 = n1
        self.value = value

    def stamp_dc(self):
        return 0

    def stamp_ac(self, omega):
        return 1j * omega * self.value

    def stamp_trans(self, solver, dt):
        raise NotImplementedError()


class Inductor(Component):
    def __init__(self, name, n0, n1, value):
        self.name = name
        self.n0 = n0
        self.n1 = n1
        self.value = value

    def stamp_dc(self):
        return 1e-9

    def stamp_ac(self, omega):
        return -1j / (omega * self.value)

    def stamp_trans(self, solver, dt):
        raise NotImplementedError()


class Netlist:
    def __init__(self):
        self.c = {}
        self.n = {}
        self.v_num = 0

    def add_node(self, node):
        self.n[node.name] = node

    def add_component(self, comp):
        if comp.n0.name not in self.n:
            self.add_node(comp.n0)
        if comp.n1.name not in self.n:
            self.add_node(comp.n1)

        self.c[comp.name] = comp

        if isinstance(comp, VoltageSource):
            self.v_num += 1

    def get_nodes(self):
        return [n for n in self.n]


class Netsolver:
    def __init__(self, net):
        self.net = net

    def solve_dc(self):
        N = len(self.net.n) - 1 + self.net.v_num
        A = np.zeros((N, N))
        I = np.zeros((N))
        v_idx = 0

        for c in self.net.c.values():
            val = c.stamp_dc()
            a = c.n0.id
            b = c.n1.id

            if isinstance(c, VoltageSource):
                idx = len(self.net.n) - 1 + v_idx
                v_idx += 1
                I[idx] = val

                if a != 0:
                    A[idx, a - 1] = 1
                    A[a - 1, idx] = 1
                if b != 0:
                    A[idx, b - 1] = -1
                    A[b - 1, idx] = -1

            elif isinstance(c, CurrentSource):
                if a != 0:
                    I[a - 1] -= val
                if b != 0:
                    I[b - 1] += val
            
            else:
                if a != 0:
                    A[a - 1, a - 1] += val
                if b != 0:
                    A[b - 1, b - 1] += val
                if a != 0 and b != 0:
                    A[a - 1, b - 1] -= val
                    A[b - 1, a - 1] -= val

        x = np.linalg.solve(A, I)
        node_count = len(self.net.n) - 1
        voltage = x[:node_count]
        current = x[node_count:]

        print(
            "nodes = "
            + str([entry.name for entry in self.net.n.values() if entry.name != "V0"])
        )
        print("voltage = " + str(voltage))
        print("current = " + str(current))

    def solve_ac(self, omega):
        N = len(self.net.n) - 1 + self.net.v_num
        A = np.zeros((N, N), dtype=complex)
        I = np.zeros((N), dtype=complex)
        v_idx = 0

        for c in self.net.c.values():
            val = c.stamp_ac(omega)
            a = c.n0.id
            b = c.n1.id

            if isinstance(c, VoltageSource):
                idx = len(self.net.n) - 1 + v_idx
                v_idx += 1
                I[idx] = val

                if a != 0:
                    A[idx, a - 1] = 1
                    A[a - 1, idx] = 1
                if b != 0:
                    A[idx, b - 1] = -1
                    A[b - 1, idx] = -1

            elif isinstance(c, CurrentSource):
                if a != 0:
                    I[a - 1] -= val
                if b != 0:
                    I[b - 1] += val
            
            else:
                if a != 0:
                    A[a - 1, a - 1] += val
                if b != 0:
                    A[b - 1, b - 1] += val
                if a != 0 and b != 0:
                    A[a - 1, b - 1] -= val
                    A[b - 1, a - 1] -= val

        x = np.linalg.solve(A, I)
        node_count = len(self.net.n) - 1
        voltage = x[:node_count]
        current = x[node_count:]

        print(
            "nodes = "
            + str([entry.name for entry in self.net.n.values() if entry.name != "V0"])
        )
        print("voltage = " + str(abs(voltage)) + " @ " + str(np.angle(voltage, deg=True)) + "°")
        print("current = " + str(abs(current)) + " @ " + str(np.angle(current, deg=True)) + "°")

    def solve_trans(self):
        pass


# test netlist

mynet = Netlist()

N = {"Vi": Node("Vi"), "V0": Node("V0"), "V1": Node("V1")}

#mynet.add_component(CurrentSource("I1", N["Vi"], N["V0"], 2e-3))
mynet.add_component(VoltageSource("V1", N["Vi"], N["V0"], 5))
mynet.add_component(Resistor("R1", N["Vi"], N["V1"], 1e3))
# mynet.add_component("R2", "V1", "V0", 1e3)
mynet.add_component(Resistor("R3", N["V1"], N["V0"], 1e3))
mynet.add_component(Capacitor("C1", N["V1"], N["V0"], 100e-9))


# test solver

mysolve = Netsolver(mynet)
#mysolve.solve_dc()
mysolve.solve_ac(1e5)