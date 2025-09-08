
import itertools
import re
from basic import FLOORS, ID, N, DisjointSet, explore_and_save, guess, initial_explore, select, normalize_graph
from primus import Data
import pydot

from matrix_tools import repair_matrix


def n_rooms(N, color):
    if color < N % 4:
        return N // 4 + 1
    return N // 4


class ExData(Data):
    def load(self, fname = 'base'):
        super().load(fname)
        self.m = len(self.rooms[0])
        self.k = len(self.rooms) * self.m
        self.ds = DisjointSet(self.k)
        return self

    def get_id(self, plan_idx, room_idx):
        return plan_idx * self.m + room_idx

    def merge_rooms(self, sources):
        targets = [[] for _ in range(6)]
        colors = [[] for _ in range(6)]
        for source in sources:
            plan_idx = source // self.m
            room_idx = source % self.m
            if room_idx == self.m - 1:
                continue
            door = self.plans[plan_idx][room_idx]
            targets[door].append(self.get_id(plan_idx, room_idx + 1))
            colors[door].append(self.rooms[plan_idx][room_idx + 1])
        roots = set()
        for c in colors:
            if len(c) > 1:
                assert min(c) == max(c)
        for t in targets:
            if len(t) <= 1:
                continue
            for x in t:
                root = self.ds.find_set(t[0])
                r2 = self.ds.find_set(x)
                if r2 != root:
                    self.ds.union_sets(r2, root)
                    roots.add(root)
        #if roots:
        #    print(f"{len(roots)} new roots")
        return roots

    def unify(self):
        M = self.m * len(self.rooms)
        sets = {}
        res = False
        for i in range(M):
            root = self.ds.find_set(i)
            if root not in sets:
                sets[root] = []
            sets[root].append(i)
        for rset in sets.values():
            if len(rset) == 1:
                continue
            if self.merge_rooms(rset):
                res = True
        return res

    def get_size(self):
        roots = set()
        for i in range(self.m * len(self.rooms)):
            root = self.ds.find_set(i)
            roots.add(root)
        return len(roots)

    def get_sizes(self):
        roots = {}
        for i in range(self.m * len(self.rooms)):
            root = self.ds.find_set(i)
            if root not in roots:
                roots[root] = 1
            else:
                roots[root] += 1
        return sorted(roots.values(), reverse=True)

    def resolve(self):
        self.resolved_rooms = [[None] * self.m for _ in range(len(self.rooms))]
        roots = {}
        for i in range(self.m * len(self.rooms)):
            root = self.ds.find_set(i)
            if root not in roots:
                roots[root] = [i]
            else:
                roots[root].append(i)
        roots = sorted(roots.values(), key=lambda x: len(x), reverse=True)[:N]
        for (i, root) in enumerate(roots):
            for idx in root:
                self.resolved_rooms[idx // self.m][idx % self.m] = i


def patch_matrix(matrix):
    # if one edge is missing, we can infer it from the other edges
    ins = [0] * N
    for i in range(N):
        for j in range(6):
            if matrix[i][j] is not None:
                ins[matrix[i][j]] += 1
    if min(ins) == 6:
        return # all edges are present
    # find the vertex with min number of edges
    min_vertex = None
    min_count = 6
    for i in range(N):
        if ins[i] < min_count:
            min_count = ins[i]
            min_vertex = i
    for i in range(N):
        for j in range(6):
            if matrix[i][j] is None:
                matrix[i][j] = min_vertex
                return # can add only one edge
    print('Ins', ins)


def get_missing(matrix):
    cnts = [0] * N
    for i in range(N):
        for j in range(6):
            if matrix[i][j] is not None:
                cnts[matrix[i][j]] += 1
    res = []
    for i in range(N):
        for _ in range(6 - cnts[i]):
            res.append(i)
    return res


def fix_matrix(matrix, perm):
    res = [list(row) for row in matrix]
    cur = 0
    for i in range(N):
        for j in range(6):
            if res[i][j] is None:
                res[i][j] = perm[cur]
                cur += 1
    return res


def main():
    data = ExData()
    data.load()

    xdoors = [0] * 4

    if N > 12:
        res = [[{} for _ in range(36)] for _ in range(4)] # room -> door -> {next_room -> plan}

        for i, (plan, rooms) in enumerate(zip(data.plans, data.rooms)):
            for j in range(len(plan) - 1):
                q = plan[j] * 6 + plan[j+1]
                res[rooms[j]][q][(rooms[j+1], rooms[j+2])] = (i, j)

        for room in range(4):
            # find a door with max number of next rooms
            max_count = 0
            for door in range(36):
                if len(res[room][door]) > max_count:
                    max_count = len(res[room][door])
            print(room, max_count)
            for dd in range(36):
                if len(res[room][dd]) == max_count:
                    xdoors[room] += 1
                    ids = {}
                    d1 = dd // 6
                    d2 = dd % 6
                    for i, (plan, rooms) in enumerate(zip(data.plans, data.rooms)):
                        for j in range(len(plan) - 1):
                            if plan[j] == d1 and plan[j+1] == d2 and rooms[j] == room:
                                key = (rooms[j+1], rooms[j+2])
                                if key not in ids:
                                    ids[key] = [data.get_id(i, j)]
                                else:
                                    ids[key].append(data.get_id(i, j))
                    for t in ids.values():
                        #if len(t) > 1:
                        #    print(f"Merging {len(t)} rooms")
                        for id in t:
                            data.ds.union_sets(id, t[0])
    else:
        res = [[{} for _ in range(6)] for _ in range(4)] # room -> door -> {next_room -> plan}

        for i, (plan, rooms) in enumerate(zip(data.plans, data.rooms)):
            cur_room = rooms[0]
            for j, (c, room) in enumerate(zip(plan, rooms[1:])):
                res[cur_room][c][room] = (i, j)
                cur_room = room

        for room in range(4):
            # find a door with max number of next rooms
            max_count = 0
            max_door = None
            for door in range(6):
                if len(res[room][door]) > max_count:
                    max_count = len(res[room][door])
                    max_door = door
            print(room, max_count)
            if max_count == 1:
                # all rooms with this color are in the same set
                t = []
                for i, rooms in enumerate(data.rooms):
                    for j, r in enumerate(rooms):
                        if r == room:
                            t.append(data.get_id(i, j))
                for id in t:
                    data.ds.union_sets(id, t[0])
            else:
                for door in range(6):
                    if len(res[room][door]) == max_count:
                        xdoors[room] += 1
                        ids = {}
                        for i, (plan, rooms) in enumerate(zip(data.plans, data.rooms)):
                            for j in range(len(plan)):
                                if plan[j] == door and rooms[j] == room:
                                    if rooms[j+1] not in ids:
                                        ids[rooms[j+1]] = [data.get_id(i, j)]
                                    else:
                                        ids[rooms[j+1]].append(data.get_id(i, j))
                        for t in ids.values():
                            #if len(t) > 1:
                            #    print(f"Merging {len(t)} rooms")
                            for id in t:
                                data.ds.union_sets(id, t[0])

    print(f"Xdoors: {xdoors}")
    print(f"Initial size: {data.get_size()}")
    print([x for x in data.get_sizes() if x > 1])
    while data.unify():
        pass
    print(f"Final size: {data.get_size()}")
    print([x for x in data.get_sizes() if x > 1])

    data.resolve()
    while True:
        matrix = data.make_matrix()
        if not data.apply_matrix(matrix):
            break
    print(matrix)
    missing = get_missing(matrix)
    print(missing)
    if 0 < len(missing) <= 5:
        permutations = list(itertools.permutations(missing))
        sols = 0
        last_sol = None
        last_fixed = None
        for perm in permutations:
            fixed = fix_matrix(matrix, perm)
            try:
                last_sol = data.get_solution(fixed)
                last_fixed = fixed
                sols += 1
            except:
                pass
        print(f"Found {sols} solutions")
        if sols == 1:
            matrix = last_fixed
        else:
            return False
    if len(missing) > 0:
        matrix = repair_matrix(matrix)
        data.apply_matrix(matrix)

    #print(data.resolved_rooms[0])

    missing = get_missing(matrix)
    if len(missing) > 0:
        print("Missing edges")
        return False

    if FLOORS == 1:
        solution = data.get_solution(matrix)
        draw_matrix_by_pydor(data, matrix)
    else:
        draw_matrix_by_pydor(data,matrix)
        solution = rip_sol(data,matrix)

    print(solution)
    if solution:
        guess(solution)
    return True


def find_ham(G, F, res, depth):
    if depth==0:
        return True
    v = res[-1]
    F[v] = True
    for u,door in G[v]:
        if not F[u]:
            res.append(u)
            if find_ham(G, F, res, depth-1):
                return True
            res.pop()
    F[v] = False
    return False


euindex = 0
def eulerize(G, v, F):
    global euindex
    F[v] = True
    for r1,d1,u,d2,i in G[v]:
        if not F[u]:
            eulerize(G, u, F)
            if len(G[u]) % 2 == 1:
                G[v].append( [v,d1,u,d2,euindex] )
                G[u].append( [u,d2,v,d1,euindex] )
                euindex += 1


def find_euler(G, F, v, res):
    while len(G[v])>0:
        r1,d1,r2,d2,i = G[v][-1]
        G[v].pop()
        if F[i]: continue
        F[i] = True
        find_euler(G, F, r2, res)
        res.append( [r2,d2,r1,d1] )


def rip_sol(data: Data, matrix):
    global euindex
    N = len(matrix)
    G1 = [ [-1] * N for _ in range(N) ] # room -> room -> door
    edges = data.get_connections(matrix)
    for e in edges:
        r1, r2 = e["from"]["room"], e["to"]["room"]
        if G1[r1][r2]==-1: G1[r1][r2] = e["from"]["door"]
        if G1[r2][r1]==-1: G1[r2][r1] = e["to"]["door"]

    # matrix room -> door -> through which door we enter the target room
    door_mat = [[None] * 6 for _ in range(N)]
    for e in edges:
        r1, r2 = e["from"]["room"], e["to"]["room"]
        d1, d2 = e["from"]["door"], e["to"]["door"]
        door_mat[r1][d1] = d2
        door_mat[r2][d2] = d1

    pyg = pydot.Dot("Simpl", graph_type="graph")
    for i in range(N): pyg.add_node(pydot.Node(f"room{i}", label=f"Room {i}"))
    for i in range(N):
        for j in range(i):
            if G1[i][j]!=-1:
                pyg.add_edge(pydot.Edge(f"room{i}", f"room{j}"))
    pyg.write_png("simpl.png")

    G2 = [ [] for _ in range(N) ]
    for i in range(N):
        for j in range(N):
            if G1[i][j]!=-1:
                G2[i].append([j,G1[i][j]])
    F = [False] * N
    start = data.get_initial_room(matrix)
    ham_path = [start]
    if not find_ham(G2, F, ham_path, N-1):
        print("No hamiltonian path")
        return False
    print("Ham", ham_path)

    colors = data.get_room_labels()
    que = ""
    for i in range(len(ham_path)):
        que += f"[{(colors[ham_path[i]] + 1) % 4}]"
        if i!=len(ham_path)-1:
            # find the door going from ham_path[i] to ham_path[i+1]
            door = None
            for d in range(6):
                if matrix[ham_path[i]][d] == ham_path[i+1]:
                    door = d
                    break
            assert door is not None
            assert G1[ham_path[i]][ham_path[i+1]] == door
            que += f"{door}"

    G3 = [ [] for _ in range(N) ]
    for i in range(len(edges)):
        e = edges[i]
        r1, r2 = e["from"]["room"], e["to"]["room"]
        d1, d2 = e["from"]["door"], e["to"]["door"]
        G3[r1].append( [r1,d1,r2,d2,i] )
        G3[r2].append( [r2,d2,r1,d1,i] )
    euindex = len(edges)
    eulerize( G3, 0, [False] * N ) # when some vertices are odd
    # print( G3 )
    euler_tour = [] # list of [room1, door1, room2, door2]
    euF = [False] * euindex
    find_euler(G3, euF, ham_path[-1], euler_tour)
    # check that all euF are True
    for f in euF:
        assert f
    # add the euler tour to the query
    cur_v = ham_path[-1]
    for r1, d1, r2, d2 in euler_tour:
        que += str(d1)
        assert matrix[cur_v][d1] == r2
        cur_v = r2

    print("Query", que)

    #print("Query", que)
    print("Euler Tour", euler_tour)
    explore_and_save([que], "stairs")
    sdata = Data()
    sdata.load("stairs")

    #cur_v = ham_path[-1]
    #cur_f = 0
    #plan = sdata.plans[0][N-1:] # skip the hamiltonian path
    colors2 = sdata.rooms[0][N-1:] # skip the hamiltonian path
    #assert len(plan) == len(rooms)

    G = [[None] * 6 for _ in range(N*2)] # FLOOR1*N+room1 -> door1 -> [FLOOR2*N+room2, door2]
    def get_floor(floors,col1,col2):
        if col1==col2: return floors-1
        return (col1-col2-1) % 4

    print(colors)
    print(colors2)
    cc = 0
    good_euler_pos = -1
    for r1,d1,r2,d2 in euler_tour:
        f1, f2 = get_floor(2, colors2[cc], colors[r1]), get_floor(2, colors2[cc+1], colors[r2])
        assert 0<=f1<2 and 0<=f2<2
        # print(f1,f2,N)
        G[f1*N+r1][d1] = [f2*N+r2, d2]
        G[f2*N+r2][d2] = [f1*N+r1, d1]
        G[(1-f1)*N+r1][d1] = [(1-f2)*N+r2, d2]
        G[(1-f2)*N+r2][d2] = [(1-f1)*N+r1, d1]
        if r1==start and f1!=0 and good_euler_pos==-1:
            good_euler_pos = cc
        cc += 1
    print("Good Euler Pos", good_euler_pos)
    if good_euler_pos==-1:
        print("No Good Euler Pos")
        return None

    draw_multifloors_by_pydor(2, G)

    print( "Floors", FLOORS )
    if FLOORS==2:
        return get_solution_by_G(2, G, data, start, matrix)

    start_floor = get_floor(2, colors2[0], colors[euler_tour[0][0]])
    end_floor = get_floor(2, colors2[-1], colors[euler_tour[-1][2]])
    print("Start/End floors of Euler Tour", start_floor, end_floor)
    if start_floor == end_floor:
        print("Bad Euler Tour")
        return None

    # the second query
    # recoloring the first floor
    que2 = ""
    for i in range(len(ham_path)):
        que2 += f"[{(colors[ham_path[i]] + 1) % 4}]"
        if i!=len(ham_path)-1:
            # find the door going from ham_path[i] to ham_path[i+1]
            door = None
            for d in range(6):
                if matrix[ham_path[i]][d] == ham_path[i+1]:
                    door = d
                    break
            assert door is not None
            assert G1[ham_path[i]][ham_path[i+1]] == door
            que2 += f"{door}"
    # go to the first room on the second floor
    for i in range(good_euler_pos):
        r1, d1, r2, d2 = euler_tour[i]
        que2 += str(d1)
    # recoloring the second floor
    for i in range(len(ham_path)):
        que2 += f"[{(colors[ham_path[i]] + 2) % 4}]"
        if i!=len(ham_path)-1:
            # find the door going from ham_path[i] to ham_path[i+1]
            door = None
            for d in range(6):
                if matrix[ham_path[i]][d] == ham_path[i+1]:
                    door = d
                    break
            assert door is not None
            assert G1[ham_path[i]][ham_path[i+1]] == door
            que2 += f"{door}"
    # two additional euler tours
    for r1, d1, r2, d2 in euler_tour * 3:
        que2 += str(d1)

    print( "Query2", que2 )

    explore_and_save([que2], "floors3")
    ssdata = Data()
    ssdata.load("floors3")

    colors3 = ssdata.rooms[0][N*2-2+good_euler_pos:] # skip recolorings
    print("colors3", colors3)
    GG = [[None] * 6 for _ in range(N*3)] # FLOOR1*N+room1 -> door1 -> [FLOOR2*N+room2, door2]

    cc = 0
    print(euler_tour)
    print(euler_tour * 2)
    for r1,d1,r2,d2 in euler_tour * 3:
        f1, f2 = get_floor(3, colors3[cc], colors[r1]), get_floor(3, colors3[cc+1], colors[r2])
        assert 0<=f1<3 and 0<=f2<3
        # print(f1,f2,N)
        print(f1, r1, d1, "->", f2, r2, d2)
        if GG[f1*N+r1][d1]:
            assert GG[f1*N+r1][d1][0]==f2*N+r2 and GG[f1*N+r1][d1][1]==d2
        GG[f1*N+r1][d1] = [f2*N+r2, d2]
        GG[f2*N+r2][d2] = [f1*N+r1, d1]
        cc += 1

    draw_multifloors_by_pydor(3, GG)

    print("GG before patch", GG[12], GG[13], GG[14])
    # patch missing edges
    for i in range(len(GG)):
        for d1 in range(6):
            if GG[i][d1]==None:
                r1, f1 = i % N, i // N
                f = (f1+1) % 3
                r, d2 = GG[f*N+r1][d1]
                r2, f2 = None, None
                for j in range(3):
                    if GG[(j * N) + (r % N)][d2]==None:
                        r2, f2 = r % N, j
                        break
                assert r2!=None and f2!=None
                GG[f1*N+r1][d1] = [f2*N+r2, d2]
                GG[f2*N+r2][d2] = [f1*N+r1, d1]
    print("GG after patch", GG[12], GG[13], GG[14])

    return get_solution_by_G(3, GG, data, start, matrix)

    def invc(col): # inverse color
        return (col + 1) % 4

    inv_mat = [[None] * 6 for _ in range(N)] # room -> door -> inverted?
    cur_v = ham_path[-1]
    assert sdata.rooms[0][N-1] == invc(colors[cur_v]) # we started from the inverted color

    inv = True

    plan = sdata.plans[0][N-1:] # skip the hamiltonian path
    rooms = sdata.rooms[0][N:] # skip the hamiltonian path
    assert len(plan) == len(rooms)
    for d, room in zip(plan, rooms):
        # curv_v -> go through door d
        new_v = matrix[cur_v][d]
        base_col = colors[new_v]
        assert room == base_col or room == invc(base_col)
        new_inv = room == invc(base_col)
        inv_mat[cur_v][d] = new_inv != inv
        other_door = door_mat[cur_v][d]
        inv_mat[new_v][other_door] = new_inv != inv
        inv = new_inv
        cur_v = new_v

    print(inv_mat)
    return data.get_solution_2f(matrix, inv_mat)


def get_solution_by_G(floors, G, data: Data, start_room, matrix):
    N = len(G)
    n = N // floors

    connections = []
    for r1 in range(N):
        for d1 in range(6):
            if G[r1][d1]:
                r2, d2 = G[r1][d1]
                if r1 < r2 or (r1==r2 and d1<=d2):
                    connections.append({"from": { "room":r1, "door":d1}, "to": { "room":r2, "door":d2} })
    result = {
            "id": ID,
            "map": {
                "rooms": data.get_room_labels() * floors,
                "startingRoom": start_room,
                "connections": connections
            }
        }
    result['map'] = normalize_graph(result['map']) # ???

    # validate the solution
    for plan, rooms in zip(data.plans, data.resolved_rooms):
        cur_room = start_room
        if rooms[0] is not None and cur_room != rooms[0]:
            assert False, f"Expected {cur_room} but got {rooms[0]}"
        for c, room in zip(plan, rooms[1:]):
            cur_room = matrix[cur_room][c]
            if room is not None and cur_room != room:
                assert False, f"Expected {cur_room} but got {room}"

    return result


def test_pydor():
    g = pydot.Dot("Test", graph_type="graph")
    g.add_node(pydot.Node("a", label="Foo", shape="circle"))
    g.add_node(pydot.Node("b", label="Bar", shape="hexagon"))
    g.add_edge(pydot.Edge("a", "b"))
    g.write_png("test.png")


def draw_matrix_by_pydor(data, matrix):
    dungeon = pydot.Dot("Dungeon", graph_type="graph")
    dungeon.set_prog("sfdp")
    # dungeon.set_prog("neato")
    # dungeon.set_prog("circo")
    # dungeon.set_prog("twopi")
    dungeon.set_node_defaults(shape="record")
    dungeon.set_graph_defaults(splines="compound")
    # dungeon.set_graph_defaults(beautify=True)

    colors = data.get_room_labels()
    for i in range(len(matrix)):
        room = pydot.Node(f"room{i}", label="{{<d0> 0|<d1> 1|<d2> 2}|" + f"<r> Room {i} ({colors[i]})" + "|{<d5> 5|<d4> 4|<d3> 3}}")
        dungeon.add_node(room)

    #for i in range(len(matrix)):
    #    for j in range(len(matrix[i])):
    #        dungeon.add_edge(pydot.Edge(f"room{i}:d{j}", f"room{matrix[i][j]}:r"))

    edges = data.get_connections(matrix)
    for e in edges:
        dungeon.add_edge(pydot.Edge(f"room{e["from"]["room"]}:d{e["from"]["door"]}", f"room{e["to"]["room"]}:d{e["to"]["door"]}"))

    # print(dungeon.to_string())
    dungeon.write_png("dungeon.png")


def draw_multifloors_by_pydor(floors, G):
    dungeon = pydot.Dot("Dungeon", graph_type="digraph", compound="true", overlap="false")
    dungeon.set_prog("fdp")
    # dungeon.set_prog("neato")
    dungeon.set_node_defaults(shape="record")
    # dungeon.set_edge_defaults(minlen="5")
    dungeon.set_graph_defaults(splines="compound")

    clusters = []
    for i in range(floors):
        clu = pydot.Cluster(f"c{i}", label=f"Floor {i}")
        clusters.append(clu)
        dungeon.add_subgraph(clu)

    N = len(G)
    n = N // floors
    for i in range(N):
        room = pydot.Node(f"room{i}", label="{{<d0> 0|<d1> 1|<d2> 2}|" + f"<r> Room {i % n} ({i // n})" + "|{<d5> 5|<d4> 4|<d3> 3}}")
        clusters[i // n].add_node(room)

    for r1 in range(N):
        for d1 in range(6):
            if G[r1][d1]:
                r2, d2 = G[r1][d1]
                #if r1 < r2 or (r1==r2 and d1<=d2):
                if r1 // n == r2 // n:
                    dungeon.add_edge(pydot.Edge(f"room{r1}:d{d1}", f"room{r2}:d{d2}"))
    print(dungeon.to_string())
    dungeon.write_png(f"floors{floors}.png")


if __name__ == "__main__":
    #test_pydor()
    select(ID, "iod")
    initial_explore(500)
    main()
