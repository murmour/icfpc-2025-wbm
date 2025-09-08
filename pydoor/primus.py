
import os
import re

from basic import ID, N, explore_and_save, guess, normalize_graph


class Data:
    def __init__(self):
        self.plans = []
        self.rooms = []
        self.resolved_rooms = []

    def load(self, fname = 'base'):
        base_plans = []
        base_rooms = []
        with open(fname + '.plan', 'r') as f:
            for plan in f.read().splitlines():
                base_plans.append(plan)
        with open(fname + '.rec', 'r') as f:
            for rooms in f.read().splitlines():
                t = [ord(c) - ord('0') for c in rooms]
                base_rooms.append(t)

        for plan, rooms in zip(base_plans, base_rooms):
            # split plan into a digit or [digit], e.g. 5, [1], 2
            res = re.findall(r'(?:\d)|(?:\[\d\])', plan)
            assert len(rooms) == len(res) + 1
            new_plan = []
            new_rooms = [rooms[0]]
            for p, r in zip(res, rooms[1:]):
                if p.startswith('['):
                    #print(f"Coloring {new_rooms[-1]} to {r}")
                    new_rooms[-1] = r
                    continue # skip coloring commands
                new_plan.append(int(p))
                new_rooms.append(r)

            self.plans.append(new_plan)
            self.rooms.append(new_rooms)
            self.resolved_rooms.append(list(new_rooms))
            assert len(new_plan) + 1 == len(new_rooms)
        return self

    def apply_identifiers(self, identifiers):
        n_total = 0
        n_resolved = 0
        for (plan, rooms, resolved_rooms) in zip(self.plans, self.rooms, self.resolved_rooms):
            for i in range(len(rooms)):
                t = None
                cur = [rooms[i]]
                if tuple(cur) in identifiers:
                    t = identifiers[tuple(cur)]
                elif i + 1 < len(rooms):
                    cur.append(plan[i])
                    cur.append(rooms[i+1])
                    if tuple(cur) in identifiers:
                        t = identifiers[tuple(cur)]
                resolved_rooms[i] = t
                if t is not None:
                    n_resolved += 1
                n_total += 1
        print(f"Resolved {n_resolved} out of {n_total} rooms")

    def make_matrix(self):
        res = [[None] * 6 for _ in range(N)]
        for plan, rooms in zip(self.plans, self.resolved_rooms):
            cur_room = rooms[0]
            for c, room in zip(plan, rooms[1:]):
                if cur_room is not None and room is not None:
                    if res[cur_room][c] is None:
                        res[cur_room][c] = room
                    else:
                        assert res[cur_room][c] == room
                cur_room = room
        bad = 0
        for i in range(N):
            for j in range(6):
                if res[i][j] is None:
                    bad += 1
        print(f"{bad} edges missing out of {N * 6}")
        return res

    def apply_matrix(self, matrix):
        updated = 0
        for plan, rooms in zip(self.plans, self.resolved_rooms):
            for i in range(len(rooms)-1):
                if rooms[i] is not None and rooms[i+1] is None and matrix[rooms[i]][plan[i]] is not None:
                    updated += 1
                    rooms[i+1] = matrix[rooms[i]][plan[i]]
        print(f"Resolved {updated} vertices")
        return updated > 0

    def make_matrix(self):
        res = [[None] * 6 for _ in range(N)]
        for plan, rooms in zip(self.plans, self.resolved_rooms):
            cur_room = rooms[0]
            for c, room in zip(plan, rooms[1:]):
                if cur_room is not None and room is not None:
                    if res[cur_room][c] is None:
                        res[cur_room][c] = room
                    else:
                        assert res[cur_room][c] == room
                cur_room = room
        bad = 0
        for i in range(N):
            for j in range(6):
                if res[i][j] is None:
                    bad += 1
        print(f"{bad} edges missing out of {N * 6}")
        return res

    def get_initial_room(self, matrix):
        # try each room as initial room
        for i in range(N):
            ok = True
            for plan, rooms in zip(self.plans, self.resolved_rooms):
                if rooms[0] is not None:
                    return rooms[0]
                cur_room = i
                for c, room in zip(plan, rooms[1:]):
                    cur_room = matrix[cur_room][c]
                    if room is not None and cur_room != room:
                        ok = False
                        break
            if ok:
                return i
        assert False, "No initial room found"

    def get_raw_connections(self, matrix):
        raw_connections = []
        for i in range(N):
            # find pairs (room, door) that lead to room i
            pairs = []
            for room in range(N):
                for door in range(6):
                    if matrix[room][door] == i:
                        pairs.append((room, door))
            assert len(pairs) == 6, f"Expected exactly 6 pairs, found {len(pairs)}"
            # for each pair, find the set of possible doors in room i that correspond to the pair
            sets = []
            for room, door in pairs:
                cur = set()
                for d in range(6):
                    if matrix[i][d] == room:
                        cur.add(d)
                sets.append(cur)

            # Find bipartite matching: select one door from each set such that all doors are different
            def find_matching(sets, used_doors=None, set_index=0):
                if used_doors is None:
                    used_doors = dict()

                if set_index == len(sets):
                    return True, used_doors.copy()

                for door in sets[set_index]:
                    if door not in used_doors:
                        used_doors[door] = set_index
                        success, result = find_matching(sets, used_doors, set_index + 1)
                        if success:
                            return True, result
                        used_doors.pop(door)

                return False, None

            success, selected_doors = find_matching(sets)
            assert success, f"No valid bipartite matching found for room {i}"

            # Create mapping from pairs to selected doors
            for j, (room, door) in enumerate(pairs):
                selected_door = None
                for d in sets[j]:
                    if d in selected_doors and selected_doors[d] == j:
                        selected_door = d
                        break
                assert selected_door is not None
                raw_connections.append((room, door, i, selected_door))

        print(f"Found {len(raw_connections)} connections")
        #for from_room, from_door, to_room, to_door in raw_connections:
        #    print(from_room, to_room)

        raw_connections.sort()
        connections = []
        for rc in raw_connections:
            from_room, from_door, to_room, to_door = rc
            if from_room < to_room or (from_room == to_room and from_door <= to_door):
                connections.append(rc)
        print(f"Found {len(connections)} unique connections")
        return connections

    def get_connections(self, matrix):
        raw_connections = self.get_raw_connections(matrix)
        connections = []
        for from_room, from_door, to_room, to_door in raw_connections:
            connections.append({
                "from": {
                    "room": from_room,
                    "door": from_door
                },
                "to": {
                    "room": to_room,
                    "door": to_door
                }
            })
        return connections

    def get_room_labels(self):
        labels = [None] * N
        for resolveed, base in zip(self.resolved_rooms, self.rooms):
            for idx, label in zip(resolveed, base):
                if idx is not None:
                    labels[idx] = label
        return labels

    def get_solution(self, matrix):
        start_room = self.get_initial_room(matrix)
        result = {
            "id": ID,
            "map": {
                "rooms": self.get_room_labels(),
                "startingRoom": start_room,
                "connections": self.get_connections(matrix)
            }
        }
        result['map'] = normalize_graph(result['map'])

        # validate the solution
        for plan, rooms in zip(self.plans, self.resolved_rooms):
            cur_room = start_room
            if rooms[0] is not None and cur_room != rooms[0]:
                assert False, f"Expected {cur_room} but got {rooms[0]}"
            for c, room in zip(plan, rooms[1:]):
                cur_room = matrix[cur_room][c]
                if room is not None and cur_room != room:
                    assert False, f"Expected {cur_room} but got {room}"
        return result

    def get_solution_2f(self, matrix, inv_mat):
        start_room = self.get_initial_room(matrix)
        labels = self.get_room_labels()
        labels.extend(labels) # second floor has the same labels
        conn0 = self.get_raw_connections(matrix)
        conn = []
        for from_room, from_door, to_room, to_door in conn0:
            assert inv_mat[from_room][from_door] is not None
            if inv_mat[from_room][from_door]:
                conn.append((from_room, from_door, to_room + N, to_door))
                conn.append((from_room + N, from_door, to_room, to_door))
            else:
                conn.append((from_room, from_door, to_room, to_door))
                conn.append((from_room + N, from_door, to_room + N, to_door))

        connections = []
        for from_room, from_door, to_room, to_door in conn:
            connections.append({
                "from": {
                    "room": from_room,
                    "door": from_door
                },
                "to": {
                    "room": to_room,
                    "door": to_door
                }
            })

        result = {
            "id": ID,
            "map": {
                "rooms": labels,
                "startingRoom": start_room,
                "connections": connections
            }
        }
        result['map'] = normalize_graph(result['map'])

        # validate the solution
        for plan, rooms in zip(self.plans, self.resolved_rooms):
            cur_room = start_room
            if rooms[0] is not None and cur_room != rooms[0]:
                assert False, f"Expected {cur_room} but got {rooms[0]}"
            for c, room in zip(plan, rooms[1:]):
                cur_room = matrix[cur_room][c]
                if room is not None and cur_room != room:
                    assert False, f"Expected {cur_room} but got {room}"
        return result


def main():
    data = Data()
    data.load()
    #conflicts = set() # (src_room, src_door, dst_room1, dst_room2, path1, path2)
    res = [[{} for _ in range(6)] for _ in range(4)] # room -> door -> {next_room -> plan}

    for i, (plan, rooms) in enumerate(zip(data.plans, data.rooms)):
        cur_room = rooms[0]
        for j, (c, room) in enumerate(zip(plan, rooms[1:])):
            res[cur_room][c][room] = (i, j)
            cur_room = room


    conflicts = []
    for room in range(4):
        # find a door with max number of next rooms
        max_count = 0
        max_door = None
        for door in range(6):
            if len(res[room][door]) > max_count:
                max_count = len(res[room][door])
                max_door = door
        conflicts.append(res[room][max_door])
        print(room, max_count)


    # explore all conflicts
    plans = []
    roots = []
    counts = []
    for (i, conflict) in enumerate(conflicts):
        if len(conflict) < 2:
            continue # no conflicts

        roots.append(i)
        counts.append(len(conflict))
        for next_room, plan in conflict.items():
            i, j = plan
            for d in range(6):
                plans.append(data.plans[i][:j] + [d])

    if not os.path.exists('conflicts.plan'):
        explore_and_save(plans, 'conflicts')

    cdata = Data()
    cdata.load('conflicts')
    cur_v = 0
    room_identifiers = {}
    for (i, (root, count)) in enumerate(zip(roots, counts)):
        baseidx = cur_v * 6
        doors = []
        for k in range(count):
            doors.append([x[-1] for x in cdata.rooms[baseidx+k*6:baseidx+k*6+6]])
        #print(doors)
        for d in range(6):
            for k in range(count):
                ok = True
                for k2 in range(count):
                    if k2 != k and doors[k2][d] == doors[k][d]:
                        ok = False
                        break
                if ok:
                    room_identifiers[(root, d, doors[k][d])] = cur_v + k
        cur_v += count

    for v0 in range(4):
        if v0 not in roots:
            room_identifiers[(v0,)] = cur_v
            cur_v += 1
    print(room_identifiers)

    data.apply_identifiers(room_identifiers)
    while True:
        matrix = data.make_matrix()
        if not data.apply_matrix(matrix):
            break
    print(matrix)
    #connections = data.get_connections(matrix)
    solution = data.get_solution(matrix)
    print(solution)
    guess(solution)
    #print(len(conflicts))


if __name__ == "__main__":
    #main()
    pass
