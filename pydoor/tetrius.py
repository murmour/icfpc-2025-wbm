
import os
from basic import ID, N, explore_and_save, guess


class Data:
    def __init__(self):
        self.plans = []
        self.rooms = []
        self.resolved_rooms = []

    @staticmethod
    def load(fname = 'base'):
        data = Data()
        with open(fname + '.plan', 'r') as f:
            for plan in f.read().splitlines():
                data.plans.append([ord(c) - ord('0') for c in plan])
        with open(fname + '.rec', 'r') as f:
            for rooms in f.read().splitlines():
                t = [ord(c) - ord('0') for c in rooms]
                data.rooms.append(t)
                data.resolved_rooms.append(list(t))
        return data

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
                    elif i + 2 < len(rooms):
                        cur.append(plan[i+1])
                        cur.append(rooms[i+2])
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
        print(f"Updated {updated} edges out of {len(self.plans) * len(self.plans[0])}")

    def get_initial_room(self):
        for rooms in self.resolved_rooms:
            if rooms[0] is not None:
                return rooms[0]
        assert False, "No initial room found"

    def get_connections(self, matrix):
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
        for from_room, from_door, to_room, to_door in raw_connections:
            print(from_room, to_room)

        raw_connections.sort()
        connections = []
        for from_room, from_door, to_room, to_door in raw_connections:
            if from_room < to_room or (from_room == to_room and from_door <= to_door):
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

        print(f"Found {len(connections)} unique connections")
        return connections

    def get_room_labels(self):
        labels = [None] * N
        for resolveed, base in zip(self.resolved_rooms, self.rooms):
            for idx, label in zip(resolveed, base):
                if idx is not None:
                    labels[idx] = label
        return labels

    def get_solution(self, matrix):
        result = {
            "id": ID,
            "map": {
                "rooms": self.get_room_labels(),
                "startingRoom": self.get_initial_room(),
                "connections": self.get_connections(matrix)
            }
        }
        return result


def main():
    data = Data.load()
    res = [[{} for _ in range(36)] for _ in range(4)] # room -> door -> {next_room -> plan}

    for i, (plan, rooms) in enumerate(zip(data.plans, data.rooms)):
        for j in range(len(plan) - 1):
            q = plan[j] * 6 + plan[j+1]
            res[rooms[j]][q][(rooms[j+1], rooms[j+2])] = (i, j)

    conflicts = []
    for room in range(4):
        # find a door with max number of next rooms
        max_count = 0
        max_door = None
        for door in range(36):
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
            for d1 in range(6):
                for d2 in range(6):
                    plans.append(data.plans[i][:j] + [d1, d2])

    if not os.path.exists('conflicts.plan'):
        explore_and_save(plans, 'conflicts')

    cdata = Data.load('conflicts')
    cur_v = 0
    room_identifiers = {}
    for (i, (root, count)) in enumerate(zip(roots, counts)):
        baseidx = cur_v * 36
        doors = []
        for k in range(count):
            doors.append([(x[-2], x[-1]) for x in cdata.rooms[baseidx+k*36:baseidx+k*36+36]])
        #print(doors)
        for d in range(36):
            for k in range(count):
                ok = True
                for k2 in range(count):
                    if k2 != k and doors[k2][d] == doors[k][d]:
                        ok = False
                        break
                if ok:
                    d1 = d // 6
                    d2 = d % 6
                    r1, r2 = doors[k][d]
                    room_identifiers[(root, d1, r1, d2, r2)] = cur_v + k
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
    main()
