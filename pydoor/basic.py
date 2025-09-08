
import requests
import json
import random
import os
import glob
import sys
import argparse


os.chdir(os.path.dirname(os.path.abspath(__file__)))

BASE_URL = "https://31pwr5t6ij.execute-api.eu-west-2.amazonaws.com/"


def register(name, pl, email):
    """
    Send a POST request to register endpoint

    Args:
        name (str): Name of the user
        pl (str): Programming language preference
        email (str): Email address

    Returns:
        dict: Response from the server
    """
    url = f"{BASE_URL}/register"

    # Request body
    data = {
        "name": name,
        "pl": pl,
        "email": email
    }

    try:
        # Send POST request
        response = requests.post(url, json=data)

        # Print the result
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")

        # Try to parse JSON response
        try:
            json_response = response.json()
            print(f"JSON Response: {json.dumps(json_response, indent=2)}")
            return json_response
        except json.JSONDecodeError:
            print("Response is not valid JSON")
            return {"raw_response": response.text}

    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")
        return {"error": str(e)}

ID = "me@murmour.me hAG1hhGeJU7UT31SRqp0ow"
CURRENT_PROBLEM = "primus"
COUNTS = {
    "probatio": 3,
    "primus": 6,
    "secundus": 12,
    "tertius": 18,
    "quartus": 24,
    "quintus": 30,
    "aleph": 12,
    "beth": 24,
    "gimel": 36,
    "daleth": 48,
    "he": 60,
    "vau": 18,
    "zain": 36,
    "hhet": 54,
    "teth": 72,
    "iod": 90
}
PROBLEM_FLOORS = {
    "probatio": 1,
    "primus": 1,
    "secundus": 1,
    "tertius": 1,
    "quartus": 1,
    "quintus": 1,
    "aleph": 2,
    "beth": 2,
    "gimel": 2,
    "daleth": 2,
    "he": 2,
    "vau": 3,
    "zain": 3,
    "hhet": 3,
    "teth": 3,
    "iod": 3
}
N = 6
FLOORS = 1
NEW_PROBLEMS = ["aleph", "beth", "gimel", "daleth", "he", "vau", "zain", "hhet", "teth", "iod"]
IS_NEW = False


def set_problem(problem_name):
    global CURRENT_PROBLEM
    global N
    CURRENT_PROBLEM = problem_name
    global IS_NEW
    IS_NEW = problem_name in NEW_PROBLEMS
    global FLOORS
    FLOORS = PROBLEM_FLOORS[problem_name]
    N = COUNTS[problem_name] // FLOORS


def load_state():
    if os.path.exists('state.txt'):
        with open('state.txt', 'r') as f:
            problem_name = f.read().strip()
            set_problem(problem_name)


class DisjointSet:
    """Disjoint Set (Union-Find) data structure with path compression and union by rank"""

    def __init__(self, n):
        self.parent = [i for i in range(n)]
        self.rank = [0 for i in range(n)]

    def find_set(self, v):
        """Find the representative of the set containing v with path compression"""
        if v == self.parent[v]:
            return v

        # Path compression: make parent[v] point directly to the root
        self.parent[v] = self.find_set(self.parent[v])
        return self.parent[v]

    def union_sets(self, a, b):
        """Union the sets containing a and b using union by rank"""
        a = self.find_set(a)
        b = self.find_set(b)

        if a != b:
            # Union by rank: attach smaller tree to larger tree
            if self.rank[a] < self.rank[b]:
                a, b = b, a  # swap a and b

            self.parent[b] = a

            # If ranks are equal, increment rank of root
            if self.rank[a] == self.rank[b]:
                self.rank[a] += 1

    def are_same_set(self, a, b):
        """Check if elements a and b are in the same set"""
        return self.find_set(a) == self.find_set(b)


def cleanup_files():
    """Delete all *.plan and *.rec files in the current directory"""
    # Find all *.plan and *.rec files
    in_files = glob.glob('*.plan')
    out_files = glob.glob('*.rec')

    # Delete *.plan files
    for file in in_files:
        os.remove(file)
        print(f"Deleted {file}")

    # Delete *.rec files
    for file in out_files:
        os.remove(file)
        print(f"Deleted {file}")

    print(f"Cleanup complete: deleted {len(in_files)} .plan files and {len(out_files)} .rec files")


def select(id, problem_name, is_cli=False):
    """
    Send a POST request to select endpoint

    Args:
        id (str): Problem ID
        problem_name (str): Name of the problem

    Returns:
        dict: Response from the server
    """
    assert problem_name in COUNTS, f"Problem {problem_name} not found"
    url = f"{BASE_URL}/select"

    # Request body
    data = {
        "id": id,
        "problemName": problem_name
    }

    with open('state.txt', 'w') as f:
        f.write(f'{problem_name}')
        set_problem(problem_name)

    if not is_cli:
        cleanup_files()

    try:
        # Send POST request
        response = requests.post(url, json=data)

        # Print the result
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")

        # Try to parse JSON response
        try:
            json_response = response.json()
            print(f"JSON Response: {json.dumps(json_response, indent=2)}")
            return json_response
        except json.JSONDecodeError:
            print("Response is not valid JSON")
            return {"raw_response": response.text}

    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")
        return {"error": str(e)}


def explore(id, plans_):
    """
    Send a POST request to explore endpoint

    Args:
        id (str): User ID
        plans (list): List of plans as strings

    Returns:
        dict: Response from the server
    """
    url = f"{BASE_URL}/explore"

    plans = []
    for plan in plans_:
        if isinstance(plan, list):
            plans.append(''.join(str(d) for d in plan))
        else:
            plans.append(plan)

    # Request body
    data = {
        "id": id,
        "plans": plans
    }

    try:
        # Send POST request
        response = requests.post(url, json=data)

        # Print the result
        print(f"Status Code: {response.status_code}")
        #print(f"Response: {response.text}")

        # Try to parse JSON response
        try:
            json_response = response.json()
            #print(f"JSON Response: {json.dumps(json_response, indent=2)}")
            return json_response
        except json.JSONDecodeError:
            print("Response is not valid JSON")
            return {"raw_response": response.text}

    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")
        return {"error": str(e)}


def main():
    #register("WILD BASHKORT MAGES", "日本語", "me@murmour.me")
    select(ID, "probatio")

    # Generate a plan with 100,000 random digits from 0 to 5
    plan = ''.join(str(random.randint(0, 5)) for _ in range(100000))

    # Save the plan to input.txt
    with open('input.txt', 'w') as f:
        f.write(plan)
    print(f"Saved plan to input.txt")

    # Call explore with the generated plan
    result = explore(ID, [plan])

    # Save results[0] to output.txt if it exists
    if 'results' in result and len(result['results']) > 0:
        with open('output.txt', 'w') as f:
            f.write(str(result['results'][0]))
        print(f"Saved results[0] to output.txt")
    else:
        print("No results found in response")


def solve():
    with open('input.txt', 'r') as f:
        plan = f.read()
    with open('output.txt', 'r') as f:
        output = f.read()
        rooms = [int(room) for room in output[1:-1].split(', ')]
    assert len(rooms) == len(plan) + 1

    plan = plan[:100]
    rooms = rooms[:101]

    res = [[None] * 6 for _ in range(N)]
    cur_room = rooms[0]
    for c_, room in zip(plan, rooms[1:]):
        c = ord(c_) - ord('0')
        if res[cur_room][c] is None:
            res[cur_room][c] = room
        else:
            assert res[cur_room][c] == room
        cur_room = room

    raw_connections = []
    for i in range(N):
        # find pairs (room, door) that lead to room i
        pairs = []
        for room in range(N):
            for door in range(6):
                if res[room][door] == i:
                    pairs.append((room, door))
        assert len(pairs) == 6, f"Expected exactly 6 pairs, found {len(pairs)}"
        # for each pair, find the set of possible doors in room i that correspond to the pair
        sets = []
        for room, door in pairs:
            cur = set()
            for d in range(6):
                if res[i][d] == room:
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
        #door_mapping = {}
        for j, (room, door) in enumerate(pairs):
            selected_door = None
            for d in sets[j]:
                if d in selected_doors and selected_doors[d] == j:
                    selected_door = d
                    break
            assert selected_door is not None
            #door_mapping[(room, door)] = selected_door
            raw_connections.append((room, door, i, selected_door))

    print(len(raw_connections))
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

    print(len(connections))

    # Format the result
    result = {
        "id": ID,
        "map": {
            "rooms": list(range(N)),
            "startingRoom": rooms[0],
            "connections": connections
        }
    }

    return result


def guess(map_result):
    """
    Send a POST request to guess endpoint with a candidate map

    Args:
        map_result (dict): The result from solve() function

    Returns:
        dict: Response from the server
    """
    url = f"{BASE_URL}/guess"

    try:
        # Send POST request
        response = requests.post(url, json=map_result)

        # Print the result
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")

        # Try to parse JSON response
        try:
            json_response = response.json()
            print(f"JSON Response: {json.dumps(json_response, indent=2)}")
            return json_response
        except json.JSONDecodeError:
            print("Response is not valid JSON")
            return {"raw_response": response.text}

    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")
        return {"error": str(e)}


def explore_and_save(plans_, fname):
    with open(fname + '.plan', 'w') as f:
        plans = []
        for plan in plans_:
            if isinstance(plan, list):
                plans.append(''.join(str(d) for d in plan))
            else:
                plans.append(plan)
        f.write('\n'.join(plans))
    print(f"Saved plans to {fname}.plan")
    result = explore(ID, plans)
    if not 'results' in result:
        print(result)
        assert False
    assert 'results' in result and len(result['results']) > 0
    with open(fname + '.rec', 'w') as f:
        for res in result['results']:
            f.write(''.join(str(r) for r in res) + '\n')
    print(f"Saved results to {fname}.rec")


def initial_explore(runs=10, file_name='base'):
    plans = []
    if IS_NEW:
        k = 6
    else:
        k = 18
    for i in range(runs):
        plans.append(''.join(str(random.randint(0, 5)) for _ in range(k * N * FLOORS)))

    explore_and_save(plans, file_name)

load_state()


def cli_select(problem_name):
    """Select a problem (CLI version without cleanup)"""
    # Validate problem name
    if problem_name not in COUNTS:
        valid_problems = list(COUNTS.keys())
        print(f"Error: Invalid problem name '{problem_name}'")
        print(f"Valid problem names are: {', '.join(valid_problems)}")
        return None

    try:
        result = select(ID, problem_name, is_cli=True)
        print(f"Selected problem: {problem_name}")
        return result
    except Exception as e:
        print(f"Error selecting problem {problem_name}: {e}")
        return None


def cli_walk(runs, file_name):
    """Run initial exploration and save results"""
    try:
        initial_explore(runs, file_name)
    except Exception as e:
        print(f"Error during walk: {e}")


def cli_guess(file_name):
    """Load JSON file and submit guess with ID added automatically"""
    try:
        # Load JSON file
        with open(file_name, 'r') as f:
            guess_data = json.load(f)

        if 'map' not in guess_data:
            print(f"Error: Missing map field")
            return None

        # Check required fields
        required_fields = ['rooms', 'connections', 'startingRoom']
        missing_fields = [field for field in required_fields if field not in guess_data['map']]
        if missing_fields:
            print(f"Error: Missing required fields in map: {missing_fields}")
            return None

        # Add ID automatically
        guess_data['id'] = ID

        # Submit guess
        result = guess(guess_data)
        print(f"Guess submitted from file: {file_name}")
        return result
    except FileNotFoundError:
        print(f"Error: File {file_name} not found")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in file {file_name}")
        return None
    except Exception as e:
        print(f"Error during guess: {e}")
        return None


def normalize_graph(m):
    """
    Renumber rooms so indices increase by room color (0..3),
    stable within each color group. Adjusts startingRoom and all connections.
    Returns a new graph in the same format.
    """
    rooms = m["rooms"]
    n = len(rooms)

    # Stable order by color, then original index
    order = sorted(range(n), key=lambda i: (rooms[i], i))  # list of old indices in new order

    # old -> new index mapping
    old_to_new = {old: new for new, old in enumerate(order)}

    # Build normalized pieces
    new_rooms = [rooms[old] for old in order]
    new_starting = old_to_new[m["startingRoom"]]

    def remap_conn(c):
        return {
            "from": {
                "room": old_to_new[c["from"]["room"]],
                "door": c["from"]["door"],
            },
            "to": {
                "room": old_to_new[c["to"]["room"]],
                "door": c["to"]["door"],
            },
        }

    new_connections = [remap_conn(c) for c in m.get("connections", [])]

    return {
        "rooms": new_rooms,
        "startingRoom": new_starting,
        "connections": new_connections,
    }


def print_help():
    """Print help message"""
    print("Usage: python3 basic.py <command> [arguments]")
    print("\nCommands:")
    print("  select <problem_name>     Select a problem (probatio, primus, secundus, tertius, quartus, quintus)")
    print("  walk <runs> <file_name>   Run initial exploration with specified number of runs")
    print("  guess <file_name>         Submit guess from JSON file")
    print("\nExamples:")
    print("  python3 basic.py select primus")
    print("  python3 basic.py walk 10 base")
    print("  python3 basic.py guess solution.json")


if __name__ == "__main__":
    """Main CLI entry point"""
    if len(sys.argv) < 2:
        print_help()
        return

    command = sys.argv[1]

    if command == "select":
        if len(sys.argv) < 3:
            print("Error: select command requires problem_name")
            print("Usage: python3 basic.py select <problem_name>")
            return
        problem_name = sys.argv[2]
        cli_select(problem_name)

    elif command == "walk":
        if len(sys.argv) < 4:
            print("Error: walk command requires runs and file_name")
            print("Usage: python3 basic.py walk <runs> <file_name>")
            return
        try:
            runs = int(sys.argv[2])
            file_name = sys.argv[3]
            cli_walk(runs, file_name)
        except ValueError:
            print("Error: runs must be a number")
            return

    elif command == "guess":
        if len(sys.argv) < 3:
            print("Error: guess command requires file_name")
            print("Usage: python3 basic.py guess <file_name>")
            return
        file_name = sys.argv[2]
        cli_guess(file_name)

    else:
        print(f"Error: Unknown command '{command}'")
        print_help()
