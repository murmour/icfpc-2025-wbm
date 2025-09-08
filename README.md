## ICFPC 2025 submission by WILD BASHKORT MAGES 🪄

The mages are:
- Damir Akhmetzyanov (damir@magelabs.jp)
- Max Mouratov (max@magelabs.jp)
- Artem Ripatti (ripatti@inbox.ru)

...

This year’s ICFPC turned out to be **especially difficult** for us. Despite that, we managed to hold our ground: in the Lightning Round we likely placed **around 10th**, and in the Main Round **around 20th**.

Even though the problem pushed us to the limit, we had a lot of fun experimenting with unusual approaches—and some of them led to perhaps **the strangest solvers** we’ve ever written.


### 🏯 The Japanese Backtracking Solver

Our first solver, **解ウス** (Kai-usu), is a backtracking algorithm, implemented in **Japanese**.

It runs a recursive backtracking search with an aggressive **branch-and-bound pruning**, supported by a disjoint set union to merge equivalent rooms, bitmasks for fast state checks, and bipartite matching to reconstruct the graph at the end.

The choice of language comes from the contest registration process, which required teams to specify a programming language in the `"pl"` field of the JSON API. Not fully understanding its purpose, we entered **日本語** (Japanese), a language that we've been studying for quite some time. Unfortunately, the language selection couldn’t be changed afterward, so we had to implement our solver **entirely in Japanese**.

We realize this is unusual, but Japanese has the advantage of being especially **clear and logical** for humans reading the code—so anyone familiar with Japanese should be able to follow our solver and even learn something from it.


### 💊 The Dreaming Python Solver

Our second solver, **Pydoor**, is what happens when you try to solve ICFPC while overdosing on random walks. Instead of carefully reasoning about the graph, Pydoor stumbles around in a haze, collecting room observations through **random walk exploration**.

Then the hallucinations kick in: Pydoor realizes that **doors reveal secrets**. If a set of doors produces as many different room observations as there are rooms of a given color, then those rooms must really be the same thing in disguise. This “drug-vision” lets Pydoor merge rooms together with a disjoint set union, **gradually unifying** the maze.

Once the trip stabilizes, Pydoor tries to **rebuild reality**: selecting the biggest hallucination clusters as the real vertices, reconstructing the adjacency matrix, and filling in missing edges with either brute-force permutations (for small gaps) or improvised “repair heuristics” (for larger ones). Finally, it snaps back to clarity with a **bipartite matching** step, assigning the correct exit doors.

But that’s only the single-floor trip. Once Pydoor enters multi-floor graphs, the hallucinations go **multi-layered**. Suddenly the maze isn't just one reality, but **stacked realities**, each tinted by a color transformation. To cope, Pydoor arms itself with **Hamiltonian paths** and **Eulerian tours** — wandering every vertex once, then every edge, to track how rooms transform as you “shift floors.”

- For **two-floor mazes**, Pydoor follows a Hamiltonian path while recoloring rooms (`color + 1 mod 4`), then launches into an Eulerian tour. By analyzing the resulting **kaleidoscope of colors**, it figures out which floor each room belongs to.
- For **three-floor mazes**, Pydoor searches for a “**good Euler position**” — a point where the trip loops back to the starting room but on another floor. With that anchor, it chains recolored Hamiltonian paths and multiple full Eulerian tours, forcing the graph to reveal its **inter-floor secrets**.

It’s chaotic, paranoid, and sometimes brilliant — but somehow, this psychedelic pipeline produces graphs that actually work. Pydoor is **less of an algorithm and more of a trip**, yet it carried us surprisingly far in the contest.
