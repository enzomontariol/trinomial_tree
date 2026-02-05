
# Trinomial Tree Pricer

Pricing and diagnostics for vanilla and selected exotics using a **recombining trinomial tree** with:

- **Kamrad–Ritchken**-style trinomial dynamics (up / mid / down).
- **Escrowed dividend adjustment** for a single discrete cash dividend.
- **Iterative backward induction** (vectorized NumPy, **O(N)** memory for values).
- Payoff **strategy pattern** (vanilla, digital, asset-or-nothing, barriers).

This repository is intentionally “quant-first”: correctness, numerical stability, and transparent modelling choices take priority over UI.

## What’s implemented

- European & American:
	- Vanilla calls/puts
	- Digital (cash-or-nothing) calls/puts
	- Asset-or-nothing calls/puts
- Barriers:
	- Knock-out (up/down)
	- Knock-in via decomposition (see theory)
- Diagnostics:
	- Terminal spot distribution from forward induction
	- Early exercise boundary extraction (American)
- Reference model:
	- Black–Scholes European pricer + Greeks

## Installation

This project is managed with `uv` (see [this page](https://docs.astral.sh/uv/getting-started/installation/) for installation) and uses a standard `src/` layout (import root: `trinomial_tree`).

### Option A: Install into an existing uv project (recommended)

From GitHub:

- `uv add "trinomial-tree @ git+https://github.com/enzomontariol/trinomial_tree.git"`

### Option B: Local development install (editable) (required to run the notebook)

Clone the repo, then from the repository root:

- `uv sync --dev`

This installs the project (and dev dependencies) into the uv-managed virtual environment.

If you prefer explicitly creating/activating the venv:

- Create: `uv venv`
- Activate (Unix/macOS): `source .venv/bin/activate`
- Activate (Windows PowerShell): `.\.venv\Scripts\Activate.ps1`
- Sync deps: `uv sync --dev`

### Verify the install

- `uv run python -c "import trinomial_tree; print('ok')"`

## Quickstart

Minimal usage with the iterative tree pricer:

```python
import datetime as dt

from trinomial_tree.pricing import (
		InductiveTree,
		MarketData,
		Option,
		Barrier,
		BarrierType,
		BarrierDirection,
)

pricing_date = dt.date(2026, 1, 2)
maturity = dt.date(2026, 7, 2)

market = MarketData(
		start_date=pricing_date,
		spot_price=100.0,
		volatility=0.20,
		interest_rate=0.03,
		discount_rate=0.03,
		dividend_ex_date=dt.date(2026, 3, 15),
		dividend_amount=1.0,
)

opt = Option(
		maturity=maturity,
		strike_price=100.0,
		is_american=False,
		is_call=True,
		pricing_date=pricing_date,
)

tree = InductiveTree(num_steps=400, market_data=market, option=opt)
px = tree.price()
print(px)
```

Barrier (knock-out) example:

```python
opt.barrier = Barrier(
		barrier_level=120.0,
		barrier_type=BarrierType.knock_out,
		barrier_direction=BarrierDirection.up,
)
print(InductiveTree(num_steps=600, market_data=market, option=opt).price())
```

## Pricer theory (technical)

### 1) Underlying model (risk-neutral)

The tree targets the standard equity diffusion under the risk-neutral measure $\mathbb{Q}$, with constant parameters:

$$
\frac{dS_t}{S_t} = (r - q)\,dt + \sigma\,dW_t^{\mathbb{Q}}.
$$

In this codebase, the “carry” is expressed via:

- `interest_rate` used for forward/capitalization,
- `discount_rate` used for discounting option values.

For most equity use-cases you set `interest_rate == discount_rate == r`.

### 2) Discrete dividend: escrowed dividend adjustment

Discrete cash dividends create a jump in the spot process at the ex-date. A plain lognormal tree built directly on $S$ will mis-handle that jump.

The implemented approach in [src/trinomial_tree/pricing/inductive_tree.py](src/trinomial_tree/pricing/inductive_tree.py) is an **escrowed dividend** model:

$$
S_t = S_t^* + \mathrm{PV}_t(D),
$$

where:

- $S_t^*$ is the “clean” price process (lognormal dynamics without dividend jumps),
- $\mathrm{PV}_t(D)$ is the present value of the future dividend _conditional on being before the ex-date_.

Implementation details:

- A single dividend amount `dividend_amount` at an ex-date `dividend_ex_date`.
- At each step $i$ (time $t_i = i\,\Delta t$),
	- if $t_i$ is before the dividend step, add $\mathrm{PV}_{t_i}(D)$ to reconstruct the “dirty” spot.

This keeps the recombining structure on $S^*$ while still restoring the correct level of $S$ prior to the ex-date.

### 3) Trinomial lattice construction (recombining)

Let $N$ be the number of time steps and $\Delta t = T/N$.

The tree uses multiplicative moves with parameter

$$
\alpha = \exp\big(\sigma\,\sqrt{\lambda}\,\sqrt{\Delta t}\big),
$$

where $\lambda$ is `alpha_parameter` (default $\lambda=3$).

At step $i$ there are $2i+1$ nodes. Index nodes by $j \in \{0,\dots,2i\}$ and define the power $p=i-j$. The spot at $(i,j)$ is computed analytically (no full spot matrix stored):

$$
S_{i,j} = (S_0 - \mathrm{PV}_0(D))\,M^i\,\alpha^{\,i-j} + \mathrm{PV}_{t_i}(D),
\quad M = e^{r\Delta t}.
$$

This formula is exactly what [src/trinomial_tree/pricing/inductive_tree.py](src/trinomial_tree/pricing/inductive_tree.py) implements in `_get_spots_at_step`, using a vectorized `j = 0..2i` construction.

### 4) Transition probabilities (moment matching)

At each node, one step forward leads to three children. Denote the probabilities $(p_u, p_m, p_d)$.

The code uses a Kamrad–Ritchken-style parametrization with

$$
p_u = \frac{1}{2\lambda} + \varepsilon,\quad
p_d = \frac{1}{2\lambda} - \varepsilon,\quad
p_m = 1 - p_u - p_d,
$$

where $\varepsilon$ is a drift correction term derived from moment matching in log-space. In the current implementation, $\varepsilon$ is computed as:

$$
\varepsilon = \frac{\sqrt{\Delta t}}{2\sigma\sqrt{\lambda}}\left(-\frac{\sigma^2}{2}\right).
$$

Two practical consequences:

- The probabilities are **state-independent**, so the backward induction is efficiently vectorizable.
- A validation gate checks: $p_u,p_m,p_d\in[0,1]$ and $p_u+p_m+p_d=1$; otherwise it raises with guidance to increase `num_steps`.

If you want to calibrate the tree to a different drift/carry convention (e.g., explicit dividend yield $q$), the natural extension is to modify $\varepsilon$ accordingly.

### 5) Pricing by backward induction (European/American)

Let $V_{i,j}$ be the option value at node $(i,j)$.

At maturity $i=N$:

$$
V_{N,j} = \Pi(S_{N,j}),
$$

where $\Pi$ is the payoff strategy (vanilla/digital/asset-or-nothing and optional barrier conditions).

For $i=N-1,\dots,0$, the continuation value is:

$$
\widetilde V_{i,j} = e^{-r\Delta t}\left(p_u V_{i+1,j} + p_m V_{i+1,j+1} + p_d V_{i+1,j+2}\right).
$$

European:

$$
V_{i,j} = \widetilde V_{i,j}.
$$

American (early exercise):

$$
V_{i,j} = \max\big(\widetilde V_{i,j},\ \Pi(S_{i,j})\big).
$$

In code:

- Values are stored as a 1D vector of length $2i+1$.
- The “children slices” use the recombining indexing:
	- `V_up = next_values[:-2]`
	- `V_mid = next_values[1:-1]`
	- `V_down = next_values[2:]`

This achieves **O(N)** memory for option values (and **O(N^2)** arithmetic, as expected for a recombining lattice).

### 6) Barriers

Barrier handling is implemented via two mechanisms:

1) **Knock-out:** applied as a node-wise condition after computing node values. For direction:
	 - Up barrier breached when $S \ge B$
	 - Down barrier breached when $S \le B$
   
	 If breached, the node value is set to $0$.

2) **Knock-in decomposition:** for knock-in options, the pricer uses the identity

$$
V_{\text{KI}} = V_{\text{Vanilla}} - V_{\text{KO}},
$$

pricing both a vanilla option and the corresponding knock-out option on the same tree configuration.

Notes:

- In the current implementation, barrier logic is **spot-based at each time slice** (not continuously monitored between slices). Increasing `num_steps` improves the monitoring approximation.

### 7) Diagnostics

- **Terminal distribution**: [src/trinomial_tree/pricing/inductive_tree.py](src/trinomial_tree/pricing/inductive_tree.py) provides `get_terminal_distribution()` using forward induction on probabilities.
- **Exercise boundary**: `get_exercise_boundary()` identifies the spot threshold where intrinsic exceeds continuation.

## Package layout (relevant modules)

- [src/trinomial_tree/pricing/inductive_tree.py](src/trinomial_tree/pricing/inductive_tree.py): main production-style pricer (iterative, vectorized, O(N) memory)
- [src/trinomial_tree/pricing/tree_node.py](src/trinomial_tree/pricing/tree_node.py): legacy node-based implementation (recursive / object graph) kept for reference
- [src/trinomial_tree/pricing/payoff.py](src/trinomial_tree/pricing/payoff.py): payoff strategy pattern
- [src/trinomial_tree/pricing/black_scholes.py](src/trinomial_tree/pricing/black_scholes.py): reference BS pricer (European)
- [src/trinomial_tree/pricing/market.py](src/trinomial_tree/pricing/market.py): market data container
- [src/trinomial_tree/pricing/option.py](src/trinomial_tree/pricing/option.py): option contract container
- [src/trinomial_tree/pricing/enums.py](src/trinomial_tree/pricing/enums.py): enums for conventions and option types
- [src/trinomial_tree/pricing/config.py](src/trinomial_tree/pricing/config.py): model/pricer configuration

## Testing & quality

- Run unit tests: `uv run pytest -n auto tests`
- Lint: `uv run ruff check .`
- Execute the notebook end-to-end (headless):
	- One-time dependency: `uv add --dev nbconvert`
	- Run: `uv run jupyter nbconvert --execute --to notebook --inplace trinomial_tree_showcase.ipynb`
	- Optional (avoid infinite runs): add `--ExecutePreprocessor.timeout=600`

The test suite includes convergence checks vs Black–Scholes, barrier validation, edge cases, and performance checks.

## Limitations (current scope)

- Single discrete dividend (one ex-date + one cash amount).
- Constant parameters (rates, vol). No local/stochastic vol, no stochastic rates.
- Barrier monitoring is at tree time steps (not continuous).
- Black–Scholes pricer is European-only (as intended).

## License

See [LICENCE.md](LICENCE.md).

## Disclaimer

This code is for learning and portfolio demonstration.

