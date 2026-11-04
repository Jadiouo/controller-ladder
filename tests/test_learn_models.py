"""The four students on toy data (CPU, tiny): they must fit, be seed-reproducible, and have matching size."""
import pytest

torch = pytest.importorskip("torch")

from arena.learn.models import KINDS, build_policy, n_params

DO, N, TA = 6, 2, 3


def toy(B=128, seed=0):
    g = torch.Generator().manual_seed(seed)
    o = torch.randn(B, DO, generator=g)
    w = torch.randn(DO, TA * N, generator=g) * 0.7
    a = torch.sin(o @ w).view(B, TA, N)                      # deterministic smooth map, unit-ish scale
    return o, a


def fit(kind, steps=1500, seed=0, width=64):
    torch.manual_seed(seed)
    o, a = toy()
    m = build_policy(kind, DO, N, TA, width=width, depth=2)
    opt = torch.optim.Adam(m.parameters(), 2e-3)
    gen = torch.Generator().manual_seed(seed)
    first = None
    for _ in range(steps):
        loss = m.loss(o, a, gen)
        first = first if first is not None else float(loss.detach())
        opt.zero_grad(); loss.backward(); opt.step()
    return m, o, a, first, float(loss.detach())


# (threshold on mean squared error of sample() vs the target; target variance is ~0.4)
THRESH = {"bc": 0.02, "mip": 0.05, "flow": 0.08, "diffusion": 0.08}


@pytest.mark.parametrize("kind", KINDS)
def test_overfits_toy_data(kind):
    m, o, a, first, last = fit(kind)
    assert last < 0.7 * first
    kw = {"steps": 10} if kind in ("flow", "diffusion") else {}
    g = torch.Generator().manual_seed(1)
    with torch.inference_mode():
        err = float(((m.sample(o, g, **kw) - a) ** 2).mean())
    assert err < THRESH[kind], (kind, err, float(a.var()))


def test_flow_one_step_runs_and_is_deterministic_given_generator():
    m, o, a, *_ = fit("flow", steps=200)
    g1, g2 = torch.Generator().manual_seed(5), torch.Generator().manual_seed(5)
    x1, x2 = m.sample(o, g1, steps=1), m.sample(o, g2, steps=1)
    assert x1.shape == (len(o), TA, N) and torch.equal(x1, x2)
    assert not torch.equal(x1, m.sample(o, torch.Generator().manual_seed(6), steps=1))
    d = build_policy("diffusion", DO, N, TA, 32, 1)
    d.eval()
    s1 = d.sample(o, torch.Generator().manual_seed(1), steps=5, eta=1.0)
    s2 = d.sample(o, torch.Generator().manual_seed(1), steps=5, eta=1.0)
    assert torch.equal(s1, s2)


@pytest.mark.parametrize("kind", ["bc", "mip", "diffusion", "flow"])
def test_same_seed_same_weights(kind):
    a = fit(kind, steps=30)[0].state_dict()
    b = fit(kind, steps=30)[0].state_dict()
    c = fit(kind, steps=30, seed=1)[0].state_dict()
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert any(not torch.equal(a[k], c[k]) for k in a if "net" in k)


def test_parameter_counts_match():
    counts = {k: n_params(build_policy(k, 48, 2, 8, 256, 4)) for k in KINDS}
    ref = counts["mip"]
    assert counts["diffusion"] == counts["flow"] == ref
    bc = n_params(build_policy("bc", 48, 2, 8, 256, 4, match_to=ref))
    assert abs(bc - ref) / ref < 0.005
