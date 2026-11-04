"""Demonstration data: shard I/O, action-chunk labels, streaming sampler, DART / DAgger collection, reproducibility."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from arena import Scenario
from arena.controllers import controller_factory
from arena.eval import DEMO_SEEDS, TRAIN_SEEDS
from arena.learn import ShardedDataset, collect
from arena.learn.features import FeatureSpec

SC = Scenario("tracking", {"ref": "lissajous", "duration": 0.5}, history=2)     # 100 steps per episode


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    d = tmp_path_factory.mktemp("demo")
    collect(SC, "ctc_lqr", TRAIN_SEEDS[:4], d / "a", shard_episodes=2)
    return d / "a"


def test_shards_roundtrip_and_meta(data):
    ds = ShardedDataset(data, chunk=1)
    assert len(ds.shards) == 2 and ds.episodes == 4 and len(ds) == 400
    assert ds.meta["teacher"] == "ctc_lqr" and ds.meta["history"] == 2
    o, a, raw = ds.load_shard(0)
    assert o.shape == (200, 3 * ds.spec.d_row) and a.shape == (200, 1, 2)
    assert set(raw["ep"].tolist()) == set(TRAIN_SEEDS[:2].tolist())
    # window = [h0, h1, current]; at k=0 the history is the first observation repeated
    w = raw["win"]
    assert np.allclose(w[0, 0], w[0, 2]) and np.allclose(w[0, 1], w[0, 2])
    assert np.allclose(w[1, 1], w[0, 2]) and np.allclose(w[5, 1], w[4, 2])      # history = previous rows
    assert np.abs(raw["act"]).max() <= Scenario("tracking").make_env().ctx.tau_limit + 1e-6


def test_chunk_labels_follow_the_episode(data):
    ds = ShardedDataset(data, chunk=5)
    _, a, raw = ds.load_shard(0)
    act, k = raw["act"], raw["k"]
    assert np.allclose(a[3, :, :], act[3:8])                       # a_k ... a_{k+4}
    last = np.flatnonzero(k == 99)[0]                              # end of the first episode: no leak into episode 2
    assert np.allclose(a[last, 0], act[last]) and np.allclose(a[last, 1:], act[last])
    assert np.allclose(a[last - 2, :], np.concatenate([act[last - 2:last + 1], np.repeat(act[last:last + 1], 2, 0)]))


def test_stats_and_sampler_resident_equals_streaming(data):
    ds = ShardedDataset(data, chunk=3)
    norm = ds.stats()
    assert norm[0].shape == (ds.spec.obs_dim,) and norm[2].shape == (2,)
    seen = {}
    for resident in (True, False):
        g = torch.Generator().manual_seed(0)
        s = ds.sampler(32, torch.device("cpu"), g, tuple(torch.as_tensor(x) for x in norm),
                       max_resident_bytes=(1 << 30) if resident else 1, group=1)
        assert s.resident == resident
        rows = [s.next() for _ in range(12)]
        seen[resident] = (s.loads, torch.cat([r[0] for r in rows]))
        assert rows[0][0].shape == (32, ds.spec.obs_dim) and rows[0][1].shape == (32, 3, 2)
    assert seen[True][0] == 1 and seen[False][0] >= 2             # streaming really reloads shards
    # normalised observations: close to zero mean / unit scale over the whole data
    o_all = torch.as_tensor(np.concatenate([ds.load_shard(i)[0] for i in range(2)]))
    z = (o_all - torch.as_tensor(norm[0])) / torch.as_tensor(norm[1])
    assert z.mean().abs() < 0.05 and abs(z.std() - 1) < 0.3


def test_collect_is_reproducible_and_dart_changes_states(tmp_path):
    collect(SC, "ctc_lqr", DEMO_SEEDS[:0] if False else TRAIN_SEEDS[:2], tmp_path / "x", dart_sigma=0.05, shard_episodes=2)
    collect(SC, "ctc_lqr", TRAIN_SEEDS[:2], tmp_path / "y", dart_sigma=0.05, shard_episodes=2)
    collect(SC, "ctc_lqr", TRAIN_SEEDS[:2], tmp_path / "z", dart_sigma=0.0, shard_episodes=2)
    ld = lambda n: np.load(tmp_path / n / "shard_0000.npz")
    assert np.array_equal(ld("x")["win"], ld("y")["win"]) and np.array_equal(ld("x")["act"], ld("y")["act"])
    assert not np.allclose(ld("x")["win"], ld("z")["win"])                     # noise moved the visited states
    assert np.allclose(ld("x")["act"][0], ld("z")["act"][0])                   # same first observation, same clean label
    with pytest.raises(FileExistsError):
        collect(SC, "ctc_lqr", TRAIN_SEEDS[:1], tmp_path / "x")


def test_dagger_round_appends_and_labels_with_teacher(data, tmp_path):
    import shutil
    d = tmp_path / "dag"
    shutil.copytree(data, d)
    # a 'student' that is just a zero-torque controller: states drift, labels stay the teacher's
    from arena.controllers.base import Controller

    class Zero(Controller):
        def act(self, obs):
            return np.zeros(2)
    collect(SC, "ctc_lqr", TRAIN_SEEDS[4:6], d, student=Zero, beta=0.0, append=True, round_id=1, tag="dagger1")
    ds = ShardedDataset(d, 1)
    assert len(ds.shards) == 3 and ds.meta["shards"][-1]["tag"] == "dagger1" and ds.episodes == 6
    _, _, raw = ds.load_shard(2)
    ref = ShardedDataset(data, 1)
    assert raw["act"].std() > 0 and len(raw["act"]) == 200


def test_rel_ref_features_are_linear_functions_of_the_row(data):
    ds = ShardedDataset(data, 1, rel_ref=True)
    raw = ShardedDataset(data, 1).load_shard(0)[2]["win"]
    f = ds.spec.expand(raw)
    d = ds.spec.d_row
    assert d == ds.spec.d_raw + 4 and f.shape[-1] == 3 * d
    sl = ds.spec._slices()
    row = f[7].reshape(3, d)[-1]
    assert np.allclose(row[ds.spec.d_raw:ds.spec.d_raw + 2], raw[7, -1, sl["q"]] - raw[7, -1, sl["ref.q"]])


def test_wrap_angles_shifts_q_and_ref_by_the_same_multiple_of_two_pi(data):
    raw = ShardedDataset(data, 1).load_shard(0)[2]["win"]
    spec = ShardedDataset(data, 1, wrap_angles=True).spec
    sl = spec._slices()
    w = raw.copy()
    w[..., sl["q"]] += 2 * np.pi * 7                    # a reference that winds up by 7 turns (the unwrapped-IK quirk)
    w[..., sl["ref.q"]] += 2 * np.pi * 7
    a, b = spec.expand(raw), spec.expand(w)
    assert np.allclose(a, b, atol=1e-4)
    assert np.abs(b.reshape(len(b), 3, -1)[..., sl["ref.q"]]).max() <= np.pi + 1e-4
