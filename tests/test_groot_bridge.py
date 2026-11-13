"""Ponte no protocolo do GR00T: conversões, contagem por episódio e ida e volta por ZMQ."""

import json
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from evalkit.groot_bridge import (
    LANGUAGE_KEY,
    WARMUP_STEPS,
    ChunkBridge,
    RecordedActions,
    action_noise,
    from_groot_action,
    from_groot_observation,
    parse_groot_url,
    to_groot_action,
    to_groot_observation,
)

STATE = np.array([1.0, -2.0, 3.0, -4.0, 5.0, 60.0], np.float32)
PIXELS = {"ego": np.full((4, 6, 3), 7, np.uint8), "external_D455": np.full((4, 6, 3), 9, np.uint8)}


class ChunkPolicy:
    """Devolve um chunk de 10 ações em que a ação t vale ``estado + t``; registra as chamadas."""

    def __init__(self, chunk_size=10, n_action_steps=4):
        self.chunk_size = chunk_size
        self.config = SimpleNamespace(
            n_action_steps=n_action_steps,
            input_features={"observation.state": None, "observation.images.ego": None,
                            "observation.images.external_D455": None})
        self.calls = []
        self.resets = 0

    def reset(self):
        self.resets += 1

    def predict_chunk(self, observation, instruction):
        self.calls.append((observation, instruction))
        return observation["agent_pos"][None] + np.arange(self.chunk_size, dtype=np.float32)[:, None]


def test_observation_matches_workshop_layout_and_round_trips():
    obs = to_groot_observation(STATE, PIXELS, "pegue o frasco")
    # o que o GR00TRemotePolicy monta: (B=1, T=1) em tudo, braço e garra separados, texto em [[...]]
    assert obs["video"]["ego"].shape == (1, 1, 4, 6, 3)
    assert obs["state"]["single_arm"].shape == (1, 1, 5) and obs["state"]["gripper"].shape == (1, 1, 1)
    assert obs["language"] == {LANGUAGE_KEY: [["pegue o frasco"]]}
    back, text = from_groot_observation(obs)
    assert text == "pegue o frasco"
    np.testing.assert_array_equal(back["agent_pos"], STATE)
    assert back["pixels"].keys() == PIXELS.keys()
    np.testing.assert_array_equal(back["pixels"]["external_D455"], PIXELS["external_D455"])


def test_observation_rejects_normalized_images_and_wrong_state():
    obs = to_groot_observation(STATE, {"ego": PIXELS["ego"].astype(np.float32) / 255}, "x")
    with pytest.raises(ValueError, match="uint8"):
        from_groot_observation(obs)
    obs = to_groot_observation(STATE, {}, "x")
    obs["state"]["gripper"] = np.zeros((1, 1, 2), np.float32)
    with pytest.raises(ValueError, match="estado"):
        from_groot_observation(obs)


def test_action_round_trip_cuts_at_horizon():
    chunk = np.arange(30, dtype=np.float32).reshape(5, 6)
    action = to_groot_action(chunk)
    assert action["single_arm"].shape == (1, 5, 5) and action["gripper"].shape == (1, 5, 1)
    np.testing.assert_array_equal(from_groot_action(action, 100), chunk)
    np.testing.assert_array_equal(from_groot_action(action, 3), chunk[:3])


def test_bridge_returns_n_action_steps_and_counts_queries_per_episode(tmp_path):
    policy, lines = ChunkPolicy(), []
    log = tmp_path / "ponte.jsonl"
    bridge = ChunkBridge(policy, log=lines.append, log_file=log)
    assert bridge.n_action_steps == 4 and bridge.cameras == ["ego", "external_D455"]
    bridge.reset()  # o cliente do workshop chama reset antes do primeiro episódio
    obs = to_groot_observation(STATE, PIXELS, "t")
    for _ in range(3):
        action, info = bridge.get_action(obs, None)
    assert info == {} and action["single_arm"].shape == (1, 4, 5)  # o chunk de 10 cortado em 4
    np.testing.assert_array_equal(from_groot_action(action, 100)[:, 5], STATE[5] + np.arange(4))
    bridge.reset()
    bridge.reset()  # o último reset do cliente não abre episódio
    bridge.get_action(obs)
    bridge.close()
    assert policy.resets == 3
    assert [line for line in lines if "episódio" in line] == [
        "[ponte] episódio 1: 3 consultas, até 12 ações (chunks de 4)",
        "[ponte] episódio 2: 1 consultas, até 4 ações (chunks de 4)",
    ]
    assert sum("observação" in line for line in lines) == 1  # só a primeira observação é impressa
    assert log.read_text(encoding="utf-8").count("\n") == 2


def test_bridge_rejects_cameras_the_policy_does_not_expect():
    bridge = ChunkBridge(ChunkPolicy(), log=lambda _: None)
    obs = to_groot_observation(STATE, {"ego": PIXELS["ego"], "front": PIXELS["ego"]}, "t")
    with pytest.raises(ValueError, match="rename_map"):
        bridge.get_action(obs)


def test_parse_groot_url():
    assert parse_groot_url("groot://localhost:5555") == ("localhost", 5555)
    assert parse_groot_url("outputs/train/x") is None
    with pytest.raises(ValueError):
        parse_groot_url("groot://localhost")


def test_server_and_client_over_zmq():
    pytest.importorskip("zmq")
    pytest.importorskip("msgpack")
    from evalkit.groot_bridge import GrootRemotePolicy, PolicyServer, pack, unpack

    policy = ChunkPolicy()
    server = PolicyServer(ChunkBridge(policy, log=lambda _: None), port=0)
    thread = threading.Thread(target=server.serve, daemon=True)
    thread.start()
    client = GrootRemotePolicy("127.0.0.1", server.port, action_horizon=100, timeout_ms=10_000)
    try:
        client.reset()
        actions = [client.act({"agent_pos": STATE, "pixels": PIXELS}, "t") for _ in range(6)]
        # 4 ações por chunk: a 5ª ação já vem de uma segunda consulta
        assert client.queries == 2 and len(policy.calls) == 2
        np.testing.assert_allclose([a[0] for a in actions], STATE[0] + np.array([0, 1, 2, 3, 0, 1]))
        assert policy.calls[0][1] == "t"
        assert client.call("get_modality_config", requires_input=False) == {}
        # erro no servidor volta ao cliente e o servidor continua de pé
        bad = to_groot_observation(STATE, {"front": PIXELS["ego"]}, "t")
        with pytest.raises(RuntimeError, match="rename_map"):
            client.call("get_action", {"observation": bad, "options": None})
        assert client.call("ping", requires_input=False)["status"] == "ok"
        # o pedido cru, como o PolicyClient do workshop o monta, e a resposta como ele a lê
        raw = pack({"endpoint": "get_action",
                    "data": {"observation": to_groot_observation(STATE, PIXELS, "t"), "options": None}})
        action, info = tuple(unpack(server.handle(raw)))
        assert action["gripper"].shape == (1, 4, 1) and info == {}
        client.call("kill", requires_input=False)
        thread.join(timeout=5)
        assert not thread.is_alive()
    finally:
        client.close()
        server.close()


def test_client_renames_cameras_like_the_workshop():
    pytest.importorskip("zmq")
    pytest.importorskip("msgpack")
    from evalkit.groot_bridge import GrootRemotePolicy, PolicyServer

    policy = ChunkPolicy()  # um servidor que espera os nomes do GR00T do curso
    policy.config.input_features = {"observation.images.front": None, "observation.images.wrist": None}
    server = PolicyServer(ChunkBridge(policy, log=lambda _: None), port=0)
    thread = threading.Thread(target=server.serve, daemon=True)
    thread.start()
    client = GrootRemotePolicy("127.0.0.1", server.port, action_horizon=16, timeout_ms=10_000,
                               rename_map={"external_D455": "front", "ego": "wrist"})
    try:
        client.act({"agent_pos": STATE, "pixels": PIXELS}, "t")
        pixels = policy.calls[0][0]["pixels"]
        assert sorted(pixels) == ["front", "wrist"]
        np.testing.assert_array_equal(pixels["front"], PIXELS["external_D455"])
        client.call("kill", requires_input=False)
        thread.join(timeout=5)
    finally:
        client.close()
        server.close()


def _fake_recording(tmp_path, lengths):
    """Registro no formato do ``scripts/isaac_recorder.py``: juntas na ordem do USD, ação em radianos."""
    from vials_lite import spec

    rec = tmp_path / "Eval.rec"
    rec.mkdir()
    names = list(reversed(spec.JOINTS_USD))  # ordem diferente da do LeRobot, para exercitar a reordenação
    (rec / "meta.json").write_text(json.dumps({"joint_names": names}), encoding="utf-8")
    (rec / "episodes.jsonl").write_text("", encoding="utf-8")
    rng = np.random.default_rng(0)
    for k, n in enumerate(lengths):
        action = rng.uniform(-0.5, 0.5, (n, 6))
        np.savez(rec / f"ep_{k:04d}.npz", action=action, joint_pos=action + 0.01 * k)
    return rec, names


def test_recorded_actions_replay_like_the_workshop_client(tmp_path):
    from vials_lite import spec
    from vials_lite.joint_mapping import lerobot_to_radians, radians_to_lerobot

    rec, names = _fake_recording(tmp_path, [130, 260])
    usd = dict(zip(spec.JOINTS_LEROBOT, spec.JOINTS_USD, strict=True))
    order = [names.index(usd[j]) for j in spec.JOINTS_LEROBOT]
    bridge = ChunkBridge(RecordedActions(rec, n_action_steps=100), log=lambda _: None,
                         log_file=tmp_path / "p.jsonl")
    bridge.reset()  # antes do 1º episódio
    for k, n in enumerate([130, 260]):
        ep = np.load(rec / f"ep_{k:04d}.npz")
        sent = []
        queue = []
        for t in range(WARMUP_STEPS, n):  # o cliente: aquecimento, depois consulta quando a fila acaba
            if not queue:
                state = radians_to_lerobot(ep["joint_pos"][t - 1][order]).astype(np.float32)
                action, _ = bridge.get_action(to_groot_observation(state, {}, "x"))
                queue = list(from_groot_action(action, 100))
            sent.append(lerobot_to_radians(queue.pop(0)))
        np.testing.assert_allclose(np.array(sent), ep["action"][WARMUP_STEPS:n][:, order], atol=1e-5)
        bridge.reset()
    bridge.reset()  # o reset final do cliente não abre episódio nem lê o ep_0002, que não existe
    bridge.close()
    rows = [json.loads(line) for line in (tmp_path / "p.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["rec_episode"] for r in rows] == [0, 1] and [r["queries"] for r in rows] == [2, 3]
    assert all(r["dev_max_deg"] < 1e-3 for r in rows)


def test_recorded_actions_hold_the_last_action_and_flag_misaligned_episodes(tmp_path):
    rec, _ = _fake_recording(tmp_path, [50, 50])
    policy = RecordedActions(rec, n_action_steps=100)
    policy.reset()
    policy.reset()  # dois resets sem consulta: continua no episódio 0
    chunk = policy.predict_chunk({"agent_pos": np.zeros(6, np.float32)}, "x")
    assert policy.episode == 0 and chunk.shape == (100, 6)
    np.testing.assert_array_equal(chunk[40:], np.repeat(chunk[39:40], 60, axis=0))  # 50 - 10 gravadas
    assert policy.episode_info()["dev_first_deg"] > 1  # o estado zero não é o gravado


def test_recorded_actions_noise_is_seeded_by_episode(tmp_path):
    from vials_lite.joint_mapping import lerobot_to_radians

    rec, _ = _fake_recording(tmp_path, [120])
    obs = {"agent_pos": np.zeros(6, np.float32)}
    clean, noisy = RecordedActions(rec), RecordedActions(rec, action_noise_deg=0.05)
    for p in (clean, noisy):
        p.reset()
    d = lerobot_to_radians(noisy.predict_chunk(obs, "x")) - lerobot_to_radians(clean.predict_chunk(obs, "x"))
    np.testing.assert_allclose(d, action_noise(120, 0, 0.05)[WARMUP_STEPS:WARMUP_STEPS + 100], atol=1e-5)
    assert 0.03 < np.degrees(d.std()) < 0.07
    assert not np.allclose(action_noise(120, 0, 0.05), action_noise(120, 1, 0.05))
