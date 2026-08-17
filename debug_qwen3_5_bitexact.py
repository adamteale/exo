"""qwen3_5 bit-exact: single-process ref vs 2-rank sharded logits. Reproduces the live garble deterministically."""
import multiprocessing as mp
import os
import json
import tempfile
import numpy as np

CFG = dict(model_type="qwen3_5", text_config=dict(
    model_type="qwen3_5", vocab_size=512, hidden_size=512, intermediate_size=1024,
    num_hidden_layers=32, num_attention_heads=16, num_key_value_heads=4, head_dim=32,
    max_position_embeddings=128, rms_norm_eps=1e-6, tie_word_embeddings=False,
    attention_bias=False, full_attention_interval=2, linear_num_value_heads=32,
    linear_num_key_heads=16, linear_key_head_dim=32, linear_value_head_dim=32,
    linear_conv_kernel_dim=4, rope_parameters={"type":"default","rope_theta":10000.0,"partial_rotary_factor":0.25,"mrope_section":[11,11,10]}))
PROMPT = [[1, 23, 45, 67, 89, 12, 34, 56]]

def build():
    import mlx.core as mx
    import mlx_lm.models.qwen3_5 as mod
    from mlx.utils import tree_map_with_path
    mx.random.seed(0)
    m = mod.Model(mod.ModelArgs(**CFG))
    m.update(tree_map_with_path(lambda p,v: v.astype(mx.bfloat16) if hasattr(v,"dtype") and v.dtype in (mx.float16,mx.float32,mx.bfloat16) else v, m.parameters()))
    mx.eval(m.parameters())
    return mx, m

def ref(out_path):
    mx, m = build()
    inputs = mx.array(PROMPT, dtype=mx.int32)
    out = m(inputs)
    mx.eval(out)
    np.savez(out_path, logits=np.asarray(out.astype(mx.float32)))
    print("[ref] saved", flush=True)

def shard(rank, out_path, hostfile):
    os.environ["MLX_RANK"] = str(rank)
    os.chdir("/Users/ateale/code/exo")
    os.environ["EXO_DASHBOARD_DIR"] = "exo-dashboard-stub"
    import mlx.core as mx
    g = mx.distributed.init(backend="ring", strict=True)
    mx, m = build()
    import exo.worker.engines.mlx.auto_parallel as ap
    from exo.shared.types.worker.shards import PipelineShardMetadata
    from exo.shared.types.backends import Backend
    from exo.shared.types.common import ModelId
    from exo.shared.types.memory import Memory
    from exo.shared.models.model_cards import ModelCard, ModelTask
    sm = PipelineShardMetadata(model_card=ModelCard(model_id=ModelId("t/q"), storage_size=Memory.from_gb(1), n_layers=32, hidden_size=512, supports_tensor=False, tasks=[ModelTask.TextGeneration], backends=[Backend.MlxMetal]), device_rank=rank, world_size=2, start_layer=rank*16, end_layer=(rank+1)*16, n_layers=4)
    gen = ap.pipeline_auto_parallel(m, g, sm)
    try:
        while True: next(gen)
    except StopIteration as stop: m = stop.value
    inputs = mx.array(PROMPT, dtype=mx.int32)
    out = m(inputs)
    mx.eval(out)
    np.savez(out_path, logits=np.asarray(out.astype(mx.float32)))
    print(f"[R{rank}] saved", flush=True)

if __name__ == "__main__":
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(["127.0.0.1:33800","127.0.0.1:33801"], f); hf=f.name
    os.environ["MLX_HOSTFILE"] = hf
    ctx = mp.get_context("spawn")
    refp, shp = "/tmp/q35_ref.npz", "/tmp/q35_shard.npz"
    r = ctx.Process(target=ref, args=(refp,)); r.start(); r.join(120)
    procs = [ctx.Process(target=shard, args=(i, shp, hf)) for i in range(2)]
    for p in procs: p.start()
    for p in procs: p.join(120)
    try:
        ref_l = np.load(refp)["logits"]; sh_l = np.load(shp)["logits"]
        diff = np.abs(ref_l - sh_l); print(f"shapes ref={ref_l.shape} shard={sh_l.shape}", flush=True)
        print(f"max_diff={float(diff.max())} mean_diff={float(diff.mean())}", flush=True)
        rt = np.argmax(ref_l[0,-1]); st = np.argmax(sh_l[0,-1])
        print(f"ref first-token={int(rt)} shard first-token={int(st)}  match={int(rt)==int(st)}", flush=True)
    except Exception as e:
        print(f"compare failed: {e}", flush=True)