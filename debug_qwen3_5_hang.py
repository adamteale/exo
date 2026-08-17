"""Debug the qwen3_5 2-rank pipeline hang. Run both ranks; prints show where each blocks."""
import multiprocessing as mp
import os
import sys
import json
import tempfile

def worker(rank, hostfile):
    os.environ["MLX_RANK"] = str(rank)
    os.chdir("/Users/ateale/code/exo")
    os.environ["EXO_DASHBOARD_DIR"] = "exo-dashboard-stub"
    import mlx.core as mx
    print(f"[R{rank}] start", flush=True)
    g = mx.distributed.init(backend="ring", strict=True)
    print(f"[R{rank}] distributed init done", flush=True)

    import mlx_lm.models.qwen3_5 as mod
    from mlx.utils import tree_map_with_path
    mx.random.seed(0)
    args = mod.ModelArgs(model_type="qwen3_5", text_config=dict(
        model_type="qwen3_5", vocab_size=512, hidden_size=512, intermediate_size=1024,
        num_hidden_layers=4, num_attention_heads=16, num_key_value_heads=4, head_dim=32,
        max_position_embeddings=128, rms_norm_eps=1e-6, tie_word_embeddings=False,
        attention_bias=False, full_attention_interval=2, linear_num_value_heads=32,
        linear_num_key_heads=16, linear_key_head_dim=32, linear_value_head_dim=32,
        linear_conv_kernel_dim=4, rope_parameters={"type":"default","rope_theta":10000.0,"partial_rotary_factor":0.25,"mrope_section":[11,11,10]}))
    m = mod.Model(args)
    m.update(tree_map_with_path(lambda p,v: v.astype(mx.bfloat16) if hasattr(v,"dtype") and v.dtype in (mx.float16,mx.float32,mx.bfloat16) else v, m.parameters()))
    mx.eval(m.parameters())
    print(f"[R{rank}] model built", flush=True)

    import exo.worker.engines.mlx.auto_parallel as ap
    from exo.shared.types.worker.shards import PipelineShardMetadata
    from exo.shared.types.backends import Backend
    from exo.shared.types.common import ModelId
    from exo.shared.types.memory import Memory
    from exo.shared.models.model_cards import ModelCard, ModelTask
    shard_meta = PipelineShardMetadata(model_card=ModelCard(model_id=ModelId("test/qwen3_5"), storage_size=Memory.from_gb(1), n_layers=4, hidden_size=512, supports_tensor=False, tasks=[ModelTask.TextGeneration], backends=[Backend.MlxMetal]), device_rank=rank, world_size=2, start_layer=rank*2, end_layer=(rank+1)*2, n_layers=4)
    gen = ap.pipeline_auto_parallel(m, g, shard_meta)
    try:
        while True:
            next(gen)
    except StopIteration as stop:
        m = stop.value
    print(f"[R{rank}] auto_parallel done, layers={[type(l).__name__ for l in ap.get_layers(ap.get_inner_model(m))]}", flush=True)

    inputs = mx.array([[1, 23, 45, 67, 89, 12, 34, 56]], dtype=mx.int32)
    print(f"[R{rank}] BEFORE forward", flush=True)
    try:
        out = m(inputs)
        mx.eval(out)
        print(f"[R{rank}] AFTER forward, out shape={out.shape}", flush=True)
    except Exception as e:
        print(f"[R{rank}] forward ERROR: {e}", flush=True)

if __name__ == "__main__":
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(["127.0.0.1:33700", "127.0.0.1:33701"], f)
        hostfile = f.name
    os.environ["MLX_HOSTFILE"] = hostfile
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=worker, args=(r, hostfile)) for r in range(2)]
    for p in procs: p.start()
    for p in procs: p.join(90)
    for p in procs:
        if p.is_alive():
            print(f"pid {p.pid} HUNG — terminating", flush=True)
            p.terminate()