# gb10-nccl-switched-fabric[README.md](https://github.com/user-attachments/files/26472718/README.md)
# GB10 NCCL over Switched RoCE Fabric — What NVIDIA's Playbook Doesn't Tell You

A practical guide to getting multi-node NCCL working on NVIDIA GB10 (DGX Spark class) systems through a **switched fabric** rather than direct-connect — and the gaps in NVIDIA's official documentation that will trip you up.

## Acknowledgments

This guide wouldn't exist without the work of the DGX Spark community, and in particular **[eugr](https://github.com/eugr)** — whose contributions to the GB10 ecosystem have been enormous. His [spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) project is effectively the community standard for running vLLM on GB10 clusters, and his [forum posts](https://forums.developer.nvidia.com/t/nccl-for-2-sparks-setup-errors/353477/7) identified critical missing NCCL configuration that NVIDIA's own playbooks omit. He also co-created [Spark Arena](https://spark-arena.com) alongside **[raphael.amorim](https://forums.developer.nvidia.com/u/raphael.amorim)** and **[dbsci](https://forums.developer.nvidia.com/u/dbsci)** — a community hub for benchmarks, recipes, and the [sparkrun](https://sparkrun.dev) orchestration tool. If you're doing anything with GB10 inference, their work is the first place to look.

Thanks also to the broader [DGX Spark / GB10 forum community](https://forums.developer.nvidia.com/c/accelerated-computing/dgx-spark-gb10/721) who have been collectively filling in the gaps that NVIDIA's documentation leaves open. Much of what's documented below was pieced together from forum threads, trial and error, and the generosity of people sharing what worked (and what didn't).

## Background

This documents my experience standing up a two-node GB10 inference cluster with tensor-parallel vLLM over NCCL/RoCE, going through a MikroTik CRS812 L2 switch rather than the direct-connect topology NVIDIA's playbooks assume.

### Hardware

| Role | Platform | Memory | Fabric IP |
|------|----------|--------|-----------|
| **Head node** | GB10 (128GB unified) | 128GB | `10.x.x.10` |
| **Worker node** | GB10 (128GB unified) | 128GB | `10.x.x.11` |
| **Fabric switch** | MikroTik CRS812-8DS-2DQ-2DDQ-RM | — | L2 only |

### Network Topology

```
┌──────────────┐         ┌──────────────┐
│  Head Node   │         │ Worker Node  │
│  10.x.x.10   │         │  10.x.x.11   │
│  CX-7 QSFP56 │         │  CX-7 QSFP56 │
└──────┬───────┘         └──────┬───────┘
       │ 200Gbps                │ 200Gbps
       │                        │
  ┌────┴────────────────────────┴────┐
  │      Fabric Switch (L2 spine)    │
  │      VLAN 30 · 10.x.x.0/24       │
  │      MTU 9200 · RoCE             │
  └──────────────────────────────────┘
```

- **Fanout cable**: FiberMall 400G→2×200G (QDD2Q56400G-PC2M) on QSFP56-DD port
- **Auto-negotiation**: Disabled; speed forced to `200G-baseCR4`
- **MTU**: L2MTU 9200 on switch fabric ports, 9000 MTU on GB10 CX-7 interfaces
- **VLAN**: PVID 30, pure L2 switching (no routing on fabric)

---

## The Official Playbooks

NVIDIA provides two playbooks for multi-node GB10 setup:

1. **[Connect Two Sparks](https://build.nvidia.com/spark/connect-two-sparks/stacked-sparks)** — Physical cabling and network config
2. **[NCCL for Two Sparks](https://build.nvidia.com/spark/nccl/stacked-sparks)** — NCCL build, test, and validation

Both assume a **direct-connect** topology with link-local or static `192.168.x.x` addressing. If you're running through a switch (which you should be for any expandable fabric), you'll hit several undocumented issues.

---

## Issue 1: The Playbook's Netplan Config Will Overwrite Your Fabric

### What the playbook says

The "Connect Two Sparks" playbook provides a netplan template that assigns `192.168.100.x` / `192.168.200.x` addresses to all four CX-7 logical interfaces.

### The problem

If you already have a switched fabric with its own addressing scheme, applying this config will:

- **Overwrite your existing netplan** for CX-7 interfaces
- Create a parallel network that bypasses your switch
- Break any existing connectivity

### The fix

Only configure the interfaces you're actually using, with your fabric's addressing:

```yaml
# /etc/netplan/40-cx7.yaml — Head Node
network:
  version: 2
  ethernets:
    enp1s0f0np0:
      addresses:
        - 10.x.x.10/24
      mtu: 9000
    enP2p1s0f0np0:
      addresses:
        - 10.x.x.110/24
      mtu: 9000
```

```yaml
# /etc/netplan/40-cx7.yaml — Worker Node
network:
  version: 2
  ethernets:
    enp1s0f0np0:
      addresses:
        - 10.x.x.11/24
      mtu: 9000
    enP2p1s0f0np0:
      addresses:
        - 10.x.x.111/24
      mtu: 9000
```

**Key point**: Both HCA interfaces (`enp1s0f0np0` *and* `enP2p1s0f0np0`) need routable IPs on the same subnet. This is critical for Issue 3 below.

---

## Issue 2: Missing NCCL Environment Variables Cause Silent Fallback to Socket Transport

### What the playbook says

The NCCL playbook instructs you to set three environment variables:

```bash
export UCX_NET_DEVICES=enp1s0f1np1
export NCCL_SOCKET_IFNAME=enp1s0f1np1
export OMPI_MCA_btl_tcp_if_include=enp1s0f1np1
```

### The problem

These variables are **incomplete**. Without explicitly specifying the IB/RoCE HCA devices and confirming IB is enabled, NCCL silently falls back to **Ethernet socket transport** instead of RDMA verbs. This causes:

- Dramatically higher latency
- Significantly lower throughput
- Outright failures on some configurations

As [noted by community member eugr on the NVIDIA forums](https://forums.developer.nvidia.com/t/nccl-for-2-sparks-setup-errors/353477/7), the playbook is incomplete and will leave performance on the table even when it does work.

### The fix

Add these two critical variables:

```bash
export NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0   # Both RDMA devices that show as Up
export NCCL_IB_DISABLE=0                       # Explicitly enable IB/RoCE transport
```

Use the RDMA device names from `ibdev2netdev` that correspond to your **Up** interfaces. On GB10, each physical CX-7 port exposes two logical RDMA devices (e.g., `rocep1s0f0` and `roceP2p1s0f0`). Include **both** for full bandwidth.

You can verify NCCL is using IB transport by setting `NCCL_DEBUG=INFO` and checking for `NET/IB` in the output rather than `NET/Socket`.

---

## Issue 3: GID Type Mismatch on Second HCA Causes `ibv_modify_qp` Failure

### Symptoms

NCCL init completes (channels set up, IB transport selected), but then crashes with:

```
Call to ibv_modify_qp failed with 22 Invalid argument,
on dev roceP2p1s0f0:1,
local GID ::ffff:10.x.x.94,
remote GID fe80::xxxx:xxxx:xxxx:xxxx
```

### The problem

The second RDMA device (`roceP2p1s0f0`, corresponding to `enP2p1s0f0np0`) doesn't have a static IPv4 address assigned. It picks up a link-local IPv6 GID while the first device has an IPv4-mapped GID. NCCL can't establish a QP connection between mismatched GID types (RoCEv2/IPv4 vs RoCEv1/link-local).

**NVIDIA's playbook never mentions that the second HCA interface needs an IP address.** They only reference a single interface throughout.

### The fix

Assign a routable IPv4 address to `enP2p1s0f0np0` on **both** nodes (see the netplan examples in Issue 1). Both HCA interfaces must be on the same subnet so their RDMA GIDs are both IPv4-mapped.

### Workaround

If you can't assign IPs to the second HCA, restrict NCCL to only the first:

```bash
export NCCL_IB_HCA=rocep1s0f0   # Single HCA only
```

This works but limits you to half the available RDMA bandwidth.

---

## Issue 4: Shell Exports Don't Propagate to Remote Ranks via mpirun

### What the playbook says

The playbook shows setting environment variables with `export` in the shell, then running `mpirun`:

```bash
export UCX_NET_DEVICES=enp1s0f1np1
export NCCL_SOCKET_IFNAME=enp1s0f1np1
mpirun -np 2 -H <node1>:1,<node2>:1 ...
```

### The problem

`mpirun` launches the remote rank via SSH. Shell exports on the local machine **do not propagate** to the remote process. The remote rank runs with default (empty) values for these variables, which can cause interface mismatches or fallback to the wrong transport.

### The fix

Pass **all** environment variables through `mpirun`'s `-x` flag:

```bash
mpirun -np 2 -H <head-ip>:1,<worker-ip>:1 \
  --mca plm_rsh_agent "ssh -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no" \
  -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH \
  -x NCCL_SOCKET_IFNAME=enp1s0f0np0 \
  -x UCX_NET_DEVICES=enp1s0f0np0 \
  -x NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0 \
  -x NCCL_IB_DISABLE=0 \
  -x NCCL_DEBUG=INFO \
  $HOME/nccl-tests/build/all_gather_perf
```

---

## Issue 5: Running mpirun from Both Nodes Simultaneously

### What the playbook implies

The playbook says to "execute the following commands on both nodes," which some users interpret as running `mpirun` from **both** machines simultaneously.

### The problem

You should only run `mpirun` from **one** node. It automatically SSHes into the other node to launch the remote rank. Running from both sides creates two overlapping NCCL jobs that fight over the same HCAs and GPU memory, producing:

- Halved bandwidth (we observed ~8.6 GB/s instead of ~20 GB/s)
- Intermittent failures
- Confusing error messages

### The fix

Pick one node as your launcher and only run `mpirun` from there.

---

## Issue 6: The Troubleshooting Page Doesn't Cover the Actual Errors

### The problem

NVIDIA's [troubleshooting page](https://build.nvidia.com/spark/nccl/troubleshooting) lists three issues:

1. mpirun hangs (SSH issues)
2. Network interface not found
3. NCCL build fails

The actual errors users encounter — GID mismatches, `ibv_modify_qp` failures, silent socket fallback, GPU resource exhaustion in Ray — are not mentioned.

---

## Validated NCCL Test Results (Through Switch)

After applying all fixes, here are the `all_gather_perf` results through the MikroTik CRS812 L2 switch with both HCAs active:

| Buffer Size | busbw (GB/s) | % of 25 GB/s Theoretical |
|-------------|-------------|--------------------------|
| 8 MB        | 15.3        | 61%                      |
| 32 MB       | 18.6        | 74%                      |
| 512 MB      | 19.5        | 78%                      |
| 1 GB        | 20.1        | 81%                      |
| 2 GB        | 21.4        | 86%                      |
| 4 GB        | 21.9        | 87%                      |
| 16 GB       | 20.3        | 81%                      |

Peak bandwidth of **~22 GB/s** (87% of theoretical) at 2–4 GB buffer sizes, consistent with community-reported results on direct-connect setups. The slight dip at 16 GB is due to memory pressure on GB10's unified memory architecture.

---

## Going from NCCL Tests to Distributed Inference

Once NCCL is validated, the path to TP=2 vLLM inference is:

1. **Use [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker)** — Community-maintained Docker setup that handles all the NCCL/RoCE/Ray plumbing automatically. Far more reliable than NVIDIA's official vLLM playbook.

2. **Build and distribute the image over the fabric**:
   ```bash
   ./build-and-copy.sh -c <worker-fabric-ip>
   ```

3. **Launch the Ray cluster** with your fabric interface names:
   ```bash
   ./launch-cluster.sh \
     --nodes "<head-ip>,<worker-ip>" \
     --eth-if enp1s0f0np0 \
     --ib-if rocep1s0f0,roceP2p1s0f0 \
     -d start
   ```

4. **Serve a model with TP=2**:
   ```bash
   ./launch-cluster.sh \
     --nodes "<head-ip>,<worker-ip>" \
     --eth-if enp1s0f0np0 \
     --ib-if rocep1s0f0,roceP2p1s0f0 \
     exec vllm serve <model-repo-id> \
     --port 8000 --host 0.0.0.0 \
     --gpu-memory-utilization 0.85 \
     -tp 2 \
     --distributed-executor-backend ray \
     --max-model-len 32768 \
     --load-format fastsafetensors \
     --trust-remote-code
   ```

### Note on model swapping

When switching models, you must **stop and restart the Ray cluster** — vLLM's Ray GPU resource allocation doesn't release cleanly between model serves. A full `stop` → `start` cycle takes about 15 seconds.

---

## Useful Diagnostic Commands

```bash
# Check which RDMA interfaces are up
ibdev2netdev

# Check IP assignments on CX-7 interfaces
ip addr show enp1s0f0np0
ip addr show enP2p1s0f0np0

# Monitor fabric throughput in real-time
nload enp1s0f0np0

# Verify NCCL is using IB (not socket) transport
# Look for "NET/IB" vs "NET/Socket" in output
mpirun ... -x NCCL_DEBUG=INFO ...
```

---

## Resources

- **[eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker)** — The community standard for vLLM on GB10 clusters (single and multi-node)
- **[sparkrun](https://sparkrun.dev)** — Unified orchestration tool for single/multi-Spark clusters, with LiteLLM proxy integration and systemd export
- **[Spark Arena](https://spark-arena.com)** — Community benchmark hub for comparing inference performance across configurations
- **[eugr/llama-benchy](https://github.com/eugr/llama-benchy)** — Benchmarking tool for GB10 inference, outputs in llama-bench format
- **[NVIDIA DGX Spark Forums](https://forums.developer.nvidia.com/c/accelerated-computing/dgx-spark-gb10/721)** — The most active source of real-world GB10 troubleshooting
- **[NVIDIA NCCL Playbook](https://build.nvidia.com/spark/nccl/stacked-sparks)** — The official starting point (with the caveats documented above)
- **[NVIDIA Connect Two Sparks Playbook](https://build.nvidia.com/spark/connect-two-sparks/stacked-sparks)** — Official network setup guide

---

## License

MIT — use this however you'd like. If it saves you a few hours of debugging, that's the goal.
