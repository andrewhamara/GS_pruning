"""
graph_eval.py

End-to-end round-trip evaluation:

    .ply file --(load)--> GaussianSplatGraph --(build KNN)--> graph
       --(write back)--> GaussianModel --(render)--> images
       --(SSIM/PSNR/LPIPS)--> metrics

This script mirrors render_and_evaluate.py but inserts a round trip through
the simple GaussianSplatGraph defined in gaussian_graph.py, so that we
can verify the graph representation faithfully preserves the original
Gaussian primitives.

Usage (same args as render_and_evaluate.py, plus --k):

    python graph_eval.py -m <model_path> --iteration -1 --k 10 \
        [--skip_train] [--skip_test] [--gpu 0]
"""

import os
import json
import time
import subprocess

import numpy as np
import torch
import torchvision

# pick least-used GPU like render_and_evaluate.py does
cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
try:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(np.argmin([int(x.split()[2]) for x in result[:-1]]))
except Exception:
    pass

from torch import nn
from tqdm import tqdm
from argparse import ArgumentParser

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.progress import (Progress, SpinnerColumn, BarColumn, TextColumn,
                           TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn)
from rich.text import Text

from scene import Scene
from gaussian_renderer import GaussianModel, render, prefilter_voxel
from arguments import ModelParams, PipelineParams, get_combined_args
from utils.general_utils import safe_state
from utils.loss_utils import ssim
from utils.image_utils import psnr
import lpips

from gaussian_graph import GaussianSplatGraph


# ----------------------------------------------------------------- round trip

def graph_from_gaussian_ply(ply_path: str, k: int) -> GaussianSplatGraph:
    print(f"\n[graph] loading primitives from {ply_path}")
    g = GaussianSplatGraph(knn_neighbors=k).load_ply(ply_path)
    print(f"[graph] {g!r}")
    print(f"[graph] building KNN graph with K={k}")
    g.build_graph(k)
    print(f"[graph] summary: {g.summary()}")
    return g


def write_graph_into_gaussian_model(g: GaussianSplatGraph, gaussians: GaussianModel) -> None:
    """
    Overwrite the GaussianModel's primitive nn.Parameters with the graph's node arrays.
    This is the "convert the graph back into the original gaussians" step.
    """
    print("[graph] writing graph nodes back into GaussianModel primitives")
    dev = "cuda"

    def as_param(arr: np.ndarray) -> nn.Parameter:
        return nn.Parameter(torch.tensor(arr, dtype=torch.float, device=dev), requires_grad=False)

    n_model = gaussians.get_xyz.shape[0]
    n_graph = len(g)
    if n_model != n_graph:
        print(f"[graph] WARNING: model has {n_model} gaussians but graph has {n_graph}. "
              f"Proceeding with graph values.")

    gaussians._xyz      = as_param(g.xyz)
    gaussians._rotation = as_param(g.rotation)
    gaussians._scaling  = as_param(g.scaling)
    gaussians._opacity  = as_param(g.opacity)
    if g.features.size:
        gaussians._feats = as_param(g.features)

    # Re-allocate buffers that depend on N.
    gaussians.max_radii2D     = torch.zeros((gaussians.get_xyz.shape[0]), device=dev)
    gaussians._neural_xyz     = torch.zeros((gaussians.get_xyz.shape[0], 3), device=dev)
    gaussians._neural_scaling = torch.zeros((gaussians.get_xyz.shape[0], 3), device=dev)
    gaussians._neural_rotation= torch.zeros((gaussians.get_xyz.shape[0], 4), device=dev)
    gaussians._neural_opacity = torch.zeros((gaussians.get_xyz.shape[0], 1), device=dev)


# --------------------------------------------------------------------- eval

def render_set_and_eval(model_path, name, iteration, views, gaussians, pipeline, background, lpips_fn):
    render_path = os.path.join(model_path, name, "graph_ours_{}".format(iteration), "renders")
    gts_path    = os.path.join(model_path, name, "graph_ours_{}".format(iteration), "gt")
    os.makedirs(render_path, exist_ok=True)
    os.makedirs(gts_path, exist_ok=True)

    ssims, psnrs, lpipss, name_list, t_list = [], [], [], [], []

    for idx, view in enumerate(tqdm(views, desc=f"Rendering {name}")):
        torch.cuda.synchronize(); t0 = time.time()
        voxel_visible_mask = prefilter_voxel(view, gaussians, pipeline, background)
        render_pkg = render(view, gaussians, pipeline, background, visible_mask=voxel_visible_mask)
        torch.cuda.synchronize(); t1 = time.time()
        t_list.append(t1 - t0)

        rendering = torch.clamp(render_pkg["render"], 0.0, 1.0)
        gt        = torch.clamp(view.original_image[0:3, :, :], 0.0, 1.0)

        fname = '{0:05d}.png'.format(idx)
        name_list.append(fname)
        torchvision.utils.save_image(rendering, os.path.join(render_path, fname))
        torchvision.utils.save_image(gt,        os.path.join(gts_path,    fname))

        ssims.append(ssim(rendering, gt))
        psnrs.append(psnr(rendering, gt).mean().double())
        lpipss.append(lpips_fn(rendering, gt).detach())

    mean_ssim  = torch.tensor(ssims).mean().item()
    mean_psnr  = torch.tensor(psnrs).mean().item()
    mean_lpips = torch.tensor(lpipss).mean().item()

    print(f"\n[{name}] SSIM : {mean_ssim:>12.7f}")
    print(f"[{name}] PSNR : {mean_psnr:>12.7f}")
    print(f"[{name}] LPIPS: {mean_lpips:>12.7f}")
    if len(t_list) > 5:
        fps = 1.0 / np.array(t_list[5:]).mean()
        print(f"[{name}] FPS  : {fps:.4f}")

    per_view = {
        "SSIM":  {n: v for v, n in zip(torch.tensor(ssims).tolist(),  name_list)},
        "PSNR":  {n: v for v, n in zip(torch.tensor(psnrs).tolist(),  name_list)},
        "LPIPS": {n: v for v, n in zip(torch.tensor(lpipss).tolist(), name_list)},
    }
    out_dir = os.path.join(model_path, name, "graph_ours_{}".format(iteration))
    with open(os.path.join(out_dir, "per_view.json"), 'w') as fp:
        json.dump(per_view, fp, indent=True)
    with open(os.path.join(out_dir, "summary.json"), 'w') as fp:
        json.dump({"SSIM": mean_ssim, "PSNR": mean_psnr, "LPIPS": mean_lpips}, fp, indent=True)

    return {"SSIM": mean_ssim, "PSNR": mean_psnr, "LPIPS": mean_lpips}


# ---------------------------------------------- prune sweep (score-only eval)

def score_set(views, gaussians, pipeline, background, lpips_fn,
              advance_fn=None, desc: str = "render"):
    """
    Render every view and compute mean SSIM / PSNR / LPIPS.

    If `advance_fn` is given, it is called once per view (used to drive a
    rich Progress bar from the outside). Otherwise we wrap `views` in tqdm.
    """
    ssims, psnrs, lpipss = [], [], []
    iter_views = views if advance_fn is not None else tqdm(views, desc=desc)
    for view in iter_views:
        voxel_visible_mask = prefilter_voxel(view, gaussians, pipeline, background)
        render_pkg = render(view, gaussians, pipeline, background, visible_mask=voxel_visible_mask)
        rendering  = torch.clamp(render_pkg["render"], 0.0, 1.0)
        gt         = torch.clamp(view.original_image[0:3, :, :], 0.0, 1.0)
        ssims.append(ssim(rendering, gt))
        psnrs.append(psnr(rendering, gt).mean().double())
        lpipss.append(lpips_fn(rendering, gt).detach())
        if advance_fn is not None:
            advance_fn()
    return {
        "SSIM":  torch.tensor(ssims).mean().item(),
        "PSNR":  torch.tensor(psnrs).mean().item(),
        "LPIPS": torch.tensor(lpipss).mean().item(),
    }


# ----------------------------------------------------------- rich TUI helpers

_METHOD_STYLE = {
    "fiedler": "cyan",
    "random":  "magenta",
}


def _make_header_panel(dataset, scene, full, lam2, mag, normalized_laplacian,
                       symmetric, methods, seed, percentages):
    txt = Text()
    txt.append(f"scene        : {dataset.model_path}\n",      style="bold")
    txt.append(f"iteration    : {scene.loaded_iter}\n")
    txt.append(f"K (knn)      : {full.knn_neighbors}   "
               f"symmetric: {symmetric}   normalized_lapl: {normalized_laplacian}\n")
    txt.append(f"N (nodes)    : {len(full):,}\n")
    txt.append(f"lambda_2     : {lam2:.6e}   "
               f"|fv| range: [{mag.min():.3e}, {mag.max():.3e}]\n", style="green")
    method_chips = Text("methods      : ")
    for i, m in enumerate(methods):
        if i: method_chips.append(" + ")
        method_chips.append(m, style=f"bold {_METHOD_STYLE[m]}")
    if "random" in methods:
        method_chips.append(f"   seed={seed}", style="dim")
    txt.append(method_chips)
    txt.append("\n")
    txt.append(f"sweep        : {percentages[0]}%..{percentages[-1]}% "
               f"step {percentages[1]-percentages[0] if len(percentages)>1 else '?'}, "
               f"{len(percentages)} cuts\n")
    return Panel(txt, title="[bold]graph_eval --prune-sweep[/]",
                 border_style="bright_blue", expand=True)


def _make_results_table(skip_train: bool, skip_test: bool):
    table = Table(show_header=True, header_style="bold white on grey23",
                  expand=True, title="[bold]Sweep Results[/]")
    table.add_column("k%",      justify="right", style="bold", width=5)
    table.add_column("method",  justify="left",                width=8)
    table.add_column("N",       justify="right",               width=10)
    table.add_column("kept",    justify="right", style="dim",  width=7)
    if not skip_test:
        table.add_column("test SSIM",  justify="right")
        table.add_column("test PSNR",  justify="right")
        table.add_column("test LPIPS", justify="right")
    if not skip_train:
        table.add_column("train SSIM",  justify="right", style="dim")
        table.add_column("train PSNR",  justify="right", style="dim")
        table.add_column("train LPIPS", justify="right", style="dim")
    return table


def _add_table_row(table, kp, method, n_kept, n_full,
                   tr, te, skip_train, skip_test, is_separator=False):
    style = _METHOD_STYLE[method]
    cells = [
        f"{kp}%",
        Text(method, style=f"bold {style}"),
        f"{n_kept:,}",
        f"{100.0 * n_kept / n_full:.1f}%",
    ]
    if not skip_test:
        if te is None:
            cells += ["—", "—", "—"]
        else:
            cells += [f"{te['SSIM']:.4f}", f"{te['PSNR']:.2f}", f"{te['LPIPS']:.4f}"]
    if not skip_train:
        if tr is None:
            cells += ["—", "—", "—"]
        else:
            cells += [f"{tr['SSIM']:.4f}", f"{tr['PSNR']:.2f}", f"{tr['LPIPS']:.4f}"]
    table.add_row(*cells, end_section=is_separator)


def prune_sweep(dataset: ModelParams, iteration: int, pipeline: PipelineParams,
                k_knn: int, symmetric: str, normalized_laplacian: bool,
                k_min: int, k_max: int, k_step: int,
                skip_train: bool, skip_test: bool,
                random_baseline: bool, random_seed: int,
                lpips_fn):
    """
    Walk prune percentages [k_min, k_max] in steps of k_step. The Fiedler vector
    is computed ONCE on the full graph; each iteration only re-thresholds |fv|
    to pick survivors. If random_baseline is True, also runs a same-percentage
    uniformly-random prune for direct comparison.
    """
    console = Console()

    with torch.no_grad():
        # 1) Bring up model + scene (loads cameras, MLPs, original .ply).
        console.print("[dim]bringing up GaussianModel + Scene ...[/]")
        gaussians = GaussianModel(
            dataset.feat_dim, dataset.knn_neighbors, dataset.voxel_size,
            dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor,
            dataset.upsampling_factors)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        # 2) Build the GaussianSplatGraph from the same .ply Scene loaded.
        ply_path = os.path.join(dataset.model_path, "point_cloud",
                                "iteration_" + str(scene.loaded_iter), "point_cloud.ply")
        console.print(f"[dim]loading .ply: {ply_path}[/]")
        full = (GaussianSplatGraph(knn_neighbors=k_knn)
                .load_ply(ply_path)
                .build_graph(k_knn, symmetric=symmetric))

        # 3) Compute the Fiedler vector ONCE on the full graph.
        console.print("[dim]computing Fiedler vector (one-shot for the whole sweep) ...[/]")
        lam2, fv = full.compute_fiedler_vector(normalized=normalized_laplacian)
        mag = np.abs(fv)

        bg_color   = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        percentages = list(range(k_min, k_max + 1, k_step))
        methods     = ["fiedler"] + (["random"] if random_baseline else [])
        rng         = np.random.default_rng(random_seed)

        header   = _make_header_panel(dataset, scene, full, lam2, mag,
                                      normalized_laplacian, symmetric, methods,
                                      random_seed, percentages)
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold]{task.description}[/]"),
            BarColumn(bar_width=None),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            transient=False,
            expand=True,
        )
        # outer sweep task: 1 step per (kp, method)
        sweep_task = progress.add_task("[bold]sweep[/]",
                                       total=len(percentages) * len(methods))
        table = _make_results_table(skip_train, skip_test)

        rows = []
        with Live(Group(header, progress, table), console=console,
                  refresh_per_second=8, vertical_overflow="visible"):
            for kp in percentages:
                kp_method_results = []
                for method in methods:
                    if method == "fiedler":
                        pruned = full.prune_by_fiedler_magnitude(mag, kp, rebuild=False)
                    else:
                        pruned = full.prune_random(kp, rng=rng, rebuild=False)

                    write_graph_into_gaussian_model(pruned, gaussians)
                    gaussians.create_knn_graph()
                    gaussians.eval()

                    row = {"prune_percent": kp, "method": method,
                           "num_nodes": len(pruned),
                           "frac_kept": len(pruned) / len(full)}

                    tr = te = None
                    if not skip_train:
                        n = len(scene.getTrainCameras())
                        t = progress.add_task(
                            f"k={kp}% [bold {_METHOD_STYLE[method]}]{method}[/] train", total=n)
                        tr = score_set(scene.getTrainCameras(), gaussians, pipeline,
                                       background, lpips_fn,
                                       advance_fn=lambda t=t: progress.advance(t))
                        progress.remove_task(t)
                        row.update({"train_SSIM": tr["SSIM"], "train_PSNR": tr["PSNR"],
                                    "train_LPIPS": tr["LPIPS"]})
                    if not skip_test:
                        n = len(scene.getTestCameras())
                        t = progress.add_task(
                            f"k={kp}% [bold {_METHOD_STYLE[method]}]{method}[/] test", total=n)
                        te = score_set(scene.getTestCameras(), gaussians, pipeline,
                                       background, lpips_fn,
                                       advance_fn=lambda t=t: progress.advance(t))
                        progress.remove_task(t)
                        row.update({"test_SSIM": te["SSIM"], "test_PSNR": te["PSNR"],
                                    "test_LPIPS": te["LPIPS"]})

                    kp_method_results.append((method, len(pruned), tr, te))
                    rows.append(row)
                    progress.advance(sweep_task)

                # Add this k% block to the table, separator between k%s.
                for i, (method, n_kept, tr, te) in enumerate(kp_method_results):
                    _add_table_row(table, kp, method, n_kept, len(full),
                                   tr, te, skip_train, skip_test,
                                   is_separator=(i == len(kp_method_results) - 1))

        # 5) Persist results.
        out_json = os.path.join(dataset.model_path, "graph_prune_sweep.json")
        out_csv  = os.path.join(dataset.model_path, "graph_prune_sweep.csv")
        with open(out_json, "w") as fp:
            json.dump({
                "iteration":            scene.loaded_iter,
                "K":                    k_knn,
                "symmetric":            symmetric,
                "normalized_laplacian": normalized_laplacian,
                "lambda_2":             lam2,
                "n_full":               len(full),
                "methods":              methods,
                "random_seed":          random_seed,
                "rows":                 rows,
            }, fp, indent=2)
        if rows:
            cols = sorted({k for r in rows for k in r.keys()},
                          key=lambda c: (c != "prune_percent", c != "method", c))
            with open(out_csv, "w") as fp:
                fp.write(",".join(cols) + "\n")
                for r in rows:
                    fp.write(",".join(str(r.get(c, "")) for c in cols) + "\n")
        console.print()
        console.print(f"[green]✓[/] wrote [bold]{out_json}[/]")
        console.print(f"[green]✓[/] wrote [bold]{out_csv}[/]")


# ---------------------------------------------------------------- main flow

def graph_eval(dataset: ModelParams, iteration: int, pipeline: PipelineParams,
               k: int, skip_train: bool, skip_test: bool, lpips_fn):
    with torch.no_grad():
        # 1) build a GaussianModel and Scene exactly like render_and_evaluate.py
        gaussians = GaussianModel(
            dataset.feat_dim, dataset.knn_neighbors, dataset.voxel_size,
            dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor,
            dataset.upsampling_factors)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        # 2) round-trip the primitives through GaussianSplatGraph
        ply_path = os.path.join(dataset.model_path, "point_cloud",
                                "iteration_" + str(scene.loaded_iter), "point_cloud.ply")
        graph = graph_from_gaussian_ply(ply_path, k)
        write_graph_into_gaussian_model(graph, gaussians)

        # 3) rebuild the model's internal KNN structure on the round-tripped values
        gaussians.create_knn_graph()
        gaussians.eval()

        # 4) render + eval
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        all_metrics = {}
        if not skip_train:
            all_metrics["train"] = render_set_and_eval(
                dataset.model_path, "train", scene.loaded_iter,
                scene.getTrainCameras(), gaussians, pipeline, background, lpips_fn)
        if not skip_test:
            all_metrics["test"] = render_set_and_eval(
                dataset.model_path, "test", scene.loaded_iter,
                scene.getTestCameras(), gaussians, pipeline, background, lpips_fn)

        with open(os.path.join(dataset.model_path, "graph_eval_results.json"), 'w') as fp:
            json.dump({"K": k, "iteration": scene.loaded_iter, "metrics": all_metrics}, fp, indent=True)
        print(f"\nWrote results to {os.path.join(dataset.model_path, 'graph_eval_results.json')}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Round-trip a Gaussian splat .ply through GaussianSplatGraph and evaluate.")
    model_params    = ModelParams(parser, sentinel=True)
    pipeline_params = PipelineParams(parser)

    parser.add_argument("--iteration", default=-1, type=int,
                        help="Iteration to load from {model_path}/point_cloud/iteration_<N>/. -1 = latest.")
    parser.add_argument("--k", default=10, type=int, help="KNN neighbors for the GaussianSplatGraph.")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test",  action="store_true")
    parser.add_argument("--quiet",      action="store_true")
    parser.add_argument("--gpu",        type=str, default='-1')

    # Prune-sweep mode: walks prune percentages and reports SSIM/PSNR/LPIPS.
    parser.add_argument("--prune-sweep", action="store_true",
                        help="Walk prune percentages and score each. Reuses one Fiedler vector across all cuts.")
    parser.add_argument("--prune-min",  type=int, default=5)
    parser.add_argument("--prune-max",  type=int, default=95)
    parser.add_argument("--prune-step", type=int, default=5)
    parser.add_argument("--symmetric",  choices=["union", "mutual", "none"], default="union",
                        help="Adjacency symmetrization mode for the Laplacian.")
    parser.add_argument("--normalized-laplacian", action="store_true",
                        help="Use the symmetric normalized Laplacian for the Fiedler vector.")
    parser.add_argument("--random-baseline", action="store_true",
                        help="In addition to the Fiedler prune, run a same-percentage uniformly-random prune for direct comparison.")
    parser.add_argument("--random-seed", type=int, default=0,
                        help="RNG seed for the random-baseline prune.")

    args = get_combined_args(parser)
    print("graph_eval on " + args.model_path)

    model_params_ext    = model_params.extract(args)
    pipeline_params_ext = pipeline_params.extract(args)

    if args.gpu != '-1':
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        os.system("echo $CUDA_VISIBLE_DEVICES")

    safe_state(args.quiet)

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    lpips_fn = lpips.LPIPS(net='vgg').to(device)

    if args.prune_sweep:
        prune_sweep(model_params_ext, args.iteration, pipeline_params_ext,
                    k_knn=args.k, symmetric=args.symmetric,
                    normalized_laplacian=args.normalized_laplacian,
                    k_min=args.prune_min, k_max=args.prune_max, k_step=args.prune_step,
                    skip_train=args.skip_train, skip_test=args.skip_test,
                    random_baseline=args.random_baseline,
                    random_seed=args.random_seed,
                    lpips_fn=lpips_fn)
    else:
        graph_eval(model_params_ext, args.iteration, pipeline_params_ext,
                   args.k, args.skip_train, args.skip_test, lpips_fn)
