# 地标视角与均匀配对

`generate_pairs.py` 从 first-key 地标生成相机和 KML，使用整数约束选出 **1000 个互不重复的无序图片组合**。每个组合只保留一个方向，JSON 中 source/target 的文件名、相机位姿和差分一致。

默认从 70 个视角开始，按 10 个递增，最多 100 个；仅在完整满足约束后写入结果。求解超时不等于数学上不可行，会尝试更多视角；最终失败会保留原有结果。当前 `0000` / `step3-v4-closeup` 使用 90 张。

## 验收指标

- ΔX、ΔY、ΔZ：每维范围 `[-40, 40]` 米。
- ΔTilt：`[-25, 25]` 度；ΔHeading：`[-180, 180)` 度，采用最短有符号角差。
- 每维 24 个固定等宽区间，每区间 41 或 42 对；同时约束 12 档为 83–84 对、6 档为 166–167 对。
- 任意两维组成的 4×4 网格，每格约束 40–85 对（理想值 62.5）。这是有容差的二维覆盖，不是宣称连续五维联合分布完全均匀。
- 1000 对不包含重复组合，也不将 A→B、B→A 同时计数。每张图片必须被使用。

## 入镜与相机参数

默认使用以 first-key Point 为中心、半径 **60 米** 的球体作为地标包围区域。每个相机必须让整个球体处于视锥内，并留出 3% NDC 边界。可用 `--landmark-radius` 调整真实建筑的包围区域；该检查保证视野覆盖，不检测地形或其他建筑的遮挡。

默认水平视场角 70°、1024×1024 图片、平均相机高度 180 米（地面以上），XYZ 各在中心 ±30 米内采样。相机高度硬上限为 210 米，绝对 tilt 为 0–25°。相机位置和取景方向分开生成，再剔除无法看到地标区域的视角，避免旧版任意旋转后丢失地标。近景尺度约束以 512×512 为参考：地标半径 60 米的水平区域宽、高都必须达到 200 像素；当前相机集合实测最小约 221 像素。输出为 1024×1024 时，实际最小约 443 像素；增加像素数不会改变视野、机位或地标在画面中的占比。求解器不能通过升高相机或扩大视野来放松该约束。相比上一版，保留近景的代价是 ΔTilt 从 ±40° 收窄到 ±25°；分布图和验证器读取 manifest 中的真实范围，不把窄范围宣称成原范围均匀。

KML `LookAt` 指向地面（altitude=0），`range = 相机高度 / cos(tilt)`，并写入 `gx:horizFov`。JSON 的 XY 使用地标位置为原点的局部东/北坐标，Z 相对当地地面；这是局部平面近似，不包含地形高程模型。外部 KML 客户端若不采用 `gx:horizFov`，需要设为相同的 70° 视野才能沿用入镜保证。

当前 step3 渲染器使用卫星瓦片的地面投影，不是真正的 Google Earth 三维建筑渲染。它会按各帧的实际经纬度和高度取景，本流程固定使用 zoom 20 瓦片。如果覆盖范围过大，会明确失败，不能自动降低瓦片分辨率，也不会通过重复边缘像素填充视野。

## 一条命令复现当前结果

```bash
bash /root/construct_dataset/step3/reproduce.sh
```

脚本明确固定当前接受的参数，并依次运行：KML/1000 组 pair 生成、1024×1024 图片渲染、分布可视化、结果校验和回归测试。默认输出到 `/root/reproduced_step3/`；其中目录结构与原结果相同，原结果不被覆盖。也可指定输出根目录：

```bash
bash /root/construct_dataset/step3/reproduce.sh /root/my_reproduction
```

使用当前 `/venv/main/bin/python`。可通过 `PYTHON_BIN` 指定其他环境，通过 `TILE_CACHE_DIR` 指定瓦片缓存根目录（其中应有 `original/` 子目录）。首次运行或缓存缺失时需要网络下载瓦片。

同环境复现应保留：

- 输入地标 `/root/first-key/0000.kml`。
- 当前代码和固定参数（种子 42；70 张起试，上限 100；高度 180±30 米；视场角 70°；tilt 0–25°）。
- 瓦片缓存 `/root/.cache/construct_dataset/google_tiles/original/`。
- Python 3.12.14 及 `requirements-reproduce.txt` 中记录的库版本。

卫星瓦片是外部数据，未来重新下载可能发生变化；固定种子不能固定远程影像。整数规划求解器跨版本、平台或运行超时也可能选择不同但满足约束的 pair，因此不承诺换环境后逐字节一致。当前环境中的复现核对结果记录在 `/root/reproduced_step3/reproduction_check.json`。`environment.json` 记录本次运行的 Python 和库版本。

## 运行

在 `/root` 下执行；Python 环境需要 numpy、scipy（含 `optimize.milp`）、Pillow、matplotlib。若缺依赖：

```bash
uv pip install --python /venv/main/bin/python -r /root/construct_dataset/step3/requirements-reproduce.txt
/venv/main/bin/python -m construct_dataset.step3.generate_pairs
/venv/main/bin/python -m construct_dataset.step3.render_images
/venv/main/bin/python -m construct_dataset.step3.visualize_pairs
/venv/main/bin/python -m construct_dataset.step3.validate_pairs
/venv/main/bin/python -m unittest construct_dataset.step3.test_geometry -v
```

`--candidate-count` 指定起始图片数，`--max-candidate-count` 设置上限（不能超过 100），`--solver-seconds` 设置每个候选规模的求解时间。`--seed` 控制可复现随机采样；`--max-camera-height` 和 `--min-landmark-pixels` 控制近景尺度，`--max-tilt` 明确角度范围。

旧数据比较（仅在各维范围相同时适合直接比较分箱计数）：

```bash
/venv/main/bin/python -m construct_dataset.step3.visualize_pairs \
  --reference-pairs /root/step3_backups/pre_uniform_v3/pairs_0000/pairs.jsonl
```

输出位于 `/root/processed_paired_json/0000/`，图片位于 `/root/processed_images/0000/`。`manifest.json` 记录约束、分箱计数、每个视角的可见性与候选规模尝试；`visualizations/validation_report.json` 记录实际文件审计结果。修改前的代码、KML、pair、图片及分布图保存在 `/root/step3_backups/pre_uniform_v3/`。

上一版远景结果及代码另存于 `/root/step3_backups/pre_closeup_v4/`，近景修改的实验与日志也保存在 `/root/step3_backups/`。

## 渲染清晰度

默认直接从 zoom 20 的原始卫星瓦片渲染为 2048×2048，再使用 Lanczos 缩采样到 1024×1024，减少投影采样的锯齿与混叠。JPEG 使用质量 98 和 4:4:4 色度，不再对缓存瓦片做额外 JPEG 重压缩；原始响应字节放在瓦片缓存的 `original/` 子目录。已有旧缓存保留，但新渲染不会使用经过重压缩的旧条目。

这是从瓦片重新采样得到的更高分辨率，原始卫星影像仍限制真实细节。未修改机位、KML、配对组合或差值分布。`manifest.json` 中的 `image_size`、`render_quality` 记录输出规格；验证器按实际尺寸检查图片和入镜位置。

```bash
/venv/main/bin/python -m construct_dataset.step3.render_images --image-size 1024 --supersample 2 --jpeg-quality 98
```

若下游强制要求 512×512，可显式传入 `--image-size 512`。提升清晰度前的 512 图片和代码保存在 `/root/step3_backups/pre_quality_v5/`。

## 批量处理 tour 0001–0100

```bash
cd /root
/venv/main/bin/python -m construct_dataset.step3.batch_tours --start 1 --end 100 --workers 12
```

批处理仍逐个调用上述生成、渲染、分布可视化和验证脚本，输出分别写入 `processed_paired_json/<tour_id>/` 和 `processed_images/<tour_id>/`。每个 tour 保留 1000 对、最多 100 张 1024×1024 图片；所有近景与均匀性约束不变。原 `0000` 不参与此次范围。

运行状态和逐 tour 日志位于 `/root/batch_runs/tours_0001_0100/`：

- `summary.json`：完成、运行、失败数量和全部 tour 状态。
- `logs/<tour_id>.log`：实际运行命令与详细日志。
- `<tour_id>.json`：该 tour 的阶段、随机种子和验收结果。

同一命令可以续跑：成功生成的数据会复用，已完整完成的 tour 会跳过；下载失败可重试。种子按 42–57 顺序尝试，实际采用值记录在各自的 `manifest.json` 中。

生成器新增 `--sampling-mode auto|uniform|conditional`。默认先用原普通采样，在 100 张以内仍无解时，尝试依据视锥侧面约束直接采样可见高度边界，补足高地标造成的 Z 差值尾部不足。条件采样不改变高度上限、地标范围、视场角、最小占图比例或任何配对均匀性约束。`manifest.json` 的 `sampling_mode` 记录实际方法；早期没有这个字段的结果使用 `uniform`。

此实例的长任务由 supervisor 管理：

```bash
supervisorctl status tour_batch_0001_0100
```

瓦片缓存使用临时文件加原子替换，以支持多个相邻 tour 同时下载同一瓦片。

批次完成后可生成独立汇总审计、CSV 清单、总体差值分布图和逐 tour 预览：

```bash
/venv/main/bin/python -m construct_dataset.step3.summarize_batch --start 1 --end 100
```

输出仍放在 `/root/batch_runs/tours_0001_0100/`，包括 `final_validation.json`、`tour_index.csv`、`aggregate_distributions.png` 和 `overview_01.png` 至 `overview_04.png`。
