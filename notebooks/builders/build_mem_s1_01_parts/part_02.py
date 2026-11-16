# Cells of build_mem_s1_01.py, part 2 of 2: run in order by that builder, with md() and code() in scope.

code(r"""
def footprint(frame: Frame, centre: np.ndarray, radius: float = 0.2) -> np.ndarray:
    angles = np.linspace(0, 2 * np.pi, 40)
    circle = centre + radius * np.stack([np.cos(angles), np.sin(angles), np.zeros_like(angles)], axis=1)
    uv, _, inside = project(circle, frame)
    return uv if inside.all() else uv[inside]


examples = []
for action in ACTIONS:  # one example per action type, the part seen with most pixels
    rows = main[(main["动作"] == action) & main["有深度"]].sort_values("像素", ascending=False)
    if len(rows):
        examples.append(rows.iloc[0])
examples = examples[:8]
fig, axes = plt.subplots(2, len(examples), figsize=(2.6 * len(examples), 7.2))
for column, row in enumerate(examples):
    i = keyframes.index(row["关键帧"])
    frame = input_frame(i, sources[MONO][0][i])
    m = motions.get(elements[row.part].annot_id)
    for r, zoom in enumerate((False, True)):
        axis = axes[r, column]
        axis.imshow(frame.rgb)
        u, v = row["uv"]
        start, normal_end = projected_axis(row["contact"], row["normal"], frame, 0.12)
        axis.annotate("", xy=normal_end, xytext=start, arrowprops=dict(arrowstyle="->", color="cyan", lw=1.8))
        if m is not None:
            _, motion_end = projected_axis(row["contact"], np.array(m["motion_dir"]), frame, 0.12)
            axis.annotate("", xy=motion_end, xytext=start, arrowprops=dict(arrowstyle="->", color="yellow", lw=1.8))
        axis.plot(u, v, "o", color="red", ms=5)
        ring = footprint(frame, row["stand"])
        if len(ring):
            axis.plot(ring[:, 0], ring[:, 1], color="lime" if row["free_gt"] else "red", lw=1.5)
        if zoom:
            axis.set_xlim(u - 80, u + 80)
            axis.set_ylim(v + 80, v - 80)
        axis.axis("off")
    axes[0, column].set_title(f"{row['动作']}\n{row['任务'][:28]}", fontsize=8)
fig.suptitle(f"红点：接触点　青：法向　黄：运动轴（oracle）　圆圈：站立点（绿=扫描中空闲）　深度：{MONO}，位姿：ARKit", fontsize=9)
plt.tight_layout()
plt.show()
""")

code(r"""
band = scan_occupied[(scan_occupied[:, 2] > gt_floor + 0.15) & (scan_occupied[:, 2] < gt_floor + 1.8)]
fig, axis = plt.subplots(figsize=(9, 9))
axis.scatter(band[:, 0], band[:, 1], s=1, c="lightgrey")
ok = main[main["有深度"]]
for _, row in ok.iterrows():
    colour = colour_of[row["动作"]]
    axis.plot(*row["contact"][:2], "o", color=colour, ms=6, mec="black", mew=0.5)
    axis.plot(*row["stand"][:2], "s", color="green" if row["free_gt"] else "red", ms=5, alpha=0.8)
    axis.annotate("", xy=row["contact"][:2], xytext=row["stand"][:2],
                  arrowprops=dict(arrowstyle="->", color="black", lw=0.6, alpha=0.6))
axis.plot(colmap_path[:, 0], colmap_path[:, 1], color="royalblue", lw=0.8, alpha=0.6)
handles = [plt.Line2D([], [], marker="o", ls="", color=colour_of[a], label=a) for a in ACTIONS]
handles += [plt.Line2D([], [], marker="s", ls="", color=c, label=l) for c, l in (("green", "站立点：扫描中空闲"), ("red", "站立点：与扫描冲突"))]
axis.legend(handles=handles, fontsize=7, loc="upper right")
axis.set_aspect("equal")
axis.set_title(f"俯视图（扫描坐标系，评测视图）：接触点、站立点和接近方向；灰色为 0.15–1.8 m 高度的扫描体素")
axis.set_xticks([])
axis.set_yticks([])
plt.show()
""")

code(r"""
def as_list(value):
    return np.asarray(value, dtype=float).round(4).tolist()


def motion_fields(annot_id: str) -> dict:
    m = motions.get(annot_id)
    if m is None:
        return {"motion_type": None}
    return {"motion_type": m["motion_type"], "motion_dir_faro": as_list(m["motion_dir"]),
            "motion_origin_faro": as_list(scan_xyz[m["motion_origin_idx"]])}


records = [{"annot_id": elements[r.part].annot_id, "action": r["动作"], "task": r["任务"], "keyframe": r["关键帧"],
            "pixel": list(r["uv"]), "pixels": int(r["像素"]), "contact_faro": as_list(r["contact"]), "normal_faro": as_list(r["normal"]),
            "stand_faro": as_list(r["stand"]), "facing_faro": as_list(r["facing"]), "stand_free_in_scan": bool(r["free_gt"]),
            "stand_moved": bool(r["moved"]), "contact_error_cm": round(float(r["接触点误差 cm"]), 2),
            **motion_fields(elements[r.part].annot_id)}
           for _, r in main[main["有深度"]].iterrows()]
# The chosen input depth and the input cameras of every keyframe, for S1-02 (humanoid display).
inputs_dir = OUT / "inputs"
inputs_dir.mkdir(exist_ok=True)
for i, t in enumerate(keyframes):
    frame = input_frame(i, sources[MONO][0][i])
    np.savez_compressed(inputs_dir / f"{t}.npz", depth=frame.depth_m.astype(np.float16), intrinsics=frame.intrinsics,
                        cam_to_world=frame.cam_to_world)
manifest = {"visit": VISIT, "video": VIDEO, "keyframes": keyframes, "pixel_step": PIXEL_STEP, "mono_model": MONO,
            "mono_model_id": MONO_MODELS[MONO.split(" × ")[0]], "lidar_scaled": " × " in MONO, "depth_table": depth_table.reset_index().to_dict("records"),
            "pose_error": {"centre_cm_median": float(pose_err.centre_cm.median()),
                           "rotation_deg_median": float(pose_err.rotation_deg.median())},
            "actionable_summary": actionable_summary.reset_index().to_dict("records"),
            "floor_input": float(results[(MONO, "ARKit")].attrs["floor"]), "up_input": as_list(UP_INPUT),
            "gt_floor": float(gt_floor),
            "note": "oracle labels and motion axes; poses expressed in the Faro frame for evaluation only"}
(OUT / "actionable_oracle.json").write_text(json.dumps(records, indent=1, ensure_ascii=False))
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False, default=float))
print("saved:", *(str((OUT / n).relative_to(ROOT)) for n in ("actionable_oracle.json", "manifest.json")), sep="\n  ")
""")

md(r"""
### 读结果的方式

- **Step 1b**：部件处的深度误差比全图更重要；单目模型的选择只在这个开发场景上做过。
- **Step 2**：同一种深度换成 COLMAP 位姿后误差下降的部分，就是 ARKit 位姿的代价；换成真值深度后下降的部分，是深度的代价。
- **Step 4**：oracle 版的误差全部来自输入深度和位姿，因此是第 7 步（模型预测）的上限。站立点在扫描中空闲的比例说明只用输入点做的空闲检查是否可信。
- 这里的数字只针对一个场景、一个视频，用来看链路和误差来源，不是方法结论。

### 本次运行记录（2026-11-10，40 个关键帧，720×960）

以下数字只针对这次执行；改动参数后应重新核对。

- **输入约定：** LiDAR 和 ARKit 相机需顺时针转 1 次才正，这只由 ARKit 位姿推算；转正后直接放大即可对上 hires 像素（中位相对误差 1.3%）。
- **深度：**
  - LiDAR 全图中位误差 2.4 cm，部件处 5.2 cm，覆盖 88% 的像素。
  - Depth Anything V2 的绝对尺度偏远约 33%，中位误差约 50 cm；逐帧乘 LiDAR 尺度后，Large 降到 4.7 cm，但部件处仍有 11.2 cm。
  - Depth Pro 在 M1 GPU 上的输出不可信，未纳入比较。
- **位姿：** ARKit 相机中心相对 COLMAP 的误差中位 7.2 cm，p90 15.8 cm，个别帧达 89 cm；朝向中位 1.2°。
- **深度 × 位姿分解（到扫描的中位距离）：**
  - 真值深度 + COLMAP 0.4 cm，换成 ARKit 位姿后 3.7 cm；
  - LiDAR + ARKit 3.3 cm，比"真值深度 + ARKit"还好，因为 LiDAR 和 ARKit 位姿来自同一台设备，彼此一致。
- **标签：** 38 个部件中 33 个至少在一个关键帧中可见；最佳关键帧里的像素数中位为 57。两个 pinch_pull 只有 2–5 个像素，没有生成条目。
- **可执行 affordance（29 个部件）：**
  - 接触点中位误差：LiDAR + ARKit 2.8 cm（54% 落在部件上）；单目 × LiDAR 尺度 + ARKit 7.7 cm（14%）；真值深度 + COLMAP 0.4 cm（97%）。
  - 最后一项说明链路本身正确，误差来自输入深度和 ARKit 位姿。
  - 支撑面法向的中位误差为 13–24°。
  - 站立点在扫描中真正空闲的比例为 75–97%。
- **对第 7 步的提示：**
  - 单目深度在小部件处的误差（约 11 cm）是主要瓶颈。
  - 只用输入点做空闲检查时，没观察到的空间被当成了空闲，违反"未知不等于空闲"，后续需要改。

**下一步：** [S1-02](S1_02_humanoid_approach.ipynb) 按 `runs/mem_s1/actionable_oracle.json` 在关键帧上贴人形（站立 + 伸手 + 运动箭头），并在扫描里检查。
""")
