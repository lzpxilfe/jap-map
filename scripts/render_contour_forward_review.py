#!/usr/bin/env python3
"""Render frozen forward-test outputs and all source samples for inspection."""

import argparse
import html
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from histcontour_core.provenance import sha256_file
from scripts.compare_contour_vector_outputs import overlay, pixel_proposals
from scripts.generate_ink_centerline_candidates import proposal_mask
from scripts.run_contour_context_experiment import read, resolve, write


def render(index_path, evaluation_path, samples_path, labels_path, output, *, illustration_ids=()):
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    index, evaluation, samples, labels = (read(path) for path in (index_path, evaluation_path, samples_path, labels_path))
    if (evaluation.get("schema") != "jap-map-contour-forward-evaluation/1" or evaluation["source_index_sha256"] != sha256_file(index_path)
            or evaluation["forward_manifest_sha256"] != sha256_file(samples_path) or evaluation["forward_labels_sha256"] != sha256_file(labels_path)):
        raise ValueError("forward display input fingerprints differ")
    tiles = {row["tile_id"]: row for row in index["tiles"] if row["split"] == "development"}
    entries = {row["tile_id"]: row for row in evaluation["tiles"]}
    if len(tiles) != 9 or set(tiles) != set(entries) or any(row["sheet_id"] == "178-gongju" for row in tiles.values()):
        raise ValueError("forward display is development-only")
    by_sid = {row["sample_id"]: row for row in samples["samples"]}
    predictions = {row["sample_id"]: row for row in evaluation["samples"]}
    label_map = {row["sample_id"]: row for row in labels["items"]}
    if set(by_sid) != set(predictions) or set(by_sid) != set(label_map) or any(sid not in by_sid for sid in illustration_ids):
        raise ValueError("forward display sample IDs differ")
    output.mkdir(parents=True, exist_ok=False)
    (output/"samples").mkdir()
    tile_blocks, sample_blocks, illustrations = [], [], {}
    for tile_id, tile in tiles.items():
        raster = resolve(tile["raster_path"])
        if sha256_file(raster) != entries[tile_id]["source_raster_sha256"]:
            raise ValueError("display source raster changed")
        with Image.open(raster) as image:
            gray = np.asarray(image.convert("L")).copy()
        masks = {}
        source_name = f"{tile_id}-source.png"
        Image.fromarray(gray).save(output/source_name)
        figures = [f'<figure><figcaption>동일 입력 원본</figcaption><img src="{source_name}" loading="lazy"></figure>']
        for name, label in (("legacy", "기존 0.1 필터"), ("candidate", "고정한 새 보수 후보")):
            entry = entries[tile_id]["methods"][name]
            vector = resolve(entry["path"])
            if sha256_file(vector) != entry["sha256"]:
                raise ValueError("display vector changed")
            masks[name] = proposal_mask(pixel_proposals(tile, read(vector)), gray.shape)
            filename = f"{tile_id}-{name}.png"
            overlay(gray, masks[name]).save(output/filename)
            figures.append(f'<figure><figcaption>{label} · {entry["count"]}선</figcaption><img src="{filename}" loading="lazy"></figure>')
        tile_blocks.append(f'<h2>{html.escape(tile_id)}</h2><div class="pair">{"".join(figures)}</div>')
        for sid, sample in by_sid.items():
            if sample["tile_id"] != tile_id:
                continue
            x1, y1, x2, y2 = sample["pixel_box"]
            source = Image.fromarray(gray[y1:y2, x1:x2]).convert("RGB")
            ImageDraw.Draw(source).line([(x-x1, y-y1) for x, y in sample["pixel_points"]], fill=(0, 100, 235), width=2)
            filename = f"samples/{sid}-source.png"
            source.save(output/filename)
            images = [source]
            panels = [f'<figure><figcaption>원본 · 파랑은 대상</figcaption><img src="{filename}" loading="lazy"></figure>']
            prediction = predictions[sid]
            for name, label in (("legacy", "기존"), ("candidate", "새 후보")):
                preview = overlay(gray[y1:y2, x1:x2], masks[name][y1:y2, x1:x2])
                filename = f"samples/{sid}-{name}.png"
                preview.save(output/filename)
                images.append(preview)
                panels.append(f'<figure><figcaption>{label} · {"유지" if prediction[name+"_retained"] else "제외"}</figcaption><img src="{filename}" loading="lazy"></figure>')
            sample_blocks.append(f'<details id="{sid}"><summary>{sid} · AI {html.escape(label_map[sid]["class"])} · 기존 {int(prediction["legacy_retained"])} / 새 후보 {int(prediction["candidate_retained"])}</summary><p>{html.escape(label_map[sid]["note"])}</p><div class="pair sample">{"".join(panels)}</div></details>')
            if sid in illustration_ids:
                panel = Image.new("RGB", (990, 320), "white")
                draw = ImageDraw.Draw(panel)
                font = ImageFont.load_default(size=17)
                captions = (f"{sid} Source; AI {label_map[sid]['class']}", f"Legacy: {'kept' if prediction['legacy_retained'] else 'excluded'}", f"New: {'kept' if prediction['candidate_retained'] else 'excluded'}")
                for column, (image, caption) in enumerate(zip(images, captions)):
                    draw.text((column*330+8, 7), caption, fill="black", font=font)
                    factor = min(310/image.width, 275/image.height)
                    enlarged = image.resize((round(image.width*factor), round(image.height*factor)), Image.Resampling.NEAREST)
                    panel.paste(enlarged, (column*330+(330-enlarged.width)//2, 38))
                illustrations[sid] = panel
    if illustration_ids:
        board = Image.new("RGB", (990, 320*len(illustration_ids)), "white")
        for number, sid in enumerate(illustration_ids):
            board.paste(illustrations[sid], (0, number*320))
        board.save(output/"forward-examples.png")
    comparison = evaluation["comparison"]
    table = '<table><tr><th>AI 새 표본 — 정식 정확도 아님</th><th>등고선 유지</th><th>비등고선 제외</th><th>도로·하천 제외</th></tr>'
    for name, label in (("legacy", "기존"), ("candidate", "새 보수 후보")):
        values = comparison[name]
        road = values["by_class"]["road_river"]
        table += f'<tr><td>{label}</td><td>{values["contours_retained"]}/{values["by_class"]["contour"]["total"]}</td><td>{values["noncontours_rejected"]}/{values["noncontours_rejected"]+values["noncontours_retained"]}</td><td>{road["total"]-road["retained"]}/{road["total"]}</td></tr>'
    table += '</table>'
    document = ('<!doctype html><html lang="ko"><meta charset="utf-8"><meta http-equiv="Content-Security-Policy" content="default-src \'none\'; img-src \'self\'; style-src \'unsafe-inline\'">'
                '<title>고정 모델의 새 선분 평가</title><style>body{font:16px system-ui;margin:24px}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #aaa}.pair{display:flex;gap:12px}figure{margin:0;flex:1;min-width:0}img{width:100%;height:auto}.sample img{object-fit:contain;max-height:320px;image-rendering:pixelated}figcaption{padding:8px}h2{margin-top:32px}details{padding:10px;border-bottom:1px solid #ccc}</style>'
                '<h1>고정한 모델 · 새로운 162개 선분</h1><p>이 화면은 AI 판독·모델 비교용입니다. 사람 검수는 별도 source-first-review.html을 먼저 보고 진행하세요.</p>'
                '<p>기존 표본과 선 ID가 겹치지 않지만 같은 세 도엽·인접 이미지 문맥입니다. 모델·임계값을 고정한 뒤 새 AI 라벨을 판독했고, 이 평가 결과로 다시 조정하지 않았습니다. 모호한 21개는 수치에서 제외합니다.</p>'
                +table+'<p>처음 검출되지 않은 선이나 전체 잘못된 연결을 모두 검증한 것은 아닙니다. 사람 승인과 기본값 승격은 없습니다. 모든 원래 도형은 보존됩니다.</p>'
                +''.join(sample_blocks)+''.join(tile_blocks)+'</html>')
    with (output/"report.html").open("x", encoding="utf-8") as handle:
        handle.write(document)
    write(output/"display-provenance.json", {"evaluation_sha256": sha256_file(evaluation_path), "sample_manifest_sha256": sha256_file(samples_path),
                                             "illustration_ids": list(illustration_ids), "illustration_scope": "post-experiment explanation only, not evaluation selection", "all_samples_displayed": len(samples["samples"])})
    print(output/"report.html", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index", type=Path)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--illustration-ids", nargs="*", default=[])
    args = parser.parse_args()
    render(args.index, args.evaluation, args.samples, args.labels, args.output, illustration_ids=args.illustration_ids)


if __name__ == "__main__":
    main()
