"""Apply the two explicitly approved frozen review spans, without network relabeling."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from histcontour_core.provenance import sha256_file
from histcontour_core.human_feedback import geometry_digest, validate_pixel_points

BASE = ROOT / 'data/derived/contour-reconstruction-2026-09-14'
CARD_HASH = '51d2429d0a5494b15353fa236124824110b7fc12c5ba879a88c4cc2d87895912'
APPROVED = {'F001': 'R015-weak-032', 'F002': 'R018-weak-017'}
QUOTE = '등고선이고 잘 연결했어. 지금 집 밖에서 원격으로 하는거라 목표 적당히 잡고 일을 마치도록 해'


def write_new(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def run():
    cards_path = BASE / 'short-human-review-001/questions.json'
    if sha256_file(cards_path) != CARD_HASH:
        raise ValueError('Frozen review cards changed')
    packet = ROOT / 'data/derived/contour-assisted-2026-09-13/human-review-final'
    cards = json.loads(cards_path.read_text())
    drawing_path = packet / 'drawing-report.json'
    if sha256_file(drawing_path) != cards['input_sha256'][str(drawing_path)]:
        raise ValueError('Drawing metadata changed')
    tiles = {t['tile_id']: t for t in json.loads(drawing_path.read_text())['tiles']}
    features = []
    for card in cards['cards']:
        if APPROVED.get(card['review_id']) != card['candidate_id']:
            raise ValueError('Unexpected review target')
        tile = tiles[card['tile_id']]
        source = (packet / tile['raster_path']).resolve()
        if packet not in source.parents or sha256_file(source) != card['source_raster_sha256']:
            raise ValueError('Source raster changed')
        if geometry_digest(card['pixel_geometry']) != card['pixel_geometry_sha256']:
            raise ValueError('Reviewed geometry changed')
        points = card['pixel_geometry']['coordinates']
        validate_pixel_points(points, tile)
        west, south, east, north = tile['bounds']
        width, height = tile['pixel_bounds'][2:]
        coords = [[west+(x+.5)*(east-west)/width, north-(y+.5)*(north-south)/height] for x,y in points]
        features.append({'type': 'Feature', 'geometry': {'type': 'LineString', 'coordinates': coords},
            'properties': {**{k: card[k] for k in ('review_id', 'candidate_id', 'tile_id', 'source_path_ids', 'pixel_geometry_sha256', 'source_raster_sha256')},
                'human_approved': True, 'semantic_class': 'contour', 'geometry_decision': 'accept_as_drawn',
                'approval_scope': 'exact_reviewed_span', 'whole_network_approved': False,
                'training_eligible': False}})
    if {f['properties']['review_id'] for f in features} != set(APPROVED) or len(features) != 2:
        raise ValueError('Approval set mismatch')
    output = BASE / 'pipeline-run-v3/integrated'
    write_new(output / 'approved-f001-f002.geojson', {'type': 'FeatureCollection',
        'crs': {'type': 'name', 'properties': {'name': 'EPSG:5132'}}, 'features': features})
    write_new(output / 'final-human-feedback.json', {'schema': 'jap-map-scoped-review-finish/1',
        'source_cards_sha256': CARD_HASH, 'user_quote': QUOTE, 'approved_review_ids': list(APPROVED),
        'semantic_approvals': 2, 'geometry_approvals': 2, 'source_cards_modified': False,
        'whole_network_approved': False, 'automatic_training': False,
        'scope': 'Record approvals, verify existing result, finish without further review requests.'})
    print('Recorded two exact local contour approvals; frozen inputs unchanged.')


if __name__ == '__main__':
    run()
