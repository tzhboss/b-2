"""Append source corroboration and cached diagnostic; do not promote unknowns."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
path = ROOT / 'representative-history-evidence.json'
e = json.loads(path.read_text(encoding='utf8'))
labels_before = {k: (j['gold_truth'], j['final_event_truth']) for k, j in e['judgments'].items()}
e['sources'].update({
    'R15_LANG_CABLE_MIRROR': {
        'url': 'https://wikileaks.jcvignoli.com/cable_09PARIS1558',
        'read_status': 'full_mirrored_attributed_cable_body_read_prior_round15_receipt_h79r15s9; current_reopen_timeout',
        'document_id': '09PARIS1558', 'document_date': '2009-11-24',
        'attributed_origin': 'US Embassy Paris; mirror not verified official release',
        'identified_visit_interval': ['2009-11-05', '2009-11-09'],
        'evidence_summary': 'Paragraph1 dates Jack Lang Beijing visit Nov5-9; paragraph7 names Dai Binguo, Wu Dawei and Liu Hongcai meetings. Corroborates Lang visit, cannot exclude another person represented by generic Envoy(France) visiting The Hague onNov5. Exact internal AP Story10260211 not linked.',
    },
    'R15_LANG_AFP_PHOTO_CAPTION': {
        'url': 'https://www.21news.be/jack-lang-largent-comme-systeme-privileges-impayes-et-chute-dun-personnage/',
        'read_status': 'actual_visible_AFP_photo_caption_read; premium_article_body_partial',
        'publication_date': '2026-02-11', 'caption_event_date': '2009-11-05',
        'evidence_summary': 'Caption attributed AFP/Frederic J.Brown places Jack Lang interview in Beijing Nov5,2009, as Sarkozy envoy for North Korea. Only caption used; not the unrelated2026 article narrative. No complete itinerary or generic-envoy exclusivity.',
    },
    'R15_PORTUGAL_ANGOLA_NAVAL_PROPOSAL': {
        'url': 'https://www.rtp.pt/noticias/pais/portugal-disponivel-para-melhorar-marinha-angolana_n28551',
        'read_status': 'actual_RTP_LUSA_body_read',
        'body_exact_date_visible': False,
        'evidence_summary': 'Luís Amado in Luanda proposes naval equipment/training cooperation during Sócrates visit, on frigate CorteReal. Search previews associate April6,2006, but web body extract lacks date. Does not establish sampledJune20 bilateral cooperation, and ongoing relationship does not prove a new daily event.',
    },
    'R15_CAIN_DURKAN_AHERN_PREVIOUS_DAY': {
        'url': 'https://cain.ulster.ac.uk/heathwood/static/2006.html',
        'read_status': 'actual_UlsterUniversity_archive_catalog_description_read; original_video_not_viewed',
        'catalog_record': '8971', 'broadcast_date': '2006-03-02',
        'evidence_summary': 'March2UTVNews catalog describes Durkan meeting Ahern in Dublin. Targeted cached gold32028isMarch3 and IrishTimesStory7376349sentence6. Related prior-daymeeting not independent exactMarch3gold support or proof there was noMarch3phone call.',
    },
    'R15_ABC_CHECHEN_OPERATION_REVIEW': {
        'url': 'https://www.abc.net.au/news/2009-04-16/chechen-war-over-russia/1653598',
        'read_status': 'actual_original_publisher_article_body_read',
        'publication_date': '2009-04-16',
        'evidence_summary': 'ScottBevan report on ending decade-long operation recalls1990s wars and1999deployment; no specificApril16rebel attack event supplied. Cached gold58659isArmedRebelRussiaUseUnconventionalViolenceMilitaryRussia2009-04-16, ABCStory8604858sentence4. Internal story linkage unproven; historical recall cannot be assumed same-day violence. Does not certify or disprove exact gold.',
    },
})
for sample_id, ids, note in [
    ('S06', ['R15_LANG_CABLE_MIRROR', 'R15_LANG_AFP_PHOTO_CAPTION'],
     'Additional attributed cable and AFPcaption corroborate JackLang Beijing presenceNov5. GenericEnvoyFrance could represent different persons; Haguecrossing remainsunknown, and gold4to5 whole-rank justification remainsunknown.'),
    ('S11', ['R15_PORTUGAL_ANGOLA_NAVAL_PROPOSAL'],
     'ActualRTP/LUSAbody documents related naval proposal, noJune20date inbody; searchAprildate cannot supplyJune20truth. Portugalcrossing remainsunknown; gold independently supported as before.'),
]:
    j = e['judgments'][sample_id]
    j['evidence_ids'] = list(dict.fromkeys(j['evidence_ids'] + ids))
    j['round15_source_note'] = note
e['round15_cached_population_artifact'] = 'cached-temporal-conflicts-round15.json'
e['round15_targeted_gold_provenance_artifact'] = 'targeted-conflict-gold-provenance-round15.json'
assert labels_before == {k: (j['gold_truth'], j['final_event_truth']) for k, j in e['judgments'].items()}
path.write_text(json.dumps(e, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
print(json.dumps({'sample_n': 40, 'sample_replaced': False, 'truth_labels_changed': 0,
                  'gold_supported_n': 9, 'final_top1_supported_n': 2,
                  'world_reasonable_fraction': None, 'world_error_fraction': None,
                  'training': False, 'neural_rescoring': False}))
