"""Persist round9 primary-source readings; separate target errors from rank errors."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
path = ROOT / 'representative-history-evidence.json'
e = json.loads(path.read_text(encoding='utf-8'))
e['sources'].update({
    'R9_OBAMA_OFFICIAL_TRAVEL': {
        'url': 'https://history.state.gov/departmenthistory/travels/president/obama-barack',
        'title': 'Barack Obama - Travels of the President',
        'read_status': 'official_travel_inventory_body_read',
        'identified_visit_interval': ['2009-11-18', '2009-11-19'],
        'evidence_summary': 'The official foreign-travel inventory lists Korea, Osan/Seoul, November18-19 in2009. The surrounding trips are Copenhagen October2 and Japan November13-14; no October13 international trip. This is an itinerary source, rather than an ordinary article whose failure to mention another event supplies no negative evidence.',
        'qualification': 'Negative judgment is conditional on the official presidential foreign-travel inventory covering physical visits and eventdate denoting actual occurrence. It does not judge an announcement on October13 about a future visit.'
    },
    'R9_OBAMA_ZAPATERO': {
        'url': 'https://obamawhitehouse.archives.gov/the-press-office/remarks-president-obama-and-president-zapatero-spain-after-meeting/',
        'title': 'Remarks by President Obama and President Zapatero of Spain after meeting',
        'read_status': 'official_transcript_body_read',
        'event_local_datetime': '2009-10-13 13:21 EDT',
        'evidence_summary': 'Transcript locates Obama at the Oval Office, after lunch with Zapatero. Corroborates domestic itinerary.',
        'qualification': 'This instant is October14 in Korea, not a simultaneous contradiction of every instant of October13 KST. Do not use timezone-blind same-day physical impossibility.'
    },
    'R9_OBAMA_FIESTA': {
        'url': 'https://obamawhitehouse.archives.gov/realitycheck/the-press-office/remarks-president-performance-white-house-fiesta-latina',
        'title': 'Remarks by the President at In Performance At the White House Fiesta Latina',
        'read_status': 'official_transcript_body_read',
        'publication_date': '2009-10-14',
        'event_local_datetime': '2009-10-13 19:30 EDT',
        'evidence_summary': 'Actual remarks begin19:30 and end19:35, SouthLawn, October13. Release header is October14. Keep occurrence date distinct from publication.',
        'qualification': 'Also October14 in Korea; corroboration only, not alone a full-day exclusion.'
    },
    'R9_CAMEO_VISIT': {
        'url': 'https://eventdata.parusanalytics.com/cameo.dir/04consult.html',
        'title': 'CAMEO: Consult, codes042/043',
        'read_status': 'codebook_body_read',
        'evidence_summary': '042 describes travel for a meeting/event and043 describes hosting a visitor at residence, office or homecountry; the two are linked. Supports literal physicalvisit interpretation. Does not establish the actual sentence or coder treatment of future-tense reports in ICEWS.'
    },
    'R9_VIETNAM_SGGP': {
        'url': 'https://www.sggp.org.vn/tphcm-mit-tinh-trong-the-ky-niem-30-nam-giai-phong-mien-nam-thong-nhat-dat-nuoc-post298206.html',
        'title': 'TPHCM mit tinh ky niem30nam giai phong mien Nam',
        'read_status': 'contemporaneous_Vietnamese_body_read',
        'publication_date': '2005-05-01',
        'identified_event_date': '2005-04-30',
        'evidence_summary': 'Article describes morningApril30 anniversary at IndependencePalace; lists Lao delegation led by defence minister Douangchay Phichit and Cambodian delegates in HoChiMinhCity. Fits raw geographicalcontext but does not confirm May3 hosting or identify ThaiNewsService Story6708924. No direct Phuc involvement established and none excluded.'
    },
    'R9_XI_FRANKFURT': {
        'url': 'https://www.mfa.gov.cn/web/wjdt_674879/gjldrhd_674881/200910/t20091014_7699752.shtml',
        'title': '习近平出席法兰克福国际书展开幕式和中国主宾国活动',
        'read_status': 'official_article_body_read',
        'publication_date': '2009-10-14',
        'event_date': '2009-10-13',
        'evidence_summary': 'Xi attends Frankfurt bookfair opening with Merkel October13 during official Germany visit, meets publishers and visits Goethehouse. S32 Xi is a genuine filtered new crossing, gap .1214561462, oldcandidate. This is contrary itinerary evidence, not a complete independently verified denial of all possible Korea hosting on that calendar date.'
    },
    'R9_NSG_FORMATION_DECREE': {
        'url': 'https://security-legislation.ly/latest-laws/resolution-no-46-of-2014-on-forming-a-national-salvation-government/',
        'title': 'Resolution46 of2014 on forming a national salvation government',
        'read_status': 'primary_decree_republication_body_read',
        'event_date': '2014-09-02',
        'evidence_summary': 'GNC decree names UmaralHasi asPM and cabinet, takes effect at issue. Establishes specific2014 Libya administration, but not identity of every generic NationalSalvationGovernment mention.'
    },
    'R9_NSG_USDOS': {
        'url': 'https://www.ecoi.net/en/document/1058946.html',
        'title': 'USDOS CountryReport2014 Libya',
        'read_status': 'official_report_republication_body_read',
        'identified_event_date': '2014-08-25',
        'evidence_summary': 'Executive summary states defunctGNC subset reconvenedTripoli and announced NationalSalvationGovernment ledbyOmaralHassi August25. Supports2014 formation of specificnamedbody; rawJune2014 uses prevent treating nameidentity as settled.'
    },
    'R9_NSG_UN_FETCH': {
        'url': 'https://unsmil.unmissions.org/en/speeches-and-statements/briefing-by-tarek-mitri-srsg-for-libya-meeting-of-the-security-0',
        'read_status': 'HTTP403_no_article_body',
        'evidence_summary': 'Searchsnippetnotusedforlabels.'
    },
    'R9_WINDSOR_PRIMARY_FETCH': {
        'url': 'https://www.aph.gov.au/binaries/hansard/senate/commttee/s8053.pdf',
        'alternative_url': 'https://www.aph.gov.au/~/media/wopapub/senate/committee/fapa_ctte/completed_inquiries/2004_07/reg_partner_prog/report/c08_pdf.ashx',
        'read_status': 'PDF_HTTP403_and_alternative_restrictedURL_no_body',
        'evidence_summary': 'Alternate primaryPDF paths didnotrecover exactFeb4 AAP story; snippets about historicalsupport are not datedgold evidence.'
    }
})

def supplement(sid, ids, rationale):
    j = e['judgments'][sid]
    j['evidence_ids'] = list(dict.fromkeys(j['evidence_ids'] + ids))
    j['rationale'] = j['rationale'].split(' Round9:')[0] + ' Round9: ' + rationale

supplement('S32', ['R9_OBAMA_OFFICIAL_TRAVEL', 'R9_OBAMA_ZAPATERO', 'R9_OBAMA_FIESTA', 'R9_CAMEO_VISIT', 'R9_XI_FRANKFURT'],
           'Under actualphysicalvisit eventdate semantics and officialforeigntravel coverage, October13 Obama visit is contradicted; actualKorea visitNovember18-19. Futurevisit announcement semantics not adjudicated because originalAFP Story10206966 remains unlinked. DomesticOct13EDT snapshots areOct14KST, so not alone a sameinstant negative. FinalKarimov truth unknown. Xi newcrossing hasGermany itinerary counterevidence but not a full-day strictnegative. Invalidgold decline cannot be classified as reasonableworldchange or modelerror by itself.')
j = e['judgments']['S32']
j['stage'] = 'detailed_source_review'
j['gold_literal_occurrence_truth'] = 'contradicted_under_actual_physical_visit_event_date_and_official_itinerary_coverage'
j['gold_truth'] = 'unknown'
j['gold_original_report_or_coder_semantics_truth'] = 'unknown'
j['evaluation_target_counterindication'] = {
    'classification': 'physical_event_date_incompatible_with_official_visit_inventory',
    'query_date': '2009-10-13',
    'documented_physical_visit_interval': ['2009-11-18', '2009-11-19'],
    'announcement_at_query_date': 'unresolved_original_AFP_body_not_linked',
    'conditions': ['043 interpreted as actualphysicalhosting', 'date interpreted as eventoccurrence', 'officialpresidentialforeigntravel inventory coversphysicalvisits'],
    'not_a_whole_rank_reasonableness_or_model_error_label': True
}
supplement('S34', ['R9_VIETNAM_SGGP'],
           'ReadApril30ceremony inHoChiMinhCity withLao/Cambodiandelegates, publishedMay1. Plausibleidentifiedretrospectivecontext but noMay3visit or originalTNSstorymatch. Phuc solecrossing truth staysunknown; no false label from firstTRAINyear or sourceabsence.')
supplement('S21', ['R9_WINDSOR_PRIMARY_FETCH'],
           'AlternateparliamentprimaryPDFs failed; historicalfundraising/snippets notusedasFeb4support. Nochange togold/finaltruth.')
supplement('S38', ['R9_NSG_FORMATION_DECREE', 'R9_NSG_USDOS', 'R9_NSG_UN_FETCH'],
           '2014LibyanNationalSalvationGovernment formation documented, but exactrawname appearsJune7 before August25formation and inJune Iraq stories taggedLibya. Generic/proposedgovernment or automatedidentityconflation unresolved; exclude as strictfifthwitness. Existing4confirmederrorcomponents unchanged.')
path.write_text(json.dumps(e, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

temporal_path = ROOT / 'temporal-competitor-evidence.json'
t = json.loads(temporal_path.read_text(encoding='utf-8'))
t['excluded_identity_ambiguous_witnesses'] = [x for x in t['excluded_identity_ambiguous_witnesses'] if x.get('candidate') not in {'National Salvation Government', 'Democratic Alliance (Africa)'}] + [
    {'sample_ids': ['S38'], 'candidate': 'National Salvation Government',
     'reason': '2014Libyanbody formation documented. Rawname hasJune7/25/26 records precedingAugustformation; generic/proposedgovernment and crosscountrycoding ambiguity. No strictmapping toAugustadministration.'},
    {'sample_ids': ['S24'], 'candidate': 'Democratic Alliance (Africa)',
     'reason': 'Rawcountryempty, SouthAfricancontexts; do not map to a laterfoundedAfricanparty to manufacture temporalnegative. ExistingSouthAfricanDApredates2009.'}
]
t['costs']['round9_raw_identity_seconds'] = json.loads((ROOT / 'competitor-identity-context-round9.json').read_text(encoding='utf-8'))['seconds']
temporal_path.write_text(json.dumps(t, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

diagnostic = {
    'fixed_sample_n': 40,
    'evaluation_target_literal_physical_date_counterindication_query_ids': ['S32'],
    'evaluation_target_literal_physical_date_counterindication_n': 1,
    'fraction_of_fixed_sample': 1 / 40,
    'population_target_error_fraction': None,
    'world_reasonable_fraction': None,
    'world_model_error_fraction': None,
    'scope': 'One conditional literal-event target contradiction. Doesnotestablish report/coderfalsehood, or reasonableafterranking, or finalKarimovtruth. This axis is distinctfromgold-supportedwrongcompetitivepositions and missingpositives.',
    'S32': j['evaluation_target_counterindication'],
    'unknowns': ['originalAFPbody and prospectiveverbtense', 'Karimov/MackySall/China/Xi actualtruths atquerydate', 'whole-rankingjustification'],
    'neural_rescoring': False,
    'training': False
}
(ROOT / 'evaluation-target-date-round9.json').write_text(json.dumps(diagnostic, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'saved_source_n':len(e['sources']), 'S32_stage':j['stage'], 'conditional_target_counterindication_n':1, 'temporal_witness_candidate_n':t['temporal_witness_candidate_n']},ensure_ascii=False))
