"""Persist source reads from rounds6/7 without assuming closed-world negatives."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
payload=json.loads(r'''{
  "sources": {
    "R6_SUDAN_COURSE": {
      "url": "https://www.mfa.gov.cn/eng/wjb/zzjg_663340/jks_665232/jkxw_665234/202406/t20240606_11404987.html",
      "title": "The Humanitarian De-mining Training Course for Sudan Sponsored by the Chinese Governement Opens in Nanjing",
      "read_status": "official_article_body_read",
      "event_date": "2008-04-07",
      "evidence_summary": "Opening ceremony in Nanjing, cooperation with Sudanese people and later equipment donation stated. Supports April7 activity, not a new May16 intention."
    },
    "R6_KIM_REUTERS": {
      "url": "https://www.brecorder.com/news/3549651/south-korea-says-critical-moment-for-north-nuclear-deal-20080906804027",
      "title": "South Korea says critical moment for North nuclear deal",
      "read_status": "Reuters_republication_article_body_read",
      "publication_date": "2008-09-06",
      "identified_event_date": "2008-09-04",
      "evidence_summary": "Kim's briefing with South Korean media explicitly Thursday; foreign ministry barred foreign media. Thursday=September4. Attendance restriction does not directly establish Expel or withdraw on September5; internal StoryID not linked."
    },
    "R6_CHINADAILY_LIST": {
      "url": "https://www.chinadaily.com.cn/cndy/2014-10/28/p3_5.html",
      "title": "Humanitarian crises demand better response (listing)",
      "read_status": "publisher_listing_opening_read_not_full_article",
      "publication_date": "2014-10-25",
      "evidence_summary": "Listing opening says IFRC representatives gathered Beijing this week. Does not establish a new October25 visit."
    },
    "R6_CHINADAILY_FETCH": {
      "url": "https://www.chinadaily.com.cn/cndy/2014-10/25/content_18801088.htm",
      "read_status": "HTTP404_full_article_not_read",
      "alternative_url": "https://family-care-foundation.net/the-challenges-facing-humanitarian-agencies-and-workers/",
      "alternative_read_status": "fetch_error_no_article_body"
    },
    "R6_VARIG_FOLHA": {
      "url": "https://feeds.folha.uol.com.br/fsp/dinheiro/fi1103200519.htm",
      "title": "AGU é contra encontro de contas na Varig",
      "read_status": "contemporaneous_Portuguese_article_body_read",
      "publication_date": "2005-03-11",
      "identified_event_date": "2005-03-10",
      "evidence_summary": "Parliamentarians met VicePresident Alencar yesterday to discuss rescue plans for Varig. Not direct same-day Varig/PartyMember negotiation. Reuters sampled story unlinked."
    },
    "R6_GASPAROVIC_LUSA": {
      "url": "https://www.rtp.pt/noticias/pais/portugal-e-eslovaquia-esperam-acordo-sobre-orcamento-ate-ao-fim-do-ano_n19258",
      "title": "Portugal e Eslováquia esperam acordo sobre orçamento até ao fim do ano",
      "read_status": "contemporaneous_LUSA_republication_body_read",
      "publication_date_from_search_metadata": "2005-10-24",
      "evidence_summary": "Actual body confirms today's Sampaio/Gasparovic meeting and Gasparovic positive assessment of deeper economic relations. Broad bilateral assessment, precise Praise Portugal target/date mapping not independently resolved by body."
    },
    "R6_DAVIS_BRIEF": {
      "url": "https://www.criticalthreats.org/briefs/pakistan-security-brief/pakistan-security-brief-february-4-2011",
      "title": "Pakistan Security Brief February 4, 2011",
      "read_status": "contemporaneous_brief_body_read",
      "publication_date": "2011-02-04",
      "identified_event_date": "2011-02-03",
      "evidence_summary": "Embassy spokesperson criticizes due process treatment of Davis Thursday. Not an Obama personal demand on February4."
    },
    "R7_IFRC_DECLARATION": {
      "url": "https://www.rcrc-resilience-southeastasia.org/wp-content/uploads/2016/08/Final_BeijingCall-for-Innovation_2014.pdf",
      "title": "The Beijing Call for Innovation",
      "read_status": "movement_primary_declaration_PDF_text_read",
      "identified_event_interval": [
        "2014-10-22",
        "2014-10-24"
      ],
      "evidence_summary": "Primary declaration explicitly gives Beijing conference October22-24. Does not rule out delegates continuing visits October25 or other IFRC delegates in Iran."
    },
    "R7_BOGAN_KPBS": {
      "url": "https://www.kpbs.org/news/border-immigration/2012/05/03/lawyer-helping-wrong-way-ammo-driver-detained-mexi",
      "title": "Lawyer Helping Wrong-Way Ammo Driver Detained In Mexico",
      "read_status": "contemporaneous_article_body_read",
      "publication_date": "2012-05-03",
      "identified_arrest_date": "2012-04-17",
      "evidence_summary": "Describes Bogan already charged after April17 customs arrest and ongoing appeal. No new May3 accusation explicitly dated; exact EFE story unlinked. Generic US men may denote other people."
    },
    "R7_CUD_AFP": {
      "url": "https://sudantribune.com/article/12361",
      "title": "Ethiopian opposition to hold mass rally for national unity govt",
      "read_status": "contemporaneous_AFP_republication_body_read",
      "publication_date": "2005-09-22",
      "identified_CUD_statement_date": "2005-09-21",
      "evidence_summary": "CUD vice president told reporters late Wednesday about October2 demonstration; Thursday government and police warned opposition. Direction cannot be inverted to prove CUD statement about police Sept22; different BBCMonitoring story unlinked."
    },
    "R7_VIETNAM_GREENLEFT": {
      "url": "https://www.greenleft.org.au/2005/626/world/vietnam-historic-victory-celebrated",
      "title": "VIETNAM: Historic victory celebrated",
      "read_status": "contemporaneous_commentary_article_body_read",
      "publication_date": "2005-05-11",
      "identified_event_date": "2005-04-30",
      "evidence_summary": "Reunification celebrations April30, source identifies RaulCastro as high-profile visitor. Does not establish or exclude separate Vietnam/Laos visit May3; exact ThaiNewsService story unlinked."
    },
    "R7_PERU_FETCH": {
      "url": "https://mreperu.blogspot.com/search/label/Ollanta%20Humala",
      "read_status": "non_retryable_fetch_error_body_not_read",
      "evidence_summary": "Search preview describes planned Humala/Xi November12 meeting, but full body inaccessible. Preview not treated as truth for sampled China/Xi November2."
    },
    "R7_BLAIR_GUARDIAN": {
      "url": "https://www.theguardian.com/politics/2007/apr/16/foreignpolicy.iran1",
      "title": "Blair backs Browne over sailors",
      "read_status": "contemporaneous_article_body_read",
      "publication_date": "2007-04-16",
      "evidence_summary": "Blair backing of Browne dated April16, does not establish April13 statement about Iran. Guardian source word limit25 applies to any displayed quotation."
    },
    "R7_IRAN_LORIENT": {
      "url": "https://www.lorientlejour.com/article/529341/La_communaute_internationale_toujours_divisee_face_a_l%2527Iran_Washington_appelle_Moscou_a_cesser_toute_cooperation_nucleaire_avec_Teheran.html",
      "title": "La communauté internationale toujours divisée face à l’Iran",
      "read_status": "visible_contemporaneous_article_body_read",
      "publication_date": "2006-04-20",
      "identified_event_date": "2006-04-19",
      "evidence_summary": "Body discusses prior-day Burns/Blair statements and divided powers over potential sanctions. Does not establish same-day April20 UNSC threat or exclude senator Obama's separate threat."
    },
    "R7_STATE_FETCH": {
      "url": "https://2009-2017.state.gov/r/pa/prs/dpb/2014/08/231008.htm",
      "title": "Daily Press Briefing August 27, 2014",
      "read_status": "technical_difficulties_no_briefing_body_read",
      "evidence_summary": "Search previews not treated as actual read statements by Iran media or Obama."
    },
    "R7_PORTUGAL_ARCHIVE_FETCH": {
      "url": "https://www.arquivo.presidencia.pt/report/?ID=141131&Locale=pt&f=3&p=%2FArcheevo6_DSDA%2FFrontOffice%2FPublic%2FMiscellaneous%2FDescriptionItemPublicReport",
      "read_status": "internal_fetch_error_no_archive_body",
      "evidence_summary": "Search metadata24-26Oct2005 state visit retained as pointer only."
    }
  },
  "updates": {
    "S01": {
      "evidence_ids": [
        "R6_SUDAN_COURSE"
      ],
      "rationale": "Actual April7 course and promise known; sampled May16 cooperation intention not established. Sudan finalTop1 not contradicted by China cooperation with Sudan."
    },
    "S08": {
      "evidence_ids": [
        "R6_KIM_REUTERS"
      ],
      "rationale": "Reuters body supplies date/relation counterindications: Thursday Sept4 briefing exclusion versus Sept5 Expel. Exact original story identity unlinked; China finalTop1 unchanged cannot be identified as newly introduced error."
    },
    "S12": {
      "evidence_ids": [
        "R7_BOGAN_KPBS"
      ],
      "rationale": "May3 source describes previous charges, not a new same-day accusation. AustraliaPolice finalTop1 unknown; absence of mention of another US man cannot prove false."
    },
    "S14": {
      "evidence_ids": [
        "R6_CHINADAILY_LIST",
        "R6_CHINADAILY_FETCH",
        "R7_IFRC_DECLARATION"
      ],
      "rationale": "Official declaration gives Beijing conference22-24Oct; ChinaDaily Oct25 opening reports this week. Gold MakeVisit exactOct25 remains unknown. Iran sole new filtered outranker, original candidate, aftergap1.910790443. Beijing gathering does not exclude simultaneous other IFRC representatives in Iran."
    },
    "S15": {
      "stage": "detailed_source_attempt_blocked",
      "evidence_ids": [
        "R7_STATE_FETCH"
      ],
      "rationale": "Official Aug27 briefing body inaccessible; snippets and later Aug29 sanctions not substituted for sampled Aug27 IranianMedia/Obama statements. No strict truth label."
    },
    "S18": {
      "evidence_ids": [
        "R7_IRAN_LORIENT"
      ],
      "rationale": "Read April20 article discusses previous-day statements; UNSC threat at sampleddate not established. Obama being senator does not exclude his ability to threaten sanctions."
    },
    "S19": {
      "stage": "detailed_source_attempt_blocked",
      "evidence_ids": [
        "R7_PERU_FETCH"
      ],
      "rationale": "Official-associated Peru blog body not acquired. Raw gold same sentence codes China intentmeetXi and Peru cooperationChina, possible argument mapping ambiguity, not proof. Unchanged SouthKorea Top1 unknown."
    },
    "S20": {
      "evidence_ids": [
        "R7_BLAIR_GUARDIAN"
      ],
      "rationale": "Actual read article datedApril16, not sampledApril13; excluded as same-day support. Unknown gold/final rather than assigning error from wrongdate article."
    },
    "S23": {
      "evidence_ids": [
        "R7_CUD_AFP"
      ],
      "rationale": "AFP report dates CUD statement WednesdaySept21; ThursdaySept22 police warning is opposite actor direction. Police/CIA/ArgentinaLawyer three new filtered outrankers all original candidates; not independently verified or contradicted. BBCMonitoring story unlinked."
    },
    "S25": {
      "evidence_ids": ["R6_DAVIS_BRIEF"],
      "rationale": "February4 brief describes Thursday February3 embassy criticism of Davis due process. Actor is US embassy spokesperson, not Obama personally; exact BBCMonitoringSouthAsia story unlinked. No same-day truth label for Obama demandPakistan or finalIran."
    },
    "S26": {
      "evidence_ids": [
        "R6_VARIG_FOLHA"
      ],
      "rationale": "Mar11 body dates parliamentarians/Alencar meeting yesterday; not same-day Varig negotiation. CitizenBrazil may include party members, without identity relation annotation cannot decide false."
    },
    "S29": {
      "evidence_ids": [
        "R6_GASPAROVIC_LUSA",
        "R7_PORTUGAL_ARCHIVE_FETCH"
      ],
      "rationale": "Actual body supports positive bilateral economic assessment during statevisit, but exact CAMEO PraisePortugal and date independently unresolved. IMF may issue same-day assessment; visit report alone cannot disprove finalTop1."
    },
    "S34": {
      "evidence_ids": [
        "R7_VIETNAM_GREENLEFT"
      ],
      "rationale": "April30 celebration commentary is not May3 ThaiNewsService visit. Sole new filtered outrankerNguyenXuanPhuc was absent2005 candidate space/firstTRAIN2014, gap3.898614883; benchmark first appearance not birth, and lack of premiership2005 does not exclude hosting as provincial official. Actual history still unresolved."
    }
  }
}''')
p=ROOT/'representative-history-evidence.json'
e=json.loads(p.read_text(encoding='utf-8'))
e['sources'].update(payload['sources'])
unknown={'final_event_truth':'unknown','original_top1_truth':'unknown','gold_truth':'unknown','reasonable_rank_adjustment':'unknown'}
for sid,j in payload['updates'].items():
 e['judgments'][sid]={**unknown,'stage':'detailed_source_review',**j}
p.write_text(json.dumps(e,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('ROUND67_SAVED', sorted(payload['updates']))
