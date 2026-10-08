"""docs/references.md (단일 원본) → 참고문헌 대장 아티팩트 HTML.
사용: python scripts/build_refs_artifact.py  → docs/references_artifact.html 생성 → Artifact 도구로 같은 URL에 재발행.
"""
import re, html, datetime, collections
SRC = "docs/references.md"; OUT = "docs/references_artifact.html"
SEC = {"A": "잠재 노드 층 (결정 1·2)", "B": "물리 연산자 (결정 3)", "C": "인코더·디코더 (결정 5·6)", "D": "readout · β (결정 7·8)", "E": "검증 설계 (결정 10)", "F": "판정 임계값 (결정 12)", "G": "대조군 (결정 11)", "H": "데이터 제품"}
STATUS_CLS = {"원문 확인": "ok", "초록 확인": "mid", "인용 금지": "ban", "**arXiv 전용": "ban", "**학회 논문집": "ban"}

def md_inline(s):
    s = html.escape(s); s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s); s = re.sub(r"~~(.+?)~~", r"<s>\1</s>", s); return s

rows = []
for ln in open(SRC, encoding="utf-8"):
    if not ln.startswith("| ") or ln.startswith("| ID") or ln.startswith("|---"): continue
    c = [x.strip() for x in ln.strip().strip("|").split("|")]
    if len(c) < 7: continue
    rows.append(dict(id=c[0], title=c[1], auth=c[2], venue=c[3], use=c[4], status=c[5], added=c[6]))
by = collections.OrderedDict((k, []) for k in SEC)
for r in rows: by[r["id"][0]].append(r)
n_ok = sum(1 for r in rows if r["status"].startswith("원문 확인")); n_ban = sum(1 for r in rows if "인용 금지" in r["status"] or "인용 불가" in r["status"])
today = datetime.date.today().isoformat()
def cls(st):
    for k, v in STATUS_CLS.items():
        if k in st: return v
    return "warn" if any(w in st for w in ("미확인", "재확인", "미열람", "미확정", "추가 필요", "프리프린트")) else "base"
body = []
for k, lst in by.items():
    if not lst: continue
    body.append(f'<section><h2><span class="sec">{k}</span>{SEC[k]}<span class="cnt">{len(lst)}건</span></h2><div class="tbl"><table><thead><tr><th>ID</th><th>논문 제목</th><th>저자 · 연도</th><th>저널 · 출처</th><th>참고 내용</th><th>확인 상태</th><th>추가일</th></tr></thead><tbody>')
    for r in lst:
        body.append(f'<tr><td class="id">{md_inline(r["id"])}</td><td class="t">{md_inline(r["title"])}</td><td>{md_inline(r["auth"])}</td><td class="v">{md_inline(r["venue"])}</td><td class="u">{md_inline(r["use"])}</td><td><span class="st {cls(r["status"])}">{md_inline(r["status"])}</span></td><td class="d">{r["added"]}</td></tr>')
    body.append("</tbody></table></div></section>")
prio = ""
m = re.search(r"## 재확인 우선순위[^\n]*\n\n(.+)", open(SRC, encoding="utf-8").read(), flags=re.S)
if m: prio = md_inline(m.group(1).strip())
page = f'''<title>NO₂–XCO₂ 참고문헌 대장</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Noto+Serif+KR:wght@500;700&family=IBM+Plex+Sans+KR:wght@300;400;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{{--paper:#f5f3ee;--surface:#fcfbf8;--ink:#1d1c19;--dim:#6d685e;--rule:#dcd7cb;--acc:#1f5f8b;--ok:#1f7a4d;--ok-bg:#dff1e6;--mid:#8a6a0f;--mid-bg:#f6ecc8;--warn:#8b4a1f;--warn-bg:#f5e1d3;--ban:#9c2b2b;--ban-bg:#f6d9d9;--base-bg:#e9e6de}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--paper:#16171b;--surface:#1e2025;--ink:#ebe8e1;--dim:#a29d93;--rule:#34373e;--acc:#7fb2e6;--ok:#7fd3a4;--ok-bg:#173324;--mid:#e3c46a;--mid-bg:#3a3014;--warn:#f0a06a;--warn-bg:#3a2416;--ban:#f08484;--ban-bg:#3d1b1b;--base-bg:#2a2c33}}}}
:root[data-theme="dark"]{{--paper:#16171b;--surface:#1e2025;--ink:#ebe8e1;--dim:#a29d93;--rule:#34373e;--acc:#7fb2e6;--ok:#7fd3a4;--ok-bg:#173324;--mid:#e3c46a;--mid-bg:#3a3014;--warn:#f0a06a;--warn-bg:#3a2416;--ban:#f08484;--ban-bg:#3d1b1b;--base-bg:#2a2c33}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--paper);color:var(--ink);font-family:"IBM Plex Sans KR","Apple SD Gothic Neo",system-ui,sans-serif;font-weight:300;font-size:14px;line-height:1.6;padding-inline:20px}}
.wrap{{max-width:1240px;margin:0 auto;padding-block:48px 96px}}
h1{{font-family:"Noto Serif KR",serif;font-weight:700;font-size:clamp(26px,4vw,36px);margin:0 0 8px;letter-spacing:-.01em}}
.lead{{color:var(--dim);max-width:70ch;margin:0 0 18px}}
.stats{{display:flex;flex-wrap:wrap;gap:10px 26px;font-family:"IBM Plex Mono",monospace;font-size:12px;color:var(--dim);margin:0 0 36px;padding:12px 0;border-top:1px solid var(--rule);border-bottom:1px solid var(--rule)}}
.stats b{{color:var(--ink);font-weight:500}}
h2{{font-family:"Noto Serif KR",serif;font-weight:500;font-size:19px;margin:38px 0 10px;display:flex;align-items:baseline;gap:12px}}
.sec{{font-family:"IBM Plex Mono",monospace;font-size:12px;color:var(--acc);letter-spacing:.1em}} .cnt{{font-family:"IBM Plex Mono",monospace;font-size:11px;color:var(--dim);font-weight:400}}
.tbl{{overflow-x:auto;border:1px solid var(--rule);border-radius:3px;background:var(--surface)}}
table{{border-collapse:collapse;width:100%;min-width:980px}} th,td{{text-align:left;padding:9px 11px;border-bottom:1px solid var(--rule);vertical-align:top}}
th{{font-family:"IBM Plex Mono",monospace;font-size:10.5px;letter-spacing:.09em;text-transform:uppercase;color:var(--dim);font-weight:500;white-space:nowrap;position:sticky;top:0;background:var(--surface)}}
tr:last-child td{{border-bottom:none}}
td.id{{font-family:"IBM Plex Mono",monospace;color:var(--acc);white-space:nowrap}} td.t{{font-weight:400;min-width:220px}} td.v{{color:var(--dim);font-size:13px;min-width:150px}} td.u{{min-width:280px}} td.d{{font-family:"IBM Plex Mono",monospace;font-size:11.5px;color:var(--dim);white-space:nowrap}}
.st{{display:inline-block;font-family:"IBM Plex Mono",monospace;font-size:11px;padding:2px 7px;border-radius:2px;white-space:normal;background:var(--base-bg);color:var(--ink)}}
.st.ok{{background:var(--ok-bg);color:var(--ok)}} .st.mid{{background:var(--mid-bg);color:var(--mid)}} .st.warn{{background:var(--warn-bg);color:var(--warn)}} .st.ban{{background:var(--ban-bg);color:var(--ban);font-weight:600}}
s{{color:var(--dim)}} strong{{font-weight:600}}
.prio{{margin-top:40px;padding:16px 20px;border-left:3px solid var(--acc);background:var(--surface);border-radius:3px;max-width:90ch;font-size:13.5px}}
.prio h3{{margin:0 0 6px;font-size:14px;font-weight:600}}
footer{{margin-top:36px;font-size:12.5px;color:var(--dim);max-width:80ch}}
</style>
<div class="wrap">
<h1>NO₂–XCO₂ 참고문헌 대장</h1>
<p class="lead">설계 결정마다 근거로 쓴 문헌과, 그 문헌에서 무엇을 가져왔는지. 원본은 저장소 <code>docs/references.md</code> — 행을 추가한 뒤 이 페이지를 다시 만든다.</p>
<div class="stats"><span>문헌 <b>{len(rows)}</b>건</span><span>원문 확인 <b>{n_ok}</b></span><span>인용 금지·불가 <b>{n_ban}</b></span><span>구획 <b>{sum(1 for v in by.values() if v)}</b></span><span>갱신 <b>{today}</b></span></div>
{"".join(body)}
<div class="prio"><h3>재확인 우선순위 (논문 집필 전)</h3>{prio}</div>
<footer>확인 상태: 원문 확인 = 본문에서 인용문 확보 · 초록 확인 = 초록/API 메타만 · 서지 확정 = DOI·서지만, 본문 미열람 · 미확인/재확인 = 2차 인용 또는 서지 불완전 · 인용 금지 = 실체 미확인 · 인용 불가 = 프리프린트/학회 논문집(저널 대체 행 참조). "참고 내용"의 <strong>결정 n</strong>은 설계 문서의 결정 번호.</footer>
</div>'''
open(OUT, "w", encoding="utf-8").write(page); print(f"{len(rows)}건 → {OUT}")
