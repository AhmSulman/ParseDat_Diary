"""
Library Dashboard — FastAPI routes over LibraryService
=======================================================
FastAPI rather than Flask: it is already a dependency and already serves the
Android client, so adding a second web framework would buy nothing.

Every number here comes from `storage.library.LibraryService`, the same module
the CLI uses. `main.py doctor` and `GET /library` therefore cannot disagree —
a divergence would be a bug in a caller, not two implementations differing.

SECURITY — THESE ROUTES DELETE FILES
------------------------------------
`run_server` defaults to host="0.0.0.0". Exposing clean/sync there would put
file deletion on every device on the network. So:

  - read-only routes are always available
  - mutating routes exist only when the server is started with --allow-admin
  - even then they refuse any request whose client is not loopback
  - and they require an explicit confirm token in the body, so a stray or
    cross-site GET can never wipe an index

The page is fully self-contained: no CDN, no external fonts, consistent with
MAAN being offline by design.
"""

from __future__ import annotations

_LOOPBACK = {"127.0.0.1", "::1", "localhost"}
CONFIRM_TOKEN = "yes-i-mean-it"


def _is_local(request) -> bool:
    client = getattr(request, "client", None)
    return bool(client and client.host in _LOOPBACK)


def register(app, *, allow_admin: bool = False) -> None:
    """Attach dashboard routes to an existing FastAPI app."""
    from fastapi import HTTPException, Request
    from fastapi.responses import HTMLResponse

    from storage.library import LibraryService

    @app.get("/library")
    def library():
        """Full reconciliation report as JSON. Read-only."""
        return LibraryService().report()

    @app.get("/ui", response_class=HTMLResponse)
    def ui():
        return _render(LibraryService().report(), allow_admin)

    def _guard(request: Request, body: dict | None):
        if not allow_admin:
            raise HTTPException(
                403, "Admin routes disabled. Restart with: main.py server --allow-admin"
            )
        if not _is_local(request):
            raise HTTPException(403, "Admin routes are loopback-only.")
        if not body or body.get("confirm") != CONFIRM_TOKEN:
            raise HTTPException(400, f"Missing confirm token '{CONFIRM_TOKEN}'.")

    @app.post("/library/sync")
    async def sync(request: Request):
        _guard(request, await _json(request))
        return {"removed": LibraryService().clean(orphans=True)}

    @app.post("/library/clean")
    async def clean(request: Request):
        body = await _json(request)
        _guard(request, body)
        scopes = {k: bool(body.get(k)) for k in
                  ("index", "checkpoint", "orphans", "quarantine", "text")}
        if not any(scopes.values()):
            raise HTTPException(400, "Select at least one scope.")
        return {"removed": LibraryService().clean(**scopes)}


async def _json(request) -> dict:
    try:
        return await request.json()
    except Exception:
        return {}


# ── rendering ─────────────────────────────────────────────────────────────────
_STATUS_STYLE = {
    "indexed": ("ok", "Indexed"),
    "hole": ("bad", "HOLE — never indexed"),
    "orphan": ("bad", "ORPHAN — PDF deleted"),
    "pending": ("warn", "Pending"),
    "quarantined": ("bad", "Quarantined"),
}


def _esc(s) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def _render(rep: dict, allow_admin: bool) -> str:
    c = rep["counts"]
    healthy = rep["healthy"]

    banner = (
        f'<div class="banner ok">Healthy — every store agrees at '
        f'{c["pdfs_on_disk"]} book(s).</div>'
        if healthy else
        f'<div class="banner bad">{rep["drift"]} inconsistency(ies). '
        f'Run <code>main.py sync</code> or <code>main.py reindex</code>.</div>'
    )

    tiles = "".join(
        f'<div class="tile"><div class="n">{v:,}</div><div class="l">{_esc(k)}</div></div>'
        for k, v in [
            ("PDFs on disk", c["pdfs_on_disk"]),
            ("Books indexed", c["books_indexed"]),
            ("Chunks", c["chunks_indexed"]),
            ("Checkpoint done", c["checkpoint_done"]),
            ("Extracted .txt", c["txt_files"]),
            ("Quarantined", c["quarantined"]),
        ]
    )

    rows = []
    for b in rep["books"]:
        cls, label = _STATUS_STYLE.get(b["status"], ("warn", b["status"]))
        rows.append(
            f'<tr><td class="src">{_esc(b["source"])}</td>'
            f'<td><span class="pill {cls}">{_esc(label)}</span></td>'
            f'<td class="num">{b["n_chunks"]:,}</td>'
            f'<td class="num">{b["n_pages"] or ""}</td>'
            f'<td>{"yes" if b["has_text"] else "no"}</td></tr>'
        )
    table = "".join(rows) or '<tr><td colspan="5">No books.</td></tr>'

    quar = ""
    if rep["quarantine"]:
        items = "".join(
            f'<li><b>{_esc(q.get("source"))}</b><ul>'
            + "".join(f"<li>{_esc(r)}</li>" for r in q.get("reasons", []))
            + "</ul></li>"
            for q in rep["quarantine"]
        )
        quar = f'<h2>Quarantined</h2><p class="muted">Failed the quality gate; never indexed.</p><ul class="q">{items}</ul>'

    s = rep.get("settings") or {}
    settings = (
        f'<p class="muted">Index built with <b>{_esc(s.get("embed_model"))}</b> '
        f'({_esc(s.get("embed_dim"))}-dim), chunks {_esc(s.get("chunk_size"))}/'
        f'{_esc(s.get("chunk_overlap"))}.</p>' if s else
        '<p class="muted">No index built yet.</p>'
    )

    admin = (
        '<p class="muted">Admin actions enabled (loopback only).</p>'
        if allow_admin else
        '<p class="muted">Read-only. Start with <code>--allow-admin</code> for '
        'clean/sync, which are restricted to localhost.</p>'
    )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MAAN Library</title>
<style>
:root {{ --bg:#faf9f7; --fg:#1a1a1a; --mut:#6b6b6b; --line:#e3e0db;
         --ok:#1a7f4b; --okbg:#e6f4ec; --bad:#a3242b; --badbg:#fbeaea;
         --warn:#8a6100; --warnbg:#fdf3e0; --card:#fff; }}
@media (prefers-color-scheme: dark) {{
  :root {{ --bg:#16151a; --fg:#eceaf0; --mut:#a09da8; --line:#2f2d36;
           --ok:#5fd39b; --okbg:#12301f; --bad:#f0868c; --badbg:#331215;
           --warn:#e8b45c; --warnbg:#332616; --card:#1e1d23; }} }}
* {{ box-sizing:border-box; }}
body {{ margin:0; padding:2rem 1.25rem; background:var(--bg); color:var(--fg);
  font:15px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif; }}
main {{ max-width:960px; margin:0 auto; }}
h1 {{ font-size:1.5rem; margin:0 0 .25rem; letter-spacing:-.01em; }}
h2 {{ font-size:1.05rem; margin:2rem 0 .5rem; }}
.muted {{ color:var(--mut); font-size:.875rem; margin:.25rem 0; }}
code {{ background:var(--card); border:1px solid var(--line); border-radius:4px;
  padding:.05rem .3rem; font-size:.85em; }}
.banner {{ padding:.7rem .9rem; border-radius:8px; margin:1rem 0;
  border:1px solid var(--line); font-weight:600; }}
.banner.ok {{ background:var(--okbg); color:var(--ok); border-color:var(--ok); }}
.banner.bad {{ background:var(--badbg); color:var(--bad); border-color:var(--bad); }}
.tiles {{ display:grid; gap:.6rem; grid-template-columns:repeat(auto-fit,minmax(130px,1fr)); }}
.tile {{ background:var(--card); border:1px solid var(--line); border-radius:8px; padding:.8rem; }}
.tile .n {{ font-size:1.5rem; font-weight:650; font-variant-numeric:tabular-nums; }}
.tile .l {{ color:var(--mut); font-size:.78rem; text-transform:uppercase;
  letter-spacing:.04em; margin-top:.15rem; }}
.wrap {{ overflow-x:auto; border:1px solid var(--line); border-radius:8px; background:var(--card); }}
table {{ border-collapse:collapse; width:100%; font-size:.875rem; }}
th,td {{ text-align:left; padding:.5rem .7rem; border-bottom:1px solid var(--line); }}
th {{ font-size:.72rem; text-transform:uppercase; letter-spacing:.05em; color:var(--mut); }}
tr:last-child td {{ border-bottom:none; }}
td.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
td.src {{ max-width:380px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
.pill {{ display:inline-block; padding:.1rem .5rem; border-radius:99px;
  font-size:.75rem; font-weight:600; white-space:nowrap; }}
.pill.ok {{ background:var(--okbg); color:var(--ok); }}
.pill.bad {{ background:var(--badbg); color:var(--bad); }}
.pill.warn {{ background:var(--warnbg); color:var(--warn); }}
ul.q {{ font-size:.875rem; }} ul.q ul {{ color:var(--mut); }}
</style></head><body><main>
<h1>MAAN Library</h1>
{settings}
{banner}
<div class="tiles">{tiles}</div>
<h2>Books</h2>
<div class="wrap"><table>
<thead><tr><th>Source</th><th>Status</th><th>Chunks</th><th>Pages</th><th>Text</th></tr></thead>
<tbody>{table}</tbody></table></div>
{quar}
<h2>Actions</h2>
{admin}
</main></body></html>"""
