import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// Thumbnail picker for H3 RefCap Select: browse models/refcaps, click the views to use, write them into the node's
// widgets. The thumbnails are the photos stored in each RefCap, served by /refcap/thumb.

const MAX_IMAGES = 9, MAX_CLIPS = 3, MAX_AUDIOS = 3;

const CSS = `
.refcap-picker {box-sizing:border-box;width:min(1180px,96vw);height:88vh;padding:20px;border:1px solid #455266;
  border-radius:14px;background:#111820;color:#e8edf5;font:14px system-ui}
.refcap-picker::backdrop {background:#000b}
.refcap-picker .layout {height:100%;display:flex;flex-direction:column;gap:14px;min-height:0}
.refcap-picker header {display:flex;gap:12px;align-items:center;justify-content:space-between}
.refcap-picker h2 {margin:0;font-size:21px}
.refcap-picker .body {flex:1;min-height:0;display:grid;grid-template-columns:270px 1fr;gap:14px}
.refcap-picker .files {overflow:auto;display:flex;flex-direction:column;gap:6px;padding-right:4px}
.refcap-picker .file {text-align:left;padding:9px 11px;border:1px solid #334254;border-radius:8px;background:#18222d;
  color:inherit;font:inherit;cursor:pointer}
.refcap-picker .file.on {border-color:#ffb257;background:#20303f}
.refcap-picker .file small {display:block;color:#9fb0c4}
.refcap-picker .views {overflow:auto;min-height:0;padding-right:4px}
.refcap-picker .grid {display:grid;grid-template-columns:repeat(auto-fill,minmax(132px,1fr));gap:10px}
.refcap-picker figure {margin:0;border:2px solid #334254;border-radius:9px;overflow:hidden;background:#18222d;
  cursor:pointer;position:relative}
.refcap-picker figure.on {border-color:#ffb257}
.refcap-picker figure img {display:block;width:100%;height:118px;object-fit:cover;background:#0c1116}
.refcap-picker figcaption {padding:5px 7px;font-size:12px;color:#c3d0de;overflow-wrap:anywhere}
.refcap-picker figure .tick {position:absolute;top:6px;right:6px;width:21px;height:21px;border-radius:50%;
  background:#ffb257;color:#171c22;font-weight:700;display:none;align-items:center;justify-content:center;font-size:13px}
.refcap-picker figure.on .tick {display:flex}
.refcap-picker .chips {display:flex;flex-wrap:wrap;gap:8px;margin:10px 0 2px}
.refcap-picker .chip {padding:7px 11px;border:1px solid #334254;border-radius:999px;background:#18222d;color:inherit;
  font:inherit;cursor:pointer}
.refcap-picker .chip.on {border-color:#ffb257;background:#20303f}
.refcap-picker input,.refcap-picker button {font:inherit;color:inherit;background:#222f3e;border:1px solid #455266;
  border-radius:7px;padding:8px 11px}
.refcap-picker input {width:100%}
.refcap-picker button {cursor:pointer}
.refcap-picker footer {display:flex;gap:12px;align-items:center;justify-content:space-between}
.refcap-picker .use {background:#ffb257;border-color:#ffb257;color:#171c22;font-weight:650}
.refcap-picker .count.over {color:#ff8a6a}
.refcap-picker .hint {color:#9fb0c4}
.refcap-picker h3 {margin:0 0 8px;font-size:15px;font-weight:600}
`;

const el = (tag, text, className) => {
    const node = document.createElement(tag);
    if (text !== undefined) node.textContent = text;
    if (className) node.className = className;
    return node;
};

function parseList(text, all) {
    const out = new Set();
    const value = (text || "").trim().toLowerCase();
    if (!value || value === "none") return out;
    if (value === "all") return new Set(all);
    for (const part of value.split(",")) {
        const range = part.trim().match(/^(\d+)\s*-\s*(\d+)$/);
        if (range) {
            for (let i = Number(range[1]); i <= Number(range[2]); i++) out.add(i);
        } else if (part.trim() !== "") {
            out.add(Number(part));
        }
    }
    return out;
}

// [0,1,2,4] -> "0-2,4"; everything -> "all"
function formatList(chosen, all) {
    const values = [...chosen].sort((a, b) => a - b);
    if (!values.length) return "";
    if (values.length === all.length) return "all";
    const parts = [];
    for (let i = 0; i < values.length;) {
        let j = i;
        while (j + 1 < values.length && values[j + 1] === values[j] + 1) j++;
        parts.push(j - i >= 2 ? `${values[i]}-${values[j]}` : values.slice(i, j + 1).join(","));
        i = j + 1;
    }
    return parts.join(",");
}

const widgetOf = (node, name) => (node.widgets || []).find(w => w.name === name);

function openPicker(node, entries) {
    const dialog = el("dialog", undefined, "refcap-picker");
    const style = el("style");
    style.textContent = CSS;
    const layout = el("div", undefined, "layout");

    const head = el("header");
    const close = el("button", "Close");
    head.append(el("h2", "RefCap picker"), close);

    const search = el("input");
    search.placeholder = "Search name or description";
    search.setAttribute("aria-label", "Search RefCaps");

    const body = el("div", undefined, "body");
    const files = el("div", undefined, "files");
    const views = el("div", undefined, "views");
    body.append(files, views);

    const footer = el("footer");
    const count = el("span", "", "count");
    const buttons = el("div");
    const cancel = el("button", "Cancel");
    const use = el("button", "Use selection", "use");
    buttons.append(cancel, use);
    buttons.style.display = "flex";
    buttons.style.gap = "10px";
    footer.append(count, buttons);

    layout.append(head, search, body, footer);
    dialog.append(style, layout);
    document.body.append(dialog);

    // one selection per file, so switching back and forth keeps what was ticked
    const picked = new Map();
    const current = widgetOf(node, "refcap")?.value;
    let active = entries.find(e => e.name === current) || entries[0];
    if (active) {
        picked.set(active.name, {
            views: parseList(widgetOf(node, "views")?.value, active.views.map(v => v.index)),
            clips: parseList(widgetOf(node, "clips")?.value, active.clips.map(c => c.number)),
            audios: parseList(widgetOf(node, "audios")?.value, active.audios.map(a => a.number)),
        });
    }
    const stateOf = name => {
        if (!picked.has(name)) picked.set(name, { views: new Set(), clips: new Set(), audios: new Set() });
        return picked.get(name);
    };

    function tokensFor(entry, state) {
        const sum = (list, key, chosen) => list.filter(x => chosen.has(x[key])).reduce((t, x) => t + (x.tokens || 0), 0);
        return sum(entry.views, "index", state.views) + sum(entry.clips, "number", state.clips)
            + sum(entry.audios, "number", state.audios);
    }

    function renderCount() {
        let images = 0, clips = 0, audios = 0, tokens = 0;
        for (const [name, state] of picked) {
            const entry = entries.find(e => e.name === name);
            if (!entry) continue;
            images += state.views.size; clips += state.clips.size; audios += state.audios.size;
            tokens += tokensFor(entry, state);
        }
        const over = images > MAX_IMAGES || clips > MAX_CLIPS || audios > MAX_AUDIOS;
        count.classList.toggle("over", over);
        count.textContent = `${images}/${MAX_IMAGES} images`
            + (clips ? `, ${clips}/${MAX_CLIPS} clips` : "") + (audios ? `, ${audios}/${MAX_AUDIOS} audio` : "")
            + ` — ${tokens} reference tokens per step`
            + (over ? "  (over the reference node's cap)" : "")
            + (picked.size > 1 ? "  •  this node writes one RefCap; chain another Select for the rest" : "");
    }

    function renderViews() {
        views.replaceChildren();
        if (!active) return;
        const state = stateOf(active.name);
        views.append(el("h3", active.name + (active.description ? ` — ${active.description}` : "")));
        if (!active.pixels && active.views.length) {
            views.append(el("p", "This RefCap stores no photos, so it has no thumbnails and cannot be replayed.", "hint"));
        }
        const grid = el("div", undefined, "grid");
        for (const view of active.views) {
            const figure = el("figure");
            figure.classList.toggle("on", state.views.has(view.index));
            const img = el("img");
            img.loading = "lazy";
            img.alt = `view ${view.index}`;
            img.src = api.apiURL(`/refcap/thumb?name=${encodeURIComponent(active.name)}&index=${view.index}`);
            const caption = `${view.index}${view.yaw !== null && view.yaw !== undefined ? `  yaw ${+view.yaw.toFixed(1)}°` : ""}`
                + `  ${view.size[0]}×${view.size[1]}`;
            const tick = el("span", "✓", "tick");
            figure.append(img, tick, el("figcaption", caption + (view.label ? `  ${view.label}` : "")));
            figure.onclick = () => {
                state.views.has(view.index) ? state.views.delete(view.index) : state.views.add(view.index);
                figure.classList.toggle("on", state.views.has(view.index));
                renderCount();
            };
            grid.append(figure);
        }
        if (active.views.length) views.append(grid);

        const chips = el("div", undefined, "chips");
        const addChip = (label, set, key) => {
            const chip = el("button", label, "chip");
            chip.classList.toggle("on", set.has(key));
            chip.onclick = () => {
                set.has(key) ? set.delete(key) : set.add(key);
                chip.classList.toggle("on", set.has(key));
                renderCount();
            };
            chips.append(chip);
        };
        for (const clip of active.clips) {
            addChip(`Video ${clip.number}: ${clip.frames} frames${clip.audio ? " + sound" : ""}`, state.clips, clip.number);
        }
        for (const audio of active.audios) addChip(`Audio ${audio.number}: ${audio.seconds}s`, state.audios, audio.number);
        if (chips.children.length) views.append(chips);

        if (active.views.length) {
            const all = el("button", "Select all views");
            const none = el("button", "Clear");
            all.onclick = () => { active.views.forEach(v => state.views.add(v.index)); renderViews(); renderCount(); };
            none.onclick = () => { state.views.clear(); state.clips.clear(); state.audios.clear(); renderViews(); renderCount(); };
            const row = el("div", undefined, "chips");
            row.append(all, none);
            views.append(row);
        }
    }

    function renderFiles() {
        const term = search.value.trim().toLowerCase();
        files.replaceChildren();
        for (const entry of entries) {
            if (term && !`${entry.name} ${entry.description}`.toLowerCase().includes(term)) continue;
            const button = el("button", undefined, "file");
            button.classList.toggle("on", active && entry.name === active.name);
            const picks = picked.get(entry.name);
            const chosen = picks ? picks.views.size + picks.clips.size + picks.audios.size : 0;
            button.append(el("span", entry.name));
            button.append(el("small", `${entry.views.length} views`
                + (entry.clips.length ? `, ${entry.clips.length} clips` : "")
                + (entry.audios.length ? `, ${entry.audios.length} audio` : "")
                + (chosen ? `  •  ${chosen} picked` : "")));
            button.onclick = () => { active = entry; renderFiles(); renderViews(); renderCount(); };
            files.append(button);
        }
    }

    search.oninput = renderFiles;
    dialog.addEventListener("close", () => dialog.remove());  // Escape closes it too
    const finish = () => dialog.close();
    close.onclick = finish;
    cancel.onclick = finish;
    use.onclick = () => {
        if (!active) return finish();
        const state = stateOf(active.name);
        const refcap = widgetOf(node, "refcap");
        if (refcap) {
            if (!refcap.options.values.includes(active.name)) refcap.options.values = [...refcap.options.values, active.name];
            refcap.value = active.name;
        }
        const set = (name, value) => { const w = widgetOf(node, name); if (w) w.value = value; };
        set("views", formatList(state.views, active.views.map(v => v.index)));
        set("clips", formatList(state.clips, active.clips.map(c => c.number)));
        set("audios", formatList(state.audios, active.audios.map(a => a.number)));
        node.setDirtyCanvas?.(true, true);
        finish();
    };

    renderFiles();
    renderViews();
    renderCount();
    dialog.showModal();
}

app.registerExtension({
    name: "RefCap.Picker",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "H3RefCapSelect") return;
        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function (...args) {
            const result = created?.apply(this, args);
            this.addWidget("button", "Pick views...", null, async () => {
                try {
                    const response = await api.fetchApi("/refcap/library");
                    const entries = await response.json();
                    if (!entries.length) return alert("No RefCaps in models/refcaps yet.");
                    openPicker(this, entries);
                } catch (error) {
                    alert(`Could not read the RefCap library: ${error}`);
                }
            });
            return result;
        };
    },
});
