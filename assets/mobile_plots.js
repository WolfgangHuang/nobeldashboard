// Mobile layout tweaks for all Plotly graphs (< 48em, same breakpoint as redesign.css).
// Plotly legends and colorbars don't scale: at phone width a legend or colorbar on the
// right eats much of the plot. On narrow viewports this
//   - moves outside-the-plot legends to a horizontal row above the plot; legends with
//     too many entries to be useful on a phone are hidden (hover/tap still shows the
//     values). Legends deliberately placed inside the plot area (prize money,
//     nominations map) are left alone;
//   - turns colorbars (coloraxis and per-trace) horizontal, above the plot;
//   - shrinks fonts/margins of parallel-categories plots so three columns fit;
//   - caps the height of orthographic globes at their width (no empty band around them);
// and grows the figure height by whatever moved on top, so the plot area keeps its size.
//
// Runs purely client-side after every draw (plotly_afterplot), so it survives filter
// callbacks and theme toggles (both re-render via Plotly.react with a fresh layout
// object) without any per-figure Python changes. Widening the window restores the
// original figure.
(function () {
    var MQ = window.matchMedia("(max-width: 47.99em)");
    var MAX_LEGEND_ENTRIES = 12;
    var LEGEND = {
        "legend.orientation": "h",
        "legend.x": 0,
        "legend.xanchor": "left",
        "legend.y": 1.02,
        "legend.yanchor": "bottom",
        "legend.font.size": 10,
        "legend.title.side": "top",
    };
    // Relative to a colorbar root ("coloraxis.colorbar" or a trace's "colorbar").
    // Flipping orientation on a drawn colorbar leaves its old (vertical) auto-margin
    // entry behind, which then squeezes the plot area (globe: ~60 px). So colorbars are
    // hidden first and re-shown with the new settings (see the `hide` step in plan()).
    var COLORBAR = {
        orientation: "h",
        x: 0.5,
        xanchor: "center",
        y: 1.02,
        yanchor: "bottom",
        len: 1,
        lenmode: "fraction",
        thickness: 10,
        "title.side": "top",
        "tickfont.size": 10,
    };
    var PARCATS_TRACE = { "tickfont.size": 9, "labelfont.size": 11 };
    var PARCATS_LAYOUT = { "margin.l": 70, "margin.r": 70 };

    // Layout objects already adapted (prevents re-entry from our own relayout's afterplot).
    var done = new WeakSet();
    // Original values + base height are kept per graph element (gd._nblMobile),
    // snapshotted from the first, unmodified figure. dcc.Graph partially writes relayout
    // changes (at least `height`) back into its figure prop, so a figure returning
    // through a State-based callback (retheme_all_plots) is no longer pristine —
    // computing the height from the element's base keeps re-applying idempotent.

    function get(obj, path) {
        var parts = path.split(".");
        for (var i = 0; i < parts.length; i++) {
            if (obj === undefined || obj === null) return null;
            obj = obj[parts[i]];
        }
        return obj === undefined ? null : obj;
    }

    function prefixed(prefix, attrs) {
        var out = {};
        Object.keys(attrs).forEach(function (k) { out[prefix + "." + k] = attrs[k]; });
        return out;
    }

    function legendEntryCount(gd) {
        var groups = {};
        var n = 0;
        (gd._fullData || []).forEach(function (t) {
            if (t.type === "pie" || t.type === "funnelarea") {
                var labels = {};
                (t.labels || []).forEach(function (l) { labels[l] = 1; });
                n += Object.keys(labels).length;
            } else if (t.showlegend) {
                if (t.legendgroup) {
                    if (!groups[t.legendgroup]) { groups[t.legendgroup] = 1; n += 1; }
                } else {
                    n += 1;
                }
            }
        });
        return n;
    }

    function legendOutsidePlot(full) {
        var lg = full.legend;
        if (!full.showlegend || !lg) return false;
        return lg.x > 1 || lg.x < 0 || lg.y > 1 || lg.y < 0;
    }

    // What to change for this figure: layout/traces updates ({index: {...}}), plus a
    // `hide` step (colorbars off) that runs before them — and before restoring.
    function plan(gd) {
        var full = gd._fullLayout;
        var p = { layout: {}, traces: {}, hide: { layout: {}, traces: {} }, globe: false };

        if (legendOutsidePlot(full)) {
            if (legendEntryCount(gd) > MAX_LEGEND_ENTRIES) {
                p.layout.showlegend = false;
            } else {
                Object.assign(p.layout, LEGEND);
            }
        }
        Object.keys(full).forEach(function (k) {
            if (/^coloraxis\d*$/.test(k) && full[k].showscale) {
                Object.assign(p.layout, prefixed(k + ".colorbar", COLORBAR));
                p.layout[k + ".showscale"] = true;
                p.hide.layout[k + ".showscale"] = false;
            }
        });
        (gd._fullData || []).forEach(function (t) {
            var upd = {};
            if (t.showscale && t.colorbar) {
                Object.assign(upd, prefixed("colorbar", COLORBAR));
                upd.showscale = true;
                p.hide.traces[t.index] = { showscale: false };
            }
            if (t.type === "parcats") {
                Object.assign(upd, PARCATS_TRACE);
                Object.assign(p.layout, PARCATS_LAYOUT);
            }
            if (Object.keys(upd).length) p.traces[t.index] = upd;
        });
        p.globe = Object.keys(full).some(function (k) {
            return /^geo\d*$/.test(k) && full[k].projection && full[k].projection.type === "orthographic";
        });
        p.empty = !Object.keys(p.layout).length && !Object.keys(p.traces).length && !p.globe;
        return p;
    }

    // Returns true if it started a relayout (whose afterplot will call update again).
    function apply(gd) {
        var layout = gd.layout;
        var full = gd._fullLayout;
        if (!layout || !full || done.has(layout)) return false;
        done.add(layout);

        var p = plan(gd);
        if (p.empty) return false;
        var st = gd._nblMobile;
        if (!st) {
            st = gd._nblMobile = {
                baseHeight: full.height,
                baseTop: full._size.t,
                layoutOrig: { height: get(layout, "height") },
                traceOrig: {},
            };
        }
        // Remember originals for every attribute we touch (first time we see it).
        Object.keys(p.layout).forEach(function (k) {
            if (!(k in st.layoutOrig)) st.layoutOrig[k] = get(layout, k);
        });
        Object.keys(p.traces).forEach(function (i) {
            var orig = st.traceOrig[i] || (st.traceOrig[i] = {});
            Object.keys(p.traces[i]).forEach(function (k) {
                if (!(k in orig)) orig[k] = get(gd.data[i], k);
            });
        });
        st.globe = p.globe;
        st.hide = p.hide;

        // Our own restyle/relayout redraws fire afterplot mid-way; don't fit the height
        // against a half-applied figure.
        gd._nblMobileBusy = true;
        run(gd, p.hide.layout, p.hide.traces).then(function () {
            return run(gd, p.layout, p.traces);
        }).then(function () {
            gd._nblMobileBusy = false;
            fitHeight(gd);
        });
        return true;
    }

    function run(gd, layoutUpd, traceUpds) {
        var steps = Object.keys(traceUpds).filter(function (i) {
            return gd.data[i];
        }).map(function (i) {
            return Plotly.restyle(gd, traceUpds[i], [Number(i)]);
        });
        if (Object.keys(layoutUpd).length) steps.push(Plotly.relayout(gd, layoutUpd));
        return Promise.all(steps);
    }

    // Height = base (globes: at most as tall as wide) + the extra top margin Plotly
    // reserves for the legend/colorbar now sitting above the plot. Checked after every
    // draw: they reflow whenever the graph width changes (rotation, resize). (Measuring
    // the drawn elements instead is unreliable — a colorbar's bbox scales with height.)
    function fitHeight(gd) {
        var st = gd._nblMobile;
        var full = gd._fullLayout;
        if (!st || !full || !full._size) return;
        var base = st.baseHeight;
        if (st.globe) base = Math.min(base, full.width);
        var target = Math.round(base + Math.max(0, full._size.t - st.baseTop));
        if (Math.abs(target - full.height) > 2) {
            Plotly.relayout(gd, { height: target });
        }
    }

    function restore(gd) {
        var st = gd._nblMobile;
        if (!st || !gd.layout) return;
        delete gd._nblMobile;
        done.add(gd.layout);
        var layoutOrig = Object.assign({}, st.layoutOrig);
        // Explicit base height: the figure may come back from Dash carrying our mobile
        // height, and autosize would then just keep the current (taller) size.
        if (layoutOrig.height === null) layoutOrig.height = st.baseHeight;
        gd._nblMobileBusy = true;
        run(gd, st.hide.layout, st.hide.traces).then(function () {
            return run(gd, layoutOrig, st.traceOrig);
        }).then(function () {
            gd._nblMobileBusy = false;
        });
    }

    function update(gd) {
        if (gd._nblMobileBusy) return;
        if (MQ.matches) {
            if (!apply(gd)) fitHeight(gd);
        } else {
            restore(gd);
        }
    }

    function hook(gd) {
        if (gd._nblMobileHooked || typeof gd.on !== "function") return;
        gd._nblMobileHooked = true;
        gd.on("plotly_afterplot", function () { update(gd); });
        update(gd);
    }

    function hookAll() {
        document.querySelectorAll(".js-plotly-plot").forEach(hook);
    }

    // Graphs are mounted (and re-mounted on page navigation) by Dash at any time.
    // Batched per frame — hover labels etc. mutate the DOM constantly.
    var pending = false;
    new MutationObserver(function () {
        if (pending) return;
        pending = true;
        requestAnimationFrame(function () { pending = false; hookAll(); });
    }).observe(document.documentElement, {
        childList: true,
        subtree: true,
    });
    // The breakpoint fires mid-resize, before dcc.Graph has resized its plots; relayouting
    // then squeezes graphs to a stale width. Let the resize settle, then resize + adapt.
    var mqTimer = null;
    MQ.addEventListener("change", function () {
        clearTimeout(mqTimer);
        mqTimer = setTimeout(function () {
            document.querySelectorAll(".js-plotly-plot").forEach(function (gd) {
                Promise.resolve(Plotly.Plots.resize(gd)).then(function () { update(gd); });
            });
        }, 300);
    });
    hookAll();
})();
