// Fits the nominations Cytoscape viewport to show every node on load/filter/reset.
// The "cola" layout runs with fit=False so the animated simulation doesn't re-center
// the viewport while the user drags nodes; this does a one-off cy.fit() instead,
// reaching the underlying cytoscape instance via react-cytoscapejs' internal `_cyreg`
// (there is no public dash-cytoscape prop for triggering a fit — this private handle
// may break on a dash-cytoscape/react-cytoscapejs upgrade).
window.dash_clientside = Object.assign({}, window.dash_clientside, {
    network: {
        fitCytoscape: function (elements, layout, resetClicks) {
            var el = document.getElementById("nom-cyto");
            if (el && el._cyreg && el._cyreg.cy) {
                var cy = el._cyreg.cy;
                var ctx = window.dash_clientside.callback_context || {};
                var triggered = ctx.triggered || [];
                var isReset = triggered.some(function (t) {
                    return (t.prop_id || "").indexOf("reset-button") !== -1;
                });
                if (isReset) {
                    // Reset doesn't start a new layout — fit right away.
                    cy.fit(undefined, 40);
                } else {
                    // New elements/layout: fit once the (finite) layout settles, so
                    // the bounding box reflects the final positions. Fallback timeout
                    // in case layoutstop already fired before this handler registered
                    // (static layouts, races).
                    var done = false;
                    var fit = function () {
                        if (!done) {
                            done = true;
                            cy.fit(undefined, 40);
                        }
                    };
                    cy.one("layoutstop", fit);
                    setTimeout(fit, 4800);
                }
            }
            return window.dash_clientside.no_update;
        }
    }
});
