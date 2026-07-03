// Fits the nominations Cytoscape viewport to show every node on load/filter/reset.
// The "cola" layout runs with fit=False so its continuous simulation doesn't fight
// the user while dragging nodes; this does a one-off cy.fit() instead, reaching the
// underlying cytoscape instance via react-cytoscapejs' internal `_cyreg` (there is no
// public dash-cytoscape prop for triggering a fit).
window.dash_clientside = Object.assign({}, window.dash_clientside, {
    network: {
        fitCytoscape: function (elements, layout, resetClicks) {
            var el = document.getElementById("nom-cyto");
            if (el && el._cyreg && el._cyreg.cy) {
                var cy = el._cyreg.cy;
                // Let the cola simulation spread out for a moment before fitting,
                // so the bounding box reflects the settled layout, not the initial cluster.
                setTimeout(function () {
                    cy.fit(undefined, 40);
                }, 400);
            }
            return window.dash_clientside.no_update;
        }
    }
});
