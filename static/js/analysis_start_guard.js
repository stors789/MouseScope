(function (global) {
    'use strict';

    function loadingIsVisible(loading) {
        return Boolean(loading && global.getComputedStyle(loading).display !== 'none');
    }

    document.addEventListener('click', event => {
        const button = event.target.closest && event.target.closest('#primaryActionBtn');
        if (!button) return;
        if (button.dataset.analysisStartPending === '1') {
            event.preventDefault();
            event.stopImmediatePropagation();
            return;
        }

        // Mark synchronously in the capture phase so a second click cannot
        // enter another calibration handler. Delay the disabled property until
        // the inline handler has begun, avoiding browser differences around
        // disabling the event target during dispatch.
        button.dataset.analysisStartPending = '1';
        global.queueMicrotask(() => { button.disabled = true; });
        const loading = document.getElementById('loading');
        let observer = null;

        const releaseIfNotStarting = () => {
            if (loadingIsVisible(loading)) return;
            button.disabled = false;
            delete button.dataset.analysisStartPending;
            if (observer) observer.disconnect();
            observer = null;
        };

        if (loading) {
            observer = new MutationObserver(releaseIfNotStarting);
            observer.observe(loading, {attributes: true, attributeFilter: ['style', 'class']});
        }
        // Validation failures return synchronously without showing the loading
        // layer; allow the user to correct the form and try again.
        global.setTimeout(releaseIfNotStarting, 0);
    }, true);
})(window);
