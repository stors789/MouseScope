(function (global) {
    'use strict';

    const activeRequests = new Map();

    function sleep(ms) {
        return new Promise(resolve => global.setTimeout(resolve, ms));
    }

    function showRetryMessage(attempt) {
        const message = document.getElementById('messageText');
        if (!message) return;
        const language = typeof global.getLang === 'function'
            ? global.getLang()
            : (global.localStorage?.getItem('mousescope_lang') || 'zh');
        message.textContent = language === 'en'
            ? `Connection interrupted. Retrying automatically (${attempt})...`
            : `连接暂时中断，正在自动重试（${attempt}）...`;
    }

    async function fetchUntilReachable(url) {
        let attempt = 0;
        let delayMs = 300;
        while (true) {
            try {
                const response = await global.fetch(url, {cache: 'no-store'});
                const contentType = String(response.headers.get('content-type') || '').toLowerCase();
                // 4xx is authoritative (notably 404 for an expired session).
                // Network failures, 5xx, and non-JSON success pages keep
                // retrying without replacing the live analysis state.
                if ((response.ok && contentType.includes('json')) || (response.status >= 400 && response.status < 500)) {
                    return response;
                }
            } catch (_) {
                // Preserve the current UI and retry below.
            }
            attempt += 1;
            showRetryMessage(attempt);
            await sleep(delayMs);
            delayMs = Math.min(5000, Math.round(delayMs * 1.7));
        }
    }

    async function fetchStatus(url) {
        let request = activeRequests.get(url);
        if (!request) {
            request = fetchUntilReachable(url).finally(() => activeRequests.delete(url));
            activeRequests.set(url, request);
        }
        // Legacy monitor pages use setInterval. Share a slow/retrying request
        // and clone its body for every consumer to prevent overlapping network
        // responses from arriving out of order.
        const response = await request;
        return response.clone();
    }

    function routeFetch(resource, options) {
        const value = String(resource || '');
        return value.startsWith('/api/status/')
            ? fetchStatus(value)
            : global.fetch(resource, options);
    }

    global.MouseScopeMonitor = Object.freeze({fetchStatus, routeFetch});
})(window);
