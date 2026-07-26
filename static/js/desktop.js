(function () {
    'use strict';

    function bridge() {
        return window.pywebview && window.pywebview.api ? window.pywebview.api : null;
    }

    async function quitApplication(button) {
        if (button) button.disabled = true;
        try {
            const response = await fetch('/api/desktop/quit', {
                method: 'POST',
                headers: {'X-Requested-With': 'MouseScopeDesktop'}
            });
            if (!response.ok) throw new Error('quit request failed');
        } catch (_error) {
            const api = bridge();
            if (api && api.quit_app) await api.quit_app();
            if (button) button.disabled = false;
        }
    }

    async function chooseDirectory(button) {
        const targetId = button.dataset.desktopDirectoryTarget;
        const target = targetId ? document.getElementById(targetId) : null;
        const api = bridge();
        if (!target || !api || !api.choose_directory) return;
        button.disabled = true;
        try {
            const selected = await api.choose_directory(target.value || '');
            if (!selected) return;
            if (target.tagName === 'TEXTAREA' && target.value.trim()) {
                target.value = `${target.value.trim()}\n${selected}`;
            } else {
                target.value = selected;
            }
            target.dispatchEvent(new Event('input', {bubbles: true}));
            target.dispatchEvent(new Event('change', {bubbles: true}));
        } finally {
            button.disabled = false;
        }
    }

    function prepareDownloadLinks(root) {
        const scope = root && root.querySelectorAll ? root : document;
        scope.querySelectorAll('a[href]').forEach((link) => {
            let url;
            try {
                url = new URL(link.href, window.location.href);
            } catch (_error) {
                return;
            }
            if (url.origin === window.location.origin && (
                url.pathname.startsWith('/api/download/')
                || url.pathname.startsWith('/api/download_all/')
                || url.pathname.startsWith('/api/history/download_all/')
            )) {
                link.setAttribute('download', '');
            }
        });
    }

    document.addEventListener('click', async (event) => {
        const quitButton = event.target.closest('[data-desktop-quit]');
        if (quitButton) {
            event.preventDefault();
            await quitApplication(quitButton);
            return;
        }

        const directoryButton = event.target.closest('[data-desktop-directory-target]');
        if (directoryButton) {
            event.preventDefault();
            await chooseDirectory(directoryButton);
            return;
        }

        const link = event.target.closest('a[href]');
        if (!link) return;
        let url;
        try {
            url = new URL(link.href, window.location.href);
        } catch (_error) {
            event.preventDefault();
            return;
        }
        if (!['http:', 'https:'].includes(url.protocol)) {
            event.preventDefault();
            return;
        }
        if (url.origin !== window.location.origin) {
            event.preventDefault();
            const api = bridge();
            if (api && api.open_external) await api.open_external(url.href);
            return;
        }
        if (link.target === '_blank' && !link.hasAttribute('download')) {
            event.preventDefault();
            window.location.assign(url.href);
        }
    });

    const nativeWindowOpen = window.open.bind(window);
    window.open = function (rawUrl, target, features) {
        if (!rawUrl) return nativeWindowOpen(rawUrl, target, features);
        let url;
        try {
            url = new URL(rawUrl, window.location.href);
        } catch (_error) {
            return null;
        }
        if (url.origin === window.location.origin) {
            window.location.assign(url.href);
        } else if (['http:', 'https:'].includes(url.protocol)) {
            const api = bridge();
            if (api && api.open_external) api.open_external(url.href);
        }
        return null;
    };

    document.addEventListener('DOMContentLoaded', () => {
        prepareDownloadLinks(document);
        const observer = new MutationObserver((mutations) => {
            mutations.forEach((mutation) => {
                mutation.addedNodes.forEach((node) => {
                    if (node.nodeType === Node.ELEMENT_NODE) prepareDownloadLinks(node);
                });
            });
        });
        observer.observe(document.body, {childList: true, subtree: true});
    });

    window.MouseScopeDesktop = {quit: quitApplication, chooseDirectory: chooseDirectory};
})();
