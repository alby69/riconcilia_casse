export function initPWA() {
    if ('serviceWorker' in navigator) {
        window.addEventListener('load', () => {
            navigator.serviceWorker.register('./sw.js').then((reg) => {
                console.log('PWA ServiceWorker registered with scope:', reg.scope);
            }).catch((err) => {
                console.log('PWA ServiceWorker registration skipped/failed:', err);
            });
        });
    }
}
