#!/usr/bin/env node

import assert from 'node:assert/strict';
import os from 'node:os';
import path from 'node:path';
import {createRequire} from 'node:module';

const require = createRequire(import.meta.url);
const baseUrl = process.env.MOUSESCOPE_E2E_URL || 'http://127.0.0.1:8080';
const videoPath = path.resolve(process.env.MOUSESCOPE_E2E_VIDEO || 'test.mp4').replaceAll('\\', '/');
const edgePath = process.env.MOUSESCOPE_EDGE_PATH || 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe';
const e2eOutputRoot = path.join(os.tmpdir(), 'mousescope-e2e-results').replaceAll('\\', '/');
const playwrightCandidates = [
    process.env.MOUSESCOPE_PLAYWRIGHT_CORE,
    path.join(os.tmpdir(), 'mousescope-browser-e2e', 'node_modules', 'playwright-core'),
    'playwright-core',
].filter(Boolean);

let chromium = null;
let lastImportError = null;
for (const candidate of playwrightCandidates) {
    try {
        ({chromium} = require(candidate));
        break;
    } catch (error) {
        lastImportError = error;
    }
}
if (!chromium) {
    throw new Error(
        'playwright-core is unavailable. Install it outside the project with: ' +
        'npm install --prefix "%TEMP%\\mousescope-browser-e2e" playwright-core\n' +
        String(lastImportError || '')
    );
}

const browser = await chromium.launch({headless: true, executablePath: edgePath});
const context = await browser.newContext({viewport: {width: 1440, height: 1000}, acceptDownloads: true});
const page = await context.newPage();
const faults = [];
let expectedTransientStatusUrl = null;

page.on('pageerror', error => faults.push(`pageerror ${page.url()} ${error.stack || error.message}`));
page.on('response', response => {
    const expectedTransient = response.status() === 503 && response.url() === expectedTransientStatusUrl;
    if (response.status() >= 400 && !expectedTransient) faults.push(`http ${response.status()} ${response.url()}`);
});

async function dismissTour() {
    await page.waitForTimeout(650);
    const skip = page.locator('[data-tour-action="skip"]');
    if (await skip.isVisible().catch(() => false)) await skip.click();
}

async function acceptLicense(nextPath = '/') {
    await page.goto(`${baseUrl}/license?next=${encodeURIComponent(nextPath)}`);
    await page.locator('#agreeCheck').check();
    await page.locator('#licenseAcceptBtn').click();
    await page.waitForURL(url => url.pathname === nextPath);
}

async function assertStandardPage(pathname) {
    await page.goto(`${baseUrl}${pathname}`);
    await page.locator('nav.navbar').waitFor();
    const geometry = await page.evaluate(() => {
        const nav = document.querySelector('nav.navbar').getBoundingClientRect();
        return {
            navTop: nav.top,
            bodyMargin: getComputedStyle(document.body).margin,
            overflow: document.documentElement.scrollWidth - document.documentElement.clientWidth,
        };
    });
    assert.equal(geometry.navTop, 0, `${pathname}: navbar must touch the top edge`);
    assert.equal(geometry.bodyMargin, '0px', `${pathname}: body margin creates a blank strip`);
    assert.ok(geometry.overflow <= 1, `${pathname}: horizontal overflow ${geometry.overflow}px`);
}

async function configureTstRoi() {
    await dismissTour();
    await page.waitForFunction(() => document.querySelector('#calibCanvas')?.width > 500);
    await page.locator('button[onclick="skipCalibration()"]') .click();
    await page.locator('#roi-step').waitFor({state: 'visible'});
    await page.waitForFunction(() => document.querySelector('#roiCanvas')?.width > 500);
    await page.locator('button[onclick="addNewROI()"]') .click();
    const canvas = page.locator('#roiCanvas');
    const box = await canvas.boundingBox();
    assert.ok(box, 'ROI canvas has no visible bounding box');
    for (const [rx, ry] of [[0.2, 0.2], [0.75, 0.2], [0.75, 0.75], [0.2, 0.75]]) {
        await canvas.click({position: {x: box.width * rx, y: box.height * ry}});
    }
    await page.locator('#finishROIBtn').click();
    assert.equal((await page.locator('#roiCount').innerText()).trim(), '1');
    await page.locator('#nextToDebugBtn').click();
    await page.locator('#debug-step').waitFor({state: 'visible'});
}

try {
    await acceptLicense('/');

    for (const pathname of ['/', '/projects', '/upload', '/tasks', '/visual-style', '/manual', '/license']) {
        await assertStandardPage(pathname);
    }

    await page.goto(`${baseUrl}/manual`);
    await page.locator('.lang-toggle-btn').click();
    assert.equal(await page.locator('html').getAttribute('lang'), 'en');
    const manualLinks = await page.locator('.manual-switch-btn').evaluateAll(links => links.map(link => link.href));
    assert.equal(manualLinks.length, 8, 'all released experiment manuals must be reachable');
    for (const href of manualLinks) {
        await page.goto(href);
        assert.ok((await page.locator('.manual-switch-btn.active').innerText()).trim());
        assert.ok((await page.locator('.manual-hero h2').first().innerText()).trim());
    }

    await page.goto(`${baseUrl}/visual-style`);
    await page.waitForFunction(() => document.querySelector('#styleStatus')?.style.display === 'none');
    const styleGeometry = await page.evaluate(() => ({
        heroHeight: document.querySelector('.settings-page-hero').getBoundingClientRect().height,
        tabTop: document.querySelector('.settings-tabs .nav-link').getBoundingClientRect().top,
    }));
    let delayedDefaultsLoad = false;
    await page.route('**/api/experiment_defaults', async route => {
        if (!delayedDefaultsLoad && route.request().method() === 'GET') {
            delayedDefaultsLoad = true;
            await new Promise(resolve => setTimeout(resolve, 500));
        }
        await route.fallback();
    });
    await page.locator('.settings-tabs a[href$="/experiment-defaults"]').click();
    await page.waitForURL(url => url.pathname === '/experiment-defaults');
    assert.equal(await page.locator('#saveBtn').isDisabled(), true, 'defaults cannot be saved before GET completes');
    await page.locator('#moduleRail .module-btn').first().waitFor();
    assert.equal(await page.locator('#saveBtn').isEnabled(), true, 'defaults save is enabled after loading');
    await page.unroute('**/api/experiment_defaults');
    const defaultsGeometry = await page.evaluate(() => ({
        heroHeight: document.querySelector('.settings-page-hero').getBoundingClientRect().height,
        tabTop: document.querySelector('.settings-tabs .nav-link').getBoundingClientRect().top,
    }));
    assert.ok(Math.abs(styleGeometry.heroHeight - defaultsGeometry.heroHeight) < 1);
    assert.ok(Math.abs(styleGeometry.tabTop - defaultsGeometry.tabTop) < 1);

    await page.goto(`${baseUrl}/upload`);
    await page.locator('#experimentType').selectOption('tst');
    await page.locator('#localPaths').fill(videoPath);
    await page.locator('#validatePathsBtn').click();
    await page.locator('#configSection').waitFor({state: 'visible'});
    await page.locator('#submitBtn').click();
    await page.waitForURL(url => url.pathname.startsWith('/analysis-config/'), {timeout: 30000});
    const sessionId = page.url().split('/').pop();

    await page.locator('#analysisOutputDir').fill(e2eOutputRoot);
    await page.locator('#video-start-0').fill('0');
    await page.locator('#video-end-0').fill('1');
    await page.locator('#video-fps-0').fill('15');
    await page.locator('.lang-toggle-btn').click();
    assert.equal(await page.locator('#video-end-0').inputValue(), '1', 'language switch must preserve an unsaved end time');
    assert.equal(await page.locator('#video-fps-0').inputValue(), '15', 'language switch must preserve an unsaved FPS');
    await page.locator('.lang-toggle-btn').click();
    assert.equal(await page.locator('#video-end-0').inputValue(), '1');
    await page.locator('button[onclick^="saveVideoSettings"]').click();
    await page.waitForTimeout(300);
    await page.reload();
    await page.waitForFunction(() => document.getElementById('video-end-0')?.value === '1');
    const reloadedVideoSettings = await page.evaluate(() => ({
        endInput: document.getElementById('video-end-0')?.value,
        fpsInput: document.getElementById('video-fps-0')?.value,
        stateSettings: latestBatchState.cards?.[0]?.settings,
        html: document.getElementById('video-end-0')?.outerHTML,
    }));
    assert.equal(reloadedVideoSettings.endInput, '1', JSON.stringify(reloadedVideoSettings));

    await page.locator('a[data-tour-id="analysis-config-roi"]').click();
    await page.waitForURL(url => url.pathname.startsWith('/calibrate/'), {timeout: 30000});
    assert.equal((await page.locator('h1').innerText()).trim(), 'Tail Suspension Test Setup');
    await configureTstRoi();
    let startRequestCount = 0;
    page.on('request', request => {
        if (request.method() === 'POST' && request.url() === `${baseUrl}/api/start_analysis/${sessionId}`) {
            startRequestCount += 1;
        }
    });
    let transientInjected = false;
    expectedTransientStatusUrl = `${baseUrl}/api/status/${sessionId}`;
    await page.route(`**/api/status/${sessionId}`, async route => {
        if (!transientInjected) {
            transientInjected = true;
            await route.fulfill({status: 503, contentType: 'application/json', body: JSON.stringify({error: 'temporary'})});
            return;
        }
        await route.fallback();
    });
    await page.locator('#primaryActionBtn').evaluate(button => { button.click(); button.click(); });
    await page.waitForURL(url => url.pathname.startsWith('/monitor/'), {timeout: 30000});
    await dismissTour();
    await page.locator('#downloadSection').waitFor({state: 'visible', timeout: 90000});
    const processedFrames = Number(await page.locator('#processedFrames').innerText());
    assert.ok(processedFrames >= 14 && processedFrames <= 16, `expected about 15 frames, got ${processedFrames}`);
    assert.equal(startRequestCount, 1, 'double click must issue exactly one start request');
    assert.equal(transientInjected, true, 'monitor transient failure was not injected');
    await page.unroute(`**/api/status/${sessionId}`);
    expectedTransientStatusUrl = null;
    assert.match(await page.locator('#downloadAllBtn').getAttribute('href'), new RegExp(`/api/download_all/${sessionId}$`));

    // Exercise task search/delete and avoid leaving the smoke run in the
    // user's task center. Result files remain under the system temp folder.
    await page.goto(`${baseUrl}/tasks`);
    await dismissTour();
    const filteredTasks = page.waitForResponse(response => {
        if (!response.url().startsWith(`${baseUrl}/api/tasks?`)) return false;
        return new URL(response.url()).searchParams.get('q') === sessionId;
    });
    await page.locator('#taskSearch').fill(sessionId);
    await filteredTasks;
    const taskCard = page.locator('.task-card');
    await taskCard.waitFor();
    assert.equal(await taskCard.count(), 1, 'the completed smoke task should be searchable');
    page.once('dialog', dialog => dialog.accept());
    const deletedTask = page.waitForResponse(response => (
        response.request().method() === 'DELETE' && response.url().includes(sessionId)
    ));
    await taskCard.locator('[data-delete-url]').click();
    await deletedTask;
    await taskCard.waitFor({state: 'detached'});

    assert.deepEqual(faults, [], `browser faults:\n${faults.join('\n')}`);
    console.log(JSON.stringify({success: true, sessionId, processedFrames, checkedPages: 7, checkedManuals: 8}, null, 2));
} finally {
    await browser.close();
}
