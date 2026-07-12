#!/usr/bin/env node

import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {createRequire} from 'node:module';

const require = createRequire(import.meta.url);
const baseUrl = process.env.MOUSESCOPE_E2E_URL || 'http://127.0.0.1:8080';
const firstVideo = path.resolve(process.env.MOUSESCOPE_E2E_VIDEO || 'test.mp4');
const secondVideo = path.join(os.tmpdir(), `mousescope-e2e-${process.pid}-second.mp4`);
const outputRoot = path.join(os.tmpdir(), 'mousescope-e2e-batch-results');
const edgePath = process.env.MOUSESCOPE_EDGE_PATH || 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe';
const playwrightPath = process.env.MOUSESCOPE_PLAYWRIGHT_CORE || path.join(os.tmpdir(), 'mousescope-browser-e2e', 'node_modules', 'playwright-core');
const {chromium} = require(playwrightPath);

fs.copyFileSync(firstVideo, secondVideo);
const browser = await chromium.launch({headless: true, executablePath: edgePath});
const context = await browser.newContext({viewport: {width: 1440, height: 1000}});
const page = await context.newPage();
const unexpected = [];
page.on('pageerror', error => unexpected.push(`pageerror ${error.stack || error.message}`));
page.on('response', response => {
    if (response.status() >= 400) unexpected.push(`http ${response.status()} ${response.url()}`);
});

async function dismissTour() {
    await page.waitForTimeout(650);
    const skip = page.locator('[data-tour-action="skip"]');
    if (await skip.isVisible().catch(() => false)) await skip.click();
}

async function configureTstVideoAndReturn() {
    await dismissTour();
    await page.waitForFunction(() => document.querySelector('#calibCanvas')?.width > 500);
    await page.locator('button[onclick="skipCalibration()"]') .click();
    await page.locator('#roi-step').waitFor({state: 'visible'});
    await page.waitForFunction(() => document.querySelector('#roiCanvas')?.width > 500);
    await page.locator('button[onclick="addNewROI()"]') .click();
    const canvas = page.locator('#roiCanvas');
    const box = await canvas.boundingBox();
    assert.ok(box);
    for (const [x, y] of [[0.2, 0.2], [0.75, 0.2], [0.75, 0.75], [0.2, 0.75]]) {
        await canvas.click({position: {x: box.width * x, y: box.height * y}});
    }
    await page.locator('#finishROIBtn').click();
    await page.locator('#nextToDebugBtn').click();
    await page.locator('#debug-step').waitFor({state: 'visible'});
    await page.locator('#primaryActionBtn').click();
    await page.waitForURL(url => url.pathname.startsWith('/analysis-config/'), {timeout: 30000});
    await page.locator('#videoCardList .video-settings').first().waitFor();
}

try {
    await page.goto(`${baseUrl}/license?next=${encodeURIComponent('/upload')}`);
    await page.locator('#agreeCheck').check();
    await page.locator('#licenseAcceptBtn').click();
    await page.waitForURL(url => url.pathname === '/upload');

    await page.locator('#experimentType').selectOption('tst');
    await page.locator('#localPaths').fill(`${firstVideo}\n${secondVideo}`.replaceAll('\\', '/'));
    await page.locator('#validatePathsBtn').click();
    await page.locator('#configSection').waitFor({state: 'visible'});
    await page.locator('#submitBtn').click();
    await page.waitForURL(url => url.pathname.startsWith('/analysis-config/'), {timeout: 30000});
    await dismissTour();
    const sessionId = page.url().split('/').pop();
    assert.equal(await page.locator('#videoCardList article').count(), 2);

    await page.locator('#analysisOutputDir').fill(outputRoot.replaceAll('\\', '/'));
    await page.locator('article[data-video-index="0"] a[data-tour-id="analysis-config-roi"]').click();
    await page.waitForURL(url => url.pathname.includes('/video/0'));
    await configureTstVideoAndReturn();

    await page.locator('article[data-video-index="1"] a[data-tour-id="analysis-config-roi"]').click();
    await page.waitForURL(url => url.pathname.includes('/video/1'));
    await configureTstVideoAndReturn();

    for (const index of [0, 1]) {
        await page.locator(`#video-start-${index}`).fill('0');
        await page.locator(`#video-end-${index}`).fill('0.5');
        await page.locator(`#video-fps-${index}`).fill('15');
    }

    let startRequests = 0;
    page.on('request', request => {
        if (request.method() === 'POST' && request.url() === `${baseUrl}/api/start_analysis/${sessionId}`) startRequests += 1;
    });
    await page.locator('#startBatchBtn').evaluate(button => { button.click(); button.click(); });
    await page.waitForURL(url => url.pathname === `/monitor/${sessionId}`, {timeout: 30000});
    await page.locator('#downloadSection').waitFor({state: 'visible', timeout: 90000});
    const processedFrames = Number(await page.locator('#processedFrames').innerText());
    assert.ok(processedFrames >= 14 && processedFrames <= 18, `expected about 16 total frames, got ${processedFrames}`);
    assert.equal(startRequests, 1);

    const status = await page.evaluate(async sid => (await fetch(`/api/status/${sid}`)).json(), sessionId);
    const snapshotPath = (status.result_files || []).find(value => /run_snapshot\.json$/i.test(value));
    assert.ok(snapshotPath, 'run snapshot is missing from result files');
    const snapshot = JSON.parse(fs.readFileSync(snapshotPath, 'utf8'));
    const configs = snapshot.resolved_config?.per_video_configs || [];
    assert.equal(configs.length, 2);
    for (const config of configs) {
        assert.equal(config.end_sec, 0.5);
        assert.equal(config.fps_override, 15);
    }

    // Remove the temporary task record/session while leaving only results in
    // the system temp directory.
    const deletion = await page.evaluate(async sid => {
        const response = await fetch(`/api/tasks/${sid}`, {method: 'DELETE'});
        return {status: response.status, body: await response.json()};
    }, sessionId);
    assert.equal(deletion.status, 200, JSON.stringify(deletion.body));
    assert.deepEqual(unexpected, []);
    console.log(JSON.stringify({success: true, sessionId, processedFrames, startRequests}, null, 2));
} finally {
    await browser.close();
    fs.rmSync(secondVideo, {force: true});
}
