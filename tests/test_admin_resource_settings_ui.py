import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = (
    ROOT / "omlx/admin/templates/dashboard/_settings.html"
).read_text()


def test_decode_priority_copy_describes_prefill_behavior():
    translations = json.loads((ROOT / "omlx/admin/i18n/en.json").read_text())

    assert (
        translations["settings.resource.decode_fairness"]
        == "Prioritize Decoding During Prefill"
    )
    assert "Prefill pauses between chunks" in translations[
        "settings.resource.decode_fairness_description"
    ]


def test_decode_priority_toggle_keeps_its_fixed_width():
    binding = "globalSettings.scheduler.decode_fairness ="
    binding_start = SETTINGS.index(binding)
    button_start = SETTINGS.rindex("<button", 0, binding_start)
    button_end = SETTINGS.index("</button>", binding_start)
    button = SETTINGS[button_start:button_end]

    assert "flex-shrink-0" in button


def test_loading_cache_settings_preserves_size_until_slider_edit():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for dashboard behavior tests")
    script = r"""
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const source = fs.readFileSync('omlx/admin/static/js/dashboard.js', 'utf8');
(async () => {
    for (const size of ['auto', '1536MB']) {
        const context = {
            localStorage: {getItem: () => null},
            THEME_STORAGE_KEY: 'theme', ENHANCED_READABILITY_KEY: 'readability',
            window: {t: key => key}, navigator: {language: 'en'}, document: {},
            console,
            fetch: async () => ({ok: true, json: async () => ({
                cache: {ssd_cache_max_size: size, ssd_cache_auto_size_bytes: 150 * 1024 ** 3},
                system: {ssd_total_bytes: 1000 * 1024 ** 3},
            })}),
        };
        const state = vm.runInNewContext(source + '\n dashboard;', context)();
        await state.loadGlobalSettings();
        assert.equal(state.globalSettings.cache.ssd_cache_max_size, size);
        assert.equal(state.loadingGlobalSettings, false);
        state.cachePercent = 20;
        state.updateCacheFromSlider();
        assert.equal(state.globalSettings.cache.ssd_cache_max_size, '200GB');
        state.globalSettings.cache.ssd_cache_max_size = 'auto';
        assert.equal(state.cacheSizeGB, 150);
    }
})().catch(error => { console.error(error); process.exitCode = 1; });
"""
    result = subprocess.run(
        [node, "-e", script], cwd=ROOT, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_settings_section_anchor_does_not_outlive_its_tab():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for dashboard behavior tests")
    script = r"""
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const source = fs.readFileSync('omlx/admin/static/js/dashboard.js', 'utf8');
const location = {};
const setUrl = (href) => {
    const url = new URL(href, 'http://h/admin/dashboard');
    Object.assign(location, {href: url.href, search: url.search, hash: url.hash,
                             origin: url.origin, pathname: url.pathname});
};
const context = {
    localStorage: {getItem: () => null},
    THEME_STORAGE_KEY: 'theme', ENHANCED_READABILITY_KEY: 'readability',
    window: {t: key => key, location, history: {replaceState: (_s, _t, url) => setUrl(String(url))}},
    navigator: {language: 'en'}, document: {}, console, URL, URLSearchParams,
};
const state = vm.runInNewContext(source + '\n dashboard;', context)();
state.globalSettings.server.distributed_inference_active = false;

setUrl('/admin/dashboard#settings-cache');
state.applyTabStateFromUrl();
assert.equal(state.mainTab, 'settings');
assert.equal(state.activeTab, 'global');
assert.equal(state.settingsActiveSection, 'settings-cache');

state.setMainTab('status');
assert.equal(location.hash, '');

setUrl('/admin/dashboard?tab=status#settings-cache');
state.applyTabStateFromUrl();
assert.equal(state.mainTab, 'status');
"""
    result = subprocess.run(
        [node, "-e", script], cwd=ROOT, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_memory_watermark_with_and_without_the_guard():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for dashboard behavior tests")
    script = r"""
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const source = fs.readFileSync('omlx/admin/static/js/dashboard.js', 'utf8');
const context = {localStorage: {getItem: () => null}, window: {t: key => key}, document: {}};
const state = vm.runInNewContext(source + '\n dashboard;', context)();
const GB = 1024 ** 3;
state.globalSettings.system.total_memory_bytes = 128 * GB;

// Guard off: the server sends 0 for every limit.
state.stats.active_models = {
    models: [{id: 'm', estimated_size: 20 * GB}], model_memory_used: 18 * GB, model_memory_max: 0,
    memory_pressure: {enabled: false, current_bytes: 0, soft_bytes: 0, hard_bytes: 0},
};
let wm = state.memoryWatermark;
assert.equal(wm.visible, true);
assert.equal(wm.hard, 0);
assert.equal(wm.hardPercent, 0);
assert.equal(Math.round(wm.actualPercent), 14);

// Guard on: limits come from the enforcer, scaled to the machine.
state.stats.active_models.memory_pressure = {
    enabled: true, current_bytes: 64 * GB, soft_bytes: 80 * GB, hard_bytes: 96 * GB,
};
wm = state.memoryWatermark;
assert.equal(wm.actualPercent, 50);
assert.equal(wm.hardPercent, 75);
assert.equal(wm.softPercent, 62.5);
"""
    result = subprocess.run(
        [node, "-e", script], cwd=ROOT, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr
