"""Real-browser checks for the local inspection workspace. Requires Playwright and Chrome/Edge."""
import base64
import json
import subprocess
import sys
import time
from pathlib import Path

import httpx
from playwright.sync_api import sync_playwright, expect

ROOT = Path(__file__).resolve().parents[1]
URL = 'http://127.0.0.1:8766'

def main():
    log = open(ROOT / 'reports/ui-server.log', 'w', encoding='utf-8')
    server = subprocess.Popen([sys.executable, '-m', 'uvicorn', 'casting_inspector.api:app', '--host', '127.0.0.1', '--port', '8766'], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0)
    checks, errors = [], []
    try:
        with httpx.Client(base_url=URL) as client:
            for _ in range(90):
                try:
                    if client.get('/health/ready').status_code == 200:
                        break
                except httpx.TransportError:
                    pass
                if server.poll() is not None:
                    raise RuntimeError('UI server exited; inspect reports/ui-server.log')
                time.sleep(1)
            else:
                raise TimeoutError('UI server did not become ready')
            assert client.get('/').status_code == 200
            assert client.get('/assets/app.js').status_code == 200
            assert client.get('/ui/info').json()['test_images'] == 715
        with sync_playwright() as p:
            chrome = Path('C:/Program Files/Google/Chrome/Application/chrome.exe')
            edge = Path('C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe')
            executable = chrome if chrome.exists() else edge if edge.exists() else None
            browser = p.chromium.launch(executable_path=str(executable) if executable else None, headless=True, args=['--disable-gpu'])
            page = browser.new_page(viewport={'width':1440, 'height':1050}, device_scale_factor=1)
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.goto(URL, wait_until='networkidle')
            expect(page.locator('#service-text')).to_have_text('Model ready')
            expect(page.locator('#accuracy-value')).to_have_text('98.60%')
            expect(page.locator('#inspect-button')).to_be_disabled()
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
            page.evaluate('document.activeElement.blur(); window.scrollTo(0, 0)')
            page.screenshot(path=str(ROOT / 'reports/figures/ui-desktop.png'), full_page=True)
            checks.append('Desktop layout, actual readiness and model-linked benchmark results')
            page.locator('[data-example="normal_1.jpg"]').click()
            expect(page.locator('#predicted-class')).to_have_text('Normal')
            expect(page.locator('#result-status')).to_have_text('Complete')
            expect(page.locator('#image-preview')).to_be_visible()
            checks.append('One-click normal sample sends real prediction')
            page.locator('[data-example="defective_1.jpg"]').click()
            expect(page.locator('#predicted-class')).to_have_text('Defective')
            expect(page.locator('#verdict-icon')).to_have_class('verdict-icon defective')
            preview = page.locator('#image-preview').bounding_box()
            zone = page.locator('#drop-zone').bounding_box()
            assert preview['height'] <= zone['height'] and preview['width'] <= zone['width']
            page.evaluate('document.activeElement.blur(); window.scrollTo(0, 0)')
            page.screenshot(path=str(ROOT / 'reports/figures/ui-result.png'), full_page=True)
            checks.append('One-click defective sample renders real score and verdict')
            with page.expect_download() as pending:
                page.locator('#download-result').click()
            saved = ROOT / 'reports/ui-download.json'
            pending.value.save_as(str(saved))
            payload = json.loads(saved.read_text())
            assert payload['predicted_class'] == 'Defective' and payload['request_id']
            saved.unlink()
            checks.append('Downloaded JSON contains actual prediction and request ID')
            page.locator('#reset-button').click()
            expect(page.locator('#empty-upload')).to_be_visible()
            expect(page.locator('#inspect-button')).to_be_disabled()
            checks.append('Reset clears preview, verdict and selection')
            page.locator('#file-input').set_input_files(str(ROOT / 'reports/examples/normal_2.jpg'))
            expect(page.locator('#inspect-button')).to_be_enabled()
            page.locator('#inspect-button').click()
            expect(page.locator('#predicted-class')).to_have_text('Normal')
            checks.append('Local file upload and manual inspection')
            page.locator('#file-input').set_input_files({'name':'not-an-image.txt','mimeType':'text/plain','buffer':b'hello'})
            expect(page.locator('#error-message')).to_contain_text('JPG or PNG')
            expect(page.locator('#result-content')).to_be_hidden()
            expect(page.locator('#inspect-button')).to_be_disabled()
            page.locator('#file-input').set_input_files({'name':'corrupt.jpg','mimeType':'image/jpeg','buffer':b'broken image'})
            expect(page.locator('#error-message')).to_contain_text('could not be opened')
            checks.append('Unsupported/corrupt uploads rejected without stale results')
            encoded = base64.b64encode((ROOT / 'reports/examples/defective_2.jpg').read_bytes()).decode()
            page.evaluate('''encoded => {
                const bytes = Uint8Array.from(atob(encoded), c => c.charCodeAt(0));
                const transfer = new DataTransfer();
                transfer.items.add(new File([bytes], 'dropped.jpg', {type:'image/jpeg'}));
                document.getElementById('drop-zone').dispatchEvent(new DragEvent('drop', {bubbles:true, cancelable:true, dataTransfer:transfer}));
            }''', encoded)
            expect(page.locator('#file-name')).to_have_text('dropped.jpg')
            page.locator('#inspect-button').click()
            expect(page.locator('#predicted-class')).to_have_text('Defective')
            checks.append('Drag-and-drop image inspection')
            page.route('**/predict', lambda route: route.fulfill(status=503, content_type='application/json', body=json.dumps({'detail':'Model temporarily unavailable'})))
            page.locator('#inspect-button').click()
            expect(page.locator('#error-message')).to_contain_text('Model temporarily unavailable')
            expect(page.locator('#inspect-button')).to_be_enabled()
            page.unroute('**/predict')
            checks.append('Server error is visible and inspection can be retried')
            page.locator('#help-button').click()
            expect(page.locator('#help-dialog')).to_be_visible()
            page.keyboard.press('Escape')
            expect(page.locator('#help-dialog')).to_be_hidden()
            checks.append('Usage dialog opens and closes using keyboard')
            page.route('**/ui/info', lambda route: route.fulfill(status=503, body='{}', content_type='application/json'))
            page.evaluate('refreshStatus()')
            expect(page.locator('#service-text')).to_have_text('Service offline')
            expect(page.locator('#inspect-button')).to_be_disabled()
            page.unroute('**/ui/info'); page.evaluate('refreshStatus()')
            expect(page.locator('#service-text')).to_have_text('Model ready')
            checks.append('Offline status disables inference; readiness recovery works')
            page.set_viewport_size({'width':390, 'height':844})
            page.locator('[data-example="defective_1.jpg"]').click()
            expect(page.locator('#result-status')).to_have_text('Complete')
            expect(page.locator('#predicted-class')).to_have_text('Defective')
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
            page.evaluate('document.activeElement.blur(); window.scrollTo(0, 0)')
            page.screenshot(path=str(ROOT / 'reports/figures/ui-mobile.png'), full_page=True)
            page.locator('#mobile-help').click()
            expect(page.locator('#help-dialog')).to_be_visible()
            page.locator('#help-done').click()
            page.set_viewport_size({'width':320, 'height':740})
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
            checks.append('390px and 320px layouts; mobile prediction and usage help')
            browser.close()
        assert not errors, errors
        report = {'checks_passed':len(checks),'checks':checks,'javascript_errors':errors,'browser':'Installed Chrome, headless; actual local model HTTP requests', 'screenshots':['reports/figures/ui-desktop.png','reports/figures/ui-result.png','reports/figures/ui-mobile.png']}
        (ROOT / 'reports/ui-verification.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
    finally:
        server.terminate(); server.wait(timeout=20); log.close()

if __name__ == '__main__':
    main()
