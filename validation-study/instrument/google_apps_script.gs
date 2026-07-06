/**
 * Rating survey submission endpoint.
 *
 * Receives form-urlencoded POSTs from the rating instrument and appends
 * one row per submission to the active spreadsheet's "Responses" sheet.
 *
 * Why this exists: Netlify Forms' free tier applies Akismet spam
 * filtering that silently discarded rapid per-event POSTs from the
 * survey. Apps Script + Sheet has no spam filter and unlimited free
 * submissions, so we switched the backend.
 *
 * Setup steps (do these once):
 *
 *   1. Open the Google Sheet you want the data to land in.
 *   2. Rename the first tab to "Responses" (exact spelling, case sensitive).
 *   3. Paste the headers row into row 1 of that tab. The exact list
 *      is at the bottom of this file under HEADERS_FOR_ROW_1.
 *   4. Extensions menu in the sheet → Apps Script. Replace the
 *      default code with the contents of this file. Save.
 *   5. Click "Deploy" → "New deployment".
 *      - Type: "Web app"
 *      - Description: "Rating survey endpoint v1"
 *      - Execute as: "Me" (your Google account)
 *      - Who has access: "Anyone"
 *      - Click Deploy. Authorize when Google asks. Accept the
 *        "advanced" / "unsafe" warning if shown — it appears because
 *        the app is unverified, which is normal for personal scripts.
 *   6. Google gives you a "Web app URL" that looks like
 *        https://script.google.com/macros/s/AKfycbX.../exec
 *      Copy it. This is the endpoint the survey will POST to.
 *   7. Paste that URL into the APPS_SCRIPT_URL constant in index.html
 *      and redeploy the HTML to Netlify.
 *
 * On re-deployment of the script (after edits): Deploy → Manage
 * deployments → pencil icon → set "Version" to "New version" → Deploy.
 * Reusing the same web app URL avoids breaking the survey.
 */

function doPost(e) {
  // Serialize concurrent writes so two raters submitting at the same
  // millisecond cannot interleave their row writes.
  const lock = LockService.getScriptLock();
  lock.tryLock(5000);

  try {
    const sheet = SpreadsheetApp.getActiveSpreadsheet().getSheetByName('Responses');
    if (!sheet) {
      throw new Error('Sheet "Responses" not found. Rename the first tab to "Responses".');
    }

    // Read the headers from row 1. We append data in header order so
    // adding new fields to the survey only requires adding a new header
    // column to row 1 of the sheet, no code change.
    const lastCol = sheet.getLastColumn();
    if (lastCol < 1) {
      throw new Error('Row 1 of the Responses tab is empty. Paste the headers row.');
    }
    const headers = sheet.getRange(1, 1, 1, lastCol).getValues()[0];

    // Build a row aligned to the headers. Missing fields become "".
    // Server timestamp goes in the received_at column if present.
    const params = e.parameter || {};
    const nowIso = new Date().toISOString();
    const row = headers.map(h => {
      if (h === 'received_at') return nowIso;
      const v = params[h];
      return (v === undefined || v === null) ? '' : v;
    });

    sheet.appendRow(row);

    return ContentService.createTextOutput(
      JSON.stringify({success: true, received_at: nowIso})
    ).setMimeType(ContentService.MimeType.JSON);

  } catch (err) {
    return ContentService.createTextOutput(
      JSON.stringify({success: false, error: String(err)})
    ).setMimeType(ContentService.MimeType.JSON);

  } finally {
    lock.releaseLock();
  }
}

/**
 * GET handler. Useful as a sanity check: open the web app URL in a
 * browser, you should see this confirmation text. Confirms the
 * deployment is live and reachable.
 */
function doGet(e) {
  return ContentService.createTextOutput(
    'Rating survey collection endpoint is live. POST form-urlencoded data here.'
  );
}

/**
 * HEADERS_FOR_ROW_1
 *
 * Copy the row below (the tab-separated string between the BEGIN and
 * END markers) into row 1 of the Responses tab of your sheet. Order
 * matters; doPost writes columns in this order.
 *
 * --- BEGIN HEADERS ---
 * received_at	event_type	participant_code	submitted_at	dialogue_id	practice_id	r5	r2	expected_r5	expected_r2	sequence_position	rating_time_seconds	n_practice_completed	n_dialogues_completed	total_session_seconds	recovered_events	comments
 * --- END HEADERS ---
 *
 * (Each value above is separated by a single TAB character so it pastes
 *  cleanly into Google Sheets as 17 columns. If your tab key got
 *  swallowed by the formatter, use the comma-separated version:
 *  received_at,event_type,participant_code,submitted_at,dialogue_id,practice_id,r5,r2,expected_r5,expected_r2,sequence_position,rating_time_seconds,n_practice_completed,n_dialogues_completed,total_session_seconds,recovered_events,comments)
 */
