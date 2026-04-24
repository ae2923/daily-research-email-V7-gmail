"""
test_gmail.py — Verify Gmail SMTP is working
=============================================
Run this BEFORE the full briefing to confirm email delivery.

Usage:
  # With env vars already set:
  python test_gmail.py

  # Or inline:
  GMAIL_ADDRESS="you@gmail.com" GMAIL_APP_PASSWORD="abcdefghijklmnop" python test_gmail.py
"""

import smtplib
import sys
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from config import GMAIL_ADDRESS, GMAIL_APP_PASSWORD


def main():
    # ── Preflight: check that env vars are set ──
    errors = []
    if not GMAIL_ADDRESS:
        errors.append("GMAIL_ADDRESS is empty — set it as an env var.")
    if not GMAIL_APP_PASSWORD:
        errors.append("GMAIL_APP_PASSWORD is empty — set it as an env var.")
    if errors:
        for e in errors:
            print(f"  FAIL: {e}")
        sys.exit(1)

    print(f"  Gmail address:      {GMAIL_ADDRESS}")
    print(f"  App password:       {'*' * 12}{GMAIL_APP_PASSWORD[-4:]}")

    # ── Build a tiny test email ──
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    html = f"""<html><body>
<h2>Gmail SMTP Test — OK</h2>
<p>This email was sent at <b>{now}</b>.</p>
<p>If you're reading this, your <code>GMAIL_ADDRESS</code> and
<code>GMAIL_APP_PASSWORD</code> are configured correctly and the
daily briefing will be able to send emails.</p>
</body></html>"""

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"[TEST] Gmail SMTP — {now}"
    msg["From"] = GMAIL_ADDRESS
    msg["To"] = GMAIL_ADDRESS
    msg.attach(MIMEText(html, "html"))

    # ── Send it ──
    print("\n  Connecting to smtp.gmail.com:465...", end=" ", flush=True)
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=15) as server:
            print("connected.", flush=True)

            print("  Logging in...", end=" ", flush=True)
            server.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
            print("authenticated.", flush=True)

            print("  Sending test email...", end=" ", flush=True)
            server.sendmail(GMAIL_ADDRESS, GMAIL_ADDRESS, msg.as_string())
            print("sent.", flush=True)

    except smtplib.SMTPAuthenticationError as e:
        print("FAILED.", flush=True)
        print(f"\n  Authentication error: {e}")
        print("\n  Common fixes:")
        print("    1. Make sure 2-Step Verification is ON for your Google account")
        print("    2. Generate an App Password at https://myaccount.google.com/apppasswords")
        print("    3. Use the 16-char App Password, NOT your regular Google password")
        print("    4. Check for typos — no spaces in the password")
        sys.exit(1)

    except smtplib.SMTPException as e:
        print("FAILED.", flush=True)
        print(f"\n  SMTP error: {e}")
        sys.exit(1)

    except TimeoutError:
        print("FAILED.", flush=True)
        print("\n  Connection timed out. Port 465 may be blocked by your network/firewall.")
        sys.exit(1)

    print(f"\n  SUCCESS — check {GMAIL_ADDRESS} for the test email.")


if __name__ == "__main__":
    print("\n=== Gmail SMTP Test ===\n")
    main()
