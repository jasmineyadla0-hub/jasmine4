"""
PhishGuard SVM - Training Script
=================================
Trains a LinearSVC phishing detector using TF-IDF features.
Run: python train_model.py
Output: phishing_model.pkl
"""

import pickle
import json
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import numpy as np

# ─── DATASET ──────────────────────────────────────────────────────────────────
# Expand this dataset with real emails for production use.
# Recommended datasets: SpamAssassin, Enron, CEAS 2008 phishing corpus

PHISHING_EMAILS = [
    "URGENT: Your account has been suspended. Verify your information immediately at http://paypal-secure.xyz/login",
    "Dear customer your bank account will be closed unless you click here to verify now",
    "Congratulations! You've won $1,000,000. Claim your prize now by providing your SSN and bank account",
    "Your password expires in 24 hours. Click the link to reset: http://microsoft-login.info/reset",
    "ALERT: Unusual sign-in activity detected. Confirm your identity immediately or your account will be locked",
    "Your PayPal account is limited. Please verify your credit card details to restore access",
    "Act now! Your Amazon order cannot be shipped until you update your payment information",
    "IRS Notice: You have unpaid taxes. Provide your SSN to avoid immediate arrest",
    "Your Apple ID has been compromised. Enter your password at apple-support-id.com",
    "Final warning: Click here within 24 hours or lose access to your account permanently",
    "Dear valued user your account shows suspicious login attempts verify now or be locked out",
    "You have a pending delivery. Pay $3.50 customs fee at this link to release your package",
    "Netflix billing issue detected. Update your credit card to avoid service interruption",
    "SECURITY ALERT: Someone tried to access your account. Confirm it was you by clicking here",
    "Your social security number has been suspended due to suspicious activity. Call immediately",
    "Congratulations winner selected for free iPhone. Provide details to claim reward today",
    "Bank account verification required within 48 hours or it will be permanently suspended",
    "Your email storage is full. Click to verify account or it will be automatically deleted",
    "CEO confidential: Process urgent wire transfer of $50,000 to new vendor immediately today",
    "HR department: Update your direct deposit information by end of day using the attached form",
    "Click to verify your email address or your account will be suspended in 24 hours urgent",
    "Your login credentials were exposed in a breach. Reset password at secure-google.net now",
    "Limited time offer: guaranteed 300% investment return no risk click to invest today",
    "DHL delivery failed. Click to reschedule and pay small fee or package returned to sender",
    "We detected unauthorized access verify identity now at http://banking-secure-portal.com",
    "Your subscription will auto-renew for $299 call now to cancel and get refund immediately",
    "Verify your account now to avoid losing all your data and stored information click here",
    "You have been selected for exclusive crypto investment opportunity guaranteed profits daily",
    "Please update KYC information immediately to avoid account suspension banking portal",
    "Your computer has virus click here to download free antivirus and remove immediately",
    "Dear user we noticed irregular activity on your account please confirm details to continue",
    "Your account access will expire respond immediately with username and password to renew",
    "Exclusive offer investment with guaranteed 200 percent returns act now limited slots",
    "Warning your email will be deactivated update account settings by clicking the link now",
    "Your debit card was used in a suspicious transaction verify your card details immediately",
    "Your account has been flagged for illegal activity. Verify immediately to avoid prosecution.",
    "Final notice: Your PayPal account will be permanently closed unless you confirm your details today.",
    "Microsoft security update required. Download patch now or risk losing all files.",
    "Unclaimed package waiting. Pay customs fee online to release shipment.",
    "Your credit card has been suspended. Enter details to restore access.",
    "Urgent payroll update required. Submit SSN and bank details to HR portal.",
    "Suspicious login detected from Russia. Confirm identity now to secure account.",
    "Your tax refund is ready. Provide bank details to receive funds instantly.",
    "Apple security alert: Your account will be locked unless you reset password here.",
    "Exclusive lottery winner! Claim $500,000 by filling out attached form.",
    "Your insurance policy has expired. Renew immediately by entering payment details.",
    "Bank alert: Unauthorized wire transfer attempt. Verify account to stop transaction.",
    "Your Gmail account storage exceeded. Click here to upgrade or lose emails.",
    "Crypto investment opportunity: Double your money in 24 hours. Sign up now.",
    "Your Gmail account will be deleted in 12 hours unless you verify immediately.",
    "Suspicious login detected from China. Confirm identity now or account locked.",
    "Your Outlook mailbox is full. Click here to upgrade storage.",
    "Password reset required: visit secure-login-update.net  to continue.",
    "Your Apple ID has been disabled. Restore access by entering credentials.",
    "Unusual activity detected on your debit card. Verify details urgently.",
    "Your account will be suspended unless you provide updated KYC information.",
    "Final warning: Your banking portal access will expire today.",
    "Security breach detected. Reset your password at secure-google-login.com.",
    "Your PayPal account is frozen. Enter credit card details to unlock.",
    "IRS refund available. Provide SSN and bank account to claim.",
    "Unpaid taxes detected. Pay immediately to avoid arrest.",
    "Your subscription auto-renewed for $399. Call now for refund.",
    "Pending wire transfer requires confirmation. Enter account details.",
    "Your insurance policy expired. Renew now with payment details.",
    "Credit card declined. Update payment info to continue service.",
    "Your loan application approved. Send processing fee to finalize.",
    "Bank alert: Unauthorized transaction. Verify account to stop.",
    "Your mortgage payment overdue. Pay online to avoid foreclosure.",
    "Your Amazon order cannot ship until payment is updated.",

]

LEGIT_EMAILS = [
    "Hi John, just wanted to follow up on the meeting notes from yesterday. Let me know if you have questions.",
    "Your GitHub pull request has been approved and merged into the main branch.",
    "Meeting reminder: Team standup tomorrow at 9 AM. Please confirm your attendance.",
    "Your Amazon order #123-456 has shipped and will arrive by Thursday.",
    "Thanks for your purchase! Your receipt is attached for your records.",
    "The quarterly report is ready for review. Please find it attached to this email.",
    "Happy birthday! Hope you have a wonderful day filled with joy.",
    "Your flight confirmation for March 15th from New York to London is attached.",
    "Newsletter: Here are this month's top articles on technology and innovation.",
    "Hi, I wanted to share this interesting article about machine learning with you.",
    "Your subscription to Netflix renews on March 1st for $15.99. Manage at netflix.com/account",
    "Project update: We've completed phase 1 and are on track for the Q2 deadline.",
    "Your doctor appointment is confirmed for Tuesday at 2:30 PM at the downtown clinic.",
    "Thanks for joining our webinar! Here's the recording link for future reference.",
    "Your tax documents are ready to download from your account at irs.gov",
    "Weekly digest: Here are the top stories from your subscriptions this week.",
    "Reminder: Your library books are due back on Friday. Renew at the library website.",
    "Your job application has been received. We will review and contact you within 5 business days.",
    "Hello, I am writing to confirm our lunch meeting for next Wednesday at noon downtown.",
    "The conference call has been rescheduled to 3 PM. Updated invite attached.",
    "Your GitHub security alert: a dependency in your repo has a known vulnerability please update",
    "Attached please find the invoice for services rendered in January as discussed.",
    "We appreciate your feedback on our service. Your response helps us improve for all customers.",
    "Hi team, please review and approve the budget proposal before Friday end of day.",
    "Your new password has been set. If you did not request this change please contact support.",
    "Congratulations on completing the course! Your certificate is ready to download.",
    "Your insurance policy renewal is coming up. No action needed if you wish to continue.",
    "Please review and sign the attached contract at your earliest convenience thank you.",
    "The system maintenance is scheduled for this Saturday from 2 to 4 AM downtime expected.",
    "Just checking in on the project status. Let me know if you need any help or resources.",
    "Hi Sarah the report you requested is now available on the shared drive for review",
    "Thank you for attending the conference your feedback form is available online",
    "Following up on our last conversation here are the next steps we discussed",
    "Your package has been delivered successfully please rate your experience with us",
    "The board meeting minutes are now available in the shared folder for all members",
    "Your LinkedIn connection request has been accepted. View their profile now."
    "Reminder: Company holiday party this Friday at 6 PM. RSVP by Wednesday.",
    "Your Uber receipt for yesterday’s ride is attached.",
    "Weekly team update: Progress on sprint goals and blockers.",
    "Your Amazon return has been processed. Refund will appear in 3–5 business days.",
    "Conference registration confirmed. Badge pickup instructions attached.",
    "Your Spotify subscription renewed successfully. Next billing date is April 1st.",
    "Hi team, please find attached the updated project timeline.",
    "Your hotel booking for April 10th in Paris is confirmed.",
    "Friendly reminder: Submit expense reports by end of month.",
    "Your Dropbox files have been shared successfully with the team.",
    "Happy holidays! Wishing you joy and peace this season.",
    "Your Zoom meeting recording is now available for download.",
    "The system upgrade was completed successfully. Services are back online."
]

# ─── FEATURE ENGINEERING ──────────────────────────────────────────────────────
def extract_meta_features(text):
    """Extract hand-crafted phishing indicators as additional features."""
    import re
    features = []
    t = text.lower()
    
    # URL patterns
    urls = re.findall(r'http[s]?://\S+', text)
    features.append(len(urls))                                    # url count
    features.append(1 if any('.xyz' in u or '.info' in u or '.tk' in u for u in urls) else 0)  # suspicious TLD
    features.append(1 if any(re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}', u) for u in urls) else 0)  # IP in URL
    
    # Urgency signals
    urgency = ['urgent', 'immediately', 'act now', 'limited time', '24 hours', 'expire', 'suspend', 'final warning']
    features.append(sum(1 for w in urgency if w in t))
    
    # Credential harvesting
    creds = ['password', 'ssn', 'social security', 'credit card', 'bank account', 'verify your']
    features.append(sum(1 for w in creds if w in t))
    
    # Threat language
    threats = ['locked', 'suspended', 'terminated', 'arrest', 'legal action', 'permanently']
    features.append(sum(1 for w in threats if w in t))
    
    # Money/reward
    money = ['won', 'winner', 'prize', 'reward', 'free', 'guaranteed', 'investment return']
    features.append(sum(1 for w in money if w in t))
    
    return features


# ─── TRAINING ─────────────────────────────────────────────────────────────────
def train():
    texts = PHISHING_EMAILS + LEGIT_EMAILS
    labels = [1] * len(PHISHING_EMAILS) + [0] * len(LEGIT_EMAILS)

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=8000,
            sublinear_tf=True,        # log normalization
            stop_words='english',
            analyzer='word',
            min_df=1
        )),
        ('svm', LinearSVC(
            C=1.0,                    # regularization — tune this
            class_weight='balanced',  # handles class imbalance
            max_iter=3000
        ))
    ])

    # Cross-validation
    cv_scores = cross_val_score(pipeline, texts, labels, cv=5, scoring='accuracy')
    print(f"Cross-validation accuracy: {cv_scores.mean()*100:.1f}% ± {cv_scores.std()*100:.1f}%")

    # Final fit
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    print(f"\nTest Accuracy: {accuracy_score(y_test, y_pred)*100:.1f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Phishing']))

    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}  TP={cm[1,1]}")

    # Top discriminative features
    tfidf = pipeline.named_steps['tfidf']
    svm_coef = pipeline.named_steps['svm'].coef_[0]
    feature_names = tfidf.get_feature_names_out()

    top_phish = [(feature_names[i], round(float(svm_coef[i]), 4))
                 for i in svm_coef.argsort()[-20:][::-1]]
    top_legit = [(feature_names[i], round(float(svm_coef[i]), 4))
                 for i in svm_coef.argsort()[:20]]

    print("\nTop Phishing Indicators:", [w for w, _ in top_phish[:10]])
    print("Top Legit Indicators:   ", [w for w, _ in top_legit[:10]])

    # Save model
    with open('phishing_model.pkl', 'wb') as f:
        pickle.dump(pipeline, f)
    print("\n✅ Model saved to phishing_model.pkl")

    # Export feature weights for visualization
    weights = {
        "phishing_words": top_phish,
        "legit_words": top_legit,
        "cv_accuracy": round(float(cv_scores.mean()), 4),
        "test_accuracy": round(float(accuracy_score(y_test, y_pred)), 4)
    }
    with open('model_weights.json', 'w') as f:
        json.dump(weights, f, indent=2)
    print("✅ Feature weights saved to model_weights.json")

    return pipeline


if __name__ == '__main__':
    train()
