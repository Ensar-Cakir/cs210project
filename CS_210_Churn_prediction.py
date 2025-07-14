pip install faker

import pandas as pd
import random
from faker import Faker
from datetime import datetime, timedelta
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings("ignore")

faker = Faker()
random.seed(42)

def generate_interactions(num_contacts=200, num_days=60, num_events=3000):
    data = []
    contact_ids = [f'C{id:03}' for id in range(1, num_contacts + 1)]
    event_types = ['call', 'email', 'meeting']

    for _ in range(num_events):
        contact_id = random.choice(contact_ids)
        event_type = random.choice(event_types)
        timestamp = datetime.now() - timedelta(days=random.randint(0, num_days))
        notes = faker.sentence()
        data.append([contact_id, event_type, timestamp, notes])

    return pd.DataFrame(data, columns=['contact_id', 'event_type', 'timestamp', 'notes'])

interactions_df = generate_interactions()
interactions_df.to_csv('interactions.csv', index=False)

# Subscriptions
subs = pd.DataFrame({
    'contact_id': [f'C{id:03}' for id in range(1, 201)],
    'subscription_start': [datetime.now() - timedelta(days=random.randint(30, 180)) for _ in range(200)],
})
subs.to_csv('subscriptions.csv', index=False)

# Support tickets
support = []
for c in subs['contact_id']:
    for _ in range(random.randint(1, 5)):
        support.append({
            'contact_id': c,
            'ticket_date': datetime.now() - timedelta(days=random.randint(1, 30)),
            'text': faker.sentence(),
        })

support_df = pd.DataFrame(support)
support_df.to_csv('support_tickets.csv', index=False)

# Load
interactions = pd.read_csv('interactions.csv', parse_dates=['timestamp'])
subs = pd.read_csv('subscriptions.csv')
support = pd.read_csv('support_tickets.csv', parse_dates=['ticket_date'])

# Aggregate Features
cutoff = datetime.now() - timedelta(days=7)
recent = interactions[interactions['timestamp'] > cutoff]
recent_events = recent.groupby('contact_id').size().reset_index(name='recent_event_count')
total_events = interactions.groupby('contact_id').size().reset_index(name='total_event_count')
last_event = interactions.groupby('contact_id')['timestamp'].max().reset_index()
last_event['days_since_last_event'] = (datetime.now() - last_event['timestamp']).dt.days
ticket_counts = support.groupby('contact_id').size().reset_index(name='support_ticket_count')

# Merge
features = subs[['contact_id']].copy()
features = features.merge(recent_events, on='contact_id', how='left')
features = features.merge(total_events, on='contact_id', how='left')
features = features.merge(last_event[['contact_id', 'days_since_last_event']], on='contact_id', how='left')
features = features.merge(ticket_counts, on='contact_id', how='left')
features.fillna(0, inplace=True)

#  Updated label logic: easier to trigger churn
features['label'] = features.apply(
    lambda row: 1 if (row['recent_event_count'] <= 2 or row['days_since_last_event'] > 14) else 0,
    axis=1
)

#  Label Distribution
print("\n Label Distribution BEFORE balancing:")
print(features['label'].value_counts(normalize=True))

#  Safe Downsampling Logic
majority = features[features.label == 0]
minority = features[features.label == 1]

downsample_n = min(len(majority), len(minority) * 2)  # prevent sampling error
majority_downsampled = majority.sample(n=downsample_n, random_state=42)

balanced_features = pd.concat([majority_downsampled, minority])
balanced_features = balanced_features.sample(frac=1, random_state=42)

#  Check after balancing
print("\n Label Distribution AFTER balancing:")
print(balanced_features['label'].value_counts(normalize=True))

# Define features/target
X = balanced_features[['recent_event_count', 'support_ticket_count', 'total_event_count', 'days_since_last_event']]
y = balanced_features['label']

# Stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

# Train
model = XGBClassifier(eval_metric='logloss', use_label_encoder=False)
model.fit(X_train, y_train)

# Evaluate
probs = model.predict_proba(X_test)[:, 1]
print(f"\n ROC-AUC: {roc_auc_score(y_test, probs):.2f}")
print("\n Classification Report (Threshold=0.5):")
print(classification_report(y_test, model.predict(X_test)))

from sklearn.metrics import classification_report

threshold = 0.4
y_pred_thresh = (probs > threshold).astype(int)

print(f"\n🧾 Classification Report (Threshold={threshold}):")
print(classification_report(y_test, y_pred_thresh))

precision, recall, thresholds = precision_recall_curve(y_test, probs)
plt.plot(recall, precision, marker='.')
plt.title("Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.grid(True)
plt.show()

def score_new_event(event_row):
    contact = event_row['contact_id']
    recent_count = interactions[
        (interactions['contact_id'] == contact) &
        (interactions['timestamp'] > datetime.now() - timedelta(days=7))
    ].shape[0]

    total_count = interactions[interactions['contact_id'] == contact].shape[0]
    last_ts = interactions[interactions['contact_id'] == contact]['timestamp'].max()
    days_since_last = (datetime.now() - last_ts).days if pd.notnull(last_ts) else 99
    ticket_count = support_df[support_df['contact_id'] == contact].shape[0]

    input_features = pd.DataFrame([[recent_count, ticket_count, total_count, days_since_last]],
                                  columns=X.columns)

    prob = model.predict_proba(input_features)[0][1]
    print(f"\n [{contact}] Churn Probability: {prob:.2f}")
    if prob >= 0.7:
        print(" ALERT: High churn risk! Take action.")
    else:
        print(" Low churn risk.")

# Simulate for a random event
sample_event = interactions_df.sample(1).iloc[0]
score_new_event(sample_event)

plt.figure(figsize=(12, 4))

# Before
plt.subplot(1, 2, 1)
sns.countplot(x=features['label'])
plt.title('Label Distribution BEFORE Balancing')
plt.xlabel('Label (0 = Not Churned, 1 = Churned)')
plt.ylabel('Count')

# After
plt.subplot(1, 2, 2)
sns.countplot(x=balanced_features['label'])
plt.title('Label Distribution AFTER Balancing')
plt.xlabel('Label (0 = Not Churned, 1 = Churned)')
plt.ylabel('Count')

plt.tight_layout()
plt.show()

for col in ['recent_event_count', 'support_ticket_count', 'days_since_last_event']:
    plt.figure(figsize=(6, 4))
    sns.kdeplot(data=balanced_features, x=col, hue='label', fill=True, common_norm=False, alpha=0.6)
    plt.title(f'Distribution of {col} by Churn Label')
    plt.xlabel(col)
    plt.ylabel('Density')
    plt.legend(['Not Churned', 'Churned'])
    plt.savefig(f'{col}_distribution.png')
    plt.show()

# Ensure SHAP explainer works
explainer = shap.Explainer(model)
shap_values = explainer(X_test)

# SHAP beeswarm summary plot
shap.plots.beeswarm(shap_values)

precision, recall, thresholds = precision_recall_curve(y_test, probs)

plt.figure(figsize=(6, 4))
plt.plot(recall, precision, marker='.')
plt.title('Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.grid(True)
plt.show()

y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

grouped = balanced_features.groupby('label')[['recent_event_count', 'support_ticket_count', 'days_since_last_event', 'total_event_count']].mean().T
grouped.columns = ['Not Churned', 'Churned']
grouped.plot(kind='bar', figsize=(8, 5), rot=0)
plt.title('Mean Feature Values by Churn Class')
plt.ylabel('Mean Value')
plt.grid(axis='y')
plt.show()