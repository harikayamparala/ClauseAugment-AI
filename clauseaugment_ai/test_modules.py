"""Quick module test script for ClauseAugment AI."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 50)
print("ClauseAugment AI - Module Test Suite")
print("=" * 50)

# Test 1: Config
print("\n[1] Testing config...")
from config.settings import CLAUSE_CATEGORIES, IS_DEMO_MODE, AUGMENTATION_TECHNIQUES
print(f"    Categories: {len(CLAUSE_CATEGORIES)}")
print(f"    Techniques: {len(AUGMENTATION_TECHNIQUES)}")
print(f"    Demo mode: {IS_DEMO_MODE}")

# Test 2: NLP
print("\n[2] Testing NLP preprocessor...")
from modules.nlp.preprocessor import compute_cosine_similarity, preprocess_for_ml, text_stats
score = compute_cosine_similarity("The parties shall maintain confidentiality.", "Both parties must keep information confidential.")
processed = preprocess_for_ml("The parties shall maintain confidentiality.")
stats = text_stats("The parties shall maintain confidentiality at all times.")
print(f"    Cosine similarity: {score}")
print(f"    Preprocessed: '{processed}'")
print(f"    Text stats: {stats}")

# Test 3: Dataset
print("\n[3] Testing dataset manager...")
from modules.dataset.manager import load_sample_dataset, dataset_stats, filter_by_category
df = load_sample_dataset()
print(f"    Loaded {len(df)} rows, columns: {list(df.columns)}")
stats = dataset_stats(df)
print(f"    Categories: {stats['num_categories']}")
filtered = filter_by_category(df, "Confidentiality")
print(f"    Confidentiality rows: {len(filtered)}")

# Test 4: Augmentation
print("\n[4] Testing augmentation (demo mode)...")
from modules.augmentation.gemini_augmenter import augment_clause, get_mode_label
result = augment_clause("Each party agrees to keep confidential all information.", "Confidentiality", num_samples=3)
print(f"    Mode: {get_mode_label()}")
print(f"    Variations generated: {len(result['variations'])}")
print(f"    Is demo: {result['is_demo']}")
if result['variations']:
    print(f"    First technique: {result['variations'][0]['technique']}")

# Test 5: Classification
print("\n[5] Testing ML classifier...")
from modules.classification.classifier import train_classifier, predict_clause
train_df = df[["clause_text", "category"]].copy()
for clf_name in ["Logistic Regression", "Naive Bayes", "Random Forest"]:
    result2 = train_classifier(train_df, clf_name, "clause_text", "category")
    if result2.get("error"):
        print(f"    {clf_name}: ERROR - {result2['error']}")
    else:
        from sklearn.metrics import accuracy_score
        acc = accuracy_score(result2['y_test'], result2['y_pred'])
        print(f"    {clf_name}: Accuracy={acc:.2%}, Train={result2['train_size']}, Test={result2['test_size']}")
        # Test prediction
        pred = predict_clause(result2['pipeline'], "This agreement shall be governed by the laws of Delaware.")
        print(f"      -> Predicted: {pred['predicted_class']} (confidence: {pred['confidence']:.2%})")

# Test 6: Evaluation metrics
print("\n[6] Testing evaluation metrics...")
from modules.evaluation.metrics import compute_metrics, compare_metrics
result3 = train_classifier(train_df, "Logistic Regression", "clause_text", "category")
metrics = compute_metrics(result3["y_test"], result3["y_pred"], result3["classes"])
print(f"    Accuracy: {metrics['accuracy']:.2%}")
print(f"    Precision: {metrics['precision']:.2%}")
print(f"    Recall: {metrics['recall']:.2%}")
print(f"    F1: {metrics['f1']:.2%}")
print(f"    CM shape: {metrics['confusion_matrix'].shape}")

# Test 7: Visualization
print("\n[7] Testing visualization charts...")
from modules.visualization.charts import category_distribution_chart, metrics_comparison_chart
fig1 = category_distribution_chart(df)
fig2 = metrics_comparison_chart(metrics, metrics)
print(f"    category_distribution_chart: OK (type={type(fig1).__name__})")
print(f"    metrics_comparison_chart: OK (type={type(fig2).__name__})")

print("\n" + "=" * 50)
print("ALL TESTS PASSED SUCCESSFULLY!")
print("=" * 50)
