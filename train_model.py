"""Tune on training posts only; evaluate once on the official test split."""
import argparse
import csv
import io
import json
from pathlib import Path
import shutil
import zipfile

import joblib
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

BASE = Path(__file__).resolve().parent


def read_split(archive, suffix):
    name = next(n for n in archive.namelist() if n.endswith(suffix))
    with archive.open(name) as stream:
        rows = list(csv.DictReader(io.TextIOWrapper(stream, encoding="utf-8-sig")))
    if any(not r['text'].strip() or r['label'] not in ('0', '1') for r in rows):
        raise ValueError('Dataset contains empty text or invalid labels')
    return rows


def metrics(model, texts, labels):
    pred = model.predict(texts)
    return {'accuracy': accuracy_score(labels, pred),
            'report': classification_report(labels, pred, output_dict=True, zero_division=0),
            'confusion_matrix': confusion_matrix(labels, pred, labels=[0, 1]).tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=BASE / 'data/dreaddit.zip')
    parser.add_argument('--jobs', type=int, default=1)
    args = parser.parse_args()
    with zipfile.ZipFile(args.data) as archive:
        train = read_split(archive, 'dreaddit-train.csv')
        test = read_split(archive, 'dreaddit-test.csv')
    if {r['post_id'] for r in train} & {r['post_id'] for r in test}:
        raise ValueError('Train/test post overlap')
    normalize = lambda s: ' '.join(s.lower().split())
    train_texts = {normalize(r['text']) for r in train}
    clean_test = [r for r in test if normalize(r['text']) not in train_texts]
    # Connect duplicate texts and post IDs so neither can leak across CV folds.
    parents = {}
    def root(x):
        parents.setdefault(x, x)
        if parents[x] != x:
            parents[x] = root(parents[x])
        return parents[x]
    for r in train:
        parents[root('text:' + normalize(r['text']))] = root('post:' + r['post_id'])
    groups = [root('post:' + r['post_id']) for r in train]
    x = np.array([r['text'] for r in train])
    y = np.array([int(r['label']) for r in train])
    pipeline = Pipeline([('tfidf', TfidfVectorizer(sublinear_tf=True)),
                         ('classifier', LogisticRegression(max_iter=2000, random_state=42))])
    grid = [
        {'tfidf__max_features': [5000, 20000], 'tfidf__ngram_range': [(1, 1), (1, 2)],
         'classifier__C': [1, 4], 'tfidf__min_df': [2]},
        {'classifier': [MultinomialNB()], 'classifier__alpha': [0.5, 1.0],
         'tfidf__max_features': [5000, 20000], 'tfidf__ngram_range': [(1, 2)]},
    ]
    search = GridSearchCV(pipeline, grid, scoring={'macro_f1': 'f1_macro', 'stress_f1': 'f1'},
                          refit='macro_f1', cv=StratifiedGroupKFold(5, shuffle=True, random_state=42),
                          n_jobs=args.jobs, error_score='raise')
    search.fit(x, y, groups=groups)
    model_path = BASE / 'model/stress_model.joblib'
    test_x = [r['text'] for r in test]
    test_y = [int(r['label']) for r in test]
    clean_x = [r['text'] for r in clean_test]
    clean_y = [int(r['label']) for r in clean_test]
    report = {'sklearn_version': sklearn.__version__, 'selection_metric': 'macro_f1',
              'train_rows': len(train), 'test_rows': len(test),
              'best_cv_macro_f1': search.best_score_,
              'best_params': {k: str(v) for k, v in search.best_params_.items()},
              'duplicate_test_rows': len(test) - len(clean_test),
              'optimized_test': metrics(search.best_estimator_, test_x, test_y),
              'optimized_deduplicated_test': metrics(search.best_estimator_, clean_x, clean_y)}
    if model_path.exists():
        backup = model_path.with_name('stress_model.original.joblib')
        if not backup.exists():
            shutil.copy2(model_path, backup)
        baseline = joblib.load(backup)
        report['previous_test'] = metrics(baseline, test_x, test_y)
        report['previous_deduplicated_test'] = metrics(baseline, clean_x, clean_y)
    # Selection is exclusively CV-based; test scores never choose a candidate.
    joblib.dump(search.best_estimator_, model_path)
    (BASE / 'model/metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
