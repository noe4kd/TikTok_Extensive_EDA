import os
import sys
import matplotlib
matplotlib.use('agg')
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Add the root directory of the project to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.main import load_and_preprocess_data, create_new_features, prepare_data, plot_roc_curve, plot_confusion_matrix

def main():
    # Set output directory with an absolute path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "outcomes")
    os.makedirs(output_dir, exist_ok=True)

    # Load, preprocess, and prepare the data
    tiktok_data = load_and_preprocess_data()
    tiktok_data = create_new_features(tiktok_data)
    features = ['Subscribers count', 'Views avg.', 'Comments avg.', 'Shares avg.']
    tiktok_data['Engagement'] = (tiktok_data['Likes avg.'] > tiktok_data['Likes avg.'].median()).astype(int)
    X_train, X_test, y_train, y_test = prepare_data(tiktok_data, features)

    # Initialize models
    rf = RandomForestClassifier(random_state=42, max_depth=20, max_features='sqrt', min_samples_leaf=4, min_samples_split=2, n_estimators=300)
    svm = SVC(probability=True, random_state=42, C=1.0, kernel='linear')
    xgb = XGBClassifier(random_state=42, colsample_bytree=0.8, learning_rate=0.1, max_depth=3, n_estimators=100, subsample=0.7)
    stacking_clf = StackingClassifier(estimators=[('rf', rf), ('svm', svm), ('xgb', xgb)], final_estimator=LogisticRegression())

    # Train the stacking classifier and evaluate its performance
    stacking_clf.fit(X_train, y_train)
    y_pred_stacking = stacking_clf.predict(X_test)

    # Print classification report
    print("Stacking Classifier Report:\n")
    print(classification_report(y_test, y_pred_stacking))
    
    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred_stacking, "Stacking Classifier", save_path=output_dir)

    
    # Plot ROC curve
    plot_roc_curve({"Stacking Classifier": stacking_clf}, X_test, y_test, save_path=output_dir)

if __name__ == "__main__":
    main()
